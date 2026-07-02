"""Camera + TensorRT detection worker (Jetson port).

Runs the Basler GigE camera (pypylon) and a TensorRT YOLO model in a background
thread, exposing the latest annotated JPEG frame plus structured detections and
timing for the web layer. This replaces the Pi's Picamera2 + Hailo-10H path; the
public DetectionWorker interface (used by app.py) is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import config
from config import (DEFAULTS, DEFAULT_MODEL, DISPLAY_SIZE, MODELS,
                    SNAPSHOT_DIR)
from labels import COCO_LABELS
from capture import should_capture, to_yolo_label
from reid import Roster
from tracking import IoUTracker, TrackSession, ZoneRegistry, hit_test
from trt_yolo import TRTYolo

# COCO classes used as auto "named places" for dwell tracking
ZONE_CLASSES = {"chair", "couch", "bed", "dining table"}


def _color_for(name):
    """Deterministic, readable BGR color per class name (stable across runs)."""
    digest = hashlib.md5(name.encode()).digest()
    return (60 + digest[0] % 180, 60 + digest[1] % 180, 60 + digest[2] % 180)


def letterbox(frame, size):
    """Resize `frame` (BGR ndarray) to fit `size`=(width, height) while preserving
    aspect ratio, padding the remainder with black. Returns an array of exactly
    (height, width, 3). Keeps non-16:9 cameras (e.g. a 4:3 Reolink) from being
    horizontally squished into the display frame."""
    tw, th = size
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((th, tw, 3), dtype=frame.dtype)
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


class CameraSource:
    """Frame-source interface. `read()` returns a BGR ndarray sized to
    DISPLAY_SIZE (or None for a transient miss); `close()` releases the device."""

    backend = ""

    def read(self):
        raise NotImplementedError

    def close(self):
        pass


class BaslerSource(CameraSource):
    """Basler GigE Vision camera via pypylon (Mono8 -> BGR). Original capture path."""

    backend = "pypylon"

    def __init__(self):
        from pypylon import pylon
        self._pylon = pylon
        cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        cam.Open()
        try:
            cam.PixelFormat.Value = "Mono8"
        except Exception:
            pass
        # keep exposure short so the frame rate stays high for live detection
        try:
            cam.ExposureAuto.Value = "Continuous"
            cam.AutoExposureTimeUpperLimit.Value = 20000.0
            cam.GainAuto.Value = "Continuous"
            cam.AcquisitionFrameRateEnable.Value = True
            cam.AcquisitionFrameRate.Value = 30.0
        except Exception:
            pass
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._cam = cam

    def read(self):
        res = self._cam.RetrieveResult(2000, self._pylon.TimeoutHandling_Return)
        if res is None or not res.GrabSucceeded():
            if res is not None:
                res.Release()
            return None
        mono = np.array(res.Array, copy=True)
        res.Release()
        bgr = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        return cv2.resize(bgr, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)

    def close(self):
        try:
            self._cam.StopGrabbing()
            self._cam.Close()
        except Exception:
            pass


class RtspSource(CameraSource):
    """RTSP/IP camera. Prefers a GStreamer pipeline using the Jetson hardware
    decoder (NVDEC); falls back to OpenCV's FFMPEG backend (CPU).

    A background thread pulls frames so the detection loop's read() never blocks:
    OpenCV's read() on a stalled appsink blocks indefinitely (e.g. when the camera
    hasn't released a prior session during a fast stream switch), which would hang
    the worker. Here the worker only ever reads the latest cached frame; if it goes
    stale, read() returns None so the worker rebuilds. Frames are letterboxed."""

    STALE_AFTER = 3.0            # seconds without a fresh frame -> report a miss
    FIRST_FRAME_TIMEOUT = 8.0    # give up opening if no frame arrives in time

    def __init__(self, url):
        self._url = url
        self.backend = "none"
        self._cap = self._open()
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(f"could not open RTSP stream: {self._safe(url)}")
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._stop = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        deadline = time.time() + self.FIRST_FRAME_TIMEOUT
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return
            time.sleep(0.05)
        self.close()
        raise RuntimeError(f"no frames within "
                           f"{self.FIRST_FRAME_TIMEOUT:.0f}s: {self._safe(url)}")

    @staticmethod
    def _safe(url):
        return re.sub(r"://[^@/]+@", "://***@", url)  # strip creds for logs

    def _gst_pipeline(self):
        # decodebin auto-plugs nvv4l2decoder for H.264/H.265 on Jetson; nvvidconv
        # moves NVMM -> system memory; drop/max-buffers=1 keeps only the freshest
        # frame. tcp-timeout bounds a dead connection so read() can't block forever.
        return (
            f'rtspsrc location="{self._url}" protocols=tcp latency=100 '
            'tcp-timeout=5000000 timeout=5000000 '
            '! rtpjitterbuffer ! decodebin ! nvvidconv '
            '! video/x-raw,format=BGRx ! videoconvert '
            '! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'
        )

    def _open(self):
        try:
            cap = cv2.VideoCapture(self._gst_pipeline(), cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self.backend = "gstreamer-nvdec"
                return cap
            cap.release()
        except Exception as e:
            print(f"[rtsp] gstreamer open failed ({e}); falling back to ffmpeg",
                  file=sys.stderr)
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                              "rtsp_transport;tcp|stimeout;5000000")
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self.backend = "ffmpeg-cpu"
            return cap
        return None

    def _read_loop(self):
        while not self._stop:
            try:
                ok, frame = self._cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                if self._stop:
                    break
                time.sleep(0.02)
                continue
            with self._lock:
                self._frame = frame
                self._stamp = time.time()

    def read(self):
        with self._lock:
            frame, stamp = self._frame, self._stamp
        if frame is None or (time.time() - stamp) > self.STALE_AFTER:
            return None
        return letterbox(frame, DISPLAY_SIZE)

    def close(self):
        self._stop = True
        try:
            self._reader.join(timeout=6.0)  # returns within tcp-timeout on a dead link
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass


def make_camera_source(source=None, stream=None):
    """Build a camera source. `source`: 'rtsp' | 'basler' (defaults to config);
    `stream`: 'main' | 'sub' for RTSP (defaults to config)."""
    src = source or config.CAMERA_SOURCE
    if src == "basler":
        return BaslerSource()
    if src == "rtsp":
        return RtspSource(config.rtsp_url(stream=stream))
    raise ValueError(f"unknown camera source {src!r} (expected 'rtsp' or 'basler')")


class DetectionWorker:
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._running = False
        self._thread = None

        self.model_key = DEFAULT_MODEL
        self.max_detections = DEFAULTS["max_detections"]
        self.threshold = DEFAULTS["threshold"]
        self.rotation = DEFAULTS["rotation"]
        self.flip_h = DEFAULTS["flip_h"]
        self.flip_v = DEFAULTS["flip_v"]
        self.paused = False

        self.camera_source = config.CAMERA_SOURCE   # 'rtsp' | 'basler'
        self.rtsp_stream = config.RTSP_STREAM        # 'main' | 'sub'
        self.backend = ""                            # active decode backend label

        self._pending_model = None
        self._pending_camera = False
        self._pending_snapshot = False
        self._last_snapshot = None

        self._jpeg = None
        self._frame_id = 0
        self.detections = []
        self.infer_ms = 0.0
        self._times = deque(maxlen=90)

        # ---- tracking state (mutated only by the worker thread) ----
        self.tracker = IoUTracker()
        self.zones = ZoneRegistry()
        self._selected_id = None
        self._session = None
        self._last_tracks = []
        self._track_snapshot = {"tracks": [], "track": None, "zones": []}
        self._fw, self._fh = DISPLAY_SIZE
        self._t_prev = None
        self._pending_select = None     # None | {"id":int} | {"x":float,"y":float}
        self._pending_stop = False
        self._pending_zone_ops = []
        self.show_trail = True
        self.show_heatmap = True
        self.show_zones = True

        # ---- cat re-identification (appearance) ----
        self.roster = Roster(threshold=0.5)
        self._pending_enroll = None     # None | {"name":str,"x":float,"y":float}
        self._enrolling = None          # {"name","tid","left"} while averaging a signature

        # ---- training-set capture (only for manually-named cats) ----
        self.capture_enabled = True
        self.capture_conf = 0.80
        self.dataset_dir = os.path.expanduser("~/catdata")
        self._img_dir = os.path.join(self.dataset_dir, "images")
        self._lbl_dir = os.path.join(self.dataset_dir, "labels")
        os.makedirs(self._img_dir, exist_ok=True)
        os.makedirs(self._lbl_dir, exist_ok=True)
        self._class_map = {}            # name -> class idx (stable)
        self._dataset_counts = {}       # name -> labeled instances saved
        self._last_capture_ts = None
        self._roster_path = os.path.join(self.dataset_dir, "roster.json")
        self.roster.load(self._roster_path)   # survive restarts

        # ---- on-Orin training job ----
        self._train_proc = None
        self._paused_for_train = False

        SNAPSHOT_DIR.mkdir(exist_ok=True)

    # ---- lifecycle ----
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    # ---- config API (called from the Flask thread) ----
    def set_config(self, max_detections=None, threshold=None, paused=None,
                   rotation=None, flip_h=None, flip_v=None,
                   track_trail=None, track_heatmap=None, track_zones=None):
        with self._lock:
            if max_detections is not None:
                self.max_detections = max(1, min(100, int(max_detections)))
            if threshold is not None:
                self.threshold = max(0.05, min(0.95, float(threshold)))
            if paused is not None:
                self.paused = bool(paused)
            if rotation is not None:
                new_rot = (int(rotation) % 360) // 90 * 90
                if new_rot != self.rotation:
                    self.rotation = new_rot
                    self._reset_tracking()        # geometry changed -> zones invalid
            if flip_h is not None:
                self.flip_h = bool(flip_h)
            if flip_v is not None:
                self.flip_v = bool(flip_v)
            if track_trail is not None:
                self.show_trail = bool(track_trail)
            if track_heatmap is not None:
                self.show_heatmap = bool(track_heatmap)
            if track_zones is not None:
                self.show_zones = bool(track_zones)
        return self.config()

    def request_model(self, model_key):
        if model_key not in MODELS:
            raise ValueError(f"unknown model: {model_key}")
        with self._lock:
            self._pending_model = model_key
        return model_key

    def request_camera(self, source=None, stream=None):
        """Switch capture backend ('rtsp'|'basler') and/or RTSP stream
        ('main'|'sub') at runtime; the worker reconnects on the next loop."""
        with self._lock:
            if source is not None:
                s = str(source).strip().lower()
                if s in ("rtsp", "basler"):
                    self.camera_source = s
            if stream is not None:
                st = str(stream).strip().lower()
                if st in ("main", "sub"):
                    self.rtsp_stream = st
            self._reset_tracking()                 # new scene -> drop session + zones
            self._pending_camera = True
        return {"camera_source": self.camera_source, "rtsp_stream": self.rtsp_stream}

    def request_snapshot(self):
        with self._lock:
            self._pending_snapshot = True
            self._last_snapshot = None

    # ---- tracking control (Flask thread queues; worker applies) ----
    def request_select(self, track_id=None, x=None, y=None):
        with self._lock:
            if track_id is not None:
                self._pending_select = {"id": int(track_id)}
            elif x is not None and y is not None:
                self._pending_select = {"x": float(x), "y": float(y)}
            else:
                return {"selected": None}
        return {"queued": True}

    def request_stop_tracking(self):
        with self._lock:
            self._pending_stop = True
        return {"queued": True}

    def zone_add(self, label, box):
        with self._lock:
            self._pending_zone_ops.append(("add", str(label), [int(v) for v in box]))
        return {"queued": True}

    def zone_rename(self, zid, label):
        with self._lock:
            self._pending_zone_ops.append(("rename", int(zid), str(label)))
        return {"queued": True}

    def zone_delete(self, zid):
        with self._lock:
            self._pending_zone_ops.append(("delete", int(zid)))
        return {"queued": True}

    def tracking_state(self):
        return self._track_snapshot      # atomically swapped reference; read-only

    # ---- worker-thread helpers (caller holds self._lock) ----
    def enroll_cat(self, name, x, y):
        with self._lock:
            nm = str(name).strip()[:24] or "cat"
            self._pending_enroll = {"name": nm, "x": float(x), "y": float(y)}
        return {"queued": True}

    def rename_cat(self, old, new):
        nm = str(new).strip()[:24]
        with self._lock:
            self.roster.rename(str(old), nm)
            if self._enrolling and self._enrolling["name"] == old:
                self._enrolling["name"] = nm
            self.roster.save(self._roster_path)
        return {"ok": True}

    def clear_cat(self, name):
        with self._lock:
            self.roster.clear(str(name))
            if self._enrolling and self._enrolling["name"] == name:
                self._enrolling = None
            self.roster.save(self._roster_path)
        return {"ok": True}

    # ---- training control ----
    def _launch_train(self, epochs):
        import subprocess
        vpy = os.path.expanduser("~/venvs/train/bin/python")
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_runner.py")
        ob = os.path.expanduser("~/oblas/root/usr/lib/aarch64-linux-gnu")
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{ob}/openblas-pthread:{ob}:" + env.get("LD_LIBRARY_PATH", "")
        log = open(os.path.join(self.dataset_dir, "train.log"), "w")
        return subprocess.Popen([vpy, script, str(int(epochs))], cwd=self.dataset_dir,
                                stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, env=env)

    def start_training(self, epochs=30):
        with self._lock:
            if self._train_proc is not None and self._train_proc.poll() is None:
                return {"error": "already training"}
            self._paused_for_train = True          # frees the GPU for the trainer
            self._train_proc = self._launch_train(epochs)
        return {"started": True, "epochs": int(epochs)}

    def cancel_training(self):
        with self._lock:
            p = self._train_proc
        if p is not None and p.poll() is None:
            p.terminate()
        self._paused_for_train = False
        return {"cancelled": True}

    def training_status(self):
        running = self._train_proc is not None and self._train_proc.poll() is None
        st = {}
        try:
            with open(os.path.join(self.dataset_dir, "train_status.json")) as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            pass
        st["running"] = running
        return st

    @staticmethod
    def _hs_signature(frame, box):
        """H-S color histogram of the box interior (center-cropped) -> L1-normed vector."""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in box]
        mx, my = int((x1 - x0) * 0.2), int((y1 - y0) * 0.2)
        cx0, cy0 = max(0, x0 + mx), max(0, y0 + my)
        cx1, cy1 = min(w, x1 - mx), min(h, y1 - my)
        if cx1 - cx0 < 3 or cy1 - cy0 < 3:
            return None
        hsv = cv2.cvtColor(frame[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
        hist = hist.flatten().astype(np.float32)
        s = float(hist.sum())
        return hist / s if s > 0 else None

    def _reid_cats(self, frame, tracked):
        """Name cat detections by matching appearance to the enrolled roster;
        reinforce the signature during an active enrollment window."""
        cats = [(i, self._hs_signature(frame, tracked[i]["box"]))
                for i, d in enumerate(tracked) if d["name"] == "cat"]
        cats = [(i, s) for i, s in cats if s is not None]
        if self.roster.names() and cats:
            assigned = self.roster.assign([s for _, s in cats])
            for (i, _), name in zip(cats, assigned):
                if name:
                    tracked[i]["label"] = name
                    tracked[i]["reid"] = name
        if self._enrolling:
            e = self._enrolling
            t = self.tracker.get(e["tid"])
            if t is not None and t.cls == "cat":
                sig = self._hs_signature(frame, t.box)
                if sig is not None:
                    self.roster.reinforce(e["name"], sig)
            e["left"] -= 1
            if e["left"] <= 0:
                self._enrolling = None
                self.roster.save(self._roster_path)

    def _maybe_capture(self, frame, tracked, now):
        """Save a labeled frame when a NAMED cat is confidently detected. Only
        manually-enrolled cats are ever collected; frame-level time-spacing dedup."""
        if not self.capture_enabled:
            return
        named = [d for d in tracked
                 if d.get("reid") and d["score"] >= self.capture_conf]
        if not named or not should_capture(now, self._last_capture_ts, 0.6):
            return
        h, w = frame.shape[:2]
        lines = []
        for d in named:
            name = d["reid"]
            if name not in self._class_map:
                self._class_map[name] = len(self._class_map)
            lines.append(to_yolo_label(d["box"], w, h, self._class_map[name]))
            self._dataset_counts[name] = self._dataset_counts.get(name, 0) + 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(os.path.join(self._img_dir, f"cap_{ts}.jpg"), frame)
        with open(os.path.join(self._lbl_dir, f"cap_{ts}.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        with open(os.path.join(self.dataset_dir, "classes.json"), "w") as fh:
            json.dump(self._class_map, fh)
        self._last_capture_ts = now

    def _reset_tracking(self):
        """Scene changed (camera/rotation): drop session + zones (keep the cat roster)."""
        self.tracker = IoUTracker()
        self.zones = ZoneRegistry()
        self._selected_id = None
        self._session = None
        self._last_tracks = []
        self._pending_select = None
        self._pending_stop = False
        self._pending_zone_ops = []
        self._pending_enroll = None
        self._enrolling = None
        self._track_snapshot = {"tracks": [], "track": None, "zones": []}

    def _apply_pending(self, tracked, frame):
        for op in self._pending_zone_ops:
            if op[0] == "add":
                self.zones.add(op[1], op[2])
            elif op[0] == "rename":
                self.zones.rename(op[1], op[2])
            elif op[0] == "delete":
                self.zones.delete(op[1])
        self._pending_zone_ops = []
        if self._pending_stop:
            if self._session is not None:
                self._session.stop()
            self._selected_id = None
            self._pending_stop = False
        if self._pending_select is not None:
            sel, self._pending_select = self._pending_select, None
            tid = sel.get("id")
            if tid is None:
                tid = hit_test(tracked, sel["x"], sel["y"], self._fw, self._fh)
            if tid is not None:
                t = self.tracker.get(int(tid))
                self._selected_id = int(tid)
                self._session = TrackSession(int(tid), t.label if t else f"#{tid}",
                                             self._fw, self._fh)
                self._t_prev = None
        if self._pending_enroll is not None:
            req, self._pending_enroll = self._pending_enroll, None
            tid = hit_test(tracked, req["x"], req["y"], self._fw, self._fh)
            if tid is not None:
                t = self.tracker.get(int(tid))
                if t is not None and t.cls == "cat":
                    sig = self._hs_signature(frame, t.box)
                    if sig is not None:
                        self.roster.enroll(req["name"], sig)
                        self.roster.save(self._roster_path)
                        self._enrolling = {"name": req["name"], "tid": int(tid),
                                           "left": 12}

    def _build_snapshot(self, tracked):
        tracks = [{"id": d["id"], "label": d["label"], "cls": d["name"],
                   "score": round(d["score"], 2), "box": [int(v) for v in d["box"]],
                   "selected": d["id"] == self._selected_id} for d in tracked]
        summary = self._session.summary() if self._session is not None else None
        return {"tracks": tracks, "track": summary, "zones": self.zones.list(),
                "cats": self.roster.names(),
                "enrolling": self._enrolling["name"] if self._enrolling else None,
                "dataset": dict(self._dataset_counts)}

    def config(self):
        return {
            "model": self.model_key,
            "models": list(MODELS.keys()),
            "max_detections": self.max_detections,
            "threshold": round(self.threshold, 2),
            "paused": self.paused,
            "rotation": self.rotation,
            "flip_h": self.flip_h,
            "flip_v": self.flip_v,
            "camera_source": self.camera_source,
            "camera_sources": ["rtsp", "basler"],
            "rtsp_stream": self.rtsp_stream,
            "rtsp_streams": ["main", "sub"],
            "backend": self.backend,
            "track_trail": self.show_trail,
            "track_heatmap": self.show_heatmap,
            "track_zones": self.show_zones,
        }

    def fps(self):
        now = time.perf_counter()
        recent = [t for t in self._times if now - t <= 2.0]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    def snapshot_path(self):
        return self._last_snapshot

    # ---- MJPEG stream ----
    def frames(self):
        last_id = -1
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._frame_id != last_id, timeout=5)
                last_id = self._frame_id
                jpeg = self._jpeg
            if jpeg is not None:
                yield jpeg

    # ---- worker thread ----
    def _run(self):
        model = TRTYolo(MODELS[self.model_key])    # GPU engine: load once
        while self._running:                       # camera: reconnect on error
            source = None
            try:
                source = make_camera_source(self.camera_source, self.rtsp_stream)
                self.backend = source.backend
                print(f"[detector] camera source={self.camera_source}"
                      + (f"/{self.rtsp_stream}" if self.camera_source == "rtsp" else "")
                      + (f" backend={self.backend}" if self.backend else ""),
                      file=sys.stderr)
                while self._running and not self._pending_camera:
                    if self._pending_model:
                        with self._lock:
                            key = self._pending_model
                            self._pending_model = None
                        model.reload(MODELS[key])
                        self.model_key = key

                    if self.paused:
                        time.sleep(0.05)
                        continue

                    if self._paused_for_train:
                        p = self._train_proc
                        if p is None or p.poll() is not None:
                            self._paused_for_train = False   # trainer finished -> resume
                        else:
                            time.sleep(0.3)
                            continue

                    frame = source.read()
                    if frame is None:
                        continue

                    k = (-(self.rotation // 90)) % 4   # degrees CW -> np.rot90 steps
                    if k:
                        frame = np.ascontiguousarray(np.rot90(frame, k))
                    if self.flip_h:
                        frame = np.ascontiguousarray(np.fliplr(frame))
                    if self.flip_v:
                        frame = np.ascontiguousarray(np.flipud(frame))

                    t0 = time.perf_counter()
                    raw = model.infer(frame, conf=self.threshold)
                    self.infer_ms = (time.perf_counter() - t0) * 1000.0

                    dets = self._to_dets(raw)
                    self._fh, self._fw = frame.shape[0], frame.shape[1]
                    now = time.perf_counter()
                    dt = (now - self._t_prev) if self._t_prev else 0.0
                    self._t_prev = now

                    with self._lock:
                        tracked = self.tracker.update(dets)
                        self._last_tracks = tracked
                        self._apply_pending(tracked, frame)
                        self._reid_cats(frame, tracked)
                        self._maybe_capture(frame, tracked, now)
                        self.zones.update_auto([d for d in tracked
                                                if d["name"] in ZONE_CLASSES])
                        if (self._selected_id is not None and self._session is not None
                                and self._session.state != "stopped"):
                            sel = self.tracker.get(self._selected_id)
                            if sel is not None:
                                self._session.update(sel.box, dt, self.zones.list())
                            else:
                                self._session.mark_lost()
                        self._track_snapshot = self._build_snapshot(tracked)

                    annotated = self._draw(frame, tracked)

                    ok, buf = cv2.imencode(".jpg", annotated,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with self._cond:
                            self._jpeg = buf.tobytes()
                            self._frame_id += 1
                            self.detections = tracked
                            self._cond.notify_all()

                    if self._pending_snapshot:
                        self._save_snapshot(annotated)

                    self._times.append(time.perf_counter())

                # left inner loop: stopping, or a camera switch was requested
                self._pending_camera = False
                self.backend = ""
                if source is not None:
                    source.close()
                    source = None
                if self._running:
                    time.sleep(1.0)  # let the camera release the prior RTSP session
            except Exception as e:
                print(f"[detector] camera/run error: {e}; retrying in 2s",
                      file=sys.stderr)
                self.backend = ""
                try:
                    if source is not None:
                        source.close()
                except Exception:
                    pass
                time.sleep(2)

    def _to_dets(self, raw):
        """raw: list of (class_id, score, [x0,y0,x1,y1]) -> sorted, capped dicts."""
        results = []
        for class_id, score, box in raw:
            name = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else str(class_id)
            results.append({"name": name, "score": score, "box": box})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:self.max_detections]

    def _draw(self, frame, dets):
        out = frame.copy()
        sess = self._session            # capture once (worker thread; avoids None-race)
        sel = self._selected_id
        cyan = (198, 214, 79)           # BGR for UI #4fd6c6
        if self.show_heatmap and sess is not None:
            out = self._render_heatmap(out, sess.heat)
        if self.show_zones:
            for z in self.zones.list():
                x0, y0, x1, y1 = z["box"]
                cv2.rectangle(out, (x0, y0), (x1, y1), (90, 200, 255), 1)
                cv2.putText(out, z["label"], (x0 + 3, y0 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 200, 255), 1, cv2.LINE_AA)
        if self.show_trail and sess is not None and len(sess.trail) > 1:
            pts = np.array(sess.trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, cyan, 2, cv2.LINE_AA)
        for d in dets:
            x0, y0, x1, y1 = [int(v) for v in d["box"]]
            is_sel = d.get("id") == sel
            color = cyan if is_sel else _color_for(d["name"])
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 3 if is_sel else 2)
            label = f'{d.get("label", d["name"])} {d["score"] * 100:.0f}%'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), color, -1)
            cv2.putText(out, label, (x0 + 2, max(10, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def _render_heatmap(self, frame, heat, alpha=0.45):
        m = float(heat.max())
        if m <= 0:
            return frame
        small = (np.clip(heat / m, 0, 1) * 255).astype(np.uint8)
        big = cv2.resize(small, (frame.shape[1], frame.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
        cmap = cv2.applyColorMap(big, cv2.COLORMAP_TURBO)
        mask = (big > 8)[:, :, None]
        return np.where(mask, cv2.addWeighted(frame, 1 - alpha, cmap, alpha, 0), frame)

    def _save_snapshot(self, frame):
        with self._lock:
            self._pending_snapshot = False
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SNAPSHOT_DIR / f"snap_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        self._last_snapshot = str(path)
