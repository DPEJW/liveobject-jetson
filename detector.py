"""Camera + TensorRT detection worker (Jetson port).

Runs the Basler GigE camera (pypylon) and a TensorRT YOLO model in a background
thread, exposing the latest annotated JPEG frame plus structured detections and
timing for the web layer. This replaces the Pi's Picamera2 + Hailo-10H path; the
public DetectionWorker interface (used by app.py) is unchanged.
"""
from __future__ import annotations

import hashlib
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
from trt_yolo import TRTYolo


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
                   rotation=None, flip_h=None, flip_v=None):
        with self._lock:
            if max_detections is not None:
                self.max_detections = max(1, min(100, int(max_detections)))
            if threshold is not None:
                self.threshold = max(0.05, min(0.95, float(threshold)))
            if paused is not None:
                self.paused = bool(paused)
            if rotation is not None:
                self.rotation = (int(rotation) % 360) // 90 * 90
            if flip_h is not None:
                self.flip_h = bool(flip_h)
            if flip_v is not None:
                self.flip_v = bool(flip_v)
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
            self._pending_camera = True
        return {"camera_source": self.camera_source, "rtsp_stream": self.rtsp_stream}

    def request_snapshot(self):
        with self._lock:
            self._pending_snapshot = True
            self._last_snapshot = None

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
                    annotated = self._draw(frame, dets)

                    ok, buf = cv2.imencode(".jpg", annotated,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with self._cond:
                            self._jpeg = buf.tobytes()
                            self._frame_id += 1
                            self.detections = dets
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
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            color = _color_for(d["name"])
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            label = f'{d["name"]} {d["score"] * 100:.0f}%'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), color, -1)
            cv2.putText(out, label, (x0 + 2, max(10, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def _save_snapshot(self, frame):
        with self._lock:
            self._pending_snapshot = False
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SNAPSHOT_DIR / f"snap_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        self._last_snapshot = str(path)
