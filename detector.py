"""Camera + Hailo-10H detection worker.

Runs Picamera2 and a Hailo YOLO model in a background thread, exposing the
latest annotated JPEG frame plus structured detections and timing for the web
layer. All Hailo calls stay on the worker thread; the Flask thread only reads
shared state and posts simple config/model/snapshot requests.
"""
import hashlib
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import Hailo

from config import (DEFAULTS, DEFAULT_MODEL, DISPLAY_SIZE, MODELS,
                    SNAPSHOT_DIR)
from labels import COCO_LABELS


def _color_for(name):
    """Deterministic, readable BGR color per class name (stable across runs)."""
    digest = hashlib.md5(name.encode()).digest()
    return (60 + digest[0] % 180, 60 + digest[1] % 180, 60 + digest[2] % 180)


class DetectionWorker:
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._running = False
        self._thread = None

        # Live config (writes guarded by _lock; scalar reads are atomic enough).
        self.model_key = DEFAULT_MODEL
        self.max_detections = DEFAULTS["max_detections"]
        self.threshold = DEFAULTS["threshold"]
        self.rotation = DEFAULTS["rotation"]   # degrees clockwise
        self.paused = False

        # Requests handled on the worker thread.
        self._pending_model = None
        self._pending_snapshot = False
        self._last_snapshot = None

        # Shared outputs.
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
                   rotation=None):
        with self._lock:
            if max_detections is not None:
                self.max_detections = max(1, min(100, int(max_detections)))
            if threshold is not None:
                self.threshold = max(0.05, min(0.95, float(threshold)))
            if paused is not None:
                self.paused = bool(paused)
            if rotation is not None:
                self.rotation = (int(rotation) % 360) // 90 * 90  # snap 0/90/180/270
        return self.config()

    def request_model(self, model_key):
        if model_key not in MODELS:
            raise ValueError(f"unknown model: {model_key}")
        with self._lock:
            self._pending_model = model_key
        return model_key

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
        """Yield the latest JPEG bytes as new frames are produced."""
        last_id = -1
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._frame_id != last_id, timeout=5)
                last_id = self._frame_id
                jpeg = self._jpeg
            if jpeg is not None:
                yield jpeg

    # ---- worker thread internals ----
    def _enable_autofocus(self, picam):
        """Continuous autofocus on the IMX708 so the model sees sharp frames."""
        try:
            from libcamera import controls as libcontrols
            picam.set_controls({
                "AfMode": libcontrols.AfModeEnum.Continuous,
                "AfSpeed": libcontrols.AfSpeedEnum.Fast,
            })
        except Exception as exc:
            print(f"autofocus unavailable: {exc}")

    def _load_model(self, key):
        hailo = Hailo(MODELS[key])
        h, w, _ = hailo.get_input_shape()
        return hailo, (w, h)

    def _run(self):
        picam = Picamera2()
        hailo, (mw, mh) = self._load_model(self.model_key)
        cfg = picam.create_preview_configuration(
            main={"size": DISPLAY_SIZE, "format": "RGB888"},
            lores={"size": (mw, mh), "format": "RGB888"},
            controls={"FrameRate": 30},
        )
        picam.configure(cfg)
        picam.start()
        self._enable_autofocus(picam)
        try:
            while self._running:
                if self._pending_model:
                    hailo, (mw, mh) = self._swap_model(picam, cfg, hailo)

                if self.paused:
                    time.sleep(0.05)
                    continue

                req = picam.capture_request()
                main = req.make_array("main")    # BGR bytes (picamera2 "RGB888")
                lores = req.make_array("lores")
                req.release()

                k = (-(self.rotation // 90)) % 4   # degrees CW -> np.rot90 steps
                if k:
                    main = np.rot90(main, k)
                    lores = np.rot90(lores, k)

                # Model expects RGB; picamera2 gives BGR -> reverse channels.
                model_in = np.ascontiguousarray(lores[:, :, ::-1])

                t0 = time.perf_counter()
                raw = hailo.run(model_in)
                self.infer_ms = (time.perf_counter() - t0) * 1000.0

                dets = self._parse(raw, main.shape[1], main.shape[0])
                annotated = self._draw(main, dets)

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
        finally:
            picam.stop()
            hailo.close()

    def _swap_model(self, picam, cfg, hailo):
        with self._lock:
            key = self._pending_model
            self._pending_model = None
        hailo.close()
        new_hailo, (mw, mh) = self._load_model(key)
        self.model_key = key
        if tuple(cfg["lores"]["size"]) != (mw, mh):
            picam.stop()
            cfg["lores"]["size"] = (mw, mh)
            picam.configure(cfg)
            picam.start()
        return new_hailo, (mw, mh)

    def _parse(self, raw, w, h):
        thr = self.threshold
        cap = self.max_detections
        results = []
        for class_id, dets in enumerate(raw):
            arr = np.asarray(dets)
            if arr.size == 0:
                continue
            for d in arr:
                score = float(d[4])
                if score < thr:
                    continue
                y0, x0, y1, x1 = float(d[0]), float(d[1]), float(d[2]), float(d[3])
                name = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else str(class_id)
                results.append({
                    "name": name,
                    "score": score,
                    "box": [int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)],
                })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:cap]

    def _draw(self, frame, dets):
        out = frame.copy()  # copy makes the rot90 view contiguous & writable
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
