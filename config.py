"""Configuration for the liveobject detection web app (Jetson + TensorRT)."""
from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = BASE_DIR / "snapshots"
MODELS_DIR = BASE_DIR / "models"

# Detection models for the runtime model-switch control.
# TensorRT FP16 engines built (by trtexec) from Ultralytics-exported ONNX.
# YOLO26 (m/l) is exported with end2end=False so it keeps the traditional
# (1, 84, 8400) head that trt_yolo.py decodes; YOLO12/YOLO11 stay listed for A/B
# comparison (all 80 COCO classes). Only engines that have actually been built
# are offered; default prefers the newest available.
_ALL_MODELS = {
    "yolo26m": str(MODELS_DIR / "yolo26m.engine"),
    "yolo26l": str(MODELS_DIR / "yolo26l.engine"),
    "yolo12m": str(MODELS_DIR / "yolo12m.engine"),
    "yolo11m": str(MODELS_DIR / "yolo11m.engine"),
}
MODELS = {k: v for k, v in _ALL_MODELS.items() if Path(v).exists()}
if not MODELS:                       # fall back so the app can still start
    MODELS = {"yolo11m": _ALL_MODELS["yolo11m"]}
DEFAULT_MODEL = next((k for k in ("yolo26m", "yolo26l", "yolo12m") if k in MODELS),
                     next(iter(MODELS)))

# Display (main) stream size in landscape orientation. Rotation is applied per
# frame at runtime (see DEFAULTS["rotation"] and the UI rotate control).
DISPLAY_SIZE = (1280, 720)

DEFAULTS = {
    "max_detections": 10,
    "threshold": 0.40,
    "rotation": 0,      # degrees clockwise (0/90/180/270); live-adjustable in UI
    "flip_h": False,
    "flip_v": False,
}

HOST = "0.0.0.0"
PORT = 8000
