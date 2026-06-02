"""Configuration for the liveobject detection web app."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = BASE_DIR / "snapshots"

# Detection models available for the runtime model-switch control.
# All are Hailo-10H HEFs with 80 COCO classes and NMS baked in.
MODELS = {
    "yolov8m": "/usr/share/hailo-models/yolov8m_h10.hef",
    "yolov11m": "/usr/share/hailo-models/yolov11m_h10.hef",
}
DEFAULT_MODEL = "yolov8m"

# Display (main) stream size BEFORE rotation, in landscape orientation.
# With rotate=True the frame is turned 90 deg clockwise to correct this Pi's
# sideways camera mount, so the served image is portrait.
DISPLAY_SIZE = (1280, 720)

DEFAULTS = {
    "max_detections": 10,
    "threshold": 0.40,
    "rotate": True,
}

HOST = "0.0.0.0"
PORT = 8000
