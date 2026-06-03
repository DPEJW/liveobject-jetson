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

# Display (main) stream size in landscape orientation. Rotation is applied per
# frame at runtime (see DEFAULTS["rotation"] and the UI rotate control).
DISPLAY_SIZE = (1280, 720)

DEFAULTS = {
    "max_detections": 10,
    "threshold": 0.40,
    # Frame rotation in degrees clockwise (0/90/180/270). 0 and 180 are
    # landscape; 90 and 270 are portrait. Adjustable live from the UI.
    "rotation": 0,
    # Mirror/flip the frame (applied after rotation). Adjustable live from the UI.
    "flip_h": False,   # horizontal mirror (left-right)
    "flip_v": False,   # vertical flip (up-down)
}

HOST = "0.0.0.0"
PORT = 8000
