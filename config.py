"""Configuration for the liveobject detection web app (Jetson + TensorRT)."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

BASE_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = BASE_DIR / "snapshots"
MODELS_DIR = BASE_DIR / "models"


def _load_env_file(path):
    """Minimal KEY=VALUE .env loader so camera credentials can live in a
    gitignored file instead of the repo or the systemd unit. Real environment
    variables win (setdefault), so `CAMERA_SOURCE=basler python3 app.py` still
    overrides the file."""
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_env_file(BASE_DIR / ".camera.env")

# Detection models for the runtime model-switch control.
# TensorRT FP16 engines built (by trtexec) from Ultralytics-exported ONNX.
# YOLO26 (m/l) is exported with end2end=False so it keeps the traditional
# (1, 84, 8400) head that trt_yolo.py decodes; YOLO12/YOLO11 stay listed for A/B
# comparison (all 80 COCO classes). Only engines that have actually been built
# are offered; default prefers the newest available.
_ALL_MODELS = {
    "yolo26x-hd": str(MODELS_DIR / "yolo26x_hd.engine"),  # 26x @ 1280 — best accuracy
    "yolo26x": str(MODELS_DIR / "yolo26x.engine"),
    "yolo26l": str(MODELS_DIR / "yolo26l.engine"),
    "yolo26m": str(MODELS_DIR / "yolo26m.engine"),
    "yolo12m": str(MODELS_DIR / "yolo12m.engine"),
    "yolo11m": str(MODELS_DIR / "yolo11m.engine"),
}
MODELS = {k: v for k, v in _ALL_MODELS.items() if Path(v).exists()}
if not MODELS:                       # fall back so the app can still start
    MODELS = {"yolo11m": _ALL_MODELS["yolo11m"]}
DEFAULT_MODEL = next((k for k in ("yolo26x-hd", "yolo26x", "yolo26l", "yolo26m",
                                  "yolo12m") if k in MODELS),
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

# ---- Camera source selection -------------------------------------------------
# "rtsp"   -> IP camera over RTSP (e.g. Reolink PoE), HW-decoded via GStreamer NVDEC
# "basler" -> Basler GigE Vision via pypylon (the original Jetson capture path)
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "rtsp").strip().lower()

# RTSP / IP-camera settings (used when CAMERA_SOURCE == "rtsp"). Credentials are
# normally supplied via .camera.env (see .camera.env.example).
RTSP_HOST = os.environ.get("RTSP_HOST", "192.168.1.104").strip()
RTSP_PORT = int(os.environ.get("RTSP_PORT", "554") or "554")
RTSP_USER = os.environ.get("RTSP_USER", "admin").strip()
RTSP_PASS = os.environ.get("RTSP_PASS", "")
RTSP_STREAM = os.environ.get("RTSP_STREAM", "main").strip().lower()  # main | sub


def build_rtsp_url(host, port, user, password, stream):
    """Construct a Reolink RTSP URL, URL-encoding credentials so special
    characters (#, @, :, /) in the password can't corrupt the URL.
    stream="sub" selects the low-res substream; anything else -> main stream."""
    path = "h264Preview_01_sub" if str(stream).lower() == "sub" else "h264Preview_01_main"
    cred = f"{quote(user, safe='')}:{quote(password or '', safe='')}@" if user else ""
    return f"rtsp://{cred}{host}:{port}/{path}"


def rtsp_url(stream=None):
    """Effective RTSP URL: full RTSP_URL override if set, else built from parts.
    `stream` ("main"|"sub") overrides RTSP_STREAM for runtime stream switching."""
    override = os.environ.get("RTSP_URL", "").strip()
    return override or build_rtsp_url(
        RTSP_HOST, RTSP_PORT, RTSP_USER, RTSP_PASS, stream or RTSP_STREAM)


HOST = "0.0.0.0"
PORT = 8000
