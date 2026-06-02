# liveobject

Live object detection web app for the Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H).

Streams the Pi camera with YOLO bounding boxes + class names drawn on the feed,
lets you control detection from the browser, and shows realtime RAM and FPS
graphs.

## Features

- **Live annotated MJPEG stream** (boxes + class name + confidence).
- **Detections panel** — live list of detected objects with confidence.
- **Backend controls:**
  - **Max detections** — cap how many objects are reported (keeps the highest-scoring).
  - **Confidence threshold** — filter weak detections.
  - **Model switch** — `yolov8m` ⇄ `yolov11m` (both Hailo-10H, 80 COCO classes).
  - **Snapshot** — save the current annotated frame to `snapshots/`.
  - **Pause / Resume** — freeze the pipeline to save CPU.
- **Realtime graphs** — RAM % and FPS (Chart.js, served locally).
- Frames are rotated 90° in software to correct this Pi's sideways camera mount.

## Requirements

All provided by the system (apt), no virtualenv needed:
`python3-picamera2`, `python3-h10-hailort` (HailoRT 5.x), `python3-flask`,
`python3-psutil`, `python3-opencv` (headless), and the Hailo-10H driver
(`/dev/hailo0`). Models live in `/usr/share/hailo-models/`.

## Run

```bash
cd ~/projects/liveobject
python3 app.py
```

Then open `http://<pi-ip>:8000/` from any device on the LAN.

## Layout

- `app.py` — Flask routes (`/`, `/stream.mjpg`, `/stats`, `/config`, `/snapshot`).
- `detector.py` — `DetectionWorker`: Picamera2 + Hailo inference thread.
- `config.py` — models, display size, defaults, host/port.
- `labels.py` — COCO 80 class names.
- `templates/index.html`, `static/` — the dashboard UI.
