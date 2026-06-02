# liveobject

Live object detection web app for the **Raspberry Pi 5 + AI HAT+ 2 (Hailo‑10H)**.

It runs a YOLO model on the Hailo NPU via Picamera2 and serves a browser
dashboard that shows the live camera feed with bounding boxes and class names,
lets you tune detection from the UI, and plots RAM and FPS in real time.

---

## Features

- **Live annotated video** — MJPEG stream with bounding boxes + class name + confidence (~22 fps, ~30 ms inference).
- **Detections panel** — live list of detected objects and their confidence.
- **Backend controls** (all live, no restart):
  - **Max detections** — cap how many objects are reported (keeps the highest‑scoring).
  - **Confidence threshold** — filter out weak detections.
  - **Model switch** — `yolov8m` ⇄ `yolov11m` (both Hailo‑10H, 80 COCO classes).
  - **Snapshot** — save the current annotated frame to `snapshots/`.
  - **Pause / Resume** — freeze the pipeline to save CPU.
- **Realtime graphs** — RAM % and FPS (Chart.js, vendored locally so it works offline), plus CPU % and CPU temperature.
- **Sideways‑mount correction** — frames are rotated 90° in software, so the feed is upright even though the camera is mounted sideways.
- **Continuous autofocus** — keeps frames sharp, which is what makes detection reliable.

---

## Requirements

Everything is provided by Raspberry Pi OS packages (apt) — **no virtualenv needed**:

| Component | Package | Notes |
|---|---|---|
| Camera stack | `python3-picamera2` | Pi Camera (tested: IMX708 / Camera Module 3) |
| Hailo runtime | `python3-h10-hailort` | HailoRT 5.x for the **Hailo‑10H** (AI HAT+ 2) |
| Web framework | `python3-flask` | |
| Metrics | `python3-psutil` | RAM / CPU / temperature |
| Image ops | `python3-opencv` | headless build is fine |

Also required: the Hailo driver loaded (`/dev/hailo0` present) and the HEF models in
`/usr/share/hailo-models/` (`yolov8m_h10.hef`, `yolov11m_h10.hef`), which ship with
the `hailo-h10-all` package.

Verify the NPU before running:

```bash
hailortcli fw-control identify     # should report Device Architecture: HAILO10H
ls /dev/hailo0
```

---

## Run

```bash
cd ~/projects/liveobject
python3 app.py          # or ./run.sh
```

Then open **`http://<pi-ip>:8000/`** from any device on the LAN.

---

## Usage

- The video panel shows the live feed with detections drawn on it.
- Use the **Controls** card to change model, max detections, and threshold — changes apply immediately.
- **Snapshot** saves the current frame and reveals a **View** link.
- **Pause** freezes detection (FPS drops to 0); **Resume** restarts it.
- The **System** card shows live RAM/CPU/temperature and the rolling RAM % and FPS charts.

---

## HTTP API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `GET` | `/stream.mjpg` | MJPEG video stream with overlays |
| `GET` | `/stats` | JSON: fps, infer_ms, ram, cpu, temp, detections, config |
| `POST` | `/config` | Set `max_detections`, `threshold`, `paused`, or `model` |
| `POST` | `/snapshot` | Save the current annotated frame; returns its path |
| `GET` | `/snapshot/latest` | Serve the most recent snapshot |

Example:

```bash
curl -X POST http://<pi-ip>:8000/config \
  -H 'Content-Type: application/json' \
  -d '{"max_detections": 5, "threshold": 0.5, "model": "yolov11m"}'
```

---

## Project layout

```
liveobject/
├── app.py                 # Flask routes
├── detector.py            # DetectionWorker: Picamera2 + Hailo inference thread
├── config.py              # models, display size, defaults, host/port
├── labels.py              # COCO 80 class names
├── templates/index.html   # dashboard markup
├── static/
│   ├── app.js             # polling, controls, Chart.js graphs
│   ├── style.css
│   └── vendor/chart.umd.min.js
├── run.sh
└── README.md
```

---

## How it works

- **Two camera streams**: a `main` stream for display and a 640×640 `lores` stream for the model. Inference runs on the lores stream; boxes are scaled onto the display frame.
- **All Hailo calls live on one worker thread** (HailoRT objects are thread‑affine). Flask only reads shared state and posts simple config/model/snapshot requests; the worker handles model swaps via a pending flag.
- **Color order**: Picamera2's `RGB888` arrays are actually BGR‑ordered, so the model is fed `frame[:, :, ::-1]` (RGB) while OpenCV drawing/encoding keep BGR.
- **NMS output**: the HEFs have YOLOv8 NMS baked in, so `Hailo.run()` returns a list of 80 class arrays, each detection `[y0, x0, y1, x1, score]` normalized 0–1.

---

## Notes

- This uses Flask's built‑in development server, which is fine for personal/LAN use. For an always‑on deployment, run it behind a production WSGI server or wrap it in a `systemd` service.
- Snapshots (`snapshots/`) and HailoRT logs (`hailort*.log`) are git‑ignored.

## License

MIT — see `LICENSE` if present, otherwise treat as MIT.
