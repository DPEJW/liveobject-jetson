# liveobject (Jetson Orin NX edition)

Live object-detection web app for the **NVIDIA Jetson Orin NX** with a **Basler
GigE Vision** camera and **TensorRT** (YOLO26 / YOLO11) inference on the GPU.

It serves a browser dashboard with the live camera feed, bounding boxes + class
names + confidence, live detection controls, and real-time FPS / inference
timing.

> Ported from the Raspberry Pi 5 + AI HAT+ (Hailo-10H) version (the original
> `liveobject` repo). The public `DetectionWorker` interface is unchanged — only
> the camera and inference backends were swapped.

---

## What changed from the Pi version

|              | Raspberry Pi 5            | Jetson Orin NX (this repo)              |
|--------------|---------------------------|-----------------------------------------|
| Camera       | Picamera2 (CSI)           | Basler GigE Vision via **pypylon**      |
| Inference    | Hailo-10H NPU (HEF)       | **TensorRT** FP16 engine, GPU (pycuda)  |
| Models       | yolov8m / yolov11m (HEF)  | **yolo26m / yolo26l / yolo11m** (.engine) |
| NMS          | baked into the HEF        | `cv2.dnn.NMSBoxes` on the decoded head  |

---

## Features

- **Live annotated video** — MJPEG stream with boxes + class + confidence.
- **Detections panel** — live list of detected objects and scores.
- **Backend controls** (all live, no restart): max detections, confidence
  threshold, **model switch** (`yolo26m` ⇄ `yolo26l` ⇄ `yolo11m`), snapshot,
  pause / resume, rotation (0/90/180/270) and horizontal / vertical flip.
- **Real-time graphs** — FPS and inference time.

---

## Hardware / software

- **Device:** Seeed reComputer Orin Industrial (Orin NX 16GB), **L4T 35.5.0**,
  Ubuntu 20.04, CUDA 11.4, **TensorRT 8.5.2**.
- **Camera:** Basler ace2 GigE (`Mono8`) on a dedicated link-local NIC.
- **Python:** pypylon, pycuda (built from source), numpy **1.24.x** (with a
  `np.bool`/`np.float` shim for the TRT 8.5 bindings — see `trt_yolo.py`),
  opencv, flask.

---

## Building the detection engines

Model binaries are **not** committed (large, and each `.engine` is specific to
this exact TensorRT version + GPU). Rebuild them like this:

**1. Export ONNX** (on a workstation with Ultralytics — keeps the Jetson
torch-free):

```bash
python3 -m venv /tmp/yolo-export && source /tmp/yolo-export/bin/activate
pip install -U ultralytics onnx onnxslim
# end2end=False keeps the traditional (1,84,8400) head that trt_yolo.py decodes.
yolo export model=yolo26m.pt format=onnx end2end=False imgsz=640 opset=17 simplify=True half=False
yolo export model=yolo26l.pt format=onnx end2end=False imgsz=640 opset=17 simplify=True half=False
scp yolo26m.onnx yolo26l.onnx orinnx1:~/projects/liveobject/models/
```

**2. Build the FP16 TensorRT engine** (on the Orin):

```bash
cd ~/projects/liveobject/models
/usr/src/tensorrt/bin/trtexec --onnx=yolo26m.onnx --fp16 --saveEngine=yolo26m.engine
/usr/src/tensorrt/bin/trtexec --onnx=yolo26l.onnx --fp16 --saveEngine=yolo26l.engine
```

`config.py` auto-detects whichever `*.engine` files exist and offers them in the
model switcher (default = newest available).

> **Why YOLO26 with `end2end=False`?** YOLO26's default head is NMS-free and
> emits `(1, 300, 6)`, which this app does not decode and which is riskier to
> build on TensorRT 8.5. `end2end=False` produces the traditional one-to-many
> head (`(1, 84, 8400)`) — a drop-in for the existing decoder, with the same mAP.

---

## Run

```bash
cd ~/projects/liveobject
python3 app.py            # or ./run.sh
```

Or via systemd (`liveobject.service`, port 8000):

```bash
sudo systemctl start liveobject
```

Open **`http://<orin-ip>:8000/`** from any device on the LAN.

> **One camera, one app:** `liveobject.service` declares
> `Conflicts=baffle-qc.service`, so starting it stops the baffle-qc vision app
> (they share the single GigE camera), and vice-versa.

---

## HTTP API

| Method | Route                | Purpose                                                        |
|--------|----------------------|----------------------------------------------------------------|
| `GET`  | `/`                  | Dashboard UI                                                   |
| `GET`  | `/stream.mjpg`       | MJPEG video stream with overlays                               |
| `GET`  | `/stats`             | JSON: fps, infer_ms, detections, config                        |
| `POST` | `/config`            | Set `max_detections`, `threshold`, `paused`, `model`, `rotation`, `flip_h`, `flip_v` |
| `POST` | `/snapshot`          | Save the current annotated frame; returns its path             |
| `GET`  | `/snapshot/latest`   | Serve the most recent snapshot                                 |

```bash
curl -X POST http://<orin-ip>:8000/config \
  -H 'Content-Type: application/json' \
  -d '{"model": "yolo26m", "threshold": 0.4, "max_detections": 10}'
```

---

## Project layout

```
liveobject/
├── app.py                 # Flask routes
├── detector.py            # DetectionWorker: Basler GigE + TensorRT worker thread
├── trt_yolo.py            # TensorRT engine load + (1,84,8400) decode + NMS
├── config.py              # model list, display size, defaults, host/port
├── labels.py              # COCO 80 class names
├── templates/index.html   # dashboard markup
├── static/                # app.js, style.css, vendored Chart.js
├── models/                # *.engine / *.onnx (gitignored — see "Building" above)
├── docs/plans/            # design docs
├── run.sh
└── README.md
```

---

## How it works

- **Mono → BGR:** the Basler camera streams `Mono8`; frames are converted to BGR
  and resized to the display size. (Grayscale input caps accuracy on
  color-dependent classes regardless of model.)
- **One worker thread:** all CUDA / TensorRT calls live on the `DetectionWorker`
  thread; the CUDA primary context is pushed/popped around every op. Flask only
  reads shared state and posts config / model / snapshot requests; model swaps
  go through a pending flag handled by the worker.
- **Decode:** the engine outputs `(1, 84, 8400)` (4 box coords + 80 class scores
  × 8400 anchors); `trt_yolo.py` takes the per-anchor argmax, thresholds, and
  runs `cv2.dnn.NMSBoxes`.

---

## License

MIT.
