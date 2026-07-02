# liveobject

Live object-detection web app with **TensorRT** (YOLO26 / YOLO11) inference on
an NVIDIA GPU. It serves a browser dashboard with the live camera feed, bounding
boxes + class names + confidence, live detection controls, object tracking,
cat re-ID + on-device retrain, and real-time FPS / inference timing.

> **Two hardware targets, one codebase:**
> - **Jetson Orin NX** (`main` / `feat/cat-reid-retrain` branches) — Basler GigE
>   camera via pypylon, TensorRT 8.5 + pycuda, CUDA 11.4, Python 3.8.
> - **NVIDIA GB10 / Grace-Blackwell** (`feat/blackwell-gb10` branch, this one) —
>   network Reolink RTSP camera (FFmpeg decode), **TensorRT 11 + cuda-python**,
>   CUDA 13, Python 3.12. See **"Blackwell / GB10 notes"** below.
>
> Originally ported from the Raspberry Pi 5 + AI HAT+ (Hailo-10H) version. The
> public `DetectionWorker` interface is unchanged across all three — only the
> camera and inference backends are swapped.

---

## Blackwell / GB10 notes

What changed from the Jetson version (all in `trt_yolo.py`, `build_engine.py`,
`train_runner.py`), and why:

| Concern | Jetson Orin NX | GB10 / Blackwell (this branch) |
|---|---|---|
| GPU / arch | Ampere, `sm_87` | **Blackwell GB10, `sm_121`** |
| CUDA / TensorRT | 11.4 / **8.5.2** | 13.0 / **11.1** |
| Python | 3.8 | **3.12** |
| CUDA memory ops | **pycuda** | **cuda-python** (`cuda.bindings.runtime`) — pycuda won't build on CUDA 13 |
| TRT tensor I/O | binding-index API (`num_bindings`, `execute_async_v2`) | **name-based API** (`num_io_tensors`, `set_tensor_address`, `execute_async_v3`) — the old API was removed in TRT 10 |
| FP16 selection | `BuilderFlag.FP16` | **strongly-typed network + FP16 ONNX** — the FP16 builder flag is gone in TRT 11 |
| Engine build tool | `trtexec` CLI | **`build_engine.py`** (TRT Python API) — trtexec isn't in the pip wheels |
| Camera | Basler GigE (pypylon) | **Reolink RTSP** over the LAN; no Basler present |
| RTSP decode | GStreamer NVDEC (Jetson `nvv4l2decoder`) | OpenCV **FFmpeg** fallback (automatic — those Jetson GStreamer elements don't exist here) |

Setup:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements-blackwell.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

Ballpark performance on the GB10: **~100–110 fps** at 640×640 (yolo11m ~96 fps,
yolo26m ~110 fps), vs ~12 fps on the Orin NX.

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

### On the GB10 / Blackwell (this branch)

Export the FP16 ONNX and build the engine with `build_engine.py` (the TRT Python
API; `trtexec` isn't shipped in the pip wheels). Both steps run on the GB10:

```bash
source .venv/bin/activate
# FP16 ONNX (half=True needs the GPU). YOLO26 auto-emits its NMS-free (1,300,6)
# head; YOLO11 emits the classic (1,84,8400) head. trt_yolo.py decodes BOTH.
python - <<'PY'
from ultralytics import YOLO
for m in ("yolo11m", "yolo26m"):
    YOLO(m + ".pt").export(format="onnx", imgsz=640, opset=17,
                           simplify=True, dynamic=False, half=True, device=0)
PY
python build_engine.py models/yolo11m.onnx models/yolo11m.engine --fp16
python build_engine.py models/yolo26m.onnx models/yolo26m.engine --fp16
```

`config.py` auto-detects whichever `*.engine` files exist and offers them in the
model switcher (default = newest available). `trt_yolo.py` auto-detects the head
layout per engine, so YOLO11 `(1,84,8400)` and YOLO26 `(1,300,6)` both work.

### On the Jetson Orin NX (`main` branch)

**1. Export ONNX** (on a workstation with Ultralytics — keeps the Jetson
torch-free):

```bash
python3 -m venv /tmp/yolo-export && source /tmp/yolo-export/bin/activate
pip install -U ultralytics onnx onnxslim
# end2end=False keeps the traditional (1,84,8400) head that trt_yolo.py decodes.
yolo export model=yolo26m.pt format=onnx end2end=False imgsz=640 opset=17 simplify=True half=False
scp yolo26m.onnx orinnx1:~/projects/liveobject/models/
```

**2. Build the FP16 TensorRT engine** (on the Orin):

```bash
cd ~/projects/liveobject/models
/usr/src/tensorrt/bin/trtexec --onnx=yolo26m.onnx --fp16 --saveEngine=yolo26m.engine
```

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
