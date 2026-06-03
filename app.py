"""Flask app: live object detection UI backed by the Hailo-10H NPU."""
import time

import psutil
from flask import (Flask, Response, jsonify, render_template, request,
                   send_file)

from config import HOST, PORT
from detector import DetectionWorker

app = Flask(__name__)
worker = DetectionWorker()
worker.start()
psutil.cpu_percent(None)  # prime the non-blocking CPU sampler


def _cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        for key in ("cpu_thermal", "coretemp"):
            if temps.get(key):
                return round(temps[key][0].current, 1)
    except Exception:
        pass
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream.mjpg")
def stream():
    def gen():
        for jpeg in worker.frames():
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                   + jpeg + b"\r\n")
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    vm = psutil.virtual_memory()
    return jsonify({
        "fps": round(worker.fps(), 1),
        "infer_ms": round(worker.infer_ms, 1),
        "ram_used_mb": round(vm.used / 1048576),
        "ram_total_mb": round(vm.total / 1048576),
        "ram_pct": vm.percent,
        "cpu_pct": psutil.cpu_percent(None),
        "cpu_temp": _cpu_temp(),
        "count": len(worker.detections),
        "detections": worker.detections,
        "config": worker.config(),
    })


@app.route("/config", methods=["POST"])
def set_config():
    data = request.get_json(force=True, silent=True) or {}
    try:
        if "model" in data:
            worker.request_model(data["model"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    cfg = worker.set_config(
        max_detections=data.get("max_detections"),
        threshold=data.get("threshold"),
        paused=data.get("paused"),
        rotation=data.get("rotation"),
    )
    return jsonify(cfg)


@app.route("/snapshot", methods=["POST"])
def snapshot():
    worker.request_snapshot()
    for _ in range(60):  # wait up to ~3s for the worker to write the file
        if worker.snapshot_path():
            break
        time.sleep(0.05)
    return jsonify({"path": worker.snapshot_path()})


@app.route("/snapshot/latest")
def snapshot_latest():
    path = worker.snapshot_path()
    if not path:
        return ("no snapshot yet", 404)
    return send_file(path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, threaded=True)
