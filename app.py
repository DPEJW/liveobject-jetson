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
    payload = {
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
    }
    payload.update(worker.tracking_state())
    return jsonify(payload)


@app.route("/config", methods=["POST"])
def set_config():
    data = request.get_json(force=True, silent=True) or {}
    try:
        if "model" in data:
            worker.request_model(data["model"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if "camera_source" in data or "rtsp_stream" in data:
        worker.request_camera(source=data.get("camera_source"),
                              stream=data.get("rtsp_stream"))
    cfg = worker.set_config(
        max_detections=data.get("max_detections"),
        threshold=data.get("threshold"),
        paused=data.get("paused"),
        rotation=data.get("rotation"),
        flip_h=data.get("flip_h"),
        flip_v=data.get("flip_v"),
        track_trail=data.get("track_trail"),
        track_heatmap=data.get("track_heatmap"),
        track_zones=data.get("track_zones"),
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


@app.route("/track", methods=["POST"])
def track():
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")
    if action == "stop":
        return jsonify(worker.request_stop_tracking())
    if action == "select":
        return jsonify(worker.request_select(track_id=data.get("id"),
                                             x=data.get("x"), y=data.get("y")))
    return jsonify({"error": "unknown action"}), 400


@app.route("/zones", methods=["POST"])
def zones():
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")
    if action == "add" and data.get("box"):
        return jsonify(worker.zone_add(data.get("label", "zone"), data["box"]))
    if action == "rename" and data.get("id") is not None:
        return jsonify(worker.zone_rename(data["id"], data.get("label", "")))
    if action == "delete" and data.get("id") is not None:
        return jsonify(worker.zone_delete(data["id"]))
    return jsonify({"error": "unknown action"}), 400


@app.route("/enroll", methods=["POST"])
def enroll():
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action", "enroll")
    if action == "enroll" and data.get("name") and data.get("x") is not None:
        return jsonify(worker.enroll_cat(data["name"], data["x"], data["y"]))
    if action == "rename" and data.get("old") and data.get("new"):
        return jsonify(worker.rename_cat(data["old"], data["new"]))
    if action == "clear" and data.get("name"):
        return jsonify(worker.clear_cat(data["name"]))
    return jsonify({"error": "bad enroll request"}), 400


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")
    if action == "start":
        return jsonify(worker.start_training(int(data.get("epochs", 30))))
    if action == "cancel":
        return jsonify(worker.cancel_training())
    return jsonify({"error": "unknown action"}), 400


@app.route("/train/status")
def train_status():
    return jsonify(worker.training_status())


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, threaded=True)
