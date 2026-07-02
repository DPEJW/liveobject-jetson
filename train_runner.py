"""On-Orin fine-tune runner (executed by the TRAINING venv, not the live service).

Launched as a subprocess by detector.py:
    ~/venvs/train/bin/python train_runner.py <epochs>    # train + export + engine
    ~/venvs/train/bin/python train_runner.py export      # export newest best.pt only

Reads the captured dataset in ~/catdata (images/ + labels/ + classes.json), makes
an 80/20 train/val split (symlinks), writes data.yaml, and fine-tunes a small model
on ONLY the manually-labeled classes. After training it exports best.pt -> ONNX ->
TensorRT engine (trtexec --fp16) and writes ~/catdata/identity/manifest.json, which
the live detector loads as the "identity model" running alongside the base model.
Progress (epoch / loss / mAP, then exporting/building states) is written to
~/catdata/train_status.json for the dashboard to poll.
"""
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

DATA = os.path.expanduser("~/catdata")
STATUS = os.path.join(DATA, "train_status.json")
IDENT = os.path.join(DATA, "identity")
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def write_status(**kw):
    kw["ts"] = time.time()
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(kw, fh)
    os.replace(tmp, STATUS)   # atomic


def load_names():
    class_map = json.load(open(os.path.join(DATA, "classes.json")))
    names = [None] * len(class_map)
    for n, i in class_map.items():
        names[int(i)] = n
    return names


def export_deploy(best, names):
    """best.pt -> ONNX -> TensorRT engine + manifest.json. Returns engine path."""
    write_status(state="exporting", names=names, msg="best.pt -> ONNX")
    from ultralytics import YOLO
    onnx_path = YOLO(best).export(format="onnx", imgsz=640, opset=12,
                                  simplify=False, device="cpu")
    os.makedirs(IDENT, exist_ok=True)
    engine = os.path.join(IDENT, "identity_%s.engine" % time.strftime("%Y%m%d_%H%M%S"))
    write_status(state="building", names=names,
                 msg="TensorRT engine build (takes a few minutes)")
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)   # ultralytics cpu-export sets it to "" —
    r = subprocess.run([TRTEXEC, "--onnx=%s" % onnx_path, "--saveEngine=%s" % engine,
                        "--fp16"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=2400, env=env)
    if r.returncode != 0 or not os.path.exists(engine):
        tail = r.stdout.decode(errors="replace")[-400:]
        write_status(state="error", msg="trtexec failed: " + tail)
        return None
    manifest = {"engine": engine, "names": names, "built": time.time(), "best": best}
    tmp = os.path.join(IDENT, "manifest.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh)
    os.replace(tmp, os.path.join(IDENT, "manifest.json"))
    return engine


def newest_best():
    cands = glob.glob(os.path.join(DATA, "train_out", "*", "weights", "best.pt"))
    return max(cands, key=os.path.getmtime) if cands else None


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        try:
            names = load_names()
        except Exception:
            write_status(state="error", msg="no classes.json")
            return
        best = newest_best()
        if not best:
            write_status(state="error", msg="no trained best.pt to export")
            return
        engine = export_deploy(best, names)
        if engine:
            write_status(state="done", names=names, best=best, engine=engine)
        return

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    imgsz = 640    # MUST match the ONNX/TensorRT export size — training at 320
                   # while deploying at 640 made the model near-blind live
    try:
        names = load_names()
    except Exception:
        write_status(state="error", msg="nothing labeled yet (no classes.json)")
        return

    imgs = sorted(glob.glob(os.path.join(DATA, "images", "*.jpg")))
    if len(imgs) < 8:
        write_status(state="error", msg="need at least 8 frames, have %d" % len(imgs))
        return

    run = os.path.join(DATA, "run")
    if os.path.exists(run):
        shutil.rmtree(run)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(run, sub))
    random.seed(0)
    random.shuffle(imgs)
    nval = max(1, len(imgs) // 5)
    kept = 0
    for k, img in enumerate(imgs):
        base = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(DATA, "labels", base + ".txt")
        if not os.path.exists(lbl):
            continue
        split = "val" if k < nval else "train"
        os.symlink(img, os.path.join(run, "images", split, os.path.basename(img)))
        os.symlink(lbl, os.path.join(run, "labels", split, base + ".txt"))
        kept += 1

    data_yaml = os.path.join(run, "data.yaml")
    with open(data_yaml, "w") as fh:
        fh.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n" % run)
        for i, n in enumerate(names):
            fh.write("  %d: %s\n" % (i, n))

    write_status(state="starting", epoch=0, epochs=epochs, images=kept, names=names)

    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    def on_epoch_end(trainer):
        try:
            m = getattr(trainer, "metrics", None) or {}
            write_status(state="training", epoch=int(trainer.epoch) + 1, epochs=epochs,
                         images=kept, names=names,
                         mAP50=round(float(m.get("metrics/mAP50(B)", 0.0)), 4),
                         loss=round(float(getattr(trainer, "loss", 0.0)), 4))
        except Exception as exc:
            write_status(state="training", epoch=-1, epochs=epochs, err=str(exc))

    model.add_callback("on_fit_epoch_end", on_epoch_end)
    try:
        model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=8, device=0,
                    workers=2, project=os.path.join(DATA, "train_out"), name="model",
                    exist_ok=True, plots=False, verbose=False)
        best = newest_best()
        if not best:
            write_status(state="error", msg="training finished but no best.pt found")
            return
        engine = export_deploy(best, names)   # ONNX -> TensorRT -> manifest
        if engine:
            write_status(state="done", epoch=epochs, epochs=epochs, names=names,
                         best=best, engine=engine)
    except Exception as exc:
        write_status(state="error", msg=str(exc)[:300])


if __name__ == "__main__":
    main()
