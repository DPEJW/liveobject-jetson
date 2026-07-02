"""On-Orin fine-tune runner (executed by the TRAINING venv, not the live service).

Launched as a subprocess by detector.py:
    ~/venvs/train/bin/python train_runner.py <epochs>

Reads the captured dataset in ~/catdata (images/ + labels/ + classes.json), makes
an 80/20 train/val split (symlinks), writes data.yaml, and fine-tunes a small model
on ONLY the manually-labeled classes. Per-epoch progress (epoch / loss / mAP) is
written to ~/catdata/train_status.json for the dashboard to poll.
"""
import glob
import json
import os
import random
import shutil
import sys
import time

DATA = os.path.expanduser("~/catdata")
STATUS = os.path.join(DATA, "train_status.json")


def write_status(**kw):
    kw["ts"] = time.time()
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(kw, fh)
    os.replace(tmp, STATUS)   # atomic


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    imgsz = 320
    try:
        class_map = json.load(open(os.path.join(DATA, "classes.json")))
    except Exception:
        write_status(state="error", msg="nothing labeled yet (no classes.json)")
        return
    names = [None] * len(class_map)
    for n, i in class_map.items():
        names[int(i)] = n

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
        best = os.path.join(DATA, "train_out", "model", "weights", "best.pt")
        write_status(state="done", epoch=epochs, epochs=epochs, names=names,
                     best=best if os.path.exists(best) else None)
    except Exception as exc:
        write_status(state="error", msg=str(exc)[:300])


if __name__ == "__main__":
    main()
