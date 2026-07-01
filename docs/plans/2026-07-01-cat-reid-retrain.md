# Cat re-ID + on-device retrain — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Name the 2 cats (click + name), stop the ID churn via appearance re-ID, then let the user retrain on-device (manual labels only) with live progress, deploying a small identity model alongside the untouched base COCO model.

**Architecture:** `reid.py` (pure-numpy matching core, Mac-testable) + worker integration for live naming; a capture/auto-label step gated by conf>0.8 + re-ID; an on-Orin Ultralytics fine-tune (background job, pauses inference) with progress surfaced to the dashboard; export → TensorRT → reversible hot-swap of a second (identity) model.

**Tech Stack:** NumPy, OpenCV (edge only), Flask, Ultralytics + PyTorch-for-Jetson (Phase 2, on Orin), TensorRT/trtexec.

**Design doc:** `docs/plans/2026-07-01-cat-reid-retrain-design.md`

**Conventions:** self-running tests (`python3 tests/test_*.py`), commit per green step, no Claude co-author trailer, SSH/scp/LAN need `dangerouslyDisableSandbox=true`, deploy = scp + `sudo systemctl restart liveobject`.

---

## Phase 0 — Environment spike (GATE: stop-and-report if it fails)

**Goal:** prove on-Orin YOLO fine-tuning is viable before building anything around it.

**Step 1 — Probe existing stack**
Run on the Orin: check Python, pip, existing torch, JetPack/CUDA, free RAM/disk, NVIDIA torch index availability.
Expected: know exactly what's missing.

**Step 2 — Install PyTorch-for-Jetson + Ultralytics (one-time, scripted)**
Install the NVIDIA torch wheel matching JetPack 5.1.3 (CUDA 11.4), a matching torchvision (from source if no wheel), and `ultralytics` (with `--no-deps` where needed to avoid pulling a wrong torch). Do it in a **venv** at `~/venvs/train` so it can't disturb the live service's Python.
Expected: `python -c "import torch, torchvision, ultralytics; print(torch.__version__, torch.cuda.is_available())"` → prints versions, CUDA True.

**Step 3 — Toy fine-tune (2 epochs)**
Download `yolo11n.pt`, make a 3-image toy dataset (or use `coco8`), run `yolo train ... epochs=2 imgsz=320 batch=2 device=0`.
Expected: training completes, writes `runs/.../weights/best.pt`, no OOM.

**Step 4 — Export path**
Export the toy `best.pt` → ONNX; build a TensorRT engine with `trtexec --fp16`.
Expected: `.engine` produced.

**CHECKPOINT:** Report timings, RAM headroom, CUDA status. **If any step can't be made robust, STOP** and present fallbacks (train on Mac, or ship re-ID only). Do not proceed to Phase 2 build until this passes. (Phase 1 does NOT depend on this and can proceed regardless.)

---

## Phase 1 — Re-ID + named roster (on-device, TDD)

### Task 1: `reid.py` matching core (pure numpy — Mac-testable)

**Files:** Create `reid.py`; Test `tests/test_reid.py`

**Step 1 — Write failing tests** (use synthetic histogram vectors; no cv2 needed):

```python
# tests/test_reid.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from reid import similarity, Roster

def _sig(peak, n=32):
    v = np.full(n, 0.01, dtype=np.float32); v[peak] = 1.0
    return v / v.sum()

def test_similarity_high_for_same_low_for_different():
    a, b, c = _sig(3), _sig(3), _sig(20)
    assert similarity(a, b) > 0.9
    assert similarity(a, c) < 0.3

def test_roster_matches_enrolled_identity():
    r = Roster(threshold=0.5)
    r.enroll("Mittens", _sig(3)); r.enroll("Shadow", _sig(20))
    name, score = r.match(_sig(3))
    assert name == "Mittens" and score > 0.5

def test_unknown_below_threshold_returns_none():
    r = Roster(threshold=0.6)
    r.enroll("Mittens", _sig(3))
    name, _ = r.match(_sig(28))
    assert name is None

def test_two_slot_assignment_no_double_naming():
    r = Roster(threshold=0.3)
    r.enroll("Mittens", _sig(3)); r.enroll("Shadow", _sig(20))
    # two live cats, slightly noisy -> each must get a distinct name
    assign = r.assign([_sig(20), _sig(3)])   # order: shadow-ish, mittens-ish
    assert assign == ["Shadow", "Mittens"]

def test_returning_cat_regains_name():
    r = Roster(threshold=0.5)
    r.enroll("Mittens", _sig(3))
    assert r.match(_sig(3))[0] == "Mittens"   # left and came back -> same name
```

**Step 2 — Run, verify fail** (`No module named 'reid'`).

**Step 3 — Implement `reid.py`** (`similarity` = histogram correlation or Bhattacharyya; `Roster.enroll/match/assign` with a threshold; `assign` = greedy best-match without reusing a name). Keep pure numpy.

**Step 4 — Run, verify pass. Step 5 — Commit** `feat(reid): appearance-signature matching core + roster`.

### Task 2: Signature extraction (cv2 edge) + worker enrollment/live naming

**Files:** Modify `detector.py` (add `hs_signature(bgr, box)` using cv2 HSV+hist; roster held in worker; enrollment via click+name applied in the pending-request pattern; re-ID assigns names to cat detections each frame; names flow into `tracked`/snapshot). Add control methods `enroll_cat(name, x, y)`, `rename_cat`, `clear_cat`. Reset roster on camera/rotation change.
**Verify:** byte-compile; live check after deploy (Task 4).
**Commit** `feat(detector): cat appearance signatures + live re-ID naming`.

### Task 3: Flask + UI for enrollment and names

**Files:** `app.py` (`POST /enroll` {name,x,y} / rename / clear; roster in `/stats`), `templates/index.html` + `static/app.js` + `static/style.css` (Enroll mode: click a cat → name prompt; enrolled-cats list with color swatch + rename/clear; Tracking list shows names).
**Commit** `feat(ui): cat enrollment (click+name) and named tracking`.

### Task 4: Deploy + verify Phase 1 on the Orin

scp changed files, restart, verify: enroll both cats, confirm names persist across exits (churn gone), re-ID doesn't swap when both visible. Headless screenshot.
**Commit** any fixes.

---

## Phase 2 — Capture + on-Orin train + live progress + deploy (detail after Phase 0 passes)

Outlined now; exact commands/model finalized using Phase-0 spike results.

- **Task 5 — Auto-capture/label:** in the worker, when a labeled cat is `conf>0.80` + re-ID match + box-sane + not-duplicate, write frame+YOLO-label to `dataset/<name>/`. Dedup via time-spacing + frame-hash. Live counters in `/stats`. TDD the pure bits (dedup decision, label formatting, box→YOLO-norm).
- **Task 6 — Review grid (optional-use):** `GET` collected samples; UI grid per cat with delete; `POST /dataset/delete`.
- **Task 7 — Training job:** `train_runner.py` launches Ultralytics fine-tune (small model, labeled classes only, val split, early-stop, augment) in the venv as a subprocess; **pauses live detection**; parses progress (epoch/loss/mAP) to a status file; `GET /train/status`; `POST /train/start|cancel`.
- **Task 8 — Progress UI:** dashboard training panel — state, epoch, losses, live mAP curve, ETA, cancel.
- **Task 9 — Export + hot-swap (reversible):** best.pt → ONNX → trtexec engine; register identity model; worker runs base + identity, merges (cat→name); keep prior engine for one-click revert; measure fps, add every-Nth-frame option if needed.
- **Task 10 — Deploy + verify full loop on the Orin; update memory; finish branch.**

---

## Done criteria
- `python3 tests/test_reid.py` green on the Mac.
- Phase 1 live: both cats named, IDs stable across exits, no swap.
- Phase 0 gate passed (or fallback chosen with the user).
- Phase 2 live: label → collect >80% frames → train with visible progress → hot-swap → detector outputs the names; base detection/zones unaffected; reversible.
