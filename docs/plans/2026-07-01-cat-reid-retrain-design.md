# Cat re-ID + on-device retrain — design

- **Date:** 2026-07-01
- **Status:** Approved (brainstorming complete; ready for implementation plan)
- **Component:** `liveobject` (Jetson Orin NX, Reolink RTSP + TensorRT YOLO26)
- **Builds on:** `2026-06-28-object-tracking-design.md` (tracker/zones/session already shipped)

## Problem

Object IDs churn (`cat #5 → #24 → #25`) because the IoU tracker retires a track after
~1 s of missed detection and mints a new id on re-appearance. The couch stays `#1`
because it's detected every frame. The user has **2 cats + 1 couch** in a fixed scene and
wants: (1) stop the churn, (2) tell the two cats apart by name, (3) better cat detection —
and specifically wants a **retrain** where **manually naming a cat** starts collecting
**>80%-confidence frames** labeled with that name, then an **on-device training run with
live progress**.

## Decisions (from brainstorming)

| Question | Decision |
|----------|----------|
| Goals | stop churn **and** name the 2 cats **and** better detection |
| Do the cats look different? | **Clearly different** → color re-ID is reliable |
| Enrollment | **Click a cat + type a name** |
| Training location | **On the Orin itself** (self-contained; pauses live detection) |
| What gets trained | **Only manually-labeled objects.** No auto-training of unlabeled objects. |
| Forgetting/zone safety | **Two-model** architecture: base COCO model untouched; a small model trained on only the labeled classes runs alongside |

## Key insight: re-ID is the auto-labeler

A raw >80% YOLO box only says "cat" — it can't say Mittens vs Shadow. **Enrollment
(click+name) + appearance re-ID is what stamps the correct name onto each high-confidence
frame.** So the two halves compose: name a cat → re-ID tags its detections → high-conf +
re-ID-matched frames become labeled samples → train. Re-ID is not thrown away; it is the
mechanism that makes per-cat labels possible without hand-labeling.

## Architecture

### Subsystem 1 — Cat re-ID + named roster (on-device, NO training)

Delivers "stop churn" + "name the cats" immediately, and is the labeler for Subsystem 2.

- **Appearance signature:** a Hue–Saturation 2-D color histogram of the detection's box
  **interior** (center-cropped ~60% to avoid couch/floor bleed), L1-normalized. HSV H–S
  (ignoring Value) is robust to brightness. Cheap, on-device, no model.
- **Enrollment:** an "enroll" action — user clicks a cat, types a name. The signature is
  **averaged over ~1 s** of frames for stability and stored as a named identity in a
  **roster** (`{name, signature, color_swatch}`). Rename/clear supported.
- **Live re-ID:** each frame, for every `cat` detection compute its signature and score it
  against each enrolled identity (histogram correlation / Bhattacharyya). Use the existing
  IoU tracker for frame-to-frame continuity as a prior; assign names by best match above a
  **similarity threshold** (else "unknown cat"); when both cats are visible use a 2-slot
  assignment so they can't both grab one name. A returning cat re-matches its signature →
  **regains its name** (kills the churn).
- **UI:** the Tracking list shows `Mittens` / `Shadow` instead of `cat #5`; an Enrollment
  panel lists enrolled cats (name + color swatch) with rename/clear.

### Subsystem 2 — Capture → train (on the Orin) → deploy

- **Auto-collect (only labeled objects):** when a labeled cat is detected **conf > 0.80
  AND** re-ID similarity is high AND the box passes sanity (size/aspect), save the raw
  frame + a YOLO-format label (box, class = the name) into a dataset dir. **Dedupe**
  (time-spaced + reject near-identical via frame hash/IoU) so we don't train on 300 copies
  of the same pose. Per-class target count with a live counter (`Mittens 143/300`).
- **Review (recommended):** a grid of collected crops per cat with delete, so a bad
  pseudo-label can be removed before training. (Honors robustness; optional to use.)
- **Environment (the big risk):** the Orin is torch-free by design. Training needs
  **PyTorch-for-Jetson + Ultralytics** (torchvision likely built from source) matched to
  JetPack 5.1.3 / CUDA 11.4. **De-risk first with an environment spike** (below).
- **Training job:** a background subprocess runs an Ultralytics fine-tune of a **small
  model** (yolo11n / yolo26n) on **only the labeled classes**, from pretrained weights.
  **Live detection is paused** for the run (GPU/RAM contention). Progress (epoch, box/cls
  loss, val mAP, ETA) is parsed from Ultralytics output/`results.csv` and exposed via a
  status endpoint; the dashboard shows a **live progress view + mAP curve**.
  Anti-overfit: val split, few epochs, early-stop on mAP, augmentation (mosaic/HSV/flip).
- **Export + deploy (reversible):** best.pt → ONNX → `trtexec` builds a TensorRT engine on
  the Orin → registered as the **identity model**. The **base COCO model is untouched**;
  the worker runs base + identity and merges (a cat box gets its trained name; everything
  else from base). The previous identity engine is kept for **one-click revert**.
- **Two-model inference cost:** the identity model is nano and can run every Nth frame if
  fps drops; measured during the spike/Phase-2.

## On-Orin training reality (accepted trade-offs)

- One-time heavy install (torch/torchvision/ultralytics on aarch64/JetPack). Scripted +
  verified; **if it can't be made robust, stop and report** rather than ship flaky.
- **Training pauses live detection** for its duration (a few minutes on a nano model).
- Single-room dataset → the fine-tune is **scene-specific** (fine for a fixed camera), and
  overfit is guarded by val mAP + early stop; the base model stays intact regardless.

## Data flow

```
click+name ─► roster{name, HS-signature}
per frame: base YOLO ─► dets ─► IoU tracker ─► re-ID names cats (vs roster)
   ├─ live: Tracking list shows names; churn gone
   └─ if labeled cat & conf>0.8 & reID-match & sane & !dup ─► dataset/{img,label}
user hits Train ─► pause detection ─► ultralytics fine-tune (small model, labeled classes)
   ─► progress (epoch/loss/mAP) ─► /train/status ─► dashboard live view
best.pt ─► onnx ─► trtexec engine ─► identity model (hot-swap, reversible)
inference = base COCO (couch/zones/etc, untouched) + identity model (Mittens/Shadow)
```

## Risks & mitigations

- **Jetson training env won't build** → env spike first; fail fast; fallback options
  (Mac training, or re-ID-only) reported to user.
- **Catastrophic forgetting / zones break** → two-model; base weights never change.
- **Bad pseudo-labels** → gate on conf + re-ID + box sanity + dedupe; optional review grid.
- **Overfit to one room** → val split, early-stop, augmentation; reversible hot-swap.
- **fps drop from 2 models** → nano identity model; optional every-Nth-frame; measured.
- **GPU OOM during training** → pause inference; small batch/imgsz; nano model.

## Phasing

- **Phase 0 — environment spike (GATE):** install torch-for-Jetson + Ultralytics on the
  Orin; run a 2-epoch toy fine-tune; confirm it completes and exports. Stop-and-report if
  not robust.
- **Phase 1 — re-ID + roster + enrollment** (on-device, TDD): stops churn + names cats;
  the labeler for Phase 2. Immediate value, no training dependency.
- **Phase 2 — capture + review + on-Orin training + live progress + export/hot-swap.**

## Deferred (v-next)

- Learned embedding re-ID (instead of color histogram) for tougher cases.
- Training on the Mac / cloud as an alternative backend.
- More than 2 identities; non-cat custom labels.
