# Single-object motion tracking — design

- **Date:** 2026-06-28
- **Status:** Approved (brainstorming complete; ready for implementation plan)
- **Component:** `liveobject` (Jetson Orin NX, Reolink RTSP + TensorRT YOLO26)
- **Related:** `2026-06-27-rtsp-camera-source-design.md`

## Problem

The app detects objects per frame but has **no notion of identity** — each frame's
detections are independent and merely sorted by score (`DetectionWorker._to_dets`
returns `{"name", "score", "box"}`). The user wants to **select one detected object**
(e.g. a cat) — by clicking it on the video or picking it from a list — and have the
app **follow that specific object over time**, reporting:

1. **How long** it was moving (vs. still).
2. **How far** it moved.
3. **Where** it spent its time — a heatmap plus a plain-language summary keyed to
   named places ("on the couch 4m12s · on the table 1m03s").

## Goals

- Persistent IDs for detected objects so one can be selected and followed.
- A live, incrementally-populated summary: duration (moving/still), distance, place dwell.
- A heatmap overlay of where the selected object spent time.
- Named places defined by a **hybrid** scheme: auto-detected furniture (YOLO classes)
  that the user can rename / delete / add to.
- Selection by **both** clicking the object on the video and picking from a list.

## Non-goals (v1) / deferred to v2

- **Real-world units (meters).** v1 reports on-screen distance (px + frame-widths);
  the data model carries an optional pixels→meters scalar so calibration drops in later.
- **Saved / replayable sessions.** v1 is live-only; Stop freezes the summary; export is
  via an annotated snapshot.
- **Multiple simultaneous tracked objects.** v1 tracks one at a time.
- **Occlusion-robust MOT** (SORT/ByteTrack). v1 uses a greedy IoU tracker behind an
  interface so it can be upgraded if ID-switches prove a problem.

## Approved decisions (from brainstorming)

| Question | Decision |
|----------|----------|
| Distance units | **Both** — on-screen now, meters later (architect for it) |
| "Where" summary | **Heatmap + text** dwell summary by named place |
| Named places | **Hybrid** — auto-detect furniture, user can rename/add/adjust |
| Selection | **Both** — click on video and pick from list |
| Tracker algorithm | **(A)** greedy IoU + centroid, dependency-free |
| Scope | **One object at a time**, **live-only** |

## Architecture

All tracking math runs **server-side in the worker thread**; the browser stays a thin
client (an `<img>` MJPEG view + a polling panel). This matches the current design, where
`_draw` annotates frames server-side.

New module **`tracking.py`** holds four cohesive, GPU-free, NumPy-only pieces:

### 1. `IoUTracker` (+ `Track`)

- Input each frame: list of detections `{name, score, box:[x0,y0,x1,y1]}`.
- Greedy match new boxes to existing tracks **of the same class** by IoU (centroid
  distance as a gate/tiebreak). Matched → update box + `last_seen`. Unmatched detection
  → new `Track` with a new incrementing `id` and a display `label` (`"cat #1"`,
  `"cat #2"`, …, numbered per class). Track unseen for `MAX_AGE` frames (~15) → expire.
- Output: detections annotated with stable `id` + `label`.
- Isolated behind a tiny interface (`update(dets) -> tracks`) so SORT can replace it.

### 2. `TrackSession` (the selected object's accumulator)

Created when the user selects a track id. Per frame the selected id is present:

- `c = centroid(box)`.
- **Distance / movement:** `step = ||c - c_prev||`. A **dead-band**
  (`step < DEAD_BAND_FRAC * frame_diagonal`, ~0.5%) treats the object as *still* and adds
  ~0 distance (kills detector jitter so a sitting cat doesn't accrue fake distance).
  Above the dead-band: add `step` to `path_px`, count `dt` as **moving** time; otherwise
  count `dt` as **still** time.
- **Heatmap:** increment a low-res grid (`64×36`, frame is 16:9) at `c`'s cell
  (optional 3×3 Gaussian spread). Cumulative over the session (no decay).
- **Place dwell:** find the named zone(s) containing `c`; if several, choose the
  smallest-area (most specific); add `dt` to that zone's dwell timer.
- **Exposed metrics:** `elapsed_s`, `moving_s`, `still_s`, `path_px`,
  `path_frames` (= `path_px / frame_width`), `net_px` / `net_frames` (start→end),
  `current_zone`, and `[{zone_label, dwell_s}]`. `meters_per_pixel` is `None` until
  calibration; when set, meters are derived.

### 3. `ZoneRegistry` (hybrid named places)

- **Auto:** each frame, furniture-class detections (default `ZONE_CLASSES =
  {chair, couch, bed, dining table}`, configurable) feed the registry. A new furniture
  box creates a zone with an auto label; a re-detected zone's box is **EMA-smoothed**
  (α≈0.2) so it doesn't jitter. Zones **persist through a long expiry** (~30 s) so the
  cat sitting on / occluding the couch doesn't drop the zone.
- **Manual (hybrid):** `rename(id, label)`, `delete(id)`, and `add(label, box)` (user
  drags a box on the video). Manual zones never auto-expire.
- Each zone: `{id, label, box, source: 'auto'|'manual'}`.
- **Assumption:** the Reolink is fixed/mounted, so the scene is static and zones stay
  valid. (A camera-source switch invalidates zones — see error handling.)

### 4. Heatmap rendering helper

Colorize the accumulation grid (normalize by max → `cv2.applyColorMap`, TURBO/JET),
upscale to frame size, alpha-blend (~0.45) where value > 0. Toggleable.

## Worker integration (`detector.py`)

- Hold a `tracker`, `zones` registry, `_selected_id`, and `_session` (or `None`).
- `_to_dets` → run `tracker.update(dets)` so every detection carries `id`/`label`;
  feed furniture subset to `zones`; if a session is active, `session.update(...)`.
- New control methods (mutate under the existing lock; no GPU/camera work, so applied
  directly rather than via the pending-flag pattern): `request_select(id=None, x=None,
  y=None)` (id, or hit-test normalized click coords against current track boxes →
  topmost/smallest containing), `request_stop()`, `zone_add/zone_rename/zone_delete`,
  and overlay toggles.
- `_draw` gains overlays: track id labels on boxes, **selected track highlighted**
  (thicker/brighter), the selected object's **trail polyline**, **zone outlines + labels**,
  and the **heatmap** blend. Each overlay gated by a toggle.
- `config()` / stats gain the new fields below.

## Server API (`app.py`)

- **Enrich `/stats`** (single existing 500 ms poll — no new polling loop):
  - `tracks`: `[{id, label, cls, score, box, selected}]`
  - `track`: `null` or `{id, label, state:'active'|'lost', elapsed_s, moving_s, still_s,
    dist_px, dist_frames, net_px, net_frames, current_zone, zones:[{label, dwell_s}]}`
  - `zones`: `[{id, label, box, source}]`
  - overlay toggle states.
- `POST /track`: `{action:'select', id}` | `{action:'select', x, y}` (normalized [0..1])
  | `{action:'stop'}`.
- `POST /zones`: `{action:'add', label, box}` | `{action:'rename', id, label}` |
  `{action:'delete', id}`.
- Overlay toggles ride on the existing `POST /config`: `track_trail`, `track_heatmap`,
  `track_zones` (bools).

## Client UI (`templates/index.html`, `static/app.js`, `static/style.css`)

- New **Tracking card** (instrument-panel styling):
  - Object list: `Cat #1 · 94%`, `Person #2 · 88%` → click to select; selected row
    highlighted; **Stop** button.
  - Live readout: elapsed · moving/still · distance (px + frame-widths) · current place.
  - Per-zone dwell list: `couch 4m12s · table 1m03s · floor 2m`.
  - Toggles: **Trail · Heatmap · Zones**.
- **Click-to-select on the video:** map a click on the `<img>` to normalized coords
  `nx=(clientX-rect.left)/rect.width`, `ny=(clientY-rect.top)/rect.height`, POST to
  `/track`. (The `<img>` shows the full 1280×720 annotated frame, so no letterbox-bar
  correction is needed.)
- **Zone editing:** rename (click a zone label → prompt), delete, and "add zone" mode
  (drag a rectangle on the video → name it).

## Data flow

```
camera → frame → YOLO → dets(+box)
      → tracker.update() → tracks(+id,label)
      → zones.update(furniture subset)
      → if session: session.update(centroid, dt, zones)
      → _draw overlays (ids, selected, trail, zones, heatmap) → MJPEG
/stats ← snapshot of {tracks, track summary, zones, toggles}   (client polls 500ms)
client → POST /track (select id|click / stop), POST /zones (add/rename/delete),
         POST /config (toggles)  → mutate under worker lock
```

## Error handling & edge cases

- **Selected object lost** (occluded / leaves frame): session `state='lost'`; pause
  accumulation (the gap counts as neither moving nor still, adds no distance). If the
  same id reappears before tracker expiry → resume. After expiry → stays lost with stats
  intact; user can Stop (freeze) or select again.
- **Camera-source switch:** scene changes → **end the session and clear zones** (their
  boxes are meaningless in the new view).
- **Model reload:** keep session + zones (same scene).
- **Jitter:** dead-band on distance; EMA on zone boxes.
- **Thread-safety:** all tracker/session/zone mutations under the worker lock; `/stats`
  returns a consistent snapshot.
- **Performance:** tracker is O(n·m) over a handful of boxes; heatmap grid is tiny;
  colormap+blend is on a small upscaled grid. Expected fps impact ≤ ~1–2 fps.

## Testing (TDD, NumPy-only, runs on the Mac like existing `tests/`)

`tests/test_tracking.py`:
- **Tracker:** overlapping boxes across frames keep the same id; a new box gets a new id;
  a track expires after `MAX_AGE` missing frames; same location but different class →
  different id.
- **Session:** stationary centroid (within dead-band) → ~0 distance, `still_s` accrues;
  moving centroid → `path_px` ≈ Σ steps, `moving_s` accrues; `net_px` = straight-line.
- **Zones:** furniture det creates a zone; EMA smoothing; survives missing frames within
  expiry; rename/add/delete.
- **Heatmap:** correct cell increments; max-normalization.
- **Click hit-test:** normalized coord → correct track; empty space → none.

## Phasing

- **v1 (this build):** tracker (A) + IDs in UI; click + list selection; session
  (duration/moving/still, distance px + frame-widths, net displacement); heatmap; hybrid
  zones (auto + rename/delete/add-by-draw) with dwell summary; overlay toggles; live
  summary, freeze-on-stop, export via snapshot; one-at-a-time; live-only.
- **v2 (later):** calibration → meters; saved/replayable sessions; multiple simultaneous
  tracks; SORT upgrade if occlusion ID-switches appear.

## Files

- **New:** `tracking.py` (IoUTracker/Track, TrackSession, ZoneRegistry, heatmap),
  `tests/test_tracking.py`.
- **Modified:** `detector.py` (wire tracker/zones/session, ids in `_to_dets`, overlays in
  `_draw`, control methods, stats fields), `app.py` (`/track`, `/zones`, `/config`
  flags, enriched `/stats`), `templates/index.html` + `static/app.js` +
  `static/style.css` (Tracking card, click-to-select, zone editing, summary, toggles).
