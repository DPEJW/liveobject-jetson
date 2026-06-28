# Object Motion Tracking — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let a user select one detected object (click on the video or pick from a list) and follow it over time, reporting how long it moved (moving vs still), how far (px + frame-widths now, meters later), and where (a heatmap + dwell time over auto-detected, user-editable named places).

**Architecture:** All tracking math runs server-side in the worker thread; the browser stays a thin MJPEG `<img>` + polling panel. A new `tracking.py` module holds four GPU-free, numpy-only pieces (greedy IoU tracker, per-object session accumulator, zone registry, click hit-test) so they are unit-testable off-device. `detector.py` wires them into the frame loop and draws overlays; `app.py` exposes control endpoints; the frontend adds a Tracking card.

**Tech Stack:** Python 3.8, NumPy, OpenCV (cv2, drawing only), Flask, TensorRT YOLO26 (existing), vanilla JS + Chart.js (existing).

**Design doc:** `docs/plans/2026-06-28-object-tracking-design.md`

**Conventions:**
- Tests are self-running (no pytest): a `__main__` block runs every `test_*` func and exits non-zero on failure. Run with `python3 tests/test_tracking.py`.
- `tracking.py` core classes must NOT import cv2 (keeps tests runnable on the Mac). cv2 is only for rendering in `detector.py`.
- Frame coordinate space = the **annotated frame** dims (`DISPLAY` 1280×720 at rotation 0). The worker stores current `self._fw/self._fh` each frame.
- Commit after every green step. Do NOT add a Claude co-author trailer (repo convention).
- Deploy is scp → `sudo systemctl restart liveobject` on `orinnx1` (passwordless systemctl). SSH/scp/LAN calls need `dangerouslyDisableSandbox=true`.
- Use @superpowers:test-driven-development for the test-first tasks and @frontend-design for the UI task.

---

## Task 1: IoU + centroid helpers and `IoUTracker`

**Files:**
- Create: `tracking.py`
- Test: `tests/test_tracking.py`

**Step 1 — Write the failing test**

```python
# tests/test_tracking.py
"""Tests for tracking.py (stable IDs, session metrics, zones, hit-test).

Pure numpy + stdlib (no cv2), run from the repo root:
    python3 tests/test_tracking.py
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking import iou, centroid, IoUTracker


def test_iou_and_centroid_basics():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0
    assert centroid([0, 0, 10, 20]) == (5.0, 10.0)


def test_same_object_keeps_id_across_frames():
    tr = IoUTracker(iou_thresh=0.3, max_age=15)
    a = tr.update([{"name": "cat", "score": .9, "box": [10, 10, 50, 50]}])
    b = tr.update([{"name": "cat", "score": .9, "box": [12, 11, 52, 51]}])
    assert a[0]["id"] == b[0]["id"]
    assert a[0]["label"] == "cat #1"


def test_new_object_gets_new_id():
    tr = IoUTracker()
    tr.update([{"name": "cat", "score": .9, "box": [10, 10, 50, 50]}])
    out = tr.update([
        {"name": "cat", "score": .9, "box": [11, 10, 51, 50]},
        {"name": "cat", "score": .8, "box": [200, 200, 240, 240]},
    ])
    ids = sorted(d["id"] for d in out)
    assert ids == [1, 2], ids


def test_track_expires_after_max_age():
    tr = IoUTracker(max_age=2)
    tr.update([{"name": "cat", "score": .9, "box": [10, 10, 50, 50]}])
    for _ in range(3):
        tr.update([])               # object missing
    out = tr.update([{"name": "cat", "score": .9, "box": [10, 10, 50, 50]}])
    assert out[0]["id"] == 2, "expired track should mint a fresh id"


def test_same_spot_different_class_is_a_different_track():
    tr = IoUTracker()
    a = tr.update([{"name": "cat", "score": .9, "box": [10, 10, 50, 50]}])
    b = tr.update([{"name": "dog", "score": .9, "box": [10, 10, 50, 50]}])
    assert a[0]["id"] != b[0]["id"]
```

**Step 2 — Run, verify it fails**

Run: `python3 tests/test_tracking.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'tracking'`.

**Step 3 — Implement `tracking.py` (helpers + tracker)**

```python
# tracking.py
"""Single-object motion tracking: stable IDs, a per-object motion/dwell session,
auto/managed named zones, and a heatmap accumulator.

Pure numpy + stdlib (NO cv2) so the logic is unit-testable off-device:
    python3 tests/test_tracking.py
"""
from __future__ import annotations
import math
import time
from collections import defaultdict


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid(box):
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


class Track:
    __slots__ = ("id", "cls", "label", "box", "score", "first_seen", "last_seen")

    def __init__(self, tid, cls, label, box, score, frame_idx):
        self.id, self.cls, self.label = tid, cls, label
        self.box, self.score = list(box), score
        self.first_seen = self.last_seen = frame_idx


class IoUTracker:
    """Greedy IoU + centroid tracker; same-class matching only."""

    def __init__(self, iou_thresh=0.3, max_age=15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = {}                      # id -> Track
        self._next_id = 1
        self._frame = 0
        self._counts = defaultdict(int)       # cls -> labels minted so far

    def update(self, dets):
        """dets: list of {'name','score','box'} -> same list with 'id','label' added."""
        self._frame += 1
        live = list(self.tracks.values())

        pairs = []
        for di, d in enumerate(dets):
            for t in live:
                if t.cls == d["name"]:
                    ov = iou(d["box"], t.box)
                    if ov >= self.iou_thresh:
                        pairs.append((ov, di, t.id))
        pairs.sort(reverse=True)

        matched, used = {}, set()
        for ov, di, tid in pairs:
            if di not in matched and tid not in used:
                matched[di] = tid
                used.add(tid)

        out = []
        for di, d in enumerate(dets):
            if di in matched:
                t = self.tracks[matched[di]]
                t.box, t.score, t.last_seen = list(d["box"]), d["score"], self._frame
            else:
                self._counts[d["name"]] += 1
                label = f'{d["name"]} #{self._counts[d["name"]]}'
                t = Track(self._next_id, d["name"], label, d["box"], d["score"], self._frame)
                self.tracks[self._next_id] = t
                self._next_id += 1
            out.append({**d, "id": t.id, "label": t.label})

        for tid in [tid for tid, t in self.tracks.items()
                    if self._frame - t.last_seen > self.max_age]:
            del self.tracks[tid]
        return out

    def get(self, tid):
        return self.tracks.get(tid)
```

**Step 4 — Run, verify pass**

Run: `python3 tests/test_tracking.py`
Expected: PASS for all `test_*` so far.

**Step 5 — Commit**

```bash
git add tracking.py tests/test_tracking.py
git commit -m "feat(tracking): greedy IoU tracker with stable per-class IDs"
```

---

## Task 2: `TrackSession` (duration, distance, dead-band, net)

**Files:**
- Modify: `tracking.py`
- Test: `tests/test_tracking.py`

**Step 1 — Add failing tests**

```python
from tracking import TrackSession   # add to imports

def test_still_object_accrues_no_distance():
    s = TrackSession(1, "cat #1", frame_w=1280, frame_h=720)
    for _ in range(10):
        s.update([100, 100, 140, 140], dt=0.1, zones=[])   # identical box
    assert s.path_px < 1.0
    assert s.still_s > 0.9 and s.moving_s == 0.0

def test_moving_object_accumulates_distance_and_moving_time():
    s = TrackSession(1, "cat #1", frame_w=1280, frame_h=720)
    x = 100
    for _ in range(10):
        s.update([x, 100, x + 40, 140], dt=0.1, zones=[])
        x += 40                                            # 40 px/step >> dead-band
    assert s.path_px > 300
    assert s.moving_s > 0.8
    assert round(s.net_px()) == 360                        # 9 steps * 40

def test_zone_dwell_counts_seconds_in_smallest_containing_zone():
    s = TrackSession(1, "cat #1", frame_w=1280, frame_h=720)
    zones = [{"label": "room", "box": [0, 0, 1280, 720]},
             {"label": "couch", "box": [80, 80, 200, 200]}]
    for _ in range(5):
        s.update([100, 100, 140, 140], dt=1.0, zones=zones)
    by = {z["label"]: z["dwell_s"] for z in s.summary()["zones"]}
    assert by.get("couch", 0) >= 4.0           # smallest containing zone wins
    assert s.summary()["current_zone"] == "couch"

def test_lost_then_reacquire_does_not_count_the_jump():
    s = TrackSession(1, "cat #1", frame_w=1280, frame_h=720)
    s.update([100, 100, 140, 140], dt=0.1, zones=[])
    s.mark_lost()
    s.update([900, 600, 940, 640], dt=0.1, zones=[])       # big jump after loss
    assert s.path_px < 1.0, "the re-acquire jump must not count as travel"
```

**Step 2 — Run, verify it fails**

Run: `python3 tests/test_tracking.py`
Expected: FAIL — `ImportError: cannot import name 'TrackSession'`.

**Step 3 — Implement `TrackSession` (append to `tracking.py`)**

```python
class TrackSession:
    def __init__(self, track_id, label, frame_w, frame_h,
                 dead_band_frac=0.005, heatmap_size=(64, 36), meters_per_pixel=None):
        import numpy as np
        self._np = np
        self.track_id, self.label = track_id, label
        self.frame_w, self.frame_h = frame_w, frame_h
        self.diag = math.hypot(frame_w, frame_h)
        self.dead_band = dead_band_frac * self.diag
        self.meters_per_pixel = meters_per_pixel

        self.state = "active"                 # 'active' | 'lost' | 'stopped'
        self.elapsed_s = self.moving_s = self.still_s = 0.0
        self.path_px = 0.0
        self._start_c = self._last_c = None
        self._skip_step = False
        self.trail, self._trail_cap = [], 256

        self.hm_w, self.hm_h = heatmap_size
        self.heat = np.zeros((self.hm_h, self.hm_w), dtype=np.float32)
        self.zone_dwell = defaultdict(float)
        self.current_zone = None

    def update(self, box, dt, zones):
        self.state = "active"
        c = centroid(box)
        if self._start_c is None:
            self._start_c = c
        self.elapsed_s += dt
        if self._last_c is not None and not self._skip_step:
            step = math.hypot(c[0] - self._last_c[0], c[1] - self._last_c[1])
            if step >= self.dead_band:
                self.path_px += step
                self.moving_s += dt
            else:
                self.still_s += dt
        self._skip_step = False
        self._last_c = c

        self.trail.append(c)
        if len(self.trail) > self._trail_cap:
            self.trail = self.trail[-self._trail_cap:]

        gx = min(self.hm_w - 1, max(0, int(c[0] / self.frame_w * self.hm_w)))
        gy = min(self.hm_h - 1, max(0, int(c[1] / self.frame_h * self.hm_h)))
        self.heat[gy, gx] += dt

        cur, best = None, None
        for z in zones:
            x0, y0, x1, y1 = z["box"]
            if x0 <= c[0] <= x1 and y0 <= c[1] <= y1:
                area = (x1 - x0) * (y1 - y0)
                if best is None or area < best:
                    best, cur = area, z["label"]
        self.current_zone = cur
        if cur is not None:
            self.zone_dwell[cur] += dt

    def mark_lost(self):
        self.state = "lost"
        self._skip_step = True            # ignore the next step on re-acquire

    def stop(self):
        self.state = "stopped"

    def net_px(self):
        if self._start_c is None or self._last_c is None:
            return 0.0
        return math.hypot(self._last_c[0] - self._start_c[0],
                          self._last_c[1] - self._start_c[1])

    def summary(self):
        d = {
            "id": self.track_id, "label": self.label, "state": self.state,
            "elapsed_s": round(self.elapsed_s, 1),
            "moving_s": round(self.moving_s, 1), "still_s": round(self.still_s, 1),
            "dist_px": round(self.path_px, 1),
            "dist_frames": round(self.path_px / self.frame_w, 2),
            "net_px": round(self.net_px(), 1),
            "net_frames": round(self.net_px() / self.frame_w, 2),
            "current_zone": self.current_zone,
            "zones": [{"label": k, "dwell_s": round(v, 1)}
                      for k, v in sorted(self.zone_dwell.items(),
                                         key=lambda kv: kv[1], reverse=True)],
        }
        if self.meters_per_pixel:
            d["dist_m"] = round(self.path_px * self.meters_per_pixel, 2)
            d["net_m"] = round(self.net_px() * self.meters_per_pixel, 2)
        return d
```

**Step 4 — Run, verify pass**  → `python3 tests/test_tracking.py` → PASS.

**Step 5 — Commit**

```bash
git add tracking.py tests/test_tracking.py
git commit -m "feat(tracking): TrackSession metrics (dead-band distance, moving/still, dwell, heatmap)"
```

---

## Task 3: `ZoneRegistry` (auto furniture + manual edits)

**Files:** Modify `tracking.py`; Test `tests/test_tracking.py`

**Step 1 — Add failing tests**

```python
from tracking import ZoneRegistry

def test_auto_zone_created_and_smoothed():
    zr = ZoneRegistry(ema=0.5, expire_s=30)
    zr.update_auto([{"name": "couch", "score": .9, "box": [100, 100, 300, 200]}], now=0)
    zr.update_auto([{"name": "couch", "score": .9, "box": [110, 100, 310, 200]}], now=1)
    z = zr.list()
    assert len(z) == 1 and z[0]["label"] == "couch" and z[0]["source"] == "auto"
    assert 104 <= z[0]["box"][0] <= 106          # EMA between 100 and 110

def test_auto_zone_survives_brief_absence_then_expires():
    zr = ZoneRegistry(expire_s=10)
    zr.update_auto([{"name": "couch", "score": .9, "box": [0, 0, 50, 50]}], now=0)
    zr.update_auto([], now=5);  assert len(zr.list()) == 1     # still there
    zr.update_auto([], now=20); assert len(zr.list()) == 0     # expired

def test_manual_zone_add_rename_delete_and_no_expire():
    zr = ZoneRegistry(expire_s=1)
    zid = zr.add("food bowl", [10, 10, 60, 60])
    zr.update_auto([], now=999)                  # manual must NOT expire
    assert any(z["label"] == "food bowl" for z in zr.list())
    zr.rename(zid, "bowl"); assert any(z["label"] == "bowl" for z in zr.list())
    zr.delete(zid); assert zr.list() == []
```

**Step 2 — Run, verify it fails** (`ImportError: ... 'ZoneRegistry'`).

**Step 3 — Implement `ZoneRegistry` (append to `tracking.py`)**

```python
class ZoneRegistry:
    def __init__(self, ema=0.2, expire_s=30.0, match_iou=0.3):
        self.ema, self.expire_s, self.match_iou = ema, expire_s, match_iou
        self.zones = {}                  # id -> dict
        self._next_id = 1
        self._counts = defaultdict(int)

    def _match(self, cls, box):
        best, best_ov = None, 0.0
        for z in self.zones.values():
            if z["source"] == "auto" and z["cls"] == cls:
                ov = iou(box, z["box"])
                if ov > best_ov:
                    best, best_ov = z, ov
        return best if best_ov >= self.match_iou else None

    def update_auto(self, furniture_dets, now=None):
        now = time.time() if now is None else now
        for d in furniture_dets:
            z = self._match(d["name"], d["box"])
            if z is None:
                self._counts[d["name"]] += 1
                n = self._counts[d["name"]]
                label = d["name"] if n == 1 else f'{d["name"]} #{n}'
                zid = self._next_id; self._next_id += 1
                self.zones[zid] = {"id": zid, "label": label, "box": list(d["box"]),
                                   "source": "auto", "cls": d["name"], "last_seen": now}
            else:
                a = self.ema
                z["box"] = [(1 - a) * o + a * n for o, n in zip(z["box"], d["box"])]
                z["last_seen"] = now
        for zid in [zid for zid, z in self.zones.items()
                    if z["source"] == "auto" and now - z["last_seen"] > self.expire_s]:
            del self.zones[zid]

    def add(self, label, box):
        zid = self._next_id; self._next_id += 1
        self.zones[zid] = {"id": zid, "label": label, "box": list(box),
                           "source": "manual", "cls": None, "last_seen": time.time()}
        return zid

    def rename(self, zid, label):
        if zid in self.zones:
            self.zones[zid]["label"] = label

    def delete(self, zid):
        self.zones.pop(zid, None)

    def list(self):
        return [{"id": z["id"], "label": z["label"],
                 "box": [int(round(v)) for v in z["box"]], "source": z["source"]}
                for z in self.zones.values()]
```

**Step 4 — Run, verify pass.  Step 5 — Commit**

```bash
git add tracking.py tests/test_tracking.py
git commit -m "feat(tracking): hybrid ZoneRegistry (auto furniture EMA/expire + manual add/rename/delete)"
```

---

## Task 4: `hit_test` (click → track id)

**Files:** Modify `tracking.py`; Test `tests/test_tracking.py`

**Step 1 — Add failing tests**

```python
from tracking import hit_test

def test_hit_test_picks_smallest_containing_box():
    tracks = [{"id": 1, "box": [0, 0, 1280, 720]},
              {"id": 2, "box": [600, 300, 700, 420]}]
    assert hit_test(tracks, 0.51, 0.5, 1280, 720) == 2     # inside small box
    assert hit_test(tracks, 0.05, 0.05, 1280, 720) == 1    # only big box
    assert hit_test(tracks, 0.99, 0.99, 1280, 720) == 1
    assert hit_test([], 0.5, 0.5, 1280, 720) is None
```

**Step 2 — Run, verify fail.  Step 3 — Implement (append to `tracking.py`)**

```python
def hit_test(tracks, nx, ny, frame_w, frame_h):
    """nx, ny normalized [0,1]; returns id of smallest box containing the point."""
    px, py = nx * frame_w, ny * frame_h
    best, best_area = None, None
    for t in tracks:
        x0, y0, x1, y1 = t["box"]
        if x0 <= px <= x1 and y0 <= py <= y1:
            area = (x1 - x0) * (y1 - y0)
            if best_area is None or area < best_area:
                best, best_area = t["id"], area
    return best
```

**Step 4 — Run, verify all green.  Step 5 — Commit**

```bash
git add tracking.py tests/test_tracking.py
git commit -m "feat(tracking): click hit-test (normalized point -> track id)"
```

---

## Task 5: Wire tracker + zones + session into the worker

**Files:** Modify `detector.py`

**Step 1 — Imports & constants.** Near the top of `detector.py` (after existing imports):

```python
from tracking import IoUTracker, TrackSession, ZoneRegistry, hit_test

# COCO classes used as auto "named places"
ZONE_CLASSES = {"chair", "couch", "bed", "dining table"}
```

**Step 2 — Worker state.** In `DetectionWorker.__init__` (after `self.detections = []`, ~line 253) add:

```python
        self.tracker = IoUTracker()
        self.zones = ZoneRegistry()
        self._selected_id = None
        self._session = None
        self._last_tracks = []           # last tracked dets (for click hit-test)
        self._fw, self._fh = 1280, 720   # current annotated-frame size
        self._t_prev = None              # perf_counter of previous frame
        self.show_trail = True
        self.show_heatmap = True
        self.show_zones = True
```

**Step 3 — Per-frame integration in `_run`.** Replace the block (currently ~lines 396-406):

```python
                    dets = self._to_dets(raw)
                    annotated = self._draw(frame, dets)

                    ok, buf = cv2.imencode(".jpg", annotated,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with self._cond:
                            self._jpeg = buf.tobytes()
                            self._frame_id += 1
                            self.detections = dets
                            self._cond.notify_all()
```

with:

```python
                    dets = self._to_dets(raw)
                    tracked = self.tracker.update(dets)
                    self._fh, self._fw = frame.shape[0], frame.shape[1]

                    now = time.perf_counter()
                    dt = (now - self._t_prev) if self._t_prev else 0.0
                    self._t_prev = now

                    self.zones.update_auto([d for d in tracked
                                            if d["name"] in ZONE_CLASSES])
                    self._update_session(tracked, dt)

                    annotated = self._draw(frame, tracked)

                    ok, buf = cv2.imencode(".jpg", annotated,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with self._cond:
                            self._jpeg = buf.tobytes()
                            self._frame_id += 1
                            self.detections = tracked
                            self._last_tracks = tracked
                            self._cond.notify_all()
```

**Step 4 — Session updater + control methods.** Add these methods to `DetectionWorker`:

```python
    def _update_session(self, tracked, dt):
        sel, sess = self._selected_id, self._session
        if sel is None or sess is None or sess.state == "stopped":
            return
        t = self.tracker.get(sel)
        if t is not None:
            sess.update(t.box, dt, self.zones.list())
        else:
            sess.mark_lost()

    def request_select(self, track_id=None, x=None, y=None):
        with self._lock:
            if track_id is None and x is not None and y is not None:
                track_id = hit_test(self._last_tracks, float(x), float(y),
                                    self._fw, self._fh)
            if track_id is None:
                return {"selected": None}
            t = self.tracker.get(int(track_id))
            label = t.label if t else f"#{track_id}"
            self._selected_id = int(track_id)
            self._session = TrackSession(int(track_id), label, self._fw, self._fh)
            self._t_prev = None
            return {"selected": self._selected_id, "label": label}

    def request_stop(self):
        with self._lock:
            if self._session is not None:
                self._session.stop()
            self._selected_id = None
            return {"selected": None}

    def zone_add(self, label, box):
        with self._lock:
            return {"id": self.zones.add(str(label), [int(v) for v in box])}

    def zone_rename(self, zid, label):
        with self._lock:
            self.zones.rename(int(zid), str(label)); return {"ok": True}

    def zone_delete(self, zid):
        with self._lock:
            self.zones.delete(int(zid)); return {"ok": True}

    def _reset_tracking(self):
        """Scene changed (camera/rotation): drop session + zones."""
        self._selected_id = None
        self._session = None
        self.zones = ZoneRegistry()
        self._last_tracks = []
```

**Step 5 — Reset on scene change.** In `set_config`, inside the `if rotation is not None:` branch, after setting `self.rotation`, add `self._reset_tracking()`. In `request_camera`, inside the lock before `self._pending_camera = True`, add `self._reset_tracking()`.

**Step 6 — Extend `config()` / stats fields.** Add to the dict returned by `config()`:

```python
            "track_trail": self.show_trail,
            "track_heatmap": self.show_heatmap,
            "track_zones": self.show_zones,
```

And add a new method used by `/stats`:

```python
    def tracking_state(self):
        with self._lock:
            tracks = [{"id": d["id"], "label": d["label"], "cls": d["name"],
                       "score": round(d["score"], 2), "box": [int(v) for v in d["box"]],
                       "selected": d["id"] == self._selected_id}
                      for d in self._last_tracks]
            summary = self._session.summary() if self._session else None
            return {"tracks": tracks, "track": summary, "zones": self.zones.list()}
```

**Step 7 — Manual smoke check** (no unit test; needs cv2/GPU). On the Mac just byte-compile:

Run: `python3 -c "import ast; ast.parse(open('detector.py').read()); print('detector.py parses')"`
Expected: `detector.py parses`

**Step 8 — Commit**

```bash
git add detector.py
git commit -m "feat(detector): wire tracker, zones, and selected-object session into the frame loop"
```

---

## Task 6: Overlays — IDs, selected highlight, trail, zones, heatmap

**Files:** Modify `detector.py` (`_draw` + a heatmap renderer)

**Step 1 — Replace `_draw`** with a version that labels track IDs, highlights the selected track, and draws trail/zones/heatmap behind the boxes:

```python
    def _draw(self, frame, dets):
        out = frame.copy()
        if self.show_heatmap and self._session is not None:
            out = self._render_heatmap(out, self._session.heat)
        if self.show_zones:
            for z in self.zones.list():
                x0, y0, x1, y1 = z["box"]
                cv2.rectangle(out, (x0, y0), (x1, y1), (90, 200, 255), 1)
                cv2.putText(out, z["label"], (x0 + 3, y0 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 200, 255), 1, cv2.LINE_AA)
        if self.show_trail and self._session is not None and len(self._session.trail) > 1:
            import numpy as np
            pts = np.array(self._session.trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, (79, 214, 198), 2, cv2.LINE_AA)
        for d in dets:
            x0, y0, x1, y1 = [int(v) for v in d["box"]]
            sel = d.get("id") == self._selected_id
            color = (79, 214, 198) if sel else _color_for(d["name"])
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 3 if sel else 2)
            label = f'{d.get("label", d["name"])} {d["score"] * 100:.0f}%'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), color, -1)
            cv2.putText(out, label, (x0 + 2, max(10, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def _render_heatmap(self, frame, heat, alpha=0.45):
        import numpy as np
        m = float(heat.max())
        if m <= 0:
            return frame
        norm = np.clip(heat / m, 0, 1)
        small = (norm * 255).astype(np.uint8)
        big = cv2.resize(small, (frame.shape[1], frame.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
        cmap = cv2.applyColorMap(big, cv2.COLORMAP_TURBO)
        mask = (big > 8)[:, :, None]
        return np.where(mask, cv2.addWeighted(frame, 1 - alpha, cmap, alpha, 0), frame)
```

**Step 2 — Toggle handling in `set_config`.** Extend the signature and body:

```python
    def set_config(self, max_detections=None, threshold=None, paused=None,
                   rotation=None, flip_h=None, flip_v=None,
                   track_trail=None, track_heatmap=None, track_zones=None):
        ...
            if track_trail is not None:   self.show_trail = bool(track_trail)
            if track_heatmap is not None: self.show_heatmap = bool(track_heatmap)
            if track_zones is not None:   self.show_zones = bool(track_zones)
```

**Step 3 — Smoke check:** `python3 -c "import ast; ast.parse(open('detector.py').read()); print('ok')"` → `ok`.

**Step 4 — Commit**

```bash
git add detector.py
git commit -m "feat(detector): draw track IDs, selected highlight, trail, zones, and heatmap"
```

---

## Task 7: Flask endpoints (`/track`, `/zones`, enriched `/stats`, toggles)

**Files:** Modify `app.py`

**Step 1 — Enrich `/stats`.** In the `stats()` view, merge tracking state into the JSON:

```python
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
```

**Step 2 — Toggles via `/config`.** In `set_config()` add to the `worker.set_config(...)` call:

```python
        track_trail=data.get("track_trail"),
        track_heatmap=data.get("track_heatmap"),
        track_zones=data.get("track_zones"),
```

**Step 3 — New endpoints** (add before `if __name__`):

```python
@app.route("/track", methods=["POST"])
def track():
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")
    if action == "stop":
        return jsonify(worker.request_stop())
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
```

**Step 4 — Smoke check:** `python3 -c "import ast; ast.parse(open('app.py').read()); print('ok')"` → `ok`.

**Step 5 — Commit**

```bash
git add app.py
git commit -m "feat(api): /track + /zones endpoints, tracking fields in /stats, overlay toggles"
```

---

## Task 8: Frontend — Tracking card (HTML/CSS)

**Files:** Modify `templates/index.html`, `static/style.css`. Read @frontend-design first to match the instrument-panel system (CSS vars `--amber`, `--cyan`, `--mono`, etc.).

**Step 1 — Add the Tracking card** in `templates/index.html` as the FIRST card in `<aside class="panel">` (before the Camera card):

```html
      <div class="card" data-reveal="1">
        <h2>Tracking <span id="trk-state" class="tag">idle</span></h2>
        <ul id="trk-list" class="trk-list"><li class="muted">no objects yet</li></ul>
        <div class="trk-readout hidden" id="trk-readout">
          <div class="ro"><span class="ro-k">ELAPSED</span><b id="trk-elapsed">0s</b></div>
          <div class="ro"><span class="ro-k">MOVING</span><b id="trk-moving">0s</b></div>
          <div class="ro"><span class="ro-k">STILL</span><b id="trk-still">0s</b></div>
          <div class="ro"><span class="ro-k">DIST</span><b id="trk-dist">0</b></div>
          <div class="ro"><span class="ro-k">PLACE</span><b id="trk-place" class="alt">—</b></div>
        </div>
        <ul id="trk-zones" class="det-list"></ul>
        <div class="btn-row">
          <button id="trk-stop" class="btn hidden">STOP TRACKING</button>
        </div>
        <div class="btn-row trk-toggles">
          <button id="tg-trail"   class="btn ghost on">TRAIL</button>
          <button id="tg-heatmap" class="btn ghost on">HEATMAP</button>
          <button id="tg-zones"   class="btn ghost on">ZONES</button>
        </div>
      </div>
```

Renumber the existing Camera/Detection/Telemetry/Detections `data-reveal` values to 2–5.

**Step 2 — Make the video clickable.** On the `<img id="stream">` element add `style="cursor:crosshair"` (the click handler is wired in Task 9).

**Step 3 — CSS** (append to `static/style.css`):

```css
/* ---- tracking card ---- */
.trk-list { list-style: none; margin: 0 0 10px; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.trk-list li { display: flex; align-items: center; gap: 8px; padding: 6px 9px; background: var(--inset);
               cursor: pointer; font-size: 12px; letter-spacing: .5px; }
.trk-list li.sel { outline: 1px solid var(--cyan); color: var(--cyan); }
.trk-list li .sc { margin-left: auto; color: var(--ink-dim); }
.trk-readout { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; margin-bottom: 8px; }
.trk-toggles .btn.on { color: var(--cyan); border-color: var(--cyan); }
.hidden { display: none !important; }
```

**Step 4 — Visual check after deploy** (Task 10). For now: `python3 -c "print('html/css edited')"`.

**Step 5 — Commit**

```bash
git add templates/index.html static/style.css
git commit -m "feat(ui): Tracking card markup + instrument-panel styling"
```

---

## Task 9: Frontend — behavior (`static/app.js`)

**Files:** Modify `static/app.js`

**Step 1 — State + helpers.** Near the top:

```javascript
let trackSel = null;
const fmtDur = (s) => s >= 60 ? `${Math.floor(s/60)}m${String(Math.round(s%60)).padStart(2,"0")}s` : `${s.toFixed(0)}s`;
function postTrack(body){ return fetch("/track",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(r=>r.json()); }
```

**Step 2 — Click-to-select on the video:**

```javascript
$("stream").addEventListener("click", (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  const nx = (e.clientX - r.left) / r.width, ny = (e.clientY - r.top) / r.height;
  postTrack({ action: "select", x: nx, y: ny });
});
$("trk-stop").addEventListener("click", () => { postTrack({ action: "stop" }); });
```

**Step 3 — Toggle buttons:**

```javascript
function wireToggle(id, key){
  $(id).addEventListener("click", () => {
    const on = !$(id).classList.contains("on");
    $(id).classList.toggle("on", on);
    postConfig({ [key]: on });
  });
}
wireToggle("tg-trail","track_trail"); wireToggle("tg-heatmap","track_heatmap"); wireToggle("tg-zones","track_zones");
```

**Step 4 — Render list + summary inside the existing `poll()`** (after detections render):

```javascript
    // ---- tracking ----
    const tracks = s.tracks || [];
    const tl = $("trk-list");
    tl.innerHTML = tracks.length
      ? tracks.map(t => `<li data-id="${t.id}" class="${t.selected?"sel":""}">`
          + `<span class="swatch" style="background:${colorFor(t.cls)}"></span>`
          + `<span class="nm">${t.label}</span><span class="sc">${(t.score*100).toFixed(0)}%</span></li>`).join("")
      : '<li class="muted">no objects yet</li>';
    tl.querySelectorAll("li[data-id]").forEach(li =>
      li.addEventListener("click", () => postTrack({ action:"select", id:+li.dataset.id })));

    const tk = s.track;
    $("trk-readout").classList.toggle("hidden", !tk);
    $("trk-stop").classList.toggle("hidden", !tk || tk.state === "stopped");
    $("trk-state").textContent = tk ? tk.state : "idle";
    if (tk) {
      $("trk-elapsed").textContent = fmtDur(tk.elapsed_s);
      $("trk-moving").textContent  = fmtDur(tk.moving_s);
      $("trk-still").textContent   = fmtDur(tk.still_s);
      $("trk-dist").textContent    = `${tk.dist_px.toFixed(0)}px · ${tk.dist_frames}×`;
      $("trk-place").textContent   = tk.current_zone || "—";
      $("trk-zones").innerHTML = (tk.zones||[]).map(z =>
        `<li><span class="nm">${z.label}</span><span class="sc">${fmtDur(z.dwell_s)}</span></li>`).join("");
    }
```

**Step 5 — Sync toggle button state in `applyConfigOnce(cfg)`:**

```javascript
  $("tg-trail").classList.toggle("on", cfg.track_trail);
  $("tg-heatmap").classList.toggle("on", cfg.track_heatmap);
  $("tg-zones").classList.toggle("on", cfg.track_zones);
```

**Step 6 — Commit**

```bash
git add static/app.js
git commit -m "feat(ui): tracking list, click-to-select, live summary, overlay toggles"
```

> Zone rename/add-by-draw UI is the last UI slice — implement after the core loop is verified live (keeps risk low). Add a "draw zone" mode that drags a rectangle on the `<img>`, prompts for a name, and POSTs `/zones {action:'add'}`; clicking a zone label prompts a rename. Commit separately.

---

## Task 10: Deploy to the Orin and verify live

**Files:** none (deploy only). Requires `dangerouslyDisableSandbox=true`.

**Step 1 — Run the unit suite on the Mac**

Run: `python3 tests/test_tracking.py`
Expected: `N/N passed`, exit 0.

**Step 2 — Copy changed files + restart**

```bash
scp tracking.py detector.py app.py orinnx1:/home/orinnx1/projects/liveobject/
scp templates/index.html orinnx1:/home/orinnx1/projects/liveobject/templates/
scp static/style.css static/app.js orinnx1:/home/orinnx1/projects/liveobject/static/
ssh orinnx1 'sudo systemctl restart liveobject; sleep 4; systemctl is-active liveobject'
```
Expected: `active`.

**Step 3 — Verify the API end-to-end** (run on the Orin via `ssh orinnx1 'python3' <<'PY' … PY`):
- `GET /stats` includes `tracks`, `track` (null initially), `zones`.
- Pick a track id from `tracks`, `POST /track {action:select,id}`, poll `/stats` ~3s, assert `track.elapsed_s > 0` and `track.state == "active"`.
- `POST /track {action:stop}`, assert `track.state == "stopped"`.
Expected: prints the asserted values; no exceptions.

**Step 4 — Headless screenshot** with the cached Playwright Chromium (as before) against `http://orinnx1:8000` (or an SSH tunnel) to confirm the Tracking card renders and a selection highlights.

**Step 5 — Commit** (nothing to commit unless deploy revealed fixes; if so, commit them with a `fix(tracking): …` message).

---

## Task 11: Update memory

**Files:** `~/.claude/projects/-Users-vicmini/memory/liveobject-jetson-yolo.md` and `MEMORY.md`

Append a dated section to the project memory summarizing: tracking feature shipped (greedy IoU IDs, TrackSession metrics, hybrid zones, heatmap), new `tracking.py` + `tests/test_tracking.py`, the `/track` and `/zones` endpoints, v2 backlog (meters/calibration, saved sessions, multi-object). Update the `MEMORY.md` index line. Commit (no push needed for the memory repo; push the code repo).

```bash
git add docs/plans/2026-06-28-object-tracking.md
git commit -m "docs: object-tracking implementation plan"
git push
```

---

## Done criteria

- `python3 tests/test_tracking.py` → all green on the Mac.
- On the Orin: select a cat (click or list) → elapsed/moving/still/distance populate; heatmap + trail render; "on the couch / on the table" dwell appears; Stop freezes the summary; toggles work; switching camera/rotation resets cleanly.
- fps stays within ~1–2 of the pre-feature baseline (~13 fps NVDEC).
