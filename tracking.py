"""Single-object motion tracking: stable IDs, a per-object motion/dwell session,
auto/managed named zones, and a heatmap accumulator.

Pure numpy + stdlib (NO cv2) so the logic is unit-testable off-device:
    python3 tests/test_tracking.py
cv2 is only used by the rendering helpers in detector.py, not here.
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
        """dets: list of {'name','score','box'} -> same dicts with 'id','label' added."""
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
                t = Track(self._next_id, d["name"], label, d["box"], d["score"],
                          self._frame)
                self.tracks[self._next_id] = t
                self._next_id += 1
            out.append({**d, "id": t.id, "label": t.label})

        for tid in [tid for tid, t in self.tracks.items()
                    if self._frame - t.last_seen > self.max_age]:
            del self.tracks[tid]
        return out

    def get(self, tid):
        return self.tracks.get(tid)


class TrackSession:
    """Accumulates motion + dwell metrics for one selected track over time."""

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
