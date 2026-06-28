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
