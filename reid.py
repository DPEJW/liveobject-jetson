"""Appearance-signature re-identification for named objects (any class).

Originally cat-only; generalized so ANY detected object (person, cat, dog, …)
can be enrolled under a name. Each identity remembers the base class it was
enrolled from, and matching/merging is gated to that class.

The matching core is pure numpy so it unit-tests off-device
(`python3 tests/test_reid.py`). Signature *extraction* from a frame (HSV
histogram of the box interior) lives in detector.py, which has cv2.

A signature is an L1-normalized histogram (1-D here / flattened H-S in practice).
`similarity` is histogram intersection in [0,1] (1.0 == identical).
"""
from __future__ import annotations

import json
import os

import numpy as np

from tracking import iou


def merge_identity(base_dets, identity_dets, base_cls=None, iou_thresh=0.45):
    """Merge identity-model detections (named objects, e.g. 'Dundun') into the
    base COCO detections: an identity det REPLACES any overlapping det of its
    BASE class (the fine-tuned confidence wins); non-overlapping identity dets
    are appended (the identity model found one the base model missed). Other
    classes (couch, zones, …) are never touched.

    `base_cls` maps identity name -> the COCO class it was enrolled from
    (e.g. {"Dundun": "cat", "Jiawei": "person"}); unknown names default to
    "cat" for backward compatibility with cat-only rosters."""
    out = list(base_dets)
    for idd in identity_dets:
        target = (base_cls or {}).get(idd["name"], "cat")
        for d in [d for d in out
                  if d["name"] == target and iou(d["box"], idd["box"]) >= iou_thresh]:
            out.remove(d)
        out.append(idd)
    return out


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    s = float(v.sum())
    return v / s if s > 0 else v


def similarity(a, b):
    """Histogram intersection of two signatures -> [0, 1] (1.0 = identical)."""
    a, b = _norm(a), _norm(b)
    n = min(a.shape[0], b.shape[0])
    return float(np.minimum(a[:n], b[:n]).sum())


class Roster:
    """A small set of named appearance signatures with thresholded matching.

    Each identity may carry the base detection class it was enrolled from
    (`cls`, e.g. "cat" / "person"); matching is then gated so a person's
    signature is never compared against a cat detection. A `cls` of None
    (legacy rosters, unit tests) matches any class."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.sigs = {}        # name -> normalized signature
        self.counts = {}      # name -> samples averaged in
        self.cls = {}         # name -> base class (None = any)

    def _cls_ok(self, name, cls):
        base = self.cls.get(name)
        return base is None or cls is None or base == cls

    def enroll(self, name, sig, cls=None):
        self.sigs[name] = _norm(sig)
        self.counts[name] = 1
        self.cls[name] = cls

    def reinforce(self, name, sig):
        """Fold another sample into an existing identity (running average)."""
        if name not in self.sigs:
            return self.enroll(name, sig)
        c = self.counts[name]
        self.sigs[name] = _norm(self.sigs[name] * c + _norm(sig))
        self.counts[name] = c + 1

    def rename(self, old, new):
        if old in self.sigs and old != new:
            self.sigs[new] = self.sigs.pop(old)
            self.counts[new] = self.counts.pop(old)
            self.cls[new] = self.cls.pop(old, None)

    def clear(self, name=None):
        if name is None:
            self.sigs.clear()
            self.counts.clear()
            self.cls.clear()
        else:
            self.sigs.pop(name, None)
            self.counts.pop(name, None)
            self.cls.pop(name, None)

    def names(self):
        return list(self.sigs.keys())

    def base_cls(self, name):
        return self.cls.get(name)

    def classes(self):
        """name -> base class map (only names that have one)."""
        return {n: c for n, c in self.cls.items() if c}

    def save(self, path):
        try:
            with open(path, "w") as fh:
                json.dump({"__v": 2,
                           "identities": {n: {"sig": s.tolist(),
                                              "cls": self.cls.get(n)}
                                          for n, s in self.sigs.items()}}, fh)
        except OSError:
            pass

    def load(self, path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and data.get("__v") == 2:
            for n, rec in (data.get("identities") or {}).items():
                self.enroll(n, np.asarray(rec["sig"], dtype=np.float32),
                            cls=rec.get("cls"))
        else:                            # legacy flat {name: [sig]} (cat era)
            for n, s in data.items():
                self.enroll(n, np.asarray(s, dtype=np.float32), cls="cat")

    def match(self, sig, cls=None):
        """Best (name, score); name is None if the best score is below threshold.
        Only identities whose base class is compatible with `cls` compete."""
        if not self.sigs:
            return (None, 0.0)
        sig = _norm(sig)
        best_name, best = None, 0.0
        for name, s in self.sigs.items():
            if not self._cls_ok(name, cls):
                continue
            sc = similarity(sig, s)
            if sc > best:
                best, best_name = sc, name
        return (best_name, best) if best >= self.threshold else (None, best)

    def assign(self, sigs, clses=None):
        """Assign each signature to a DISTINCT enrolled name (greedy, global-best,
        threshold-gated). Returns a list aligned to `sigs`; unmatched -> None.
        `clses` (optional, aligned to `sigs`) gates candidates by base class."""
        names = list(self.sigs.keys())
        result = [None] * len(sigs)
        pairs = []
        for i, sg in enumerate(sigs):
            sg = _norm(sg)
            cls = clses[i] if clses else None
            for name in names:
                if not self._cls_ok(name, cls):
                    continue
                pairs.append((similarity(sg, self.sigs[name]), i, name))
        pairs.sort(reverse=True, key=lambda p: p[0])
        used_i, used_n = set(), set()
        for sc, i, name in pairs:
            if i in used_i or name in used_n or sc < self.threshold:
                continue
            result[i] = name
            used_i.add(i)
            used_n.add(name)
        return result
