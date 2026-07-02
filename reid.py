"""Appearance-signature re-identification for named objects (cats).

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
    """A small set of named appearance signatures with thresholded matching."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.sigs = {}        # name -> normalized signature
        self.counts = {}      # name -> samples averaged in

    def enroll(self, name, sig):
        self.sigs[name] = _norm(sig)
        self.counts[name] = 1

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

    def clear(self, name=None):
        if name is None:
            self.sigs.clear()
            self.counts.clear()
        else:
            self.sigs.pop(name, None)
            self.counts.pop(name, None)

    def names(self):
        return list(self.sigs.keys())

    def save(self, path):
        try:
            with open(path, "w") as fh:
                json.dump({n: s.tolist() for n, s in self.sigs.items()}, fh)
        except OSError:
            pass

    def load(self, path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return
        for n, s in data.items():
            self.enroll(n, np.asarray(s, dtype=np.float32))

    def match(self, sig):
        """Best (name, score); name is None if the best score is below threshold."""
        if not self.sigs:
            return (None, 0.0)
        sig = _norm(sig)
        best_name, best = None, 0.0
        for name, s in self.sigs.items():
            sc = similarity(sig, s)
            if sc > best:
                best, best_name = sc, name
        return (best_name, best) if best >= self.threshold else (None, best)

    def assign(self, sigs):
        """Assign each signature to a DISTINCT enrolled name (greedy, global-best,
        threshold-gated). Returns a list aligned to `sigs`; unmatched -> None."""
        names = list(self.sigs.keys())
        result = [None] * len(sigs)
        pairs = []
        for i, sg in enumerate(sigs):
            sg = _norm(sg)
            for name in names:
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
