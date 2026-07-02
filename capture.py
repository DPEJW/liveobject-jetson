"""Auto-capture helpers for building a per-cat training set.

Pure stdlib so it unit-tests off-device (`python3 tests/test_capture.py`).
The actual frame/label file writing + gating lives in detector.py.
"""
from __future__ import annotations


def to_yolo_label(box, frame_w, frame_h, cls_idx):
    """Format one detection as a YOLO label line: '<idx> cx cy w h' (normalized)."""
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2.0) / frame_w
    cy = ((y0 + y1) / 2.0) / frame_h
    w = (x1 - x0) / frame_w
    h = (y1 - y0) / frame_h
    return f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def should_capture(now, last_ts, min_interval=0.6):
    """Time-spacing dedup: capture only if enough time passed since the last one
    for this identity (keeps the set varied instead of 300 near-identical frames)."""
    return last_ts is None or (now - last_ts) >= min_interval
