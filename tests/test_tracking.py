"""Tests for tracking.py (stable IDs, session metrics, zones, hit-test).

Pure numpy + stdlib (no cv2), run from the repo root:
    python3 tests/test_tracking.py
"""
import os
import sys

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


if __name__ == "__main__":
    import traceback

    funcs = [f for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for f in funcs:
        try:
            f()
            print(f"PASS {f.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {f.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
