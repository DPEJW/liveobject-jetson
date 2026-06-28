"""Tests for tracking.py (stable IDs, session metrics, zones, hit-test).

Pure numpy + stdlib (no cv2), run from the repo root:
    python3 tests/test_tracking.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking import iou, centroid, IoUTracker, TrackSession


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


def test_still_object_accrues_no_distance():
    s = TrackSession(1, "cat #1", frame_w=1280, frame_h=720)
    for _ in range(10):
        s.update([100, 100, 140, 140], dt=0.1, zones=[])   # identical box
    assert s.path_px < 1.0
    assert s.still_s > 0.8 and s.moving_s == 0.0


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
