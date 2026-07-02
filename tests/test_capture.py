"""Tests for capture.py (YOLO label formatting + capture dedup decision).

Pure stdlib; run from the repo root:
    python3 tests/test_capture.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture import to_yolo_label, should_capture


def test_yolo_label_center_normalized():
    # box covering the top-left quarter of a 1280x720 frame, class 0
    assert to_yolo_label([0, 0, 640, 360], 1280, 720, 0) == \
        "0 0.250000 0.250000 0.500000 0.500000"


def test_yolo_label_uses_class_index():
    lab = to_yolo_label([100, 100, 200, 200], 1000, 1000, 3)
    assert lab.startswith("3 ")
    assert lab == "3 0.150000 0.150000 0.100000 0.100000"


def test_should_capture_first_time_true():
    assert should_capture(10.0, None, 0.6) is True


def test_should_capture_respects_min_interval():
    assert should_capture(10.0, 9.9, 0.6) is False    # only 0.1s elapsed
    assert should_capture(10.0, 9.3, 0.6) is True      # 0.7s elapsed


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
