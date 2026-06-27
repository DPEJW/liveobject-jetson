"""Tests for detector.letterbox (aspect-preserving resize+pad).

Needs numpy + cv2, so run on the Jetson, from the repo root:
    python3 tests/test_letterbox.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from detector import letterbox


def test_output_has_exact_target_shape():
    src = np.full((1920, 2560, 3), 255, dtype=np.uint8)  # 4:3
    out = letterbox(src, (1280, 720))
    assert out.shape == (720, 1280, 3), out.shape


def test_four_three_into_sixteen_nine_pillarboxes():
    # 4:3 content into 16:9 frame -> black bars left/right, content in the middle.
    src = np.full((1920, 2560, 3), 255, dtype=np.uint8)
    out = letterbox(src, (1280, 720))
    assert out[:, 0].sum() == 0, "left edge should be black padding"
    assert out[:, -1].sum() == 0, "right edge should be black padding"
    assert out[360, 640].sum() > 0, "center should contain image content"


def test_matching_aspect_fills_frame_without_bars():
    src = np.full((720, 1280, 3), 255, dtype=np.uint8)  # already 16:9
    out = letterbox(src, (1280, 720))
    assert out.shape == (720, 1280, 3)
    assert out[:, 0].sum() > 0 and out[:, -1].sum() > 0, "no padding expected"


def test_portrait_into_landscape_pillarboxes():
    src = np.full((1920, 1080, 3), 255, dtype=np.uint8)  # 9:16 portrait
    out = letterbox(src, (1280, 720))
    assert out.shape == (720, 1280, 3)
    assert out[:, 0].sum() == 0, "portrait source should get side bars"


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
