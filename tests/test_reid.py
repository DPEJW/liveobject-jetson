"""Tests for reid.py (appearance-signature matching + named roster).

Pure numpy + stdlib (no cv2): signature extraction from real frames uses cv2 in
detector.py, but the matching core here is tested with synthetic histograms.
    python3 tests/test_reid.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from reid import similarity, Roster, merge_identity


def _sig(peak, n=32):
    v = np.full(n, 0.01, dtype=np.float32)
    v[peak] = 1.0
    return v / v.sum()


def test_similarity_high_for_same_low_for_different():
    a, b, c = _sig(3), _sig(3), _sig(20)
    assert similarity(a, b) > 0.9, similarity(a, b)
    assert similarity(a, c) < 0.3, similarity(a, c)


def test_roster_matches_enrolled_identity():
    r = Roster(threshold=0.5)
    r.enroll("Mittens", _sig(3))
    r.enroll("Shadow", _sig(20))
    name, score = r.match(_sig(3))
    assert name == "Mittens" and score > 0.5, (name, score)


def test_unknown_below_threshold_returns_none():
    r = Roster(threshold=0.6)
    r.enroll("Mittens", _sig(3))
    name, _ = r.match(_sig(28))
    assert name is None, name


def test_two_slot_assignment_no_double_naming():
    r = Roster(threshold=0.3)
    r.enroll("Mittens", _sig(3))
    r.enroll("Shadow", _sig(20))
    assign = r.assign([_sig(20), _sig(3)])   # shadow-ish, then mittens-ish
    assert assign == ["Shadow", "Mittens"], assign


def test_returning_cat_regains_name():
    r = Roster(threshold=0.5)
    r.enroll("Mittens", _sig(3))
    assert r.match(_sig(3))[0] == "Mittens"     # left view and came back


def test_merge_identity_replaces_overlapping_cat():
    base = [{"name": "cat", "score": 0.55, "box": [100, 100, 200, 200]},
            {"name": "couch", "score": 0.9, "box": [50, 50, 400, 300]}]
    ident = [{"name": "Dundun", "score": 0.86, "box": [105, 102, 205, 198]}]
    out = merge_identity(base, ident)
    names = sorted(d["name"] for d in out)
    assert names == ["Dundun", "couch"], names           # cat replaced, couch kept
    d = next(d for d in out if d["name"] == "Dundun")
    assert d["score"] == 0.86


def test_merge_identity_keeps_non_overlapping_cat_and_appends_new():
    base = [{"name": "cat", "score": 0.6, "box": [800, 500, 900, 600]}]
    ident = [{"name": "Dundun", "score": 0.9, "box": [100, 100, 200, 200]}]
    out = merge_identity(base, ident)
    names = sorted(d["name"] for d in out)
    assert names == ["Dundun", "cat"], names             # both survive (different spots)


def test_merge_identity_never_touches_non_cat_classes():
    base = [{"name": "dog", "score": 0.7, "box": [100, 100, 200, 200]}]
    ident = [{"name": "Dundun", "score": 0.9, "box": [100, 100, 200, 200]}]
    out = merge_identity(base, ident)
    names = sorted(d["name"] for d in out)
    assert names == ["Dundun", "dog"], names


def test_enroll_averages_multiple_samples():
    r = Roster(threshold=0.5)
    r.enroll("Mittens", _sig(3))
    r.reinforce("Mittens", _sig(4))             # slightly shifted sample
    # still closest to the 3/4 region, not to a far peak
    assert r.match(_sig(3))[0] == "Mittens"
    assert r.match(_sig(4))[0] == "Mittens"


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
