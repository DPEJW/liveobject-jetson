"""Tests for camera-source configuration / RTSP URL construction.

Pure-stdlib tests (no cv2/numpy), runnable with plain python:
    python3 tests/test_camera_config.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def test_build_main_stream_url():
    url = config.build_rtsp_url("192.168.1.104", 554, "admin", "secret", "main")
    assert url == "rtsp://admin:secret@192.168.1.104:554/h264Preview_01_main", url


def test_build_sub_stream_url():
    url = config.build_rtsp_url("192.168.1.104", 554, "admin", "secret", "sub")
    assert url == "rtsp://admin:secret@192.168.1.104:554/h264Preview_01_sub", url


def test_password_special_chars_are_url_encoded():
    # '#' must become %23 or the RTSP URL parser truncates the password.
    url = config.build_rtsp_url("10.0.0.5", 554, "admin", "RZCxbs#2736", "main")
    assert "RZCxbs%232736" in url, url
    assert "#" not in url, url


def test_username_and_at_sign_are_encoded():
    url = config.build_rtsp_url("10.0.0.5", 554, "a@b", "p:w/d", "main")
    assert "a%40b" in url, url
    assert "p%3Aw%2Fd" in url, url
    # host/path separators must survive
    assert url.endswith("@10.0.0.5:554/h264Preview_01_main"), url


def test_no_user_means_no_credentials_block():
    url = config.build_rtsp_url("10.0.0.5", 554, "", "", "main")
    assert url == "rtsp://10.0.0.5:554/h264Preview_01_main", url


def test_unknown_stream_defaults_to_main():
    url = config.build_rtsp_url("10.0.0.5", 554, "u", "p", "whatever")
    assert url.endswith("/h264Preview_01_main"), url


def test_rtsp_url_respects_full_override_env():
    os.environ["RTSP_URL"] = "rtsp://custom-host/path0"
    try:
        assert config.rtsp_url() == "rtsp://custom-host/path0"
    finally:
        del os.environ["RTSP_URL"]


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
