# Configurable camera source (Basler GigE + RTSP/Reolink)

Date: 2026-06-27

## Goal

Let `liveobject` capture from either the original Basler GigE camera or an RTSP
IP camera (Reolink RLC PoE at `192.168.1.104`), selectable by configuration,
without removing the Basler path. Side benefit: with the Reolink, liveobject no
longer needs the shared GigE camera, so it can coexist with `baffle-qc`.

## Approach

Introduce a small capture abstraction so the rest of the pipeline (model switch,
rotation/flip, YOLO inference, annotation, MJPEG, snapshots, reconnect loop) is
unchanged and camera-agnostic.

- `CameraSource` interface: `read() -> BGR ndarray @ DISPLAY_SIZE | None`, `close()`.
- `BaslerSource` — the existing pypylon Mono8→BGR logic, moved verbatim.
- `RtspSource` — GStreamer NVDEC pipeline via OpenCV `CAP_GSTREAMER`
  (`rtspsrc ! rtpjitterbuffer ! decodebin ! nvvidconv ! BGRx ! videoconvert ! BGR
  ! appsink drop=true max-buffers=1`), with a CPU `CAP_FFMPEG` fallback. Frames
  are letterboxed to `DISPLAY_SIZE` so the 4:3 stream isn't squished into 16:9.
- `make_camera_source()` factory keyed on `CAMERA_SOURCE` (`rtsp` | `basler`).
- `DetectionWorker._run()` changes only at the edges: `source = make_camera_source()`
  / `source.read()` / `source.close()`. Reconnect-on-error loop is reused.

## Configuration

Env-driven, loaded from a gitignored `.camera.env` by `config.py` (no systemd
edit, no secret in the repo). Keys: `CAMERA_SOURCE`, `RTSP_HOST`, `RTSP_PORT`,
`RTSP_USER`, `RTSP_PASS`, `RTSP_STREAM` (main|sub), or `RTSP_URL` override.
Credentials are URL-encoded (`config.build_rtsp_url`) so `#`/`@`/`:` in a
password can't corrupt the URL. Default `CAMERA_SOURCE=rtsp`.

## Stream facts (probed 2026-06-27)

Reolink, MAC `ec:71:db:*`. RTSP/ONVIF were disabled by default; enabled via the
desktop client. Main `h264Preview_01_main` = 2560×1920 @ 25 fps; sub = 640×480
@ 10 fps. Ports open after enabling: 80, 443, 554. NVDEC plugins present on the
Orin (`nvv4l2decoder`, `nvvidconv`). Inference (~12 fps at 15 W) is the
bottleneck, so main-stream resolution is essentially free on the HW decoder.

## Testing

- Unit (TDD): `tests/test_camera_config.py` (URL build + credential encoding),
  `tests/test_letterbox.py` (aspect-preserving pad). Run on the Orin (cv2/numpy).
- Integration: run `app.py` with `CAMERA_SOURCE=rtsp`, confirm live feed + boxes
  at `:8000`, confirm `backend=gstreamer-nvdec`, confirm Basler branch still
  selectable. Verified without disturbing `baffle-qc` (different camera).

## Deployment notes

Edit on the Mac repo (`liveobject-jetson`), push, scp changed files to the Orin
(`~/projects/liveobject`). NB: the Orin checkout points at the older `liveobject`
remote with uncommitted changes — reconciling those remotes is out of scope here.
