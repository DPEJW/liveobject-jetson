"""TensorRT YOLO (v8/v11/v12/26) detector — Blackwell / CUDA 13 edition.

Ported from the Jetson Orin (TensorRT 8.5 + pycuda) to the NVIDIA GB10
(Grace-Blackwell, CUDA 13, TensorRT 10/11). Two things changed vs. the Orin
version; the detection logic (letterbox, (1,84,8400) decode, NMS) is identical:

  * Engine/tensor API: TensorRT 8.5's binding-index API (num_bindings,
    get_binding_shape, execute_async_v2) was removed in TensorRT 10. This uses
    the name-based I/O API (num_io_tensors / get_tensor_name / get_tensor_shape
    / set_tensor_address / execute_async_v3).
  * CUDA plumbing: pycuda won't build against CUDA 13, so device memory, streams
    and copies go through NVIDIA's official cuda-python (cuda.bindings.runtime).

Loads a serialized .engine (build one with build_engine.py from an
Ultralytics-exported ONNX), runs inference on the GPU, and decodes the
(1, 84, 8400) detection head into COCO boxes. (YOLO26 must be exported with
end2end=False to keep this traditional head; its default NMS-free end2end head
emits (1, 300, 6) instead and is not decoded here.)

All CUDA/TensorRT calls must happen on the thread that created this object (the
DetectionWorker thread) — cuda-python binds the primary context per-thread on
first use, and the execution context is not thread-safe.
"""
from __future__ import annotations

import cv2
import numpy as np

import tensorrt as trt
from cuda.bindings import runtime as cudart

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _check(err, what=""):
    """Raise on a non-success cudaError_t (cuda-python returns (err, *rest))."""
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        name = cudart.cudaGetErrorString(err)[1]
        raise RuntimeError(f"CUDA error in {what}: {name}")


class _IOTensor:
    __slots__ = ("name", "shape", "dtype", "nbytes", "host", "dev")

    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.host = np.empty(shape, dtype=dtype)
        self.nbytes = self.host.nbytes
        err, self.dev = cudart.cudaMalloc(self.nbytes)
        _check(err, f"cudaMalloc({name})")


class TRTYolo:
    def __init__(self, engine_path: str, input_size: int = 640, iou: float = 0.45):
        self.input_size = input_size
        self.iou = iou
        _check(cudart.cudaSetDevice(0), "cudaSetDevice")
        err, self.stream = cudart.cudaStreamCreate()
        _check(err, "cudaStreamCreate")
        self.engine = None
        self.context = None
        self.inp = None
        self.out = None
        self._load(engine_path)

    # ---- engine (re)loading ----
    def _free_io(self):
        for t in (self.inp, self.out):
            if t is not None and t.dev is not None:
                cudart.cudaFree(t.dev)
        self.inp = self.out = None

    def _load(self, engine_path: str) -> None:
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(_TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        context = engine.create_execution_context()

        self._free_io()
        inp = out = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = tuple(engine.get_tensor_shape(name))
            # Resolve a dynamic batch/spatial dim to a concrete size (1, imgsz).
            if is_input and any(d < 0 for d in shape):
                shape = tuple(self.input_size if d < 0 and j >= 2 else (1 if d < 0 else d)
                              for j, d in enumerate(shape))
                context.set_input_shape(name, shape)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            t = _IOTensor(name, shape, dtype)
            context.set_tensor_address(name, int(t.dev))
            if is_input:
                inp = t
            else:
                # Output shape may only be known after the input shape is set.
                oshape = tuple(context.get_tensor_shape(name))
                if oshape != t.shape:
                    cudart.cudaFree(t.dev)
                    t = _IOTensor(name, oshape, dtype)
                    context.set_tensor_address(name, int(t.dev))
                out = t

        if inp is None or out is None:
            raise RuntimeError("engine must expose one input and one output tensor")
        # The engine knows its true input resolution — trust it over the ctor
        # default so 960/1280 high-res engines letterbox correctly.
        if len(inp.shape) == 4 and inp.shape[-1] > 0:
            self.input_size = int(inp.shape[-1])
        # Swap in the new engine/context/buffers only once everything is wired.
        self.engine, self.context, self.inp, self.out = engine, context, inp, out

    def reload(self, engine_path: str) -> None:
        """Hot-swap the engine in place (used by the identity-model swap)."""
        self._load(engine_path)

    # ---- preprocessing ----
    def _letterbox(self, im, color=(114, 114, 114)):
        h, w = im.shape[:2]
        s = self.input_size
        r = min(s / h, s / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        left = (s - nw) // 2
        top = (s - nh) // 2
        out = cv2.copyMakeBorder(resized, top, s - nh - top, left, s - nw - left,
                                 cv2.BORDER_CONSTANT, value=color)
        return out, r, left, top

    # ---- inference + decode ----
    def infer(self, bgr, conf: float = 0.4):
        """Return a list of (class_id, score, [x0,y0,x1,y1]) in bgr-pixel coords."""
        img, r, dw, dh = self._letterbox(bgr)
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0  # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob, dtype=self.inp.dtype)

        np.copyto(self.inp.host, blob.reshape(self.inp.shape))
        _check(cudart.cudaMemcpyAsync(
            self.inp.dev, self.inp.host.ctypes.data, self.inp.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream), "H2D")
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 failed")
        _check(cudart.cudaMemcpyAsync(
            self.out.host.ctypes.data, self.out.dev, self.out.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream), "D2H")
        _check(cudart.cudaStreamSynchronize(self.stream), "streamSync")
        out = self.out.host

        # Two head layouts are supported (auto-detected by output shape):
        #   * (1, 84, 8400)  traditional one-to-many head -> argmax + cv2 NMS.
        #   * (1, N, 6)      YOLO26 NMS-free "end2end" head, already decoded to
        #                    [x0, y0, x1, y1, score, class] rows in letterbox
        #                    space -> just filter + rescale.
        if out.ndim == 3 and out.shape[2] == 6:
            return self._decode_end2end(out, conf, r, dw, dh, bgr.shape[:2])

        # (1, 84, 8400) -> (8400, 84)
        p = out[0].transpose(1, 0)
        xywh = p[:, :4]
        cls_scores = p[:, 4:]
        class_ids = cls_scores.argmax(1)
        confs = cls_scores.max(1)
        keep = confs >= conf
        if not np.any(keep):
            return []
        xywh, confs, class_ids = xywh[keep], confs[keep], class_ids[keep]

        cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x0 = (cx - w / 2 - dw) / r
        y0 = (cy - h / 2 - dh) / r
        x1 = (cx + w / 2 - dw) / r
        y1 = (cy + h / 2 - dh) / r

        rects = np.stack([x0, y0, x1 - x0, y1 - y0], 1).tolist()  # x,y,w,h for NMS
        idxs = cv2.dnn.NMSBoxes(rects, confs.tolist(), conf, self.iou)
        H, W = bgr.shape[:2]
        results = []
        for i in np.array(idxs).flatten() if len(idxs) else []:
            results.append((
                int(class_ids[i]),
                float(confs[i]),
                [int(max(0, x0[i])), int(max(0, y0[i])),
                 int(min(W, x1[i])), int(min(H, y1[i]))],
            ))
        return results

    def _decode_end2end(self, out, conf, r, dw, dh, hw):
        """Decode the YOLO26 NMS-free head (1, N, 6): rows are
        [x0, y0, x1, y1, score, class] in letterbox pixels, already NMS'd."""
        rows = out[0].astype(np.float32)
        keep = rows[:, 4] >= conf
        rows = rows[keep]
        if rows.shape[0] == 0:
            return []
        H, W = hw
        results = []
        for x0, y0, x1, y1, score, cls in rows:
            x0 = (x0 - dw) / r
            y0 = (y0 - dh) / r
            x1 = (x1 - dw) / r
            y1 = (y1 - dh) / r
            results.append((
                int(cls), float(score),
                [int(max(0, x0)), int(max(0, y0)),
                 int(min(W, x1)), int(min(H, y1))],
            ))
        return results

    def __del__(self):
        try:
            self._free_io()
            if getattr(self, "stream", None) is not None:
                cudart.cudaStreamDestroy(self.stream)
        except Exception:
            pass
