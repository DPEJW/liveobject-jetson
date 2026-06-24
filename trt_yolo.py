"""TensorRT YOLO (v8/v11/v12/26) detector for the Jetson.

Replaces the Pi's Hailo NPU path. Loads a serialized .engine built by trtexec
from an Ultralytics-exported ONNX, runs inference on the GPU via pycuda, and
decodes the (1, 84, 8400) detection head into COCO boxes. (YOLO26 must be
exported with end2end=False to keep this traditional head; its default NMS-free
end2end head emits (1, 300, 6) instead and is not decoded here.)

All CUDA calls must happen on the thread that created this object (the
DetectionWorker thread); the primary context is pushed/popped around every op.
"""
from __future__ import annotations

import cv2
import numpy as np

# numpy>=1.24 removed these aliases that TensorRT 8.5's python bindings still use.
for _a, _t in (("bool", bool), ("float", float), ("int", int), ("object", object)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)

import tensorrt as trt
import pycuda.driver as cuda

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTYolo:
    def __init__(self, engine_path: str, input_size: int = 640, iou: float = 0.45):
        cuda.init()
        self.ctx = cuda.Device(0).retain_primary_context()
        self.input_size = input_size
        self.iou = iou
        self.ctx.push()
        try:
            self.stream = cuda.Stream()
            self._load(engine_path)
        finally:
            self.ctx.pop()

    # ---- engine (re)loading ----
    def _load(self, engine_path: str) -> None:
        with open(engine_path, "rb") as f, trt.Runtime(_TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.bindings = []
        self.inp = None
        self.out = None
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(dev))
            slot = {"host": host, "dev": dev, "shape": shape, "dtype": dtype}
            if self.engine.binding_is_input(i):
                self.inp = slot
            else:
                self.out = slot

    def reload(self, engine_path: str) -> None:
        self.ctx.push()
        try:
            self._load(engine_path)
        finally:
            self.ctx.pop()

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
        blob = np.ascontiguousarray(blob).ravel()

        self.ctx.push()
        try:
            np.copyto(self.inp["host"], blob.astype(self.inp["dtype"], copy=False))
            cuda.memcpy_htod_async(self.inp["dev"], self.inp["host"], self.stream)
            self.context.execute_async_v2(self.bindings, self.stream.handle)
            cuda.memcpy_dtoh_async(self.out["host"], self.out["dev"], self.stream)
            self.stream.synchronize()
            out = np.array(self.out["host"]).reshape(self.out["shape"])
        finally:
            self.ctx.pop()

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
