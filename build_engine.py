"""Build a TensorRT engine from an ONNX file using the TensorRT Python API.

Replaces the `trtexec` CLI, which is not shipped with the pip TensorRT wheels
used on the GB10 (CUDA 13). Same result: an FP16 serialized engine.

Usage:
    python build_engine.py model.onnx model.engine [--fp16] [--workspace GiB]

Importable:
    from build_engine import build_engine
    build_engine("model.onnx", "model.engine", fp16=True)
"""
from __future__ import annotations

import argparse
import os
import sys

import tensorrt as trt

_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True,
                 workspace_gib: float = 8.0) -> str:
    """Parse ONNX and serialize an (optionally FP16) TensorRT engine to disk."""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(onnx_path)

    builder = trt.Builder(_LOGGER)
    # Precision model differs by TensorRT major version:
    #   * TRT <= 8/9: weakly typed, precision via BuilderFlag.FP16.
    #   * TRT 10/11 (Blackwell/CUDA 13): the FP16 builder flag is gone; the
    #     engine honors the ONNX tensor dtypes, so FP16 is realized by feeding a
    #     half-precision ONNX into a STRONGLY_TYPED network.
    has_fp16_flag = hasattr(trt.BuilderFlag, "FP16")
    strongly_typed = fp16 and not has_fp16_flag and hasattr(
        trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")

    # TensorRT 10+ dropped the EXPLICIT_BATCH flag — explicit batch is the only
    # mode now, so create_network() takes no flags unless we want strong typing.
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if strongly_typed else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, _LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed:\n{errs}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                              int(workspace_gib * (1 << 30)))
    if fp16 and has_fp16_flag:
        cfg.set_flag(trt.BuilderFlag.FP16)

    # Static-shape ONNX (fixed imgsz) needs no optimization profile; add one only
    # if the network has a dynamic dimension.
    inp = network.get_input(0)
    if any(d < 0 for d in inp.shape):
        prof = builder.create_optimization_profile()
        c = [1 if d < 0 and i == 0 else (640 if d < 0 else d)
             for i, d in enumerate(inp.shape)]
        prof.set_shape(inp.name, c, c, c)
        cfg.add_optimization_profile(prof)

    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None (build failed)")
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return engine_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a TensorRT engine from ONNX.")
    ap.add_argument("onnx")
    ap.add_argument("engine")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--workspace", type=float, default=8.0, help="workspace in GiB")
    a = ap.parse_args(argv)
    out = build_engine(a.onnx, a.engine, fp16=a.fp16, workspace_gib=a.workspace)
    print(f"[build_engine] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
