#!/usr/bin/env python3
# IRON program: bf16 gemv y[N] = W[N,K].x[K] on XDNA2 (NPU2/AIE2P).
# x[K] resident; W streams in R-row blocks; y streams out one R-block per step.
# Runtime buffers: x [K] bf16, W [N*K] bf16 (row-major), y [N] fp32.

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(K, N, R, wdepth=2):
    assert N % R == 0, "N must be a multiple of R"
    NB = N // R
    x_ty  = np.ndarray[(K,),     np.dtype[bfloat16]]
    wb_ty = np.ndarray[(R * K,), np.dtype[bfloat16]]
    yb_ty = np.ndarray[(R,),     np.dtype[np.float32]]

    X_ty = x_ty
    W_ty = np.ndarray[(N * K,), np.dtype[bfloat16]]
    Y_ty = np.ndarray[(N,),     np.dtype[np.float32]]

    flags = [f"-DGEMV_K={K}", f"-DGEMV_R={R}"]
    src = os.path.join(HERE, "gemv.cc")
    gemv = ExternalFunction("xdna_gemv", source_file=src, arg_types=[x_ty, wb_ty, yb_ty],
                            object_file_name="gemv.o", include_dirs=[HERE], compile_flags=flags)

    ofX = ObjectFifo(x_ty,  name="X")
    ofW = ObjectFifo(wb_ty, name="W", depth=wdepth)
    ofY = ObjectFifo(yb_ty, name="Y")

    def core_fn(x, w, y, gemv_k):
        e_x = x.acquire(1)
        for _ in range_(NB):
            e_w = w.acquire(1)
            e_y = y.acquire(1)
            gemv_k(e_x, e_w, e_y)
            w.release(1)
            y.release(1)
        x.release(1)

    worker = Worker(core_fn, [ofX.cons(), ofW.cons(), ofY.prod(), gemv], stack_size=0x1000)

    rt = Runtime()
    with rt.sequence(X_ty, W_ty, Y_ty) as (X, W, Y):
        rt.start(worker)
        rt.fill(ofX.prod(), X)
        rt.fill(ofW.prod(), W)
        rt.drain(ofY.cons(), Y, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=2560)
    ap.add_argument("--N", type=int, default=2560)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--wdepth", type=int, default=2)
    ap.add_argument("--mlir", default="")
    args = ap.parse_args()
    module = build_module(args.K, args.N, args.R, args.wdepth)
    text = str(module)
    if args.mlir:
        os.makedirs(os.path.dirname(os.path.abspath(args.mlir)), exist_ok=True)
        with open(args.mlir, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
