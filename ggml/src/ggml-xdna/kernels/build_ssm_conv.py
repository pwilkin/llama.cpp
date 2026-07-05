#!/usr/bin/env python3
# IRON program: causal depthwise conv1d (SSM_CONV), one token, on XDNA2.
# Channel blocks [x0..x{KW-1}|w0..w{KW-1}] (each R) stream in; out (R) streams out.
# Runtime buffers: IN [NC*2*KW] bf16, OUT [NC] fp32.

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(NC, KW, R):
    assert NC % R == 0
    NB = NC // R
    blk = 2 * KW * R
    ib_ty = np.ndarray[(blk,), np.dtype[bfloat16]]
    ob_ty = np.ndarray[(R,),   np.dtype[np.float32]]
    IN_ty  = np.ndarray[(NB * blk,), np.dtype[bfloat16]]
    OUT_ty = np.ndarray[(NC,),       np.dtype[np.float32]]

    flags = [f"-DKW={KW}", f"-DCONV_R={R}"]
    src = os.path.join(HERE, "ssm_conv.cc")
    conv = ExternalFunction("xdna_ssm_conv", source_file=src, arg_types=[ib_ty, ob_ty],
                            object_file_name="ssm_conv.o", include_dirs=[HERE], compile_flags=flags)

    ofI = ObjectFifo(ib_ty, name="I")
    ofO = ObjectFifo(ob_ty, name="O")

    def core_fn(i, o, conv_k):
        for _ in range_(NB):
            e_i = i.acquire(1)
            e_o = o.acquire(1)
            conv_k(e_i, e_o)
            i.release(1)
            o.release(1)

    worker = Worker(core_fn, [ofI.cons(), ofO.prod(), conv], stack_size=0x800)

    rt = Runtime()
    with rt.sequence(IN_ty, OUT_ty) as (IN, OUT):
        rt.start(worker)
        rt.fill(ofI.prod(), IN)
        rt.drain(ofO.cons(), OUT, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--NC", type=int, default=8192)
    ap.add_argument("--KW", type=int, default=4)
    ap.add_argument("--R", type=int, default=256)
    ap.add_argument("--mlir", default="")
    args = ap.parse_args()
    module = build_module(args.NC, args.KW, args.R)
    text = str(module)
    if args.mlir:
        os.makedirs(os.path.dirname(os.path.abspath(args.mlir)), exist_ok=True)
        with open(args.mlir, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
