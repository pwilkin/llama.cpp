#!/usr/bin/env python3
# IRON program: fused single-core attention on XDNA2 (NPU2 / AIE2P).
# Streams Q [NQ,DH], K [NKV,DH], V [NKV,DH] into one compute core which computes
# O = softmax(scale * Q.K^T) . V and streams O [NQ,DH] back out. bf16 in/out.
# Emits MLIR; compile to xclbin with aiecc (see the build helper).

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(NQ, NKV, DH, scale):
    # Q,K,V packed into one input buffer -> a single input DMA channel.
    qkv_ty = np.ndarray[((NQ + 2 * NKV) * DH,), np.dtype[bfloat16]]
    o_ty   = np.ndarray[(NQ * DH,),            np.dtype[bfloat16]]

    flags = [f"-DNQ={NQ}", f"-DNKV={NKV}", f"-DDH={DH}", f"-DATTN_SCALE={scale!r}f"]
    attn = ExternalFunction("xdna_attn_bf16", source_file=os.path.join(HERE, "attn.cc"),
                            arg_types=[qkv_ty, o_ty], include_dirs=[HERE], compile_flags=flags)

    ofQKV = ObjectFifo(qkv_ty, name="QKV")
    ofO   = ObjectFifo(o_ty, name="O")

    def core_fn(qkv, o, kern):
        e_qkv = qkv.acquire(1)
        e_o = o.acquire(1)
        kern(e_qkv, e_o)
        qkv.release(1)
        o.release(1)

    worker = Worker(core_fn, [ofQKV.cons(), ofO.prod(), attn], stack_size=0xE00)

    rt = Runtime()
    with rt.sequence(qkv_ty, o_ty) as (QKV, O):
        rt.start(worker)
        rt.fill(ofQKV.prod(), QKV)
        rt.drain(ofO.cons(), O, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--NQ", type=int, default=32)
    ap.add_argument("--NKV", type=int, default=32)
    ap.add_argument("--DH", type=int, default=32)
    ap.add_argument("--scale", type=float, default=0.0)
    ap.add_argument("--mlir", default="")
    args = ap.parse_args()
    scale = args.scale if args.scale > 0 else 1.0 / (args.DH ** 0.5)

    module = build_module(args.NQ, args.NKV, args.DH, scale)
    text = str(module)
    if args.mlir:
        os.makedirs(os.path.dirname(os.path.abspath(args.mlir)), exist_ok=True)
        with open(args.mlir, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
