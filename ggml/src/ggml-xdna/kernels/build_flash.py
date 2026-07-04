#!/usr/bin/env python3
# IRON program: online-softmax flash attention on one XDNA2 (NPU2/AIE2P) core.
# Input packs [Q(NQ*DH) | K(KV*DH) | V(KV*DH) | mask(NQ*KV)], output O(NQ*DH).
# (First validation form: K/V resident. The streaming form feeds K/V blocks via
# an ObjectFifo so any n_kv fits — the memory-efficient flash design.)

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(NQ, DH, BLK, NBLK, scale):
    KV = NBLK * BLK
    in_len = NQ * DH + 2 * KV * DH + NQ * KV
    in_ty = np.ndarray[(in_len,),  np.dtype[bfloat16]]
    o_ty  = np.ndarray[(NQ * DH,), np.dtype[bfloat16]]

    flags = [f"-DNQ={NQ}", f"-DDH={DH}", f"-DBLK={BLK}", f"-DNBLK={NBLK}", f"-DATTN_SCALE={scale!r}f"]
    flash = ExternalFunction("xdna_flash_bf16", source_file=os.path.join(HERE, "flash.cc"),
                             arg_types=[in_ty, o_ty], include_dirs=[HERE], compile_flags=flags)

    ofIN = ObjectFifo(in_ty, name="IN")
    ofO  = ObjectFifo(o_ty, name="O")

    def core_fn(inp, o, kern):
        e_in = inp.acquire(1)
        e_o = o.acquire(1)
        kern(e_in, e_o)
        inp.release(1)
        o.release(1)

    worker = Worker(core_fn, [ofIN.cons(), ofO.prod(), flash], stack_size=0x1000)

    rt = Runtime()
    with rt.sequence(in_ty, o_ty) as (IN, O):
        rt.start(worker)
        rt.fill(ofIN.prod(), IN)
        rt.drain(ofO.cons(), O, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--NQ", type=int, default=16)
    ap.add_argument("--DH", type=int, default=64)
    ap.add_argument("--BLK", type=int, default=16)
    ap.add_argument("--NBLK", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.0)
    ap.add_argument("--mlir", default="")
    args = ap.parse_args()
    scale = args.scale if args.scale > 0 else 1.0 / (args.DH ** 0.5)
    module = build_module(args.NQ, args.DH, args.BLK, args.NBLK, scale)
    text = str(module)
    if args.mlir:
        os.makedirs(os.path.dirname(os.path.abspath(args.mlir)), exist_ok=True)
        with open(args.mlir, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
