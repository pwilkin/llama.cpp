#!/usr/bin/env python3
# IRON program: 2-in/1-out elementwise binary op (add/mul) on XDNA2 (NPU2/AIE2P).
# Streams A and B (two input DMA channels) through one core -> C. Emits MLIR;
# compile to xclbin with aiecc (see Makefile).

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

OPS = {"add": "xdna_add_bf16", "mul": "xdna_mul_bf16"}
HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(op, length, tile):
    fn = OPS[op]
    tile_ty   = np.ndarray[(tile,),   np.dtype[bfloat16]]
    tensor_ty = np.ndarray[(length,), np.dtype[bfloat16]]

    kernel = ExternalFunction(fn, source_file=os.path.join(HERE, "binary.cc"),
                              arg_types=[tile_ty, tile_ty, tile_ty],
                              include_dirs=[HERE], compile_flags=[f"-DTILE_N={tile}"])

    ofA = ObjectFifo(tile_ty, name="inA")
    ofB = ObjectFifo(tile_ty, name="inB")
    ofC = ObjectFifo(tile_ty, name="outC")

    def core_fn(a, b, c, k):
        for _ in range_(length // tile):
            e_c = c.acquire(1)
            e_a = a.acquire(1)
            e_b = b.acquire(1)
            k(e_a, e_b, e_c)
            a.release(1)
            b.release(1)
            c.release(1)

    worker = Worker(core_fn, [ofA.cons(), ofB.cons(), ofC.prod(), kernel])

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(ofA.prod(), A)
        rt.fill(ofB.prod(), B)
        rt.drain(ofC.cons(), C, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=sorted(OPS))
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--mlir", default="")
    args = ap.parse_args()
    module = build_module(args.op, args.length, args.tile)
    text = str(module)
    if args.mlir:
        os.makedirs(os.path.dirname(os.path.abspath(args.mlir)), exist_ok=True)
        with open(args.mlir, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
