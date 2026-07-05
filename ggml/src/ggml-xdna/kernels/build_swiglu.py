#!/usr/bin/env python3
# IRON program: SWIGLU (silu(gate)*up) on XDNA2. Two inputs packed per tile as
# [gate(TILE)|up(TILE)] stream in; [TILE] streams out. xdna_swiglu_bf16 reads
# a[0:TILE]=gate, a[TILE:2TILE]=up. Runtime buffers: IN [N*2] bf16 (N=NB*TILE
# interleaved per tile), OUT [N] bf16.

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(TILE, NB):
    in_ty  = np.ndarray[(2 * TILE,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(TILE,),     np.dtype[bfloat16]]
    IN_ty  = np.ndarray[(NB * 2 * TILE,), np.dtype[bfloat16]]
    OUT_ty = np.ndarray[(NB * TILE,),     np.dtype[bfloat16]]

    kernel = ExternalFunction(
        "xdna_swiglu_bf16",
        source_file=os.path.join(HERE, "unary.cc"),
        arg_types=[in_ty, out_ty],
        include_dirs=[HERE],
        compile_flags=[f"-DTILE_N={TILE}"],
    )

    ofI = ObjectFifo(in_ty,  name="in")
    ofO = ObjectFifo(out_ty, name="out")

    def core_fn(a_in, c_out, k):
        for _ in range_(NB):
            e_i = a_in.acquire(1)
            e_o = c_out.acquire(1)
            k(e_i, e_o)
            a_in.release(1)
            c_out.release(1)

    worker = Worker(core_fn, [ofI.cons(), ofO.prod(), kernel], stack_size=0x800)

    rt = Runtime()
    with rt.sequence(IN_ty, OUT_ty) as (IN, OUT):
        rt.start(worker)
        rt.fill(ofI.prod(), IN)
        rt.drain(ofO.cons(), OUT, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--TILE", type=int, default=1024)
    ap.add_argument("--NB", type=int, default=9)
    ap.add_argument("--mlir", default="")
    args = ap.parse_args()
    module = build_module(args.TILE, args.NB)
    text = str(module)
    if args.mlir:
        os.makedirs(os.path.dirname(os.path.abspath(args.mlir)), exist_ok=True)
        with open(args.mlir, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
