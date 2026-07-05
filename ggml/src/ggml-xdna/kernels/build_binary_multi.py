#!/usr/bin/env python3
# IRON program: multi-op binary on XDNA2. One xclbin serves mul/add via a runtime
# op-code (op[0]); consolidates hw-contexts. op [16] int32 resident, a/b/c [TILE]
# bf16 streamed. Runtime buffers: OP [16] int32, A/B/C [LENGTH] bf16.

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(length, tile):
    assert length % tile == 0
    NB = length // tile
    op_ty = np.ndarray[(16,),       np.dtype[np.int32]]
    ab_ty = np.ndarray[(2 * tile,), np.dtype[bfloat16]]
    c_ty  = np.ndarray[(tile,),     np.dtype[bfloat16]]
    OP_ty  = op_ty
    AB_ty  = np.ndarray[(NB * 2 * tile,), np.dtype[bfloat16]]
    C_ty   = np.ndarray[(length,),        np.dtype[bfloat16]]

    kernel = ExternalFunction(
        "xdna_binary_multi",
        source_file=os.path.join(HERE, "binary.cc"),
        arg_types=[op_ty, ab_ty, c_ty],
        include_dirs=[HERE],
        compile_flags=[f"-DTILE_N={tile}"],
    )

    ofOP = ObjectFifo(op_ty, name="op")
    ofAB = ObjectFifo(ab_ty, name="inAB")
    ofC  = ObjectFifo(c_ty,  name="outC")

    def core_fn(op, ab, c, k):
        e_op = op.acquire(1)
        for _ in range_(NB):
            e_ab = ab.acquire(1)
            e_c = c.acquire(1)
            k(e_op, e_ab, e_c)
            ab.release(1)
            c.release(1)
        op.release(1)

    worker = Worker(core_fn, [ofOP.cons(), ofAB.cons(), ofC.prod(), kernel], stack_size=0x800)

    rt = Runtime()
    with rt.sequence(OP_ty, AB_ty, C_ty) as (OP, AB, C):
        rt.start(worker)
        rt.fill(ofOP.prod(), OP)
        rt.fill(ofAB.prod(), AB)
        rt.drain(ofC.cons(), C, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--mlir", default="")
    args = ap.parse_args()
    module = build_module(args.length, args.tile)
    text = str(module)
    if args.mlir:
        os.makedirs(os.path.dirname(os.path.abspath(args.mlir)), exist_ok=True)
        with open(args.mlir, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
