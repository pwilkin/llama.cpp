#!/usr/bin/env python3
# IRON program: multi-op unary on XDNA2. One xclbin serves several unary ops via
# a runtime op-code (op[0]); consolidates hw-contexts (the NPU caps concurrent
# contexts). op [16] int32 resident, in/out [TILE] bf16 streamed (NB tiles/run).
# Runtime buffers: OP [16] int32, IN [LENGTH] bf16, OUT [LENGTH] bf16.

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
    op_ty  = np.ndarray[(16,),   np.dtype[np.int32]]
    in_ty  = np.ndarray[(tile,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(tile,), np.dtype[bfloat16]]
    OP_ty  = op_ty
    IN_ty  = np.ndarray[(length,), np.dtype[bfloat16]]
    OUT_ty = np.ndarray[(length,), np.dtype[bfloat16]]

    kernel = ExternalFunction(
        "xdna_unary_multi",
        source_file=os.path.join(HERE, "unary.cc"),
        arg_types=[op_ty, in_ty, out_ty],
        include_dirs=[HERE],
        compile_flags=[f"-DTILE_N={tile}"],
    )

    ofOP = ObjectFifo(op_ty,  name="op")
    ofI  = ObjectFifo(in_ty,  name="in")
    ofO  = ObjectFifo(out_ty, name="out")

    def core_fn(op, a_in, c_out, k):
        e_op = op.acquire(1)
        for _ in range_(NB):
            e_i = a_in.acquire(1)
            e_o = c_out.acquire(1)
            k(e_op, e_i, e_o)
            a_in.release(1)
            c_out.release(1)
        op.release(1)

    worker = Worker(core_fn, [ofOP.cons(), ofI.cons(), ofO.prod(), kernel], stack_size=0x800)

    rt = Runtime()
    with rt.sequence(OP_ty, IN_ty, OUT_ty) as (OP, IN, OUT):
        rt.start(worker)
        rt.fill(ofOP.prod(), OP)
        rt.fill(ofI.prod(), IN)
        rt.drain(ofO.cons(), OUT, wait=True)

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
