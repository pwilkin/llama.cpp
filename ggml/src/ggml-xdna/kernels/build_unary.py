#!/usr/bin/env python3
# IRON program that builds a single-core XDNA2 (NPU2 / AIE2P) elementwise-unary
# design and lowers it to MLIR. One op per xclbin; the op's compute is the
# matching `extern "C"` kernel in unary.cc.
#
# Usage:
#   python build_unary.py --op relu --length 4096 --tile 1024 --mlir build/relu.mlir
# then compile the emitted MLIR to an xclbin with aiecc.py (see Makefile).
#
# Modeled on mlir-aie programming_examples/basic/vector_exp/vector_exp.py.

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

# op name -> extern "C" entry point in unary.cc
OPS = {
    "relu":       "xdna_relu_bf16",
    "neg":        "xdna_neg_bf16",
    "abs":        "xdna_abs_bf16",
    "exp":        "xdna_exp_bf16",
    "tanh":       "xdna_tanh_bf16",
    "sigmoid":    "xdna_sigmoid_bf16",
    "silu":       "xdna_silu_bf16",
    "gelu":       "xdna_gelu_bf16",
    "gelu_quick": "xdna_gelu_quick_bf16",
}

HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(op: str, length: int, tile: int):
    fn = OPS[op]
    tile_ty   = np.ndarray[(tile,),   np.dtype[bfloat16]]
    tensor_ty = np.ndarray[(length,), np.dtype[bfloat16]]

    kernel = ExternalFunction(
        fn,
        source_file=os.path.join(HERE, "unary.cc"),
        arg_types=[tile_ty, tile_ty],
        include_dirs=[HERE],
        compile_flags=[f"-DTILE_N={tile}"],
    )

    of_in  = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_fn(a_in, c_out, k):
        for _ in range_(length // tile):
            e_out = c_out.acquire(1)
            e_in  = a_in.acquire(1)
            k(e_in, e_out)
            a_in.release(1)
            c_out.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod(), kernel])

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a, c):
        rt.start(worker)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), c, wait=True)

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
