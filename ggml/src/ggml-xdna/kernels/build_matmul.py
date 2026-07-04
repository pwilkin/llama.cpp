#!/usr/bin/env python3
# IRON program: single-tile XDNA2 (NPU2 / AIE2P) matmul, C[M,N] = A[M,K] @ B[K,N]
# with B laid out column-major (N x K) so the host buffers map 1:1 onto ggml's
# mul_mat operands (A = activations b[K,M] viewed [M,K]; B = weights a[K,N] as
# N x K; C = result [N,M] viewed [M,N]). Inputs bf16, output f32.
#
# The shim-DMA `dims_to_stream` access patterns (a_dims/b_dims/c_dims) do all the
# sub-tile scatter/gather, so the XRT host buffers stay plain row-major. Tiling
# constants (R,S,T = aie::mmul dims for bf16->f32 on aie2p) and the patterns are
# taken verbatim from mlir-aie programming_examples matrix_multiplication/single_core.
#
# Emits MLIR; compile to xclbin with aiecc (see Makefile). This first version is
# single-tile (m=M, k=K, n=N) — one compute core, one block.

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

R, S, T = 4, 8, 8  # aie::mmul<r,s,t> for bfloat16 -> float on aie2p
HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(M: int, K: int, N: int):
    m, k, n = M, K, N  # single tile: the whole matrix is one block
    assert m % (2 * R) == 0, "M must be a multiple of 8"
    assert k % S == 0,       "K must be a multiple of 8"
    assert n % (2 * T) == 0, "N must be a multiple of 16"

    a_tile = np.ndarray[(m, k), np.dtype[bfloat16]]
    b_tile = np.ndarray[(k, n), np.dtype[bfloat16]]
    c_tile = np.ndarray[(m, n), np.dtype[np.float32]]

    flags = ["-Dbf16_f32_ONLY", f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}", "-DB_COL_MAJ"]
    # both entry points live in mm.cc -> one object file (a Worker links one binary)
    mm = ExternalFunction("matmul_bf16_f32", object_file_name="mm_kernel.o",
                          source_file=os.path.join(HERE, "mm.cc"),
                          arg_types=[a_tile, b_tile, c_tile], include_dirs=[HERE], compile_flags=flags)
    zero = ExternalFunction("zero_f32", object_file_name="mm_kernel.o",
                            source_file=os.path.join(HERE, "mm.cc"),
                            arg_types=[c_tile], include_dirs=[HERE], compile_flags=flags)

    inA = ObjectFifo(a_tile, name="inA")
    a_dims = [(m // R, R * k), (k // S, S), (R, k), (S, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    inB = ObjectFifo(b_tile, name="inB")
    b_dims = [(n // T, T * k), (k // S, S), (T, k), (S, 1)]  # B column-major
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_tile, name="memC")
    c_dims = [(m // R, R * n), (R, T), (n // T, R * T), (T, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    def core_fn(of_a, of_b, of_c, zero_k, mm_k):
        e_c = of_c.acquire(1)
        zero_k(e_c)                 # C must be zeroed: the kernel does C += A*B
        e_a = of_a.acquire(1)
        e_b = of_b.acquire(1)
        mm_k(e_a, e_b, e_c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    worker = Worker(core_fn, [memA.cons(), memB.cons(), memC.prod(), zero, mm], stack_size=0xD00)

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(inA.prod(), A)
        rt.fill(inB.prod(), B)
        rt.drain(outC.cons(), C, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--mlir", default="")
    args = ap.parse_args()

    module = build_module(args.M, args.K, args.N)
    text = str(module)
    if args.mlir:
        os.makedirs(os.path.dirname(os.path.abspath(args.mlir)), exist_ok=True)
        with open(args.mlir, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
