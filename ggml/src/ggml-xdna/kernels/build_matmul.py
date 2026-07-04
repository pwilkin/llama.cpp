#!/usr/bin/env python3
# IRON program: XDNA2 (NPU2 / AIE2P) matmul, C[M,N] = A[M,K] @ B[K,N], B laid out
# column-major (N x K) so the host buffers map 1:1 onto ggml's mul_mat operands
# (A = activations b[K,M] viewed [M,K]; B = weights a[K,N] as N x K; C = result
# [N,M] viewed [M,N]). Inputs bf16, output f32.
#
# Multi-tile: the M x K x N matrix is tiled into m x k x n inner blocks (default
# 32) that fit one compute core's L1; the core zeroes each C tile and MACs over
# the K blocks; the Runtime streams the right A/B tiles (TensorTiler2D taps) and
# drains C with a 4-row ping-pong double buffer. The shim-DMA dims_to_stream
# patterns do all sub-tile scatter/gather so host buffers stay plain row-major.
# Structure and constants taken from mlir-aie matrix_multiplication/single_core.

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib import TensorTiler2D

R, S, T = 4, 8, 8  # aie::mmul<r,s,t> for bfloat16 -> float on aie2p
HERE = os.path.dirname(os.path.abspath(__file__))


def ceildiv(a, b):
    return -(-a // b)


def build_module(M, K, N, m=32, k=32, n=32):
    assert M % m == 0 and K % k == 0 and N % n == 0, "M,K,N must be multiples of the inner tile"
    assert m % (2 * R) == 0 and k % S == 0 and n % (2 * T) == 0, "inner tile must satisfy mmul dims"

    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n
    tiles = M_div_m * N_div_n

    a_ty = np.ndarray[(m, k), np.dtype[bfloat16]]
    b_ty = np.ndarray[(k, n), np.dtype[bfloat16]]
    c_ty = np.ndarray[(m, n), np.dtype[np.float32]]
    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K * N,), np.dtype[bfloat16]]
    C_ty = np.ndarray[(M * N,), np.dtype[np.float32]]

    flags = ["-Dbf16_f32_ONLY", f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}", "-DB_COL_MAJ"]
    mm = ExternalFunction("matmul_bf16_f32", object_file_name="mm_kernel.o",
                          source_file=os.path.join(HERE, "mm.cc"),
                          arg_types=[a_ty, b_ty, c_ty], include_dirs=[HERE], compile_flags=flags)
    zero = ExternalFunction("zero_f32", object_file_name="mm_kernel.o",
                            source_file=os.path.join(HERE, "mm.cc"),
                            arg_types=[c_ty], include_dirs=[HERE], compile_flags=flags)

    inA = ObjectFifo(a_ty, name="inA")
    a_dims = [(m // R, R * k), (k // S, S), (R, k), (S, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    inB = ObjectFifo(b_ty, name="inB")
    b_dims = [(n // T, T * k), (k // S, S), (T, k), (S, 1)]  # B column-major
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    c_dims = [(m // R, R * n), (R, T), (n // T, R * T), (T, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    def core_fn(of_a, of_b, of_c, zero_k, mm_k):
        for _ in (range_(tiles) if tiles > 1 else range(1)):
            e_c = of_c.acquire(1)
            zero_k(e_c)
            for _ in (range_(K_div_k) if K_div_k > 1 else range(1)):
                e_a = of_a.acquire(1)
                e_b = of_b.acquire(1)
                mm_k(e_a, e_b, e_c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(core_fn, [memA.cons(), memB.cons(), memC.prod(), zero, mm], stack_size=0xD00)

    rows_per_block = 4
    A_tiles = TensorTiler2D.group_tiler((M, K), (m, k), (1, K_div_k),
                                        pattern_repeat=N_div_n, prune_step=False)
    b_tap = TensorTiler2D.group_tiler((N, K), (n, k), (N_div_n, K_div_k), prune_step=False)[0]
    C_tiles = TensorTiler2D.group_tiler((M, N), (m, n), (rows_per_block // 2, N_div_n), prune_step=False)
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        tgs = []
        for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
            for pingpong in [0, 1]:
                row_base = tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                num_tile_rows = min(rows_per_block // 2, M_div_m - row_base)
                if num_tile_rows <= 0:
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])
                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])
                rt.drain(outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True)
                c_index += 1
                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]
        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--mlir", default="")
    args = ap.parse_args()
    module = build_module(args.M, args.K, args.N, args.m, args.k, args.n)
    text = str(module)
    if args.mlir:
        os.makedirs(os.path.dirname(os.path.abspath(args.mlir)), exist_ok=True)
        with open(args.mlir, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
