#!/usr/bin/env python3
# IRON program: streaming online-softmax flash attention on XDNA2 (NPU2/AIE2P).
# Q and the fp32 state buffer are acquired once; K/V/mask blocks stream through
# an ObjectFifo (one block per iteration), so the core holds only one block and
# one xclbin serves any n_kv. State = [Oacc(NQ*DH) | m(NQ) | l(NQ)] fp32, output.
#
# Runtime buffers: Q [NQ*DH] bf16, KV [NBLK*(2*BLK*DH + NQ*BLK)] bf16 (blocks
# packed [K|V|mask]), O [NQ*DH + 2*NQ] fp32 (host reads [0:NQ*DH]).

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(NQ, DH, BLK, NBLK, scale):
    blk_elems = 2 * BLK * DH + NQ * BLK
    q_ty     = np.ndarray[(NQ * DH,),   np.dtype[bfloat16]]
    kvblk_ty = np.ndarray[(blk_elems,), np.dtype[bfloat16]]
    state_ty = np.ndarray[(NQ * DH + 2 * NQ,), np.dtype[np.float32]]

    Q_ty  = q_ty
    KV_ty = np.ndarray[(NBLK * blk_elems,), np.dtype[bfloat16]]
    O_ty  = state_ty

    flags = [f"-DNQ={NQ}", f"-DDH={DH}", f"-DBLK={BLK}", f"-DATTN_SCALE={scale!r}f"]
    src = os.path.join(HERE, "flash_stream.cc")
    common = dict(object_file_name="flash_stream.o", source_file=src, include_dirs=[HERE], compile_flags=flags)
    init  = ExternalFunction("xdna_flash_init",  arg_types=[state_ty], **common)
    block = ExternalFunction("xdna_flash_block", arg_types=[q_ty, kvblk_ty, state_ty], **common)
    final = ExternalFunction("xdna_flash_final", arg_types=[state_ty], **common)

    ofQ  = ObjectFifo(q_ty, name="Q")
    ofKV = ObjectFifo(kvblk_ty, name="KV")
    ofO  = ObjectFifo(state_ty, name="O")

    def core_fn(q, kv, o, init_k, block_k, final_k):
        e_q = q.acquire(1)
        e_o = o.acquire(1)
        init_k(e_o)
        for _ in range_(NBLK):
            e_k = kv.acquire(1)
            block_k(e_q, e_k, e_o)
            kv.release(1)
        final_k(e_o)
        q.release(1)
        o.release(1)

    worker = Worker(core_fn, [ofQ.cons(), ofKV.cons(), ofO.prod(), init, block, final], stack_size=0x2000)

    rt = Runtime()
    with rt.sequence(Q_ty, KV_ty, O_ty) as (Q, KV, O):
        rt.start(worker)
        rt.fill(ofQ.prod(), Q)
        rt.fill(ofKV.prod(), KV)
        rt.drain(ofO.cons(), O, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--NQ", type=int, default=16)
    ap.add_argument("--DH", type=int, default=64)
    ap.add_argument("--BLK", type=int, default=16)
    ap.add_argument("--NBLK", type=int, default=4)
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
