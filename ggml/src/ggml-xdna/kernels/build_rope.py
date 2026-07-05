#!/usr/bin/env python3
# IRON program: NEOX rotary embedding on XDNA2 (host precomputes cos/sin).
# cossin [2*ND2] resident, src head-rows stream in NR-row blocks, out streams out.
# Runtime buffers: COSSIN [2*ND2] bf16, SRC [NB*NR*HD] bf16, OUT [NB*NR*HD] bf16.

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(HD, NDIMS, NR, NB):
    ND2 = NDIMS // 2
    cs_ty  = np.ndarray[(2 * ND2,), np.dtype[bfloat16]]
    blk_ty = np.ndarray[(NR * HD,), np.dtype[bfloat16]]
    CS_ty  = cs_ty
    S_ty   = np.ndarray[(NB * NR * HD,), np.dtype[bfloat16]]

    flags = [f"-DHD={HD}", f"-DNDIMS={NDIMS}", f"-DNR={NR}"]
    kernel = ExternalFunction("xdna_rope", source_file=os.path.join(HERE, "rope.cc"),
                              arg_types=[cs_ty, blk_ty, blk_ty], include_dirs=[HERE], compile_flags=flags)

    ofCS = ObjectFifo(cs_ty,  name="cs")
    ofI  = ObjectFifo(blk_ty, name="in")
    ofO  = ObjectFifo(blk_ty, name="out")

    def core_fn(cs, i, o, k):
        e_cs = cs.acquire(1)
        for _ in range_(NB):
            e_i = i.acquire(1)
            e_o = o.acquire(1)
            k(e_cs, e_i, e_o)
            i.release(1)
            o.release(1)
        cs.release(1)

    worker = Worker(core_fn, [ofCS.cons(), ofI.cons(), ofO.prod(), kernel], stack_size=0x800)

    rt = Runtime()
    with rt.sequence(CS_ty, S_ty, S_ty) as (CS, S, O):
        rt.start(worker)
        rt.fill(ofCS.prod(), CS)
        rt.fill(ofI.prod(), S)
        rt.drain(ofO.cons(), O, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--HD", type=int, default=256)
    ap.add_argument("--NDIMS", type=int, default=64)
    ap.add_argument("--NR", type=int, default=4)
    ap.add_argument("--NB", type=int, default=1)
    ap.add_argument("--mlir", default="")
    args = ap.parse_args()
    module = build_module(args.HD, args.NDIMS, args.NR, args.NB)
    text = str(module)
    if args.mlir:
        os.makedirs(os.path.dirname(os.path.abspath(args.mlir)), exist_ok=True)
        with open(args.mlir, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
