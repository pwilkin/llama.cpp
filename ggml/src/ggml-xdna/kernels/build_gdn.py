#!/usr/bin/env python3
# IRON program: one gated-delta-net recurrent step (one head, one token) on
# XDNA2. params [g|k|q|beta|scale] resident; the SV-row state streams in R-row
# blocks [M(R*SV)|v(R)] and out [M(R*SV)|attn(R)], all bf16.

import argparse
import os
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer

HERE = os.path.dirname(os.path.abspath(__file__))


def build_module(SV, R):
    assert SV % R == 0
    NB = SV // R
    par_ty = np.ndarray[(3 * SV + 16,), np.dtype[bfloat16]]
    bi_ty  = np.ndarray[(R * SV + R,),  np.dtype[bfloat16]]
    bo_ty  = np.ndarray[(R * SV + R,),  np.dtype[bfloat16]]

    P_ty = par_ty
    I_ty = np.ndarray[(NB * (R * SV + R),), np.dtype[bfloat16]]
    O_ty = np.ndarray[(NB * (R * SV + R),), np.dtype[bfloat16]]

    flags = [f"-DSV={SV}", f"-DGDN_R={R}"]
    src = os.path.join(HERE, "gdn.cc")
    gdn = ExternalFunction("xdna_gdn", source_file=src, arg_types=[par_ty, bi_ty, bo_ty],
                           object_file_name="gdn.o", include_dirs=[HERE], compile_flags=flags)

    ofP = ObjectFifo(par_ty, name="P")
    ofI = ObjectFifo(bi_ty,  name="I")
    ofO = ObjectFifo(bo_ty,  name="O")

    def core_fn(p, i, o, gdn_k):
        e_p = p.acquire(1)
        for _ in range_(NB):
            e_i = i.acquire(1)
            e_o = o.acquire(1)
            gdn_k(e_p, e_i, e_o)
            i.release(1)
            o.release(1)
        p.release(1)

    worker = Worker(core_fn, [ofP.cons(), ofI.cons(), ofO.prod(), gdn], stack_size=0x1000)

    rt = Runtime()
    with rt.sequence(P_ty, I_ty, O_ty) as (P, I, O):
        rt.start(worker)
        rt.fill(ofP.prod(), P)
        rt.fill(ofI.prod(), I)
        rt.drain(ofO.cons(), O, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--SV", type=int, default=128)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--mlir", default="")
    args = ap.parse_args()
    module = build_module(args.SV, args.R)
    text = str(module)
    if args.mlir:
        os.makedirs(os.path.dirname(os.path.abspath(args.mlir)), exist_ok=True)
        with open(args.mlir, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
