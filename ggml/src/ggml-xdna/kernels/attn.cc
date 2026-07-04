//===- attn.cc — fused single-core attention for AIE2P (XDNA2) ------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fused scaled-dot-product attention on one AIE compute core, computing
//   S = scale * (Q . K^T);  P = softmax(S);  O = P . V
// with Q,K,V,O resident in core L1 (so shapes must be small: NQ,NKV,DH).
// bfloat16 throughout, fp32 accumulation. This is the FLASH_ATTN_EXT op with
// scores materialized in core memory (not the memory-optimal online-softmax
// form). Q,K,V are laid out [rows, DH] row-major; K and V are [NKV, DH].
//
// The Q.K^T and P.V contractions use scalar fp32 MACs (correctness-first); the
// softmax over each score row is vectorized (aie::exp2 / reduce_max / reduce_add),
// matching aie_kernels/aie2p/softmax.cc idioms.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef NQ
#    define NQ 32
#endif
#ifndef NKV
#    define NKV 32
#endif
#ifndef DH
#    define DH 32
#endif
#ifndef ATTN_SCALE
#    define ATTN_SCALE 0.17677669529663687f // 1/sqrt(32)
#endif

static constexpr int VL = 16;

// QKV is Q [NQ,DH] then K [NKV,DH] then V [NKV,DH], packed into one buffer, so
// the core needs only a single input DMA channel (an AIE2 tile has just 2 in).
extern "C" void xdna_attn_bf16(bfloat16 *restrict QKV, bfloat16 *restrict O) {
    bfloat16 *restrict Q  = QKV;
    bfloat16 *restrict K  = QKV + NQ * DH;
    bfloat16 *restrict Vv = QKV + NQ * DH + NKV * DH;

    const aie::vector<bfloat16, VL> log2e = aie::broadcast<bfloat16, VL>(1.44269504089f);

    for (int iq = 0; iq < NQ; iq++) {
        bfloat16 s[NKV];
        bfloat16 e[NKV];

        // scores: s[ik] = scale * dot(Q[iq,:], K[ik,:])
        for (int ik = 0; ik < NKV; ik++) {
            float acc = 0.f;
            for (int d = 0; d < DH; d++) {
                acc += (float) Q[iq * DH + d] * (float) K[ik * DH + d];
            }
            s[ik] = (bfloat16) (acc * ATTN_SCALE);
        }

        // softmax over the NKV score row (vectorized)
        auto it_m = aie::cbegin_vector<VL>(s);
        aie::vector<bfloat16, VL> vmax = *it_m;
        for (int j = 0; j < NKV; j += VL) {
            vmax = aie::max(vmax, *it_m++);
        }
        const aie::vector<bfloat16, VL> vm = aie::broadcast<bfloat16, VL>(aie::reduce_max(vmax));

        auto it_e = aie::cbegin_vector<VL>(s);
        bfloat16 *pe = e;
        aie::accum<accfloat, VL> vsum;
        vsum.from_vector(aie::zeros<float, VL>());
        for (int j = 0; j < NKV; j += VL) {
            aie::vector<bfloat16, VL> xm = aie::sub(*it_e++, vm);
            aie::accum<accfloat, VL>  sc = aie::mul(xm, log2e);
            aie::vector<bfloat16, VL> ev = aie::exp2<bfloat16>(sc.to_vector<float>());
            aie::store_v(pe, ev);
            pe += VL;
            vsum = aie::add(vsum, ev);
        }
        const float inv = aie::inv(aie::reduce_add(vsum.to_vector<float>()));

        // output: O[iq,d] = (1/sum) * dot(e[:], V[:,d])
        for (int d = 0; d < DH; d++) {
            float acc = 0.f;
            for (int ik = 0; ik < NKV; ik++) {
                acc += (float) e[ik] * (float) Vv[ik * DH + d];
            }
            O[iq * DH + d] = (bfloat16) (acc * inv);
        }
    }
}
