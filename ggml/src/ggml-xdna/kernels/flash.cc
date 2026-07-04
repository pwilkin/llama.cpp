//===- flash.cc — online-softmax flash attention for AIE2P (XDNA2) --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Streaming flash attention over one query block Q[NQ,DH] against NBLK key/value
// blocks of BLK keys each (max n_kv = NBLK*BLK). K/V for a full sequence exceed
// a core's L1, so they are streamed block by block while the running softmax
// state (max m, denom l, output accumulator O in fp32) is kept resident and
// corrected by exp(m_old - m_new) each block — the flash / online-softmax
// recurrence. Shorter sequences pad with mask = -inf (contributes 0).
//
// Layout (packed into one input buffer): [ Q(NQ*DH) | K(NBLK*BLK*DH) |
// V(NBLK*BLK*DH) | mask(NQ*NBLK*BLK) ]. Output O[NQ*DH]. bf16 in/out, fp32 accum.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef NQ
#    define NQ 32
#endif
#ifndef DH
#    define DH 128
#endif
#ifndef BLK
#    define BLK 32
#endif
#ifndef NBLK
#    define NBLK 16 // max_n_kv = NBLK * BLK
#endif
#ifndef ATTN_SCALE
#    define ATTN_SCALE 0.08838834764831845f // 1/sqrt(128)
#endif

static constexpr int  KV  = NBLK * BLK;
static constexpr float LOG2E = 1.44269504089f;

// scalar exp2 via a length-1 lane of a broadcast vector (no scalar exp on AIE;
// exp2 on XDNA2 is bf16-output only). Clamp the argument: softmax args are <= 0,
// and masked/first-block values are hugely negative — feeding those to the LUT
// garbages out, so treat anything below the bf16 underflow point as 0.
static inline float sexp2(float x) {
    if (x <= -88.0f) return 0.0f;
    aie::vector<bfloat16, 16> r = aie::exp2<bfloat16>(aie::broadcast<float, 16>(x));
    return (float) r[0];
}

extern "C" void xdna_flash_bf16(bfloat16 *restrict IN, bfloat16 *restrict O) {
    bfloat16 *restrict Q = IN;
    bfloat16 *restrict K = IN + NQ * DH;
    bfloat16 *restrict V = IN + NQ * DH + KV * DH;
    bfloat16 *restrict M = IN + NQ * DH + 2 * KV * DH;

    float m[NQ];
    float l[NQ];
    float Oacc[NQ * DH];
    for (int iq = 0; iq < NQ; iq++) {
        m[iq] = -1e30f;
        l[iq] = 0.0f;
        for (int d = 0; d < DH; d++) Oacc[iq * DH + d] = 0.0f;
    }

    for (int b = 0; b < NBLK; b++) {
        for (int iq = 0; iq < NQ; iq++) {
            const bfloat16 *restrict qr = Q + iq * DH;

            // scores over the BLK keys of this block (scalar dot, into bf16)
            bfloat16 sc[BLK];
            for (int ic = 0; ic < BLK; ic++) {
                const int k = b * BLK + ic;
                const bfloat16 *restrict kr = K + k * DH;
                float acc = 0.0f;
                for (int d = 0; d < DH; d++) acc += (float) qr[d] * (float) kr[d];
                sc[ic] = (bfloat16) (acc * ATTN_SCALE + (float) M[iq * KV + k]);
            }

            // vectorized block max
            aie::vector<bfloat16, 16> vmax = aie::broadcast<bfloat16, 16>((bfloat16) -3.0e38f);
            {
                auto it = aie::cbegin_vector<16>(sc);
                for (int j = 0; j < BLK; j += 16) vmax = aie::max(vmax, *it++);
            }
            const float m_blk = aie::reduce_max(vmax);
            const float m_new = m[iq] > m_blk ? m[iq] : m_blk;

            // vectorized P = exp2((sc - m_new)*log2e), psum = sum
            const aie::vector<bfloat16, 16> vmn    = aie::broadcast<bfloat16, 16>((bfloat16) m_new);
            const aie::vector<bfloat16, 16> vlog2e = aie::broadcast<bfloat16, 16>((bfloat16) LOG2E);
            bfloat16 P[BLK];
            aie::accum<accfloat, 16> vsum;
            vsum.from_vector(aie::zeros<float, 16>());
            {
                auto it = aie::cbegin_vector<16>(sc);
                bfloat16 * po = P;
                for (int j = 0; j < BLK; j += 16) {
                    aie::vector<bfloat16, 16> xm = aie::sub(*it++, vmn);
                    aie::accum<accfloat, 16>  s2 = aie::mul(xm, vlog2e);
                    aie::vector<bfloat16, 16> e  = aie::exp2<bfloat16>(s2.to_vector<float>());
                    aie::store_v(po, e);
                    po += 16;
                    vsum = aie::add(vsum, e);
                }
            }
            const float psum = aie::reduce_add(vsum.to_vector<float>());

            // cross-block correction (0 on the first block: running max is -inf)
            const float corr = (m[iq] <= -1.0e29f) ? 0.0f : sexp2((m[iq] - m_new) * LOG2E);
            l[iq] = corr * l[iq] + psum;

            for (int d = 0; d < DH; d++) {
                float ov = corr * Oacc[iq * DH + d];
                for (int ic = 0; ic < BLK; ic++) {
                    ov += (float) P[ic] * (float) V[(b * BLK + ic) * DH + d];
                }
                Oacc[iq * DH + d] = ov;
            }
            m[iq] = m_new;
        }
    }

    for (int iq = 0; iq < NQ; iq++) {
        const float inv = 1.0f / l[iq];
        for (int d = 0; d < DH; d++) {
            O[iq * DH + d] = (bfloat16) (Oacc[iq * DH + d] * inv);
        }
    }
}
