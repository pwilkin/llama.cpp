//===- flash_stream.cc — streaming online-softmax flash attention (AIE2P) -===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Streaming form of the flash kernel: K/V arrive one BLK-sized block at a time
// through an ObjectFifo, so the core only ever holds a single block and one
// xclbin serves any n_kv. The running softmax state lives in a persistent fp32
// buffer STATE = [ Oacc(NQ*DH) | m(NQ) | l(NQ) ] (the acquired output fifo
// element, held across the whole block loop — like the matmul C accumulator), so
// it is NOT on the core stack.
//
//   xdna_flash_init  — zero Oacc, m=-inf, l=0
//   xdna_flash_block — fold one K/V/mask block into the state (online softmax)
//   xdna_flash_final — Oacc /= l  (Oacc becomes the fp32 output)
//
// A per-block input element packs [ K(BLK*DH) | V(BLK*DH) | mask(NQ*BLK) ] bf16.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef NQ
#    define NQ 16
#endif
#ifndef DH
#    define DH 64
#endif
#ifndef BLK
#    define BLK 16
#endif
#ifndef ATTN_SCALE
#    define ATTN_SCALE 0.125f
#endif

static constexpr float LOG2E = 1.44269504089f;

extern "C" {

void xdna_flash_init(float *restrict STATE) {
    for (int i = 0; i < NQ * DH; i++) STATE[i] = 0.0f;
    for (int iq = 0; iq < NQ; iq++) {
        STATE[NQ * DH + iq]      = -1.0e30f; // m
        STATE[NQ * DH + NQ + iq] = 0.0f;     // l
    }
}

void xdna_flash_block(bfloat16 *restrict Q, bfloat16 *restrict KV, float *restrict STATE) {
    bfloat16 *restrict K = KV;
    bfloat16 *restrict V = KV + BLK * DH;
    bfloat16 *restrict M = KV + 2 * BLK * DH; // mask [NQ, BLK]
    float *restrict Oacc = STATE;
    float *restrict m = STATE + NQ * DH;
    float *restrict l = STATE + NQ * DH + NQ;

    const aie::vector<bfloat16, 16> vlog2e = aie::broadcast<bfloat16, 16>((bfloat16) LOG2E);

    for (int iq = 0; iq < NQ; iq++) {
        const bfloat16 *restrict qr = Q + iq * DH;

        alignas(32) bfloat16 sc[BLK];
        for (int ic = 0; ic < BLK; ic++) {
            const bfloat16 *restrict kr = K + ic * DH;
            float acc = 0.0f;
            for (int d = 0; d < DH; d++) acc += (float) qr[d] * (float) kr[d];
            sc[ic] = (bfloat16) (acc * ATTN_SCALE + (float) M[iq * BLK + ic]);
        }

        aie::vector<bfloat16, 16> vmax = aie::broadcast<bfloat16, 16>((bfloat16) -3.0e38f);
        { auto it = aie::cbegin_vector<16>(sc); for (int j = 0; j < BLK; j += 16) vmax = aie::max(vmax, *it++); }
        const float m_blk = aie::reduce_max(vmax);
        const float m_new = m[iq] > m_blk ? m[iq] : m_blk;

        const aie::vector<bfloat16, 16> vmn = aie::broadcast<bfloat16, 16>((bfloat16) m_new);
        alignas(32) bfloat16 P[BLK];
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

        float corr = 0.0f;
        if (m[iq] > -1.0e29f) {
            aie::vector<bfloat16, 16> dv = aie::broadcast<bfloat16, 16>((bfloat16) (m[iq] - m_new));
            aie::accum<accfloat, 16>  da = aie::mul(dv, vlog2e);
            aie::vector<bfloat16, 16> ce = aie::exp2<bfloat16>(da.to_vector<float>());
            corr = (float) ce[0];
        }
        l[iq] = corr * l[iq] + psum;

        for (int d = 0; d < DH; d++) {
            float ov = corr * Oacc[iq * DH + d];
            for (int ic = 0; ic < BLK; ic++) ov += (float) P[ic] * (float) V[ic * DH + d];
            Oacc[iq * DH + d] = ov;
        }
        m[iq] = m_new;
    }
}

void xdna_flash_final(float *restrict STATE) {
    float *restrict Oacc = STATE;
    float *restrict l = STATE + NQ * DH + NQ;
    for (int iq = 0; iq < NQ; iq++) {
        const float inv = 1.0f / l[iq];
        for (int d = 0; d < DH; d++) Oacc[iq * DH + d] *= inv;
    }
}

} // extern "C"
