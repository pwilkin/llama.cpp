//===- gdn.cc — gated delta net (linear attention), one token, AIE2P --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One recurrent step of the fused GATED_DELTA_NET op for a single (head, seq),
// mirroring ggml_compute_forward_gated_delta_net_one_chunk with n_tokens=1.
//
// The state is kept transposed: buffer row j (M[j], SV elements) = column j of S.
// Every op is per-row of M, so rows stream in SV/R blocks (like gemv). Per row j:
//   M[j][i] *= exp(g[i])                       (gate decay, kda: g is a vector)
//   delta    = (v[j] - dot(M[j], k)) * beta    (delta rule)
//   M[j][i] += delta * k[i]                    (rank-1 update)
//   attn[j]  = dot(M[j], q) * scale            (output)
//
// params  = [ g(SV) | k(SV) | q(SV) | beta @ 3*SV | scale @ 3*SV+1 ]  bf16 (resident)
// block_in  = [ M_rows(R*SV) | v(R) ]  bf16
// block_out = [ M_rows(R*SV) | attn(R) ]  bf16
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef SV
#    define SV 128
#endif
#ifndef GDN_R
#    define GDN_R 16
#endif


extern "C" void xdna_gdn(bfloat16 *restrict params, bfloat16 *restrict in, bfloat16 *restrict out) {
    // params[0..SV) is exp(g) precomputed host-side (accurate); the AIE exp2 LUT
    // is too coarse here — its error amplifies through the recurrent dot products.
    const bfloat16 *restrict expg = params;
    const bfloat16 *restrict k = params + SV;
    const bfloat16 *restrict q = params + 2 * SV;
    const float beta  = (float) params[3 * SV];
    const float scl   = (float) params[3 * SV + 1]; // 1/sqrt(SV), computed host-side

    for (int r = 0; r < GDN_R; r++) {
        bfloat16 *restrict M  = in + r * SV;
        bfloat16 *restrict Mo = out + r * SV;
        const float vj = (float) in[GDN_R * SV + r];

        // pass 1: M *= expg ; dk = dot(M, k)
        aie::accum<accfloat, 16> adk;
        adk.from_vector(aie::zeros<float, 16>());
        {
            auto mi = aie::cbegin_vector<16>(M);
            auto ei = aie::cbegin_vector<16>(expg);
            auto ki = aie::cbegin_vector<16>(k);
            bfloat16 * mo = M;
            for (int i = 0; i < SV; i += 16) {
                aie::accum<accfloat, 16>  pm = aie::mul(*mi++, *ei++);
                aie::vector<bfloat16, 16> mv = pm.to_vector<bfloat16>();
                aie::store_v(mo, mv);
                mo += 16;
                adk = aie::mac(adk, mv, *ki++);
            }
        }
        const float dk    = aie::reduce_add(adk.to_vector<float>());
        const float delta = (vj - dk) * beta;

        // pass 2: M += delta*k ; dq = dot(M, q)  -> attn
        const aie::vector<bfloat16, 16> vdelta = aie::broadcast<bfloat16, 16>((bfloat16) delta);
        aie::accum<accfloat, 16> adq;
        adq.from_vector(aie::zeros<float, 16>());
        {
            auto mi = aie::cbegin_vector<16>(M);
            auto ki = aie::cbegin_vector<16>(k);
            auto qi = aie::cbegin_vector<16>(q);
            bfloat16 * mo = Mo;
            for (int i = 0; i < SV; i += 16) {
                aie::accum<accfloat, 16>  pm = aie::mul(vdelta, *ki++);
                aie::vector<bfloat16, 16> mv = aie::add(*mi++, pm.to_vector<bfloat16>());
                aie::store_v(mo, mv);
                mo += 16;
                adq = aie::mac(adq, mv, *qi++);
            }
        }
        const float dq = aie::reduce_add(adq.to_vector<float>());
        out[GDN_R * SV + r] = (bfloat16) (dq * scl);
    }
}
