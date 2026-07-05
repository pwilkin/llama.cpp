//===- rope.cc — NEOX rotary embedding (host-precomputed cos/sin), AIE2P ----===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The XDNA2 AIE has no vector sin/cos, so the host precomputes cos/sin per token
// (mirroring ggml's rope; for text, imrope == NEOX) and this kernel does only the
// rotation. NEOX pairing (j, j+ND2) for j in [0,ND2), rotate the first NDIMS dims,
// copy [NDIMS, HD). Processes NR head-rows per call.
//   src     = [NR * HD]  bf16
//   cossin  = [cos(ND2) | sin(ND2)]  bf16   (same for all heads of a token)
//   out     = [NR * HD]  bf16
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef HD
#    define HD 256   // head dim
#endif
#ifndef NDIMS
#    define NDIMS 64 // rotary dims (n_rot)
#endif
#ifndef NR
#    define NR 4     // head-rows per call
#endif

static constexpr int ND2 = NDIMS / 2;

// arg order matches the IRON design (cossin resident first, then the src block).
extern "C" void xdna_rope(bfloat16 *restrict cossin, bfloat16 *restrict src, bfloat16 *restrict out) {
    for (int r = 0; r < NR; r++) {
        const bfloat16 *restrict s = src + r * HD;
        bfloat16 *restrict o = out + r * HD;
        for (int j = 0; j < ND2; j += 16) {
            aie::vector<bfloat16, 16> x0 = aie::load_v<16>(s + j);
            aie::vector<bfloat16, 16> x1 = aie::load_v<16>(s + j + ND2);
            aie::vector<bfloat16, 16> c  = aie::load_v<16>(cossin + j);
            aie::vector<bfloat16, 16> sn = aie::load_v<16>(cossin + ND2 + j);
            aie::accum<accfloat, 16> a0 = aie::mul(x0, c);   // x0*c
            a0 = aie::msc(a0, x1, sn);                       // x0*c - x1*sn
            aie::accum<accfloat, 16> a1 = aie::mul(x0, sn);  // x0*sn
            a1 = aie::mac(a1, x1, c);                        // x0*sn + x1*c
            aie::store_v(o + j,       a0.to_vector<bfloat16>());
            aie::store_v(o + j + ND2, a1.to_vector<bfloat16>());
        }
        for (int i = NDIMS; i < HD; i += 16) {
            aie::store_v(o + i, aie::load_v<16>(s + i));
        }
    }
}
