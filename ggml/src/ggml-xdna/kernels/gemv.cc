//===- gemv.cc — bf16 matrix-vector product (M=1 matmul) for AIE2P ---------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// y[N] = W[N,K] . x[K].  Inference matmuls are M=1 (decode) / M=small, i.e.
// gemv — the tiled mmul kernel (needs M%64) is useless there. x[K] stays
// resident; W streams in R-row blocks through an ObjectFifo, y streams out one
// R-block per iteration. bf16 in, fp32 out (fp32 dot accumulation).
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef GEMV_K
#    define GEMV_K 2560
#endif
#ifndef GEMV_R
#    define GEMV_R 4
#endif

extern "C" void xdna_gemv(bfloat16 *restrict x, bfloat16 *restrict W, float *restrict y) {
    for (int r = 0; r < GEMV_R; r++) {
        const bfloat16 *restrict wr = W + r * GEMV_K;
        aie::accum<accfloat, 16> acc;
        acc.from_vector(aie::zeros<float, 16>());
        auto xi = aie::cbegin_vector<16>(x);
        auto wi = aie::cbegin_vector<16>(wr);
        for (int kk = 0; kk < GEMV_K; kk += 16) {
            acc = aie::mac(acc, *wi++, *xi++);
        }
        y[r] = aie::reduce_add(acc.to_vector<float>());
    }
}
