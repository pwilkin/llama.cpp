//===- ssm_conv.cc — causal depthwise conv1d (SSM_CONV), one token, AIE2P ---===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// out[c] = sum_{i=0..KW-1} x[i,c] * w[i,c]   (per-channel, KW-tap causal conv),
// for one output token. The host separates the KW taps into contiguous channel
// arrays so the reduction vectorizes over channels. A channel block of R:
//   in = [ x0(R) | x1(R) | .. | x{KW-1}(R) | w0(R) | .. | w{KW-1}(R) ]  bf16
//   out = [ R ]  fp32
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef KW
#    define KW 4
#endif
#ifndef CONV_R
#    define CONV_R 256
#endif

extern "C" void xdna_ssm_conv(bfloat16 *restrict in, float *restrict out) {
    for (int c = 0; c < CONV_R; c += 16) {
        aie::accum<accfloat, 16> acc;
        acc.from_vector(aie::zeros<float, 16>());
        for (int t = 0; t < KW; t++) {
            const aie::vector<bfloat16, 16> xv = aie::load_v<16>(in + t * CONV_R + c);
            const aie::vector<bfloat16, 16> wv = aie::load_v<16>(in + (KW + t) * CONV_R + c);
            acc = aie::mac(acc, xv, wv);
        }
        aie::store_v(out + c, acc.to_vector<float>());
    }
}
