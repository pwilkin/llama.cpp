//===- unary.cc — elementwise unary activation kernels for AIE2P (XDNA2) --===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Vectorized elementwise unary kernels for a single AIE-ML (AIE2P) compute tile,
// operating on bfloat16 (the NPU's native type). Each `extern "C"` entry point
// processes one TILE of TILE_N elements streamed in via an ObjectFifo; the IRON
// program (build_unary.py) binds one of these per op and compiles it to an
// .xclbin with the Peano backend.
//
// Idioms (accum vs vector conversions, exp via 2^(log2e*x), the tanh LUT) follow
// the AMD reference kernels bundled with mlir-aie, aie_kernels/aie2p/{silu,gelu,
// bf16_exp,relu}.cc. aie::mul/add/sub return an aie::accum; `.to_vector<T>()`
// (or assignment to an aie::vector) narrows it back to bf16.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef TILE_N
#    define TILE_N 1024 // elements per invocation (multiple of the vector width)
#endif

static constexpr int V = 16; // AIE2P bf16 activation vector lanes (fp32 accum)

// sigmoid(x) = 0.5 * (tanh(0.5x) + 1)  — reuses the hardware tanh, as in silu.cc
static inline aie::vector<bfloat16, V> vsigmoid(const aie::vector<bfloat16, V> &x,
                                                const aie::vector<bfloat16, V> &half,
                                                const aie::vector<bfloat16, V> &one) {
    aie::accum<accfloat, V>  half_x = aie::mul(x, half);
    aie::vector<bfloat16, V> t      = aie::tanh<bfloat16>(half_x.to_vector<float>());
    aie::vector<bfloat16, V> t1     = aie::add(t, one);
    return aie::mul(t1, half);
}

extern "C" {

void xdna_relu_bf16(bfloat16 *restrict a, bfloat16 *restrict c) {
    auto it_in  = aie::cbegin_vector<V>(a);
    auto it_out = aie::begin_vector<V>(c);
    aie::vector<bfloat16, V> zero = aie::broadcast<bfloat16, V>(0.0f);
    for (int i = 0; i < TILE_N; i += V) {
        *it_out++ = aie::max(*it_in++, zero);
    }
}

void xdna_neg_bf16(bfloat16 *restrict a, bfloat16 *restrict c) {
    auto it_in  = aie::cbegin_vector<V>(a);
    auto it_out = aie::begin_vector<V>(c);
    aie::vector<bfloat16, V> zero = aie::broadcast<bfloat16, V>(0.0f);
    for (int i = 0; i < TILE_N; i += V) {
        aie::vector<bfloat16, V> r = aie::sub(zero, *it_in++);
        *it_out++ = r;
    }
}

void xdna_abs_bf16(bfloat16 *restrict a, bfloat16 *restrict c) {
    auto it_in  = aie::cbegin_vector<V>(a);
    auto it_out = aie::begin_vector<V>(c);
    aie::vector<bfloat16, V> zero = aie::broadcast<bfloat16, V>(0.0f);
    for (int i = 0; i < TILE_N; i += V) {
        aie::vector<bfloat16, V> x = *it_in++;
        aie::vector<bfloat16, V> nx = aie::sub(zero, x);
        *it_out++ = aie::max(x, nx); // |x| = max(x, -x)
    }
}

// e^x = 2^(log2e * x)   (as in aie_kernels/aie2p/bf16_exp.cc)
void xdna_exp_bf16(bfloat16 *restrict a, bfloat16 *restrict c) {
    auto it_in  = aie::cbegin_vector<V>(a);
    auto it_out = aie::begin_vector<V>(c);
    aie::vector<bfloat16, V> log2e = aie::broadcast<bfloat16, V>(1.44269504089f);
    for (int i = 0; i < TILE_N; i += V) {
        aie::accum<accfloat, V> s = aie::mul(*it_in++, log2e);
        *it_out++ = aie::exp2<bfloat16>(s.to_vector<float>());
    }
}

void xdna_tanh_bf16(bfloat16 *restrict a, bfloat16 *restrict c) {
    auto it_in  = aie::cbegin_vector<V>(a);
    auto it_out = aie::begin_vector<V>(c);
    aie::vector<bfloat16, V> one = aie::broadcast<bfloat16, V>(1.0f);
    for (int i = 0; i < TILE_N; i += V) {
        aie::accum<accfloat, V> s = aie::mul(*it_in++, one); // -> fp32 accum
        *it_out++ = aie::tanh<bfloat16>(s.to_vector<float>());
    }
}

void xdna_sigmoid_bf16(bfloat16 *restrict a, bfloat16 *restrict c) {
    auto it_in  = aie::cbegin_vector<V>(a);
    auto it_out = aie::begin_vector<V>(c);
    aie::vector<bfloat16, V> half = aie::broadcast<bfloat16, V>(0.5f);
    aie::vector<bfloat16, V> one  = aie::broadcast<bfloat16, V>(1.0f);
    for (int i = 0; i < TILE_N; i += V) {
        *it_out++ = vsigmoid(*it_in++, half, one);
    }
}

void xdna_silu_bf16(bfloat16 *restrict a, bfloat16 *restrict c) {
    auto it_in  = aie::cbegin_vector<V>(a);
    auto it_out = aie::begin_vector<V>(c);
    aie::vector<bfloat16, V> half = aie::broadcast<bfloat16, V>(0.5f);
    aie::vector<bfloat16, V> one  = aie::broadcast<bfloat16, V>(1.0f);
    for (int i = 0; i < TILE_N; i += V) {
        aie::vector<bfloat16, V> x = *it_in++;
        aie::vector<bfloat16, V> s = vsigmoid(x, half, one);
        *it_out++ = aie::mul(x, s).to_vector<bfloat16>();
    }
}

void xdna_gelu_quick_bf16(bfloat16 *restrict a, bfloat16 *restrict c) {
    auto it_in  = aie::cbegin_vector<V>(a);
    auto it_out = aie::begin_vector<V>(c);
    aie::vector<bfloat16, V> half  = aie::broadcast<bfloat16, V>(0.5f);
    aie::vector<bfloat16, V> one   = aie::broadcast<bfloat16, V>(1.0f);
    aie::vector<bfloat16, V> k1702 = aie::broadcast<bfloat16, V>(1.702f);
    for (int i = 0; i < TILE_N; i += V) {
        aie::vector<bfloat16, V> x  = *it_in++;
        aie::vector<bfloat16, V> xs = aie::mul(x, k1702);
        aie::vector<bfloat16, V> s  = vsigmoid(xs, half, one);
        *it_out++ = aie::mul(x, s).to_vector<bfloat16>();
    }
}

// gelu tanh approx: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
void xdna_gelu_bf16(bfloat16 *restrict a, bfloat16 *restrict c) {
    auto it_in  = aie::cbegin_vector<V>(a);
    auto it_out = aie::begin_vector<V>(c);
    aie::vector<bfloat16, V> half   = aie::broadcast<bfloat16, V>(0.5f);
    aie::vector<bfloat16, V> one    = aie::broadcast<bfloat16, V>(1.0f);
    aie::vector<bfloat16, V> s2opi  = aie::broadcast<bfloat16, V>(0.79788456f);
    aie::vector<bfloat16, V> beta   = aie::broadcast<bfloat16, V>(0.044715f);
    for (int i = 0; i < TILE_N; i += V) {
        aie::vector<bfloat16, V> x     = *it_in++;
        aie::vector<bfloat16, V> x2    = aie::mul(x, x);
        aie::vector<bfloat16, V> x3    = aie::mul(x2, x);
        aie::vector<bfloat16, V> x3b   = aie::mul(x3, beta);
        aie::vector<bfloat16, V> inner = aie::add(x, x3b);
        aie::accum<accfloat, V>  arg   = aie::mul(inner, s2opi);
        aie::vector<bfloat16, V> t     = aie::tanh<bfloat16>(arg.to_vector<float>());
        aie::vector<bfloat16, V> t1    = aie::add(t, one);
        aie::vector<bfloat16, V> hx    = aie::mul(x, half);
        *it_out++ = aie::mul(hx, t1).to_vector<bfloat16>();
    }
}

} // extern "C"
