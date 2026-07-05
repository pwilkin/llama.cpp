//===- binary.cc — elementwise binary ops for AIE2P (XDNA2) --------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Two-input elementwise ops: c = a (op) b over TILE_N bf16 elements. Broadcast
// (e.g. multiplying a row by a per-column weight) is handled host-side by the
// dispatch, which repeats the smaller operand into the full b buffer.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef TILE_N
#    define TILE_N 1024
#endif

static constexpr int V = 16;

extern "C" {

void xdna_add_bf16(bfloat16 *restrict a, bfloat16 *restrict b, bfloat16 *restrict c) {
    auto ia = aie::cbegin_vector<V>(a);
    auto ib = aie::cbegin_vector<V>(b);
    auto ic = aie::begin_vector<V>(c);
    for (int i = 0; i < TILE_N; i += V) {
        *ic++ = aie::add(*ia++, *ib++);
    }
}

void xdna_mul_bf16(bfloat16 *restrict a, bfloat16 *restrict b, bfloat16 *restrict c) {
    auto ia = aie::cbegin_vector<V>(a);
    auto ib = aie::cbegin_vector<V>(b);
    auto ic = aie::begin_vector<V>(c);
    for (int i = 0; i < TILE_N; i += V) {
        *ic++ = aie::mul(*ia++, *ib++).to_vector<bfloat16>();
    }
}

// Multi-op binary: one xclbin serves mul/add via a runtime op-code (op[0]).
// Both operands arrive packed in one input (ab = [a(TILE_N) | b(TILE_N)]) because
// an AIE compute tile has only 2 input DMA channels (op + ab). Codes: 0=mul 1=add.
void xdna_binary_multi(int32_t *restrict op, bfloat16 *restrict ab, bfloat16 *restrict c) {
    const int o = op[0];
    auto ia = aie::cbegin_vector<V>(ab);
    auto ib = aie::cbegin_vector<V>(ab + TILE_N);
    auto ic = aie::begin_vector<V>(c);
    if (o == 0) {
        for (int i = 0; i < TILE_N; i += V) *ic++ = aie::mul(*ia++, *ib++).to_vector<bfloat16>();
    } else {
        for (int i = 0; i < TILE_N; i += V) *ic++ = aie::add(*ia++, *ib++);
    }
}

} // extern "C"
