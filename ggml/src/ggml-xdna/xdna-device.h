#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// On-NPU execution path for the XDNA backend (XRT + AIE .xclbin kernels).
//
// Attempts to run a unary op on the NPU. Returns true if it executed there,
// false if the NPU path is unavailable (no XRT at build time, no device, no
// kernels found, unsupported op) — in which case the caller runs the host
// kernel. The kernel .xclbin directory is taken from $GGML_XDNA_KERNELS.
bool ggml_xdna_npu_try_unary(enum ggml_unary_op op, const float * src, float * dst, int64_t n);

// Attempt ggml mul_mat (dst = a^T-contracted-with-b) on the NPU. `a` is the
// weight [K,N] (any type with a to_float dequantizer), `b` the f32 activations
// [K,M]; dst is f32 [N,M]. Returns false if no xclbin for this (M,K,N) shape or
// the op is otherwise unsupported, in which case the caller falls back.
bool ggml_xdna_npu_try_mul_mat(const struct ggml_tensor * a, const struct ggml_tensor * b, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
