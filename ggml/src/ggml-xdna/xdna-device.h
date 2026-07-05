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

// Per-row reduction (op = "rms_norm" | "softmax"): applies the kernel to each
// contiguous row of `row_len` elements over the flat `src`/`dst` of `n` total
// elements. Uses <op>_<row_len>.xclbin. Returns false if unavailable.
bool ggml_xdna_npu_try_rows(const char * op, int row_len, const float * src, float * dst, int64_t n);

// Elementwise binary (op = "add" | "mul"): c = a (op) b over n elements. The
// caller must pre-broadcast b to full size. Returns false if unavailable.
bool ggml_xdna_npu_try_binary(const char * op, const float * a, const float * b, float * c, int64_t n);

// Fused flash attention on the NPU (streaming online-softmax kernel), per head
// with GQA. Handles the eligible case (DK==DV, f16 mask, no ALiBi/sinks/softcap,
// NKV a multiple of the block size, a matching flash_<NQ>_<DK>_<NKV>.xclbin).
// Returns false otherwise so the caller uses the host path.
bool ggml_xdna_npu_try_flash(const struct ggml_tensor * q, const struct ggml_tensor * k,
                             const struct ggml_tensor * v, const struct ggml_tensor * mask,
                             struct ggml_tensor * dst, float scale);

// Fused gated-delta-net (linear attention) on the NPU, one recurrent step per
// (head, seq). Handles the decode case (n_tokens==1, head dim 128, vector gate,
// a matching gdn_<SV>.xclbin). Returns false otherwise for the host path.
bool ggml_xdna_npu_try_gdn(struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
