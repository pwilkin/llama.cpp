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

#ifdef __cplusplus
}
#endif
