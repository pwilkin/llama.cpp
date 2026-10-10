#pragma once
#include "common.cuh"

// Quantized-weight BF16 WMMA GEMM for wide matmul batches on RDNA3.5 (gfx1151).

bool ggml_cuda_mmw_supported_mm  (const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
bool ggml_cuda_mmw_supported_mmid(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, int cc);

// true when the kernels are compiled in and enabled for this compute capability
bool ggml_cuda_mmw_available(int cc);

void ggml_cuda_mul_mat_mmw   (ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cuda_mul_mat_id_mmw(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
