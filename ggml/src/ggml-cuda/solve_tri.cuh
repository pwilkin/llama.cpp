#include "common.cuh"

// shapes above these limits take the rocBLAS path in ggml_cuda_op_solve_tri
#define MAX_N_FAST 64
#define MAX_K_FAST 32

void ggml_cuda_op_solve_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// returns true when ggml_cuda_op_solve_tri takes the rocBLAS path, which is not
// stream-capture safe: rocblas trsm_batched allocates temporary device memory,
// which invalidates an in-flight HIP graph capture
// ref: https://github.com/ROCm/rocBLAS/issues/1240
// [TAG_SOLVE_TRI_CUDA_GRAPHS]
bool ggml_cuda_solve_tri_needs_sync(const ggml_tensor * dst);
