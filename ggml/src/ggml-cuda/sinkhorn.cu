#include "sinkhorn.cuh"

#include <cstring>

// Fused Sinkhorn-Knopp normalization for the DeepSeek-V4 mHC residual mixing.
// One thread per stacked n x n matrix. Matches build_hc_sinkhorn (after the softmax):
//   M += eps; normalize over ne1; then (normalize over ne0; normalize over ne1) repeated n_iter-1 times.
// eps is also added to every normalization denominator. Element (i = ne0, j = ne1) lives at i + j*n.

#define CUDA_SINKHORN_MAX_N 16
#define CUDA_SINKHORN_BLOCK 64

static __global__ void sinkhorn_kernel(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int n, const int n_iter, const float eps, const int64_t nmat) {
    const int64_t m = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= nmat) {
        return;
    }

    const int nn = n*n;
    float c[CUDA_SINKHORN_MAX_N*CUDA_SINKHORN_MAX_N];

    const float * s = src + m*nn;
    for (int e = 0; e < nn; ++e) {
        c[e] = s[e] + eps;
    }

    // normalize over ne1 (j): each ne0-row i divided by (eps + sum_j)
    auto norm_ne1 = [&]() {
        for (int i = 0; i < n; ++i) {
            float sum = eps;
            for (int j = 0; j < n; ++j) sum += c[i + j*n];
            for (int j = 0; j < n; ++j) c[i + j*n] /= sum;
        }
    };
    // normalize over ne0 (i): each ne1-col j divided by (eps + sum_i)
    auto norm_ne0 = [&]() {
        for (int j = 0; j < n; ++j) {
            float sum = eps;
            for (int i = 0; i < n; ++i) sum += c[i + j*n];
            for (int i = 0; i < n; ++i) c[i + j*n] /= sum;
        }
    };

    norm_ne1();
    for (int it = 1; it < n_iter; ++it) {
        norm_ne0();
        norm_ne1();
    }

    float * d = dst + m*nn;
    for (int e = 0; e < nn; ++e) {
        d[e] = c[e];
    }
}

void ggml_cuda_op_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));
    GGML_ASSERT(dst->ne[0] == dst->ne[1]); // square matrices

    const int     n    = (int) dst->ne[0];
    const int64_t nmat = dst->ne[2]*dst->ne[3];
    GGML_ASSERT(n <= CUDA_SINKHORN_MAX_N);

    float eps;
    memcpy(&eps, &dst->op_params[0], sizeof(float));
    const int n_iter = dst->op_params[1];

    cudaStream_t stream = ctx.stream();
    const int64_t num_blocks = (nmat + CUDA_SINKHORN_BLOCK - 1) / CUDA_SINKHORN_BLOCK;

    sinkhorn_kernel<<<num_blocks, CUDA_SINKHORN_BLOCK, 0, stream>>>(
        (const float *) src0->data, (float *) dst->data, n, n_iter, eps, nmat);
}
