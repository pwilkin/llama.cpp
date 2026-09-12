#include "idx-relu-sum.cuh"
#include <cstdlib>

// Lightning-indexer head reduction: relu each head's block score, then sum the heads in graph order.
// Reproduces unary op_relu (fmaxf(x, 0)) followed by (((r0 + r1) + r2) + ...) with the same rounding, so the result is
// bitwise identical to the RELU + CONT + ADD chain it replaces, while reading the scores once instead of four times.
static __global__ void idx_relu_sum_f32(const float * __restrict__ src, float * __restrict__ dst,
                                        const int n_blocks, const int heads) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const int64_t tt = blockIdx.y;                         // token within stream, streams stacked
    const float * p  = src + tt * (int64_t) heads * n_blocks + b;
    float acc = fmaxf(p[0], 0.0f);
    for (int h = 1; h < heads; ++h) {
        acc = acc + fmaxf(p[(int64_t) h * n_blocks], 0.0f);
    }
    dst[tt * (int64_t) n_blocks + b] = acc;
}

bool ggml_cuda_idx_relu_sum_enabled() {
    static const int v = getenv("LLAMA_IDX_RELU_SUM") ? atoi(getenv("LLAMA_IDX_RELU_SUM")) : 0;
    return v != 0;
}

void ggml_cuda_op_idx_relu_sum(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_relu_sum_args & args) {
    const ggml_tensor * s = args.score;
    const int n_blocks = (int) s->ne[0];
    const int64_t rows = s->ne[2] * s->ne[3];
    constexpr int threads = 256;
    const dim3 grid((n_blocks + threads - 1) / threads, (unsigned) rows, 1);
    idx_relu_sum_f32<<<grid, threads, 0, ctx.stream()>>>((const float *) s->data, (float *) args.dst->data, n_blocks, args.heads);
    CUDA_CHECK(cudaGetLastError());
    static unsigned hits = 0;
    if (hits++ < 2) fprintf(stderr, "IDX_RELU_SUM blocks=%d heads=%d rows=%lld\n", n_blocks, args.heads, (long long) rows);
}
