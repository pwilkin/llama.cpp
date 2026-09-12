#include "norm-gated.cuh"
#include "ggml-impl.h"
#include <cstdlib>
#include <cstdio>

// Wave-per-row RMS norm for narrow rows (ncols <= 256): 8 rows per 256-thread block instead of one block per row
// (the model's per-head norms have ~786k rows of 256 -> the upstream launch is block-scheduling bound at ~75 GB/s).
// The sum reproduces norm.cu's rms_norm_f32<256,...> + block_reduce order exactly: per-warp xor trees over 32
// consecutive columns, then a xor tree over the 8 partials padded with zeros -> bitwise identical results.
// Optional sigmoid gate: dst = (scale * x * w) * (1 / (1 + expf(-z)))  (== op_sigmoid then binbcast mul).
static __device__ __forceinline__ float xor_tree(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor(v, off); }
    return v;
}
template <bool GATE>
static __global__ void __launch_bounds__(256) rms_rows_f32(const float * x, const float * w, const float * z, float * dst,
        const int ncols, const int64_t nrows, const int64_t nchannels, const int64_t total_rows,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int lane = threadIdx.x & 31;
    const int64_t g = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (g >= total_rows) return;
    const int64_t row = g % nrows, channel = (g / nrows) % nchannels, sample = g / (nrows * nchannels);
    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += g * ncols; if (GATE) z += g * ncols;
    float part[8];
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) {
        const int col = 32 * wv + lane;
        const float xi = col < ncols ? x[col] : 0.0f;
        part[wv] = xor_tree(xi * xi);
    }
    float v = 0.0f;
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) { v = lane == wv ? part[wv] : v; }
    const float tmp = xor_tree(v);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) {
        const int col = 32 * wv + lane;
        if (col < ncols) {
            const float t = scale * x[col] * w[col];
            if (GATE) { const float s = 1.0f / (1.0f + expf(-z[col])); dst[col] = t * s; } else { dst[col] = t; }
        }
    }
}

static bool norm_gated_enabled() { static const int v = getenv("LLAMA_NORM_GATED") ? atoi(getenv("LLAMA_NORM_GATED")) : 0; return v != 0; }
static bool norm_rows_enabled()  { static const int v = getenv("LLAMA_NORM_ROWS")  ? atoi(getenv("LLAMA_NORM_ROWS"))  : 0; return v != 0; }

static bool rows_shape_ok(const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * out) {
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32) return false;
    if (x->ne[0] > 256 || x->ne[0] % 32 != 0 || x->nb[0] != 4 || !ggml_is_contiguous(w) || !ggml_is_contiguous(out)) return false;
    if (w->ne[0] != x->ne[0] || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1) return false;
    if (x->ne[1] * x->ne[2] * x->ne[3] < 4096) return false;
    return ggml_nelements(out) == ggml_nelements(x);
}

int ggml_cuda_norm_gated_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m) {
    if (!norm_gated_enabled() || i + 3 >= cgraph->n_nodes) return 0;
    const ggml_tensor * rms = cgraph->nodes[i], * mul = cgraph->nodes[i+1];
    {   // debug: node sequence after narrow-row norms (LLAMA_NORM_GATED_DEBUG)
        static const int dbg = getenv("LLAMA_NORM_GATED_DEBUG") ? 1 : 0; static unsigned cnt = 0;
        if (dbg && rms->op == GGML_OP_RMS_NORM && rms->src[0]->ne[0] == 128 && cnt++ < 6) {
            char buf[512]; int n = snprintf(buf, sizeof(buf), "NORM_GATED seq @%d (rows %lld):", i, (long long) ggml_nrows(rms->src[0]));
            for (int k = i; k < cgraph->n_nodes && k < i + 7; ++k) n += snprintf(buf + n, sizeof(buf) - n, " %s(%s)", ggml_op_name(cgraph->nodes[k]->op), cgraph->nodes[k]->name);
            fprintf(stderr, "%s\n", buf);
        }
    }
    if (rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) return 0;
    int k = i + 2; m.pre = -1;
    while (k + 1 < cgraph->n_nodes && k < i + 6) {
        const ggml_op op = cgraph->nodes[k]->op;
        if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE) { ++k; continue; }
        if (op == GGML_OP_MUL_MAT && m.pre < 0) { m.pre = k; ++k; continue; }
        break;
    }
    if (k + 1 >= cgraph->n_nodes) return 0;
    const ggml_tensor * sig = cgraph->nodes[k]; ggml_tensor * out = cgraph->nodes[k+1];
    if (sig->op != GGML_OP_UNARY || out->op != GGML_OP_MUL) return 0;
    if (ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) return 0;
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : (mul->src[1] == rms ? mul->src[0] : nullptr);
    if (!w) return 0;
    const ggml_tensor * z = sig->src[0];
    if (m.pre >= 0) { const ggml_tensor * zr = z->view_src ? z->view_src : z; if (zr != cgraph->nodes[m.pre]) return 0; }
    if (!((out->src[0] == mul && out->src[1] == sig) || (out->src[0] == sig && out->src[1] == mul))) return 0;
    if (!ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, i+1, 1) || !ggml_node_has_n_uses(cgraph, k, 1)) return 0;
    if (!rows_shape_ok(x, w, out) || z->type != GGML_TYPE_F32 || !ggml_is_contiguous(z) || !ggml_is_contiguous(mul)) return 0;
    if (!ggml_are_same_shape(z, out) || !ggml_are_same_shape(mul, out) || !ggml_are_same_shape(rms, out)) return 0;
    float eps; memcpy(&eps, rms->op_params, sizeof(float)); if (eps < 0.0f) return 0;
    m.x = x; m.w = w; m.z = z; m.dst = out; m.eps = eps;
    return (k + 1) - i;
}

// plain RMS_NORM -> MUL(w) on narrow rows
int ggml_cuda_norm_rows_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m) {
    if (!norm_rows_enabled() || i + 1 >= cgraph->n_nodes) return 0;
    const ggml_tensor * rms = cgraph->nodes[i]; ggml_tensor * mul = cgraph->nodes[i+1];
    if (rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) return 0;
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : (mul->src[1] == rms ? mul->src[0] : nullptr);
    if (!w || !ggml_node_has_n_uses(cgraph, i, 1)) return 0;
    if (!rows_shape_ok(x, w, mul) || !ggml_are_same_shape(rms, mul)) return 0;
    float eps; memcpy(&eps, rms->op_params, sizeof(float)); if (eps < 0.0f) return 0;
    m.x = x; m.w = w; m.z = nullptr; m.dst = mul; m.eps = eps;
    return 1;
}

void ggml_cuda_op_norm_gated(ggml_backend_cuda_context & ctx, const ggml_cuda_norm_gated_match & m) {
    const ggml_tensor * x = m.x;
    static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "NORM_ROWS%s fused: ncols=%d rows=%lld\n", m.z ? "+GATE" : "", (int) x->ne[0], (long long) (x->ne[1]*x->ne[2]*x->ne[3]));
    const int64_t total = x->ne[1] * x->ne[2] * x->ne[3];
    const dim3 grid((unsigned) ((total + 7) / 8)), block(256);
    const ggml_cuda_kernel_launch_params lp(grid, block, 0, ctx.stream());
    if (m.z) ggml_cuda_kernel_launch(rms_rows_f32<true>, lp, (const float *) x->data, (const float *) m.w->data, (const float *) m.z->data, (float *) m.dst->data,
        (int) x->ne[0], x->ne[1], x->ne[2], total, (int64_t) (x->nb[1] / 4), (int64_t) (x->nb[2] / 4), (int64_t) (x->nb[3] / 4), m.eps);
    else     ggml_cuda_kernel_launch(rms_rows_f32<false>, lp, (const float *) x->data, (const float *) m.w->data, (const float *) nullptr, (float *) m.dst->data,
        (int) x->ne[0], x->ne[1], x->ne[2], total, (int64_t) (x->nb[1] / 4), (int64_t) (x->nb[2] / 4), (int64_t) (x->nb[3] / 4), m.eps);
}
