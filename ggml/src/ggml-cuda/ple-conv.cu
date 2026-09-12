#include "ple-conv.cuh"
#include "unary.cuh"
#include <cstdlib>
static bool ple_conv_enabled() { static const int v = getenv("LLAMA_PLE_CONV") ? atoi(getenv("LLAMA_PLE_CONV")) : 0; return v != 0; }
#if defined(__HIP_PLATFORM_AMD__)
static __device__ __forceinline__ float ple_mul_rn(const float a, const float b) { float r; asm("v_mul_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
static __device__ __forceinline__ float ple_add_rn(const float a, const float b) { float r; asm("v_add_f32_e32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r; }
#else
static __device__ __forceinline__ float ple_mul_rn(const float a, const float b) { return __fmul_rn(a, b); }
static __device__ __forceinline__ float ple_add_rn(const float a, const float b) { return __fadd_rn(a, b); }
#endif

static __global__ void ple_concat_tail(const float * __restrict__ state, const float * __restrict__ x, float * __restrict__ out,
                                       const int C, const int T, const int H, const int tail_from, const int row_stride) {
    const int ncols = T + H - tail_from;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) C * ncols) return;
    const int c = (int) (idx / ncols), j = tail_from + (int) (idx % ncols);
    out[(size_t) c * row_stride + j] = (j < H) ? state[c * H + j] : x[(size_t) (j - H) * C + c];
}

template <int K, int DIL, int TT>
static __global__ void __launch_bounds__(256) ple_conv_kernel(const float * __restrict__ state, const float * __restrict__ x,
        const half * __restrict__ w, float * __restrict__ y, const int C, const int T) {
    constexpr int H = (K - 1) * DIL, WIN = H + 1;
    const int c = blockIdx.x * 256 + threadIdx.x, t0 = blockIdx.y * TT;
    if (c >= C) return;
    float wr[K];
#pragma unroll
    for (int k = 0; k < K; ++k) wr[k] = __half2float(w[c * K + k]);
    float win[WIN];
    auto ld = [&](int jp) -> float { return (jp < H) ? state[c * H + jp] : ((jp - H) < T ? x[(size_t) (jp - H) * C + c] : 0.0f); };
#pragma unroll
    for (int k = 0; k < WIN; ++k) win[k] = ld(t0 + k);
    const int tend = min(T, t0 + TT);
    for (int t = t0; t < tend; ++t) {
        float s = ple_mul_rn(win[0], wr[0]);
#pragma unroll
        for (int k = 1; k < K; ++k) s = ple_add_rn(s, ple_mul_rn(win[k * DIL], wr[k]));
        y[(size_t) t * C + c] = ggml_cuda_op_silu_single(s);
#pragma unroll
        for (int k = 0; k < WIN - 1; ++k) win[k] = win[k + 1];
        win[WIN - 1] = ld(t + WIN);
    }
}

static bool ple_conv_check(const ggml_cgraph * cgraph, int i, ggml_cuda_ple_conv_match & m) {
    if (!ple_conv_enabled() || i < 0 || i + 2 >= cgraph->n_nodes) return false;
    const ggml_tensor * cc = cgraph->nodes[i];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cc, 0) != 0) return false;
    const ggml_tensor * st = cc->src[0]; const ggml_tensor * tr = cc->src[1];
    if (!st || !tr || st->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32 || tr->op != GGML_OP_TRANSPOSE || !tr->view_src) return false;
    const ggml_tensor * x = tr->view_src;
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1) return false;
    const int64_t C = x->ne[0], T = x->ne[1], H = st->ne[0];
    if (H != 9 || st->ne[1] != C || st->ne[2] != 1 || !ggml_is_contiguous(st)) return false;
    if (tr->ne[0] != T || tr->ne[1] != C || tr->view_offs != 0) return false;
    if (cc->ne[0] != T + H || cc->ne[1] != C || cc->ne[2] != 1 || !ggml_is_contiguous(cc) || T < 256 || C % 256 != 0) return false;
    if (cgraph->nodes[i + 1]->op != GGML_OP_VIEW || cgraph->nodes[i + 1]->view_src != cc) return false;
    int64_t tail_from = T + H; int first_tap = -1, ntaps = 0; const ggml_tensor * wroot = nullptr; const ggml_tensor * chain = nullptr; int silu = -1;
    const ggml_tensor * tap_out[4] = {nullptr,nullptr,nullptr,nullptr};
    for (int n = i + 1; n < cgraph->n_nodes && n < i + 96; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SILU && chain && t->src[0] == chain) { silu = n; break; }
        if (t->op == GGML_OP_VIEW && t->view_src == cc && t->ne[0] == H) { tail_from = std::min(tail_from, (int64_t)(t->view_offs / sizeof(float))); continue; }
        if (t->op == GGML_OP_CONT && t->src[0]->op == GGML_OP_VIEW && t->src[0]->view_src == cc && t->src[0]->ne[0] == H) continue;
        if (t->op == GGML_OP_CPY) continue;
        if (t->op == GGML_OP_CONT && t->src[0]->op == GGML_OP_TRANSPOSE && t->src[0]->view_src == cc) {
            if (ntaps >= 4 || t->ne[0] != C || t->ne[1] != T) return false;
            m.starts[ntaps] = (int) (t->src[0]->view_offs / sizeof(float));
            if (m.starts[ntaps] != ntaps * 3) return false;
            if (first_tap < 0) first_tap = n;
            tap_out[ntaps] = t; ++ntaps; continue;
        }
        if (t->op == GGML_OP_CONT || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE) continue;
        if (t->op == GGML_OP_MUL) {
            // MUL(tap_out[k], w_k) with w_k an F32 [C] cast of a view of the F16 model weight
            const ggml_tensor * a = t->src[0]; const ggml_tensor * b = t->src[1];
            int k = -1; for (int q = 0; q < ntaps; ++q) if (tap_out[q] == a || tap_out[q] == b) k = q;
            if (k < 0) return false;
            const ggml_tensor * wk = (tap_out[k] == a) ? b : a;
            if (wk->type != GGML_TYPE_F32 || ggml_nelements(wk) != C) return false;
            // resolve the model weight: cast(CPY) <- reshape/view <- cont <- view of W[K, C] F16
            const ggml_tensor * src = wk->src[0]; while (src && (src->op == GGML_OP_RESHAPE || src->op == GGML_OP_VIEW)) src = src->src[0];
            if (!src || src->op != GGML_OP_CONT) return false;
            const ggml_tensor * wv = src->src[0]; if (!wv || wv->op != GGML_OP_VIEW || !wv->view_src) return false;
            const ggml_tensor * W = wv->view_src;
            if (W->type != GGML_TYPE_F16 || W->ne[0] != 4 || W->ne[1] != C || !ggml_is_contiguous(W)) return false;
            if (wv->view_offs != (size_t) k * sizeof(ggml_fp16_t) || wv->nb[1] != W->nb[1]) return false;
            if (wroot && wroot != W) return false; wroot = W;
            if (k == 0) chain = t; else { /* add comes next */ }
            tap_out[k] = t;
            continue;
        }
        if (t->op == GGML_OP_ADD) {
            if (!chain) return false;
            const ggml_tensor * a = t->src[0]; const ggml_tensor * b = t->src[1];
            int k = -1; for (int q = 1; q < ntaps; ++q) if (tap_out[q] == b && a == chain) k = q;
            if (k < 0) return false;
            chain = t; continue;
        }
        return false;
    }
    if (silu < 0 || ntaps != 4 || !wroot || first_tap < 0) return false;
    const ggml_tensor * su = cgraph->nodes[silu];
    if (su->type != GGML_TYPE_F32 || !ggml_is_contiguous(su) || su->ne[0] != C || su->ne[1] != T) return false;
    m.concat_idx = i; m.first_tap_idx = first_tap; m.silu_idx = silu; m.x = x; m.state = st; m.concat = cc; m.w = wroot; m.out = cgraph->nodes[silu];
    m.C = C; m.T = T; m.H = H; m.K = 4; m.dil = 3; m.tail_from = tail_from;
    return true;
}
bool ggml_cuda_ple_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_ple_conv_match & m) { return ple_conv_check(cgraph, i, m); }
bool ggml_cuda_ple_conv_match_at_tap(const ggml_cgraph * cgraph, int i, ggml_cuda_ple_conv_match & m) {
    if (!ple_conv_enabled()) return false;
    const ggml_tensor * t = cgraph->nodes[i];
    if (t->op != GGML_OP_CONT || !t->src[0] || t->src[0]->op != GGML_OP_TRANSPOSE || !t->src[0]->view_src) return false;
    const ggml_tensor * cc = t->src[0]->view_src;
    for (int c = i - 1; c >= 0 && c >= i - 96; --c) if (cgraph->nodes[c] == cc) return ple_conv_check(cgraph, c, m) && m.first_tap_idx == i;
    return false;
}
void ggml_cuda_ple_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_ple_conv_match & m) {
    const int ncols = (int) (m.T + m.H - m.tail_from); if (ncols <= 0) return;
    const int64_t n = m.C * ncols;
    ple_concat_tail<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (float *) m.concat->data, (int) m.C, (int) m.T, (int) m.H, (int) m.tail_from, (int) (m.concat->nb[1] / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}
void ggml_cuda_ple_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_ple_conv_match & m) {
    constexpr int TT = 128;
    dim3 grid((unsigned) (m.C / 256), (unsigned) ((m.T + TT - 1) / TT));
    ple_conv_kernel<4, 3, TT><<<grid, 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (const half *) m.w->data, (float *) m.out->data, (int) m.C, (int) m.T);
    CUDA_CHECK(cudaGetLastError());
}
