#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

constexpr int gdn_cols_per_warp = 4;

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// Chunked prefill for the KDA (per-key-dim gate) recurrence with S_v = S_k = 128 and a final state only.

static constexpr int gdn_chunk_d = 128;

// On HIP the second __launch_bounds__ argument is the minimum number of waves per SIMD: 8 caps the VGPRs at 192, which keeps all blocks of a 64-head layer (four per WGP) resident at once.
static constexpr __device__ int gdn_kda_chunked_min_waves() {
#ifdef GGML_USE_HIP
    return 8;
#else
    return 1;
#endif // GGML_USE_HIP
}

#if defined(GGML_USE_HIP) && defined(RDNA)
// value of x in the lane whose index differs in bit `mask`; DPP and permlanex16 stay off the LDS crossbar that __shfl_xor goes through
template <int mask>
static __device__ __forceinline__ float gdn_xor(float x) {
    static_assert(mask <= 16, "within a wave32");
    if constexpr (mask == 16) {
        return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xfedcba98, true, false));
    } else {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x160 | mask, 0xf, 0xf, true));
    }
}

// barrier that orders shared memory only: __syncthreads() also waits for every outstanding global load, which would end the next chunk's prefetch at each barrier
static __device__ __forceinline__ void gdn_sync_lds() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// sum over aligned groups of `width` consecutive lanes
template <int width>
static __device__ __forceinline__ float gdn_group_sum(float x) {
    static_assert(width <= 16, "groups are within 16 lanes");
    if constexpr (width >= 2) {
        x += gdn_xor<1>(x);
    }
    if constexpr (width >= 4) {
        x += gdn_xor<2>(x);
    }
    if constexpr (width >= 8) {
        x += gdn_xor<4>(x);
    }
    if constexpr (width >= 16) {
        x += gdn_xor<8>(x);
    }
    return x;
}

// one reduce-scatter step between the lanes that differ in bit `mask`: v[0, 2K) -> v[0, K), summed over both lanes; the lane with the bit set keeps the upper half
template <int mask, int K>
static __device__ __forceinline__ void gdn_reduce_scatter_step(float * v, const int lane) {
    const bool hi = lane & mask;
#pragma unroll
    for (int j = 0; j < K; ++j) {
        const float keep = hi ? v[K + j] : v[j];
        const float send = hi ? v[j] : v[K + j];
        v[j] = keep + gdn_xor<mask>(send);
    }
}

// The pairs (t, s <= t) of a chunk are split between the two halves of a block by row pairs (t, C-1-t), C+1 pairs per row pair. A half lists its k.k sums (s < t) first, then its q.k sums (s <= t).
template <int C>
static constexpr __device__ int gdn_row_half(const int t) {
    return (t < C - 1 - t ? t : C - 1 - t) % 2;
}

template <int C>
static constexpr __device__ int gdn_half_slots(const int hh) {
    int n = 0;
    for (int r = 0; r < C; ++r) {
        n += gdn_row_half<C>(r) == hh ? 2 * r + 1 : 0;
    }
    return n;
}

template <int C>
static constexpr __device__ int gdn_pair_slot(const int t, const int s, const bool qk) {
    const int hh = gdn_row_half<C>(t);
    int na = 0;
    int nq = 0;
    int ka = 0;
    for (int r = 0; r < C; ++r) {
        if (gdn_row_half<C>(r) == hh) {
            na += r < t ? r : 0;
            nq += r < t ? r + 1 : 0;
            ka += r;
        }
    }
    return qk ? ka + nq + s : na + s;
}

// this key dim's terms of the pair sums of half HH, in slot order
template <int C, int HH>
static __device__ __forceinline__ void gdn_pair_terms(const float * gcum, const float * kv, const float * qv, float * v) {
#pragma unroll
    for (int t = 0; t < C; ++t) {
        if (gdn_row_half<C>(t) != HH) {
            continue;
        }
#pragma unroll
        for (int s = 0; s <= t; ++s) {
            const float e = kv[s] * __expf(gcum[t] - gcum[s]);
            if (s < t) {
                v[gdn_pair_slot<C>(t, s, false)] = kv[t] * e;
            }
            v[gdn_pair_slot<C>(t, s, true)] = qv[t] * e;
        }
    }
}
#endif // defined(GGML_USE_HIP) && defined(RDNA)

// One block per (head, seq) owns the 128 x 128 state and walks the chunks in order. Lane (cg, r) keeps state rows [r * D/RG, (r + 1) * D/RG) of value columns [cg * CPT, (cg + 1) * CPT) in registers; the RG lanes of a column group are adjacent, so the products over the key dim are reduced with gdn_group_sum. To stage a chunk, each half of the block gates k and q of key dim tid % D for half of the tokens and sums that dim's terms of half of the token pairs of L and P; warp 0 then solves for T while the others start on the products with S.
template <int C, int RG, int CPT>
__global__ void __launch_bounds__(gdn_chunk_d / CPT * RG, gdn_kda_chunked_min_waves())
gated_delta_net_kda_chunked_cuda(const float * __restrict__ q,
                                 const float * __restrict__ k,
                                 const float * __restrict__ v,
                                 const float * __restrict__ g,
                                 const float * __restrict__ beta,
                                 const float * __restrict__ curr_state,
                                 float * __restrict__       dst,
                                 float * __restrict__       state,
                                 int64_t                    H,
                                 int64_t                    n_tokens,
                                 int64_t                    sq1,
                                 int64_t                    sq2,
                                 int64_t                    sq3,
                                 int64_t                    sv1,
                                 int64_t                    sv2,
                                 int64_t                    sv3,
                                 int64_t                    sb1,
                                 int64_t                    sb2,
                                 int64_t                    sb3,
                                 const uint3                neqk1_magic,
                                 const uint3                rq3_magic,
                                 float                      scale) {
#if defined(GGML_USE_HIP) && defined(RDNA)
    constexpr int D     = gdn_chunk_d;
    constexpr int NT    = D / CPT * RG;
    constexpr int RPL   = D / RG;     // state rows per lane
    constexpr int KP    = D + 4 * RG; // padded rows: the RG row groups of a read land in different banks
    constexpr int WS    = ggml_cuda_get_physical_warp_size();
    constexpr int NDW   = D / WS;     // warps per half of the block
    constexpr int VPT   = C * D / NT; // v values staged per thread
    constexpr int TPL   = C / RG;     // output tokens per lane
    static_assert(RPL % 4 == 0 && NT == 2 * D && (C * D) % NT == 0 && C % 2 == 0, "bad tile");
    static_assert(C <= WS && D % WS == 0, "the solve runs on one warp");
    static_assert(gdn_half_slots<C>(0) == WS && gdn_half_slots<C>(1) == WS, "one reduce-scatter per half");
    static_assert(C % RG == 0, "the lanes of a row group split the output tokens of a chunk");

    __shared__ __align__(16) float ku[C * KP];
    __shared__ __align__(16) float qu[C * KP];
    __shared__ __align__(16) float kd[C * KP];
    __shared__ __align__(16) float gl[KP];
    __shared__ __align__(16) float vs[C * D];
    __shared__ __align__(16) float tps[2 * C * C];
    __shared__ float red[2 * NDW][WS];
    __shared__ float bs[C];

    const int      tid      = threadIdx.x;
    const int      lane     = tid % WS;
    const int      r        = tid % RG;
    const int      cg       = tid / RG;
    const uint32_t h        = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      j0       = cg * CPT; // first value column of this lane
    const int      i0       = r * RPL;  // first state row of this lane
    const int      io       = i0 + 4 * r;
    const int64_t  n_chunks = (n_tokens + C - 1) / C;

    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const float * q_h = q + iq3 * sq3 + iq1 * sq1;
    const float * k_h = k + iq3 * sq3 + iq1 * sq1;
    const float * v_h = v + sequence * sv3 + h * sv1;
    float *       o_h = dst + (sequence * n_tokens * H + h) * D + j0;

    ggml_cuda_pdl_sync();

    float s[RPL][CPT];
    {
        const float * s_in = curr_state + (sequence * H + h) * D * D;
#pragma unroll
        for (int n = 0; n < CPT; ++n) {
#pragma unroll
            for (int m = 0; m < RPL; m += 4) {
                const float4 x4 = *(const float4 *) (s_in + (j0 + n) * D + i0 + m);
                s[m + 0][n] = x4.x;
                s[m + 1][n] = x4.y;
                s[m + 2][n] = x4.z;
                s[m + 3][n] = x4.w;
            }
        }
    }

    const int dim  = tid % D;
    const int half = __builtin_amdgcn_readfirstlane(tid / D); // whole warps, so the loads below get scalar offsets
    float pf[3 * C];
    float pv[VPT];
    float pb;
    auto load = [&](const int64_t chunk) {
        const int64_t t0 = chunk * C;
#pragma unroll
        for (int t = 0; t < C; ++t) {
            const int64_t tt = t0 + t < n_tokens ? t0 + t : n_tokens - 1;
            pf[t]         = g[(sequence * sb3 + tt * sb2 + h * sb1) * D + dim];
            pf[C + t]     = k_h[tt * sq2 + dim];
            pf[2 * C + t] = q_h[tt * sq2 + dim];
        }
        const int64_t tb = t0 + tid % C < n_tokens ? t0 + tid % C : n_tokens - 1;
        pb = beta[sequence * sb3 + tb * sb2 + h * sb1];
        const float * v_c = v_h + t0 * sv2;
#pragma unroll
        for (int e = 0; e < VPT; ++e) {
            const int t = e * (NT / D) + half;
            pv[e] = v_c[(t0 + t < n_tokens ? t : 0) * sv2 + dim];
        }
    };
    auto stage = [&](const int64_t t0) {
#pragma unroll
        for (int t = 0; t < C; ++t) {
            if (t0 + t >= n_tokens) {
                pf[t]         = 0.0f;
                pf[C + t]     = 0.0f;
                pf[2 * C + t] = 0.0f;
            }
        }
#pragma unroll
        for (int t = 1; t < C; ++t) {
            pf[t] += pf[t - 1];
        }
        const int   pi    = dim + dim / RPL * 4;
        const float glast = pf[C - 1]; // padding tokens have g = 0, so this is the last valid one
        if (half == 0) {
            gl[pi] = __expf(glast);
        }
#pragma unroll
        for (int t = 0; t < C; ++t) {
            if (t / (C / 2) == half) {
                const float eg = __expf(pf[t]);
                ku[t * KP + pi] = pf[C + t] * eg;
                qu[t * KP + pi] = scale * pf[2 * C + t] * eg;
                kd[t * KP + pi] = pf[C + t] * __expf(glast - pf[t]);
            }
        }
        float ps[WS];
        if (half == 0) {
            gdn_pair_terms<C, 0>(pf, pf + C, pf + 2 * C, ps);
        } else {
            gdn_pair_terms<C, 1>(pf, pf + C, pf + 2 * C, ps);
        }
        gdn_reduce_scatter_step<16, 16>(ps, lane);
        gdn_reduce_scatter_step< 8,  8>(ps, lane);
        gdn_reduce_scatter_step< 4,  4>(ps, lane);
        gdn_reduce_scatter_step< 2,  2>(ps, lane);
        gdn_reduce_scatter_step< 1,  1>(ps, lane);
        red[tid / WS][lane] = ps[0];
        if (tid < C) {
            bs[tid] = t0 + tid < n_tokens ? pb : 0.0f;
        }
#pragma unroll
        for (int e = 0; e < VPT; ++e) {
            const int t = e * (NT / D) + half;
            vs[t * D + dim] = t0 + t < n_tokens ? pv[e] : 0.0f;
        }
    };

    load(0);
    for (int64_t chunk = 0; chunk < n_chunks; ++chunk) {
        const int64_t t0 = chunk * C;

        gdn_sync_lds();
        stage(t0);
        gdn_sync_lds();

        if (tid < WS) {
            if (lane < C) {
                int col = lane;
                asm volatile("" : "+v"(col));
                float x[C];
#pragma unroll
                for (int t = 0; t < C; ++t) {
                    float acc = t == col ? 1.0f : 0.0f;
#pragma unroll
                    for (int s2 = 0; s2 < t; ++s2) {
                        float l = 0.0f;
#pragma unroll
                        for (int w = 0; w < NDW; ++w) {
                            l += red[gdn_row_half<C>(t) * NDW + w][gdn_pair_slot<C>(t, s2, false)];
                        }
                        acc = fmaf(-bs[t] * l, x[s2], acc);
                    }
                    x[t] = acc;
                }
#pragma unroll
                for (int t = 0; t < C; ++t) {
                    tps[t * C + col] = x[t] * bs[col];
                }
            } else {
                for (int j = lane - C; j < C * C; j += WS - C) {
                    const int t  = j / C;
                    const int s2 = j % C;
                    float     p  = 0.0f;
                    if (s2 <= t) {
#pragma unroll
                        for (int w = 0; w < NDW; ++w) {
                            p += red[gdn_row_half<C>(t) * NDW + w][gdn_pair_slot<C>(t, s2, true)];
                        }
                    }
                    tps[C * C + j] = scale * p;
                }
            }
        }
        load(chunk + 1 < n_chunks ? chunk + 1 : chunk);

        float x[C][CPT];
        float y[C][CPT];
#pragma unroll
        for (int t = 0; t < C; ++t) {
#pragma unroll
            for (int n = 0; n < CPT; ++n) {
                x[t][n] = 0.0f;
                y[t][n] = 0.0f;
            }
        }
#pragma unroll
        for (int m = 0; m < RPL; m += 4) {
#pragma unroll
            for (int t = 0; t < C; ++t) {
                const float4 a = *(const float4 *) (ku + t * KP + io + m);
                const float4 b = *(const float4 *) (qu + t * KP + io + m);
#pragma unroll
                for (int n = 0; n < CPT; ++n) {
                    x[t][n] = fmaf(a.x, s[m + 0][n], x[t][n]);
                    x[t][n] = fmaf(a.y, s[m + 1][n], x[t][n]);
                    x[t][n] = fmaf(a.z, s[m + 2][n], x[t][n]);
                    x[t][n] = fmaf(a.w, s[m + 3][n], x[t][n]);
                    y[t][n] = fmaf(b.x, s[m + 0][n], y[t][n]);
                    y[t][n] = fmaf(b.y, s[m + 1][n], y[t][n]);
                    y[t][n] = fmaf(b.z, s[m + 2][n], y[t][n]);
                    y[t][n] = fmaf(b.w, s[m + 3][n], y[t][n]);
                }
            }
        }
        gdn_sync_lds();

#pragma unroll
        for (int t = 0; t < C; ++t) {
#pragma unroll
            for (int n = 0; n < CPT; ++n) {
                x[t][n] = vs[t * D + j0 + n] - gdn_group_sum<RG>(x[t][n]);
            }
        }
        float * yv = &y[0][0];
        if constexpr (RG >= 16) {
            gdn_reduce_scatter_step<8, 8 * TPL * CPT>(yv, r);
        }
        if constexpr (RG >= 8) {
            gdn_reduce_scatter_step<4, 4 * TPL * CPT>(yv, r);
        }
        if constexpr (RG >= 4) {
            gdn_reduce_scatter_step<2, 2 * TPL * CPT>(yv, r);
        }
        if constexpr (RG >= 2) {
            gdn_reduce_scatter_step<1, TPL * CPT>(yv, r);
        }

#pragma unroll
        for (int t = C - 1; t >= 0; --t) {
#pragma unroll
            for (int n = 0; n < CPT; ++n) {
                float u = 0.0f;
#pragma unroll
                for (int s2 = 0; s2 <= t; ++s2) {
                    u = fmaf(tps[t * C + s2], x[s2][n], u);
                }
                x[t][n] = u;
            }
        }

#pragma unroll
        for (int tl = 0; tl < TPL; ++tl) {
            const int t = r * TPL + tl;
            if (t0 + t < n_tokens) {
                const float * p_t = tps + C * C + t * C;
                float o[CPT];
#pragma unroll
                for (int n = 0; n < CPT; ++n) {
                    o[n] = yv[tl * CPT + n];
                }
#pragma unroll
                for (int s2 = 0; s2 < C; ++s2) {
                    const float p_ts = p_t[s2];
#pragma unroll
                    for (int n = 0; n < CPT; ++n) {
                        o[n] = fmaf(p_ts, x[s2][n], o[n]);
                    }
                }
#pragma unroll
                for (int n = 0; n < CPT; ++n) {
                    o_h[(t0 + t) * H * D + n] = o[n];
                }
            }
        }

#pragma unroll
        for (int m = 0; m < RPL; m += 4) {
            const float4 d4 = *(const float4 *) (gl + io + m);
#pragma unroll
            for (int n = 0; n < CPT; ++n) {
                s[m + 0][n] *= d4.x;
                s[m + 1][n] *= d4.y;
                s[m + 2][n] *= d4.z;
                s[m + 3][n] *= d4.w;
            }
#pragma unroll
            for (int t = 0; t < C; ++t) {
                const float4 a = *(const float4 *) (kd + t * KP + io + m);
#pragma unroll
                for (int n = 0; n < CPT; ++n) {
                    s[m + 0][n] = fmaf(a.x, x[t][n], s[m + 0][n]);
                    s[m + 1][n] = fmaf(a.y, x[t][n], s[m + 1][n]);
                    s[m + 2][n] = fmaf(a.z, x[t][n], s[m + 2][n]);
                    s[m + 3][n] = fmaf(a.w, x[t][n], s[m + 3][n]);
                }
            }
        }
    }

    float * s_out = state + (sequence * H + h) * D * D;
#pragma unroll
    for (int n = 0; n < CPT; ++n) {
#pragma unroll
        for (int m = 0; m < RPL; m += 4) {
            *(float4 *) (s_out + (j0 + n) * D + i0 + m) = make_float4(s[m + 0][n], s[m + 1][n], s[m + 2][n], s[m + 3][n]);
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, g, beta, curr_state, dst, state, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                     sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(RDNA)
}

static constexpr int gdn_kda_chunk = 8;
static constexpr int gdn_kda_rg    = 4;
static constexpr int gdn_kda_cpt   = 2;

// One block per (head, seq) walks the tokens in order, so the kernel needs enough of them to fill the GPU: measured on gfx1151 it beats the per-token kernel from one chunk (8 tokens) with 16 or more blocks, and loses at every length with 8.
static bool gated_delta_net_kda_chunked_supported(int cc, bool kda, int K, int64_t S_v, int64_t S_k,
        int64_t H, int64_t n_tokens, int64_t n_seqs) {
    return GGML_CUDA_CC_IS_RDNA3_5(cc) && kda && K == 1 && S_v == gdn_chunk_d && S_k == gdn_chunk_d &&
        n_tokens >= gdn_kda_chunk && H * n_seqs >= 16;
}

static void launch_gated_delta_net_kda_chunked(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale, cudaStream_t stream) {
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    constexpr int nt = gdn_chunk_d / gdn_kda_cpt * gdn_kda_rg;
    const ggml_cuda_kernel_launch_params launch_params(dim3(H, n_seqs, 1), dim3(nt, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_kda_chunked_cuda<gdn_kda_chunk, gdn_kda_rg, gdn_kda_cpt>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (gated_delta_net_kda_chunked_supported(cc, kda, K, S_v, neq0, H, n_tokens, n_seqs)) {
        launch_gated_delta_net_kda_chunked(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
