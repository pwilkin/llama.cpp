#include "mmw.cuh"
#include "mmw-config-rdna3-5.cuh"
#include "mmid.cuh"

#include <cstdlib>

// shared-memory tile geometry and block size; also used by the host-side launch configuration
constexpr int MMW_BK = 64, MMW_NT = 256, MMW_LDS_STRIDE = MMW_BK + 8;

#if defined(AMD_WMMA_AVAILABLE)
namespace {

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint16_t mmw_f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
// same RNE as mmw_f2bf on both halves, the two high halves joined by one v_perm_b32 (no mov_b16 + and_or)
__device__ __forceinline__ uint32_t mmw_pack2(float a, float b) {
    uint32_t ua = __float_as_uint(a), ub = __float_as_uint(b);
    ua += 0x7fffu + ((ua >> 16) & 1u); ub += 0x7fffu + ((ub >> 16) & 1u);
    return __builtin_amdgcn_perm(ub, ua, 0x07060302u);
}
__device__ __forceinline__ float mmw_h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }

} // namespace

#include "mmw-quant.cuh"

namespace {

template <typename DRowFn>
__device__ __forceinline__ void mmw_store_tile(const v8f & acc, float * __restrict__ stg, float * __restrict__ D,
        const int M, DRowFn drow, const int n_base, const int m_base, const int lane) {
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int e = 0; e < 8; ++e) { stg[(2 * e + cn) * 16 + cm] = acc[e]; }
    __syncthreads();
    const int n = lane >> 1, half = lane & 1;
    const int dr = drow(n_base + n);
    if (dr >= 0) {
        const float4 v0 = *(const float4 *)(stg + n * 16 + half * 8), v1 = *(const float4 *)(stg + n * 16 + half * 8 + 4);
        const size_t base = (size_t)dr * M + m_base + half * 8;
        const bool full = m_base + 16 <= M;
        if (full && (M & 3) == 0) {
            *(float4 *)(D + base) = v0; *(float4 *)(D + base + 4) = v1;
        } else {
            const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
            for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) D[base + k] = vv[k]; }
        }
    }
    __syncthreads();
}

template <int N> using mmw_pos = std::integral_constant<int, N>;   // a compile-time K loop position

// LDS-only barrier for the K loop, which shares only shared memory: __syncthreads() also waits for every outstanding global load
__device__ __forceinline__ void mmw_sync_lds() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// one BM x BN output tile: the weights are dequantized to BF16 in shared memory while the WMMA loop runs
template <int BM, int BN, int WTM, int WTN, int WTYPE, bool TAIL, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmw_tile_gemm(const uint8_t * __restrict__ Wbase, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * As, uint16_t * Bs) {
    constexpr int WAVES_M = BM / WTM, WAVES = WAVES_M * (BN / WTN), TM = WTM / 16, TN = WTN / 16;
    constexpr int B_ITEMS = (BN * 8) / MMW_NT;
    static_assert(BM % WTM == 0 && BN % WTN == 0 && WAVES <= MMW_NT / 32, "the wave tiles must cover the block tile");
    static_assert((BM * 2) % MMW_NT == 0 && B_ITEMS >= 1, "two threads per weight row and at least one activation item per thread");
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    const bool wact = WAVES == MMW_NT / 32 || wave < WAVES;   // waves past the tile only load and decode
    const int nks = K / MMW_BK;
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMW_NT; brow[i] = max(xrow(c >> 3), 0); }
    v4u bst[2][B_ITEMS];

    // grid-based quants (mmw_lb): two threads per weight row, each prefetches its 32-weight block of the 4 slices of a super-block. The grid is filled into LDS once; store_w decodes with the same fp32 math, so the weights are bit-identical.
    constexpr int WR = (BM * 2) / MMW_NT;   // weight rows per thread
    using LBT = mmw_lb<WTYPE>;
    constexpr bool LB = LBT::ok;
    typename LBT::regs lbr[2][WR];
    __shared__ typename LBT::grid_t lbgrid[LBT::N];
    __shared__ uint2 lbsign[LB ? 16 : 1];
    if constexpr (LB) {
        for (int i = tid; i < LBT::N; i += MMW_NT) lbgrid[i] = LBT::entry(i);
        if (tid < 16) lbsign[tid] = mmw_sign_masks(tid);
    }
    // Q2_0: two threads per weight row, each prefetches 16 bytes of the 18-byte block, store_w decodes its 32 weights
    constexpr bool Q20 = WTYPE == 32 + GGML_TYPE_Q2_0;
    mmw_q20_half q20[2][WR];

    // the other quants (mmw_rq): two threads per weight row, each prefetches its 32-weight half of the row's slices. store_w decodes them with the same fp32 math, so the weights are bit-identical.
    using RQT = std::conditional_t<(BN < 64), mmw_rq_narrow<WTYPE>, mmw_rq<WTYPE>>;
    constexpr bool RQ = RQT::ok;
    typename RQT::regs rqr[2][WR];

    // A weight fetch covers NS slices. With NS = 1 slice ks is in set ks & 1, else every fetch goes to set 0. The next fetch (two slices ahead) overwrites set 0 early, so the last slice of a group is decoded from a copy in set 1.
    constexpr int NS = LB ? 4 : Q20 ? 1 : RQT::ns;
    constexpr int U = NS == 1 ? 2 : NS;   // the loop unroll: set and slot indices must be constants, or they go to scratch memory

    auto load_b = [&](const int k, v4u (&b)[B_ITEMS]) {
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMW_NT;
            b[i] = *(const v4u *)(Xh + (size_t)brow[i] * K + k * MMW_BK + (c & 7) * 8);
        }
    };
    auto store_b = [&](const v4u (&b)[B_ITEMS]) {
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMW_NT; *(v4u *)(Bs + (c >> 3) * MMW_LDS_STRIDE + (c & 7) * 8) = b[i]; }
    };
    auto wrow = [&](const int r) { return Wbase + (size_t)min((tid >> 1) + r * (MMW_NT / 2), a_rows - 1) * wrow_bytes; };
    // the weights of the NS slices from k into set s
    auto load_w = [&](const int k, auto set) {
        constexpr int s = decltype(set)::value;
#pragma unroll
        for (int r = 0; r < WR; ++r) {
            if constexpr (LB) LBT::fetch(wrow(r), k, tid & 1, lbr[s][r]);
            if constexpr (Q20) mmw_q20_fetch((const block_q2_0 *) wrow(r) + k, tid & 1, q20[s][r]);
            if constexpr (RQ) RQT::fetch(wrow(r), k, nks, tid & 1, rqr[s][r]);
        }
    };
    // decode the weights of the slice at loop position p into shared memory
    auto store_w = [&](auto pos) {
        constexpr int p = decltype(pos)::value, S = NS == 1 ? 0 : p, s = NS == 1 ? p : p == NS - 1;
#pragma unroll
        for (int r = 0; r < WR; ++r) {
            uint16_t * row = As + ((tid >> 1) + r * (MMW_NT / 2)) * MMW_LDS_STRIDE;
            if constexpr (LB) LBT::template decode<S>(lbgrid, lbsign, lbr[s][r], tid & 1, row + 32 * (tid & 1));
            if constexpr (Q20) mmw_dq_q20_half(q20[s][r], tid & 1, row + 32 * (tid & 1));
            if constexpr (RQ) RQT::template decode<S>(rqr[s][r], tid & 1, row);
            if constexpr (NS > 1 && p == NS - 2) {
                if constexpr (LB) lbr[1][r] = lbr[0][r];
                if constexpr (RQ) rqr[1][r] = rqr[0][r];
            }
        }
    };

    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = wact && (!TAIL || (wn * WTN + j * 16) < n_cols);
    auto mma = [&]() {
#pragma unroll
        for (int kk = 0; kk < MMW_BK; kk += 16) {
            v16s a[TM], b[TN]; const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) { const uint16_t * p = As + (wm * WTM + i * 16 + r) * MMW_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); a[i] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMW_LDS_STRIDE + kk;
                const uint4 v0 = *(const uint4 *)p, v1 = *(const uint4 *)(p + 8); b[j] = __builtin_bit_cast(v16s, (uint4[2]){v0, v1}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], a[i], acc[i][j]); }
        }
    };
    // slice ks at loop position p: load slice ks + 2 (past the end, the last slice again), multiply slice ks, store slice ks + 1. The last slice exits before its store: a skipped store would make the next iteration wait for its loads.
    auto slice = [&](const int ks, auto pos) {
        constexpr int p = decltype(pos)::value;
        const int kl = min(ks + 2, nks - 1);
        load_b(kl, bst[p & 1]);
        if constexpr (NS == 1) load_w(kl, mmw_pos<p>{});
        else if constexpr (p == U - 2) load_w(kl, mmw_pos<0>{});
        if constexpr (!TAIL) __builtin_amdgcn_sched_barrier(0);
        if (wact) mma();
        mmw_sync_lds();
        if (ks + 1 == nks) return false;
        store_w(mmw_pos<(p + 1) % U>{});
        store_b(bst[(p + 1) & 1]);
        mmw_sync_lds();
        return true;
    };

    load_b(0, bst[0]);
    load_w(0, mmw_pos<0>{});
    load_b(min(1, nks - 1), bst[1]);
    if constexpr (NS == 1) load_w(min(1, nks - 1), mmw_pos<1>{});
    if constexpr (LB) mmw_sync_lds();
    store_w(mmw_pos<0>{});
    store_b(bst[0]);
    mmw_sync_lds();
    for (int ks = 0;; ks += U) {
        if (!slice(ks, mmw_pos<0>{}) || !slice(ks + 1, mmw_pos<1>{})) break;
        if constexpr (U == 4) {
            if (!slice(ks + 2, mmw_pos<2>{}) || !slice(ks + 3, mmw_pos<3>{})) break;
        }
    }
    float * stg = (float *)((BN * MMW_LDS_STRIDE * 2 >= MMW_NT / 32 * 1024) ? Bs : As) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) mmw_store_tile(acc[i][j], stg, D, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            else { __syncthreads(); __syncthreads(); }
        }
    }
}

} // namespace
#endif // AMD_WMMA_AVAILABLE

__global__ void mmw_cvt_f32_bf16(const float * __restrict__ x, uint16_t * __restrict__ y, const size_t n) {
#if defined(AMD_WMMA_AVAILABLE)
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= n) {
        const float4 a = *(const float4 *)(x + i), b = *(const float4 *)(x + i + 4);
        uint4 o; o.x = mmw_pack2(a.x, a.y); o.y = mmw_pack2(a.z, a.w); o.z = mmw_pack2(b.x, b.y); o.w = mmw_pack2(b.z, b.w);
        *(uint4 *)(y + i) = o;
    } else {
        for (; i < n; ++i) y[i] = mmw_f2bf(x[i]);
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(n);
    NO_DEVICE_CODE;
#endif
}

template <int BM, int BN, int WTM, int WTN, int WTYPE>
__global__ void __launch_bounds__(MMW_NT, 2)
mmw_dense_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Xh, float * __restrict__ D, const int M, const int K, const int T) {
#if defined(AMD_WMMA_AVAILABLE)
    __shared__ __align__(16) uint16_t As[BM * MMW_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMW_LDS_STRIDE];
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const size_t wrow_bytes = mmw_row_bytes<WTYPE>(K);
    mmw_tile_gemm<BM, BN, WTM, WTN, WTYPE, false>(W + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, D, M, [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, m0, T - t0, As, Bs);
#else
    GGML_UNUSED(W); GGML_UNUSED(Xh); GGML_UNUSED(D); GGML_UNUSED(M); GGML_UNUSED(K); GGML_UNUSED(T);
    NO_DEVICE_CODE;
#endif
}

template <int BM, int BN, int WTM, int WTN, int WTYPE>
__global__ void __launch_bounds__(MMW_NT, 2)
mmw_routed_kernel(const uint8_t * __restrict__ W, const size_t expert_bytes, const uint16_t * __restrict__ Xh, float * __restrict__ D,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
#if defined(AMD_WMMA_AVAILABLE)
    __shared__ __align__(16) uint16_t As[BM * MMW_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMW_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = mmw_row_bytes<WTYPE>(K);
    mmw_tile_gemm<BM, BN, WTM, WTN, WTYPE, true>(W + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, As, Bs);
#else
    GGML_UNUSED(W); GGML_UNUSED(expert_bytes); GGML_UNUSED(Xh); GGML_UNUSED(D);
    GGML_UNUSED(ids_src); GGML_UNUSED(ids_dst); GGML_UNUSED(bounds); GGML_UNUSED(desc); GGML_UNUSED(M); GGML_UNUSED(K);
    NO_DEVICE_CODE;
#endif
}

// two tile classes: experts with at least thresh rows get BN-row tiles, the rest BN_SMALL-row tiles (fewer wasted rows). One thread per expert, so E must not exceed the block size.
__global__ void mmw_build_desc2(const int32_t * __restrict__ bounds, uint32_t * __restrict__ desc_big, uint32_t * __restrict__ desc_small,
        const int E, const int nbig_max, const int nsmall_max, const int BN_BIG, const int BN_SMALL, const int thresh) {
#if defined(AMD_WMMA_AVAILABLE)
    __shared__ int sb[1024], ss[1024];
    const int e = threadIdx.x;
    for (int i = e; i < nbig_max;   i += blockDim.x) desc_big[i]   = UINT32_MAX;
    for (int i = e; i < nsmall_max; i += blockDim.x) desc_small[i] = UINT32_MAX;
    int cnt = (e < E) ? bounds[e + 1] - bounds[e] : 0;
    const bool big = cnt >= thresh;
    const int tb = big ? (cnt + BN_BIG - 1) / BN_BIG : 0;
    const int ts = big ? 0 : (cnt + BN_SMALL - 1) / BN_SMALL;
    sb[e] = tb; ss[e] = ts;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int vb = (e >= off) ? sb[e - off] : 0, vs = (e >= off) ? ss[e - off] : 0;
        __syncthreads();
        sb[e] += vb; ss[e] += vs;
        __syncthreads();
    }
    const int bb = sb[e] - tb, bs = ss[e] - ts;
    for (int jt = 0; jt < tb; ++jt) { const int idx = bb + jt; if (idx < nbig_max)   desc_big[idx]   = (uint32_t)e | ((uint32_t)jt << 16); }
    for (int jt = 0; jt < ts; ++jt) { const int idx = bs + jt; if (idx < nsmall_max) desc_small[idx] = (uint32_t)e | ((uint32_t)jt << 16); }
#else
    GGML_UNUSED(bounds); GGML_UNUSED(desc_big); GGML_UNUSED(desc_small); GGML_UNUSED(E);
    GGML_UNUSED(nbig_max); GGML_UNUSED(nsmall_max); GGML_UNUSED(BN_BIG); GGML_UNUSED(BN_SMALL); GGML_UNUSED(thresh);
    NO_DEVICE_CODE;
#endif
}

// dense GEMMs: the two types measured faster than MMQ on RDNA3.5
static bool mmw_quant_type_dense(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL;
}

// routed experts: the types measured faster than MMQ at prefill token counts on RDNA3.5
static bool mmw_quant_type_routed(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
            return true;
        default:
            return false;
    }
}

// decoder id per weight type: 0 = IQ4_NL, 1 = Q8_0, 32 + ggml_type for the others
template <typename Fn>
static void mmw_dispatch_quant(ggml_type type, Fn fn) {
    switch (type) {
        case GGML_TYPE_Q8_0:    fn(std::integral_constant<int, 1>{}); break;
        case GGML_TYPE_IQ4_NL:  fn(std::integral_constant<int, 0>{}); break;
        case GGML_TYPE_Q2_0:    fn(std::integral_constant<int, 32 + GGML_TYPE_Q2_0>{}); break;
        case GGML_TYPE_Q5_1:    fn(std::integral_constant<int, 32 + GGML_TYPE_Q5_1>{}); break;
        case GGML_TYPE_Q4_K:    fn(std::integral_constant<int, 32 + GGML_TYPE_Q4_K>{}); break;
        case GGML_TYPE_Q5_K:    fn(std::integral_constant<int, 32 + GGML_TYPE_Q5_K>{}); break;
        case GGML_TYPE_Q6_K:    fn(std::integral_constant<int, 32 + GGML_TYPE_Q6_K>{}); break;
        case GGML_TYPE_IQ2_XXS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_XXS>{}); break;
        case GGML_TYPE_IQ2_XS:  fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_XS>{}); break;
        case GGML_TYPE_IQ2_S:   fn(std::integral_constant<int, 32 + GGML_TYPE_IQ2_S>{}); break;
        case GGML_TYPE_IQ3_XXS: fn(std::integral_constant<int, 32 + GGML_TYPE_IQ3_XXS>{}); break;
        case GGML_TYPE_IQ3_S:   fn(std::integral_constant<int, 32 + GGML_TYPE_IQ3_S>{}); break;
        case GGML_TYPE_MXFP4:   fn(std::integral_constant<int, 32 + GGML_TYPE_MXFP4>{}); break;
        default:
            GGML_ABORT("unsupported mmw quant type");
    }
}

// the kernels are wave32 WMMA with a 32-lane epilogue, and only RDNA3.5 has a tuned configuration
static bool mmw_enabled(const int cc) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_MMW") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_MMW"));
    return !disabled && GGML_CUDA_CC_IS_RDNA3_5(cc) &&
        ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size == 32;
}

bool ggml_cuda_mmw_available(const int cc) {
    return mmw_enabled(cc);
}

bool ggml_cuda_mmw_supported_mm(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    if (!mmw_enabled(cc)) return false;
    if (!mmw_quant_type_dense(src0->type)) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    const int64_t K = src0->ne[0], M = src0->ne[1];
    if (K % 64 != 0 || src1->ne[0] != K || dst->ne[0] != M) return false;
    const int64_t T = src1->ne[1] * src1->ne[2] * src1->ne[3];
    if (T < mmw_cfg_rdna3_5::min_t || T > INT32_MAX / 4) return false;
    return ggml_nrows(dst) == T;
}

bool ggml_cuda_mmw_supported_mmid(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, const int cc) {
    if (!mmw_enabled(cc)) return false;
    if (!mmw_quant_type_routed(src0->type)) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    const int64_t K = src0->ne[0], M = src0->ne[1], E = src0->ne[2];
    if (src0->ne[3] != 1 || K % 64 != 0 || E < 1 || E > 1024) return false;
    if (K % ggml_blck_size(src0->type) != 0) return false;
    const int64_t n_used = ids->ne[0], T = ids->ne[1];
    if (src1->ne[0] != K || src1->ne[3] != 1 || src1->ne[2] != T) return false;
    if (src1->ne[1] != 1 && src1->ne[1] != n_used) return false;
    if (dst->ne[0] != M || dst->ne[1] != n_used || dst->ne[2] != T || dst->ne[3] != 1) return false;
    if (ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[3] != 1) return false;
    if (T < mmw_cfg_rdna3_5::min_t || n_used > 64 || (T * n_used) >> 16 >= 1024) return false;
    if (T * n_used < E * (mmw_cfg_rdna3_5::routed_bn / 2)) return false;
    return true;
}

void ggml_cuda_mul_mat_mmw(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], M = (int) src0->ne[1];
    const int64_t T = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const int64_t n = T * K;

    ggml_cuda_pool_alloc<uint16_t> xh(ctx.pool(), n);
    mmw_cvt_f32_bf16<<<(unsigned) ((n / 8 + 255) / 256), 256, 0, stream>>>((const float *) src1->data, xh.get(), n);

    const uint8_t * W = (const uint8_t *) src0->data;
    float * D = (float *) dst->data;
    constexpr int BM = mmw_cfg_rdna3_5::dense_bm, BN = mmw_cfg_rdna3_5::dense_bn;
    const dim3 grid((M + BM - 1) / BM, (unsigned) ((T + BN - 1) / BN));
    mmw_dispatch_quant(src0->type, [&](auto tag) {
        constexpr int WT = decltype(tag)::value;
        mmw_dense_kernel<BM, BN, mmw_cfg_rdna3_5::dense_wtm, mmw_cfg_rdna3_5::dense_wtn, WT><<<grid, MMW_NT, 0, stream>>>(W, xh.get(), D, M, K, (int) T);
    });
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_id_mmw(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], E = (int) src0->ne[2];
    const int ne11 = (int) src1->ne[1], T = (int) src1->ne[2], n_used = (int) ids->ne[0];
    const int n_rows_x = ne11 * T, n_rows = n_used * T;

    ggml_cuda_pool_alloc<uint16_t> xh(ctx.pool(), (size_t) n_rows_x * K);
    mmw_cvt_f32_bf16<<<(unsigned) (((int64_t) n_rows_x * K / 8 + 255) / 256), 256, 0, stream>>>((const float *) src1->data, xh.get(), (size_t) n_rows_x * K);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), E + 1);
    const int si1  = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
        E, T, n_used, ne11, si1, sis1, /*write_inverse =*/ false, stream);

    constexpr int BN = mmw_cfg_rdna3_5::routed_bn, BN_SMALL = mmw_cfg_rdna3_5::routed_small_bn;
    constexpr int THRESH = mmw_cfg_rdna3_5::routed_thresh;
    const int nbig_max   = n_rows / BN + E + 1;
    const int nsmall_max = E * ((THRESH + BN_SMALL - 1) / BN_SMALL) + 1;
    ggml_cuda_pool_alloc<uint32_t> desc_big(ctx.pool(), nbig_max);
    ggml_cuda_pool_alloc<uint32_t> desc_small(ctx.pool(), nsmall_max);
    mmw_build_desc2<<<1, 1024, 0, stream>>>(bounds.get(), desc_big.get(), desc_small.get(), E, nbig_max, nsmall_max, BN, BN_SMALL, THRESH);

    const uint8_t * W = (const uint8_t *) src0->data;
    float * D = (float *) dst->data;
    const size_t eb = (size_t) src0->nb[2];
    constexpr int BM = mmw_cfg_rdna3_5::routed_bm, BM_SMALL = mmw_cfg_rdna3_5::routed_small_bm;
    const dim3 gbig((M + BM - 1) / BM, nbig_max), gsmall((M + BM_SMALL - 1) / BM_SMALL, nsmall_max);
    mmw_dispatch_quant(src0->type, [&](auto tag) {
        constexpr int WT = decltype(tag)::value;
        mmw_routed_kernel<BM, BN, mmw_cfg_rdna3_5::routed_wtm, mmw_cfg_rdna3_5::routed_wtn, WT>
            <<<gbig, MMW_NT, 0, stream>>>(W, eb, xh.get(), D, ids_src1.get(), ids_dst.get(), bounds.get(), desc_big.get(), M, K);
        mmw_routed_kernel<BM_SMALL, BN_SMALL, mmw_cfg_rdna3_5::routed_small_wtm, mmw_cfg_rdna3_5::routed_small_wtn, WT>
            <<<gsmall, MMW_NT, 0, stream>>>(W, eb, xh.get(), D, ids_src1.get(), ids_dst.get(), bounds.get(), desc_small.get(), M, K);
    });
    CUDA_CHECK(cudaGetLastError());
}
