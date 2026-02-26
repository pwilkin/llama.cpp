// test-quant-iq3kl.cpp
// Quantization accuracy test for IQ3_KL (H3-style per-tensor Lloyd-Max).

#include "ggml-backend.h"
#include "ggml.h"
#include <immintrin.h>

extern "C" {
void         iq3kl_train_levels(const float * data,
                                int64_t       nrow,
                                int64_t       n_per_row,
                                const float * imatrix,
                                float         levels_out[8]);
void         iq3kl_set_levels(const float * levels);
const float * iq3kl_get_tensor_levels(const void * data_ptr);
size_t       quantize_iq3_kl(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void         quantize_row_q8_K_ref(const float * x, void * y, int64_t k);
}

#define IQ3KL_N_LEVELS 8

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#define __AVX__
#define __AVX2__

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float rmse(const float * a, const float * b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double) a[i] - (double) b[i];
        s += d * d;
    }
    return (float) std::sqrt(s / (double) n);
}

static float std_quant_rmse(ggml_type type, const float * data, size_t nrow, size_t n_per_row) {
    const size_t         rs = ggml_row_size(type, n_per_row);
    std::vector<uint8_t> qb(nrow * rs);
    std::vector<float>   dq(nrow * n_per_row);
    ggml_quantize_chunk(type, data, qb.data(), 0, nrow, n_per_row, nullptr);
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    for (size_t r = 0; r < nrow; ++r) {
        tr->to_float(qb.data() + r * rs, dq.data() + r * n_per_row, (int64_t) n_per_row);
    }
    return rmse(data, dq.data(), nrow * n_per_row);
}

// Run actual IQ3_KL quantization: train levels, set, quantize, dequantize, return RMSE
static float iq3kl_rmse_actual(const float * data, size_t nrow, size_t n_per_row) {
    float levels[8];
    iq3kl_train_levels(data, (int64_t) nrow, (int64_t) n_per_row, nullptr, levels);
    iq3kl_set_levels(levels);
    const size_t         rs = ggml_row_size(GGML_TYPE_IQ3_KL, n_per_row);
    std::vector<uint8_t> qb(nrow * rs);
    std::vector<float>   dq(nrow * n_per_row);
    quantize_iq3_kl(data, qb.data(), (int64_t) nrow, (int64_t) n_per_row, nullptr);
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_IQ3_KL);
    for (size_t r = 0; r < nrow; ++r) {
        tr->to_float(qb.data() + r * rs, dq.data() + r * n_per_row, (int64_t) n_per_row);
    }
    return rmse(data, dq.data(), nrow * n_per_row);
}

// Forward declaration for optimized training function
static void iq3kl_train_levels_opt(const float * data,
                                   int64_t       nrow,
                                   int64_t       n_per_row,
                                   const float * imatrix,
                                   float         levels_out[8],
                                   int           max_iters,
                                   float         conv_epsilon,
                                   bool          use_binning,
                                   int           n_bins);

// Run IQ3_KL with optimized levels computation: train levels (optimized), set, quantize, dequantize, return RMSE
static float iq3kl_rmse_actual_opt(const float * data, size_t nrow, size_t n_per_row) {
    float levels[8];
    iq3kl_train_levels_opt(data, (int64_t) nrow, (int64_t) n_per_row, nullptr, levels, 1000, 1e-12f, true, 32768);
    iq3kl_set_levels(levels);
    const size_t         rs = ggml_row_size(GGML_TYPE_IQ3_KL, n_per_row);
    std::vector<uint8_t> qb(nrow * rs);
    std::vector<float>   dq(nrow * n_per_row);
    quantize_iq3_kl(data, qb.data(), (int64_t) nrow, (int64_t) n_per_row, nullptr);
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_IQ3_KL);
    for (size_t r = 0; r < nrow; ++r) {
        tr->to_float(qb.data() + r * rs, dq.data() + r * n_per_row, (int64_t) n_per_row);
    }
    return rmse(data, dq.data(), nrow * n_per_row);
}

// ---------------------------------------------------------------------------
// Optimized scale levels computation with early convergence + binning
// ---------------------------------------------------------------------------
static void iq3kl_train_levels_opt(const float * data,
                                   int64_t       nrow,
                                   int64_t       n_per_row,
                                   const float * imatrix,
                                   float         levels_out[8],
                                   int           max_iters,
                                   float         conv_epsilon,
                                   bool          use_binning,
                                   int           n_bins) {
    const int64_t n_sub   = n_per_row / 16;
    const int64_t n_total = nrow * n_sub * 16;

    // Allocate temporary arrays
    float * vals = (float *) malloc(n_total * sizeof(float));
    float * wts  = (float *) malloc(n_total * sizeof(float));
    GGML_ASSERT(vals && wts);

    // Collect affine-normalized values: t = (x - sub_min) / sub_range in [0,1]
    int64_t idx = 0;
    for (int64_t row = 0; row < nrow; ++row) {
        const float * xrow = data + row * n_per_row;
        for (int64_t ib = 0; ib < n_sub; ++ib) {
            const float * xb       = xrow + ib * 16;
            const int     col_base = (int) (ib * 16);
            float         sb_min = xb[0], sb_max = xb[0];
            for (int j = 1; j < 16; ++j) {
                if (xb[j] < sb_min) {
                    sb_min = xb[j];
                }
                if (xb[j] > sb_max) {
                    sb_max = xb[j];
                }
            }
            const float sb_range = sb_max - sb_min;
            for (int j = 0; j < 16; ++j) {
                float w = 1.0f;
                if (imatrix) {
                    w = imatrix[col_base + j];
                    if (w < 1e-10f) {
                        w = 1e-10f;
                    }
                }
                if (sb_range > 1e-6f) {
                    w *= sb_range;
                    vals[idx] = (xb[j] - sb_min) / sb_range;
                } else {
                    w         = 0.0f;
                    vals[idx] = 0.5f;
                }
                wts[idx] = w;
                idx++;
            }
        }
    }
    GGML_ASSERT(idx == n_total);

    // Initialize 8 levels uniformly in [0, 1]
    float levels[IQ3KL_N_LEVELS];
    for (int k = 0; k < IQ3KL_N_LEVELS; ++k) {
        levels[k] = (float) k / (IQ3KL_N_LEVELS - 1);
    }

    // Early convergence: keep previous levels
    float prev_levels[IQ3KL_N_LEVELS];

    if (use_binning && n_bins > IQ3KL_N_LEVELS) {
        // === BINNING ACCELERATION ===
        // Precompute bin centers
        std::vector<float> bin_center(n_bins);
        for (int b = 0; b < n_bins; ++b) {
            bin_center[b] = (b + 0.5f) / n_bins;
        }

        // First pass: accumulate weighted sums per bin
        float bin_w[32768]  = { 0 };  // Use fixed size for now, max 8192 bins
        float bin_wt[32768] = { 0 };
        GGML_ASSERT(n_bins <= 32768 && "Maximum 8192 bins supported");

        for (int64_t i = 0; i < n_total; ++i) {
            if (wts[i] < 1e-12f) {
                continue;
            }
            int bin_idx = (int) (vals[i] * n_bins);
            if (bin_idx >= n_bins) {
                bin_idx = n_bins - 1;
            }
            bin_w[bin_idx] += wts[i];
            bin_wt[bin_idx] += wts[i] * vals[i];
        }

        // Lloyd-Max iterations using bins instead of individual values
        for (int iter = 0; iter < max_iters; ++iter) {
            memcpy(prev_levels, levels, sizeof(levels));

            float sum_w[IQ3KL_N_LEVELS]  = { 0 };
            float sum_wt[IQ3KL_N_LEVELS] = { 0 };

            // Assign each bin to nearest cluster
            for (int b = 0; b < n_bins; ++b) {
                if (bin_w[b] < 1e-12f) {
                    continue;
                }
                float t       = bin_center[b];
                int   best    = 0;
                float best_d2 = (t - levels[0]) * (t - levels[0]);
                for (int k = 1; k < IQ3KL_N_LEVELS; ++k) {
                    float d2 = (t - levels[k]) * (t - levels[k]);
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best    = k;
                    }
                }
                sum_w[best] += bin_w[b];
                sum_wt[best] += bin_w[b] * t;
            }

            // Update cluster centers
            for (int k = 0; k < IQ3KL_N_LEVELS; ++k) {
                if (sum_w[k] > 1e-12f) {
                    levels[k] = sum_wt[k] / sum_w[k];
                }
            }

            // Keep levels sorted (insertion sort — 8 elements)
            for (int k = 1; k < IQ3KL_N_LEVELS; ++k) {
                float v = levels[k];
                int   m = k - 1;
                while (m >= 0 && levels[m] > v) {
                    levels[m + 1] = levels[m];
                    m--;
                }
                levels[m + 1] = v;
            }

            // Check convergence
            float max_delta = 0.0f;
            for (int k = 0; k < IQ3KL_N_LEVELS; ++k) {
                float delta = fabsf(levels[k] - prev_levels[k]);
                if (delta > max_delta) {
                    max_delta = delta;
                }
            }
            if (max_delta < conv_epsilon) {
                break;
            }
        }
    } else {
        // === ORIGINAL ALGORITHM (with early convergence) ===
        for (int iter = 0; iter < max_iters; ++iter) {
            memcpy(prev_levels, levels, sizeof(levels));
            float sum_w[IQ3KL_N_LEVELS]  = { 0 };
            float sum_wt[IQ3KL_N_LEVELS] = { 0 };

            for (int64_t i = 0; i < n_total; ++i) {
                if (wts[i] < 1e-12f) {
                    continue;
                }
                const float t       = vals[i];
                int         best    = 0;
                float       best_d2 = (t - levels[0]) * (t - levels[0]);
                for (int k = 1; k < IQ3KL_N_LEVELS; ++k) {
                    float d2 = (t - levels[k]) * (t - levels[k]);
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best    = k;
                    }
                }
                sum_w[best] += wts[i];
                sum_wt[best] += wts[i] * t;
            }

            for (int k = 0; k < IQ3KL_N_LEVELS; ++k) {
                if (sum_w[k] > 1e-12f) {
                    levels[k] = sum_wt[k] / sum_w[k];
                }
            }

            // Keep levels sorted (insertion sort — 8 elements)
            for (int k = 1; k < IQ3KL_N_LEVELS; ++k) {
                float v = levels[k];
                int   m = k - 1;
                while (m >= 0 && levels[m] > v) {
                    levels[m + 1] = levels[m];
                    m--;
                }
                levels[m + 1] = v;
            }

            // Check convergence
            float max_delta = 0.0f;
            for (int k = 0; k < IQ3KL_N_LEVELS; ++k) {
                float delta = fabsf(levels[k] - prev_levels[k]);
                if (delta > max_delta) {
                    max_delta = delta;
                }
            }
            if (max_delta < conv_epsilon) {
                break;
            }
        }
    }

    memcpy(levels_out, levels, IQ3KL_N_LEVELS * sizeof(float));
    free(vals);
    free(wts);
}

// ---------------------------------------------------------------------------
// Validation: compare original vs optimized
// ---------------------------------------------------------------------------
struct TestCase {
    std::string        name;
    std::vector<float> data;
    size_t             nrow, n_per_row;
};

static void validate_optimization(const float * data, size_t nrow, size_t n_per_row) {
    float levels_orig[8];
    float levels_opt[8];

    // Time original
    clock_t t0 = clock();
    iq3kl_train_levels(data, (int64_t) nrow, (int64_t) n_per_row, nullptr, levels_orig);
    clock_t t1 = clock();

    // Time optimized with binning (256 bins)
    iq3kl_train_levels_opt(data, (int64_t) nrow, (int64_t) n_per_row, nullptr, levels_opt, 100, 1e-6f, true, 256);
    clock_t t2 = clock();

    double orig_time = (double) (t1 - t0) / CLOCKS_PER_SEC;
    double opt_time  = (double) (t2 - t1) / CLOCKS_PER_SEC;
    double speedup   = orig_time / opt_time;

    // Compare levels (should be very close)
    float max_diff = 0.0f;
    for (int k = 0; k < 8; ++k) {
        float diff = fabsf(levels_orig[k] - levels_opt[k]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    printf("  Original: %.3f s, Optimized: %.3f s, Speedup: %.2fx, Max level diff: %.6e\n", orig_time, opt_time,
           speedup, max_diff);
}

// ---------------------------------------------------------------------------
// Test runner for optimization validation
// ---------------------------------------------------------------------------
static void run_optimization_tests() {
    std::mt19937 rng(0xdeadbeef);

    std::vector<TestCase> cases;
    {
        TestCase tc;
        tc.name      = "Gaussian(0,0.02) 64x4096";
        tc.nrow      = 64;
        tc.n_per_row = 4096;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0, 0.02f);
        for (auto & v : tc.data) {
            v = nd(rng);
        }
        cases.push_back(std::move(tc));
    }
    {
        TestCase tc;
        tc.name      = "Gaussian(0,0.05) 32x8192";
        tc.nrow      = 32;
        tc.n_per_row = 8192;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0, 0.05f);
        for (auto & v : tc.data) {
            v = nd(rng);
        }
        cases.push_back(std::move(tc));
    }
    {
        TestCase tc;
        tc.name      = "Gaussian(0,0.01) 128x2048";
        tc.nrow      = 128;
        tc.n_per_row = 2048;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0, 0.01f);
        for (auto & v : tc.data) {
            v = nd(rng);
        }
        cases.push_back(std::move(tc));
    }
    {
        TestCase tc;
        tc.name      = "Laplace(0,0.01) 64x4096";
        tc.nrow      = 64;
        tc.n_per_row = 4096;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::exponential_distribution<float> ed(100.f);
        std::bernoulli_distribution          sgnd(0.5f);
        for (auto & v : tc.data) {
            v = ed(rng);
            if (sgnd(rng)) {
                v = -v;
            }
        }
        cases.push_back(std::move(tc));
    }

    printf("Optimization validation (comparing original vs optimized levels computation)\n\n");
    for (auto & tc : cases) {
        printf("%s:\n", tc.name.c_str());
        validate_optimization(tc.data.data(), tc.nrow, tc.n_per_row);
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// vec_dot correctness: verbatim copies of the real functions with explicit
// target attributes so they can be called directly without GGML dispatch.
// ---------------------------------------------------------------------------

// Definitions needed to compile the copied functions
#ifndef QK_K
#  define QK_K 256
#endif
#define GGML_RESTRICT       __restrict__
#define GGML_CPU_FP16_TO_FP32(x) ggml_fp16_to_fp32(x)
#define GGML_ASSERT_VD(x)   assert(x)

// Block struct definitions matching the internal layout (sizes verified below)
typedef struct { uint16_t d, dmin; uint8_t scales[3*QK_K/32]; uint8_t qs[3*QK_K/8]; } vd_block_iq3_kl;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } vd_block_q8_K;
static_assert(sizeof(vd_block_iq3_kl) == 124, "vd_block_iq3_kl size mismatch");
static_assert(sizeof(vd_block_q8_K)   == 292, "vd_block_q8_K size mismatch");

// hsum_float_8 — verbatim copy from arch/x86/quants.c
#if defined(__AVX__) || defined(__AVX2__)
static inline float vd_hsum_float_8(const __m256 x) __attribute__((target("avx")));
static inline float vd_hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
#endif

// ── GENERIC (verbatim copy of ggml_vec_dot_iq3_kl_q8_K_generic) ──────────
static void vd_generic(int n, float * GGML_RESTRICT s, size_t bs,
                        const void * GGML_RESTRICT vx, size_t bx,
                        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;
    const vd_block_iq3_kl * GGML_RESTRICT x = (const vd_block_iq3_kl *)vx;
    const vd_block_q8_K   * GGML_RESTRICT y = (const vd_block_q8_K   *)vy;
    const int nb = n / QK_K;
    const float * levels = iq3kl_get_tensor_levels(vx);
    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float xd    = GGML_CPU_FP16_TO_FP32(x[i].d);
        const float xdmin = GGML_CPU_FP16_TO_FP32(x[i].dmin);
        const float yd    = y[i].d;
        const uint8_t * sc = x[i].scales;
        const uint8_t * qs = x[i].qs;
        const int8_t  * q8 = y[i].qs;
        float block_sum = 0.f;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const int sbit0  = ib * 6,              sbyte0 = sbit0 / 8,  soff0 = sbit0 % 8;
            const int sbit1  = (ib + QK_K/16) * 6,  sbyte1 = sbit1 / 8,  soff1 = sbit1 % 8;
            uint8_t qrange = (sc[sbyte0] >> soff0) & 0x3F;
            if (soff0 > 2) { qrange |= (uint8_t)((sc[sbyte0+1] << (8 - soff0)) & 0x3F); }
            uint8_t qnmin  = (sc[sbyte1] >> soff1) & 0x3F;
            if (soff1 > 2) { qnmin  |= (uint8_t)((sc[sbyte1+1] << (8 - soff1)) & 0x3F); }
            const float range   = xd    * (float)qrange;
            const float sub_min = -xdmin * (float)qnmin;
            float sum_lq = 0.f;
            for (int j = 0; j < 16; ++j) {
                const int qk    = ib * 16 + j;
                const int qbit  = qk * 3;
                const int qbyte = qbit / 8;
                const int qoff  = qbit % 8;
                int q = (qs[qbyte] >> qoff) & 0x7;
                if (qoff > 5) { q |= (int)((qs[qbyte+1] << (8 - qoff)) & 0x7); }
                sum_lq += levels[q] * (float)q8[qk];
            }
            block_sum += sum_lq * range + sub_min * (float)y[i].bsums[ib];
        }
        sumf += block_sum * yd;
    }
    *s = sumf;
}

// ── AVX2 variant (verbatim copy of the #if defined(__AVX2__) branch) ──────
#if defined(__AVX2__) || defined(__AVX__)
static void vd_avx2(int n, float * GGML_RESTRICT s, size_t bs,
                    const void * GGML_RESTRICT vx, size_t bx,
                    const void * GGML_RESTRICT vy, size_t by, int nrc)
    __attribute__((target("avx2,fma")));
static void vd_avx2(int n, float * GGML_RESTRICT s, size_t bs,
                    const void * GGML_RESTRICT vx, size_t bx,
                    const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;
    const vd_block_iq3_kl * GGML_RESTRICT x = (const vd_block_iq3_kl *)vx;
    const vd_block_q8_K   * GGML_RESTRICT y = (const vd_block_q8_K   *)vy;
    const int nb = n / QK_K;
    const float * levels = iq3kl_get_tensor_levels(vx);

    // Load levels into a register once - used for permute-based selection
    const __m256 levels_vec = _mm256_loadu_ps(levels);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float yd    = y[i].d;
        const float xd    = GGML_CPU_FP16_TO_FP32(x[i].d);
        const float xdmin = GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT scales = x[i].scales;
        const uint8_t * GGML_RESTRICT qs     = x[i].qs;
        const int8_t  * GGML_RESTRICT q8     = y[i].qs;
        const int16_t * GGML_RESTRICT bsums  = y[i].bsums;

        __m256 block_sum = _mm256_setzero_ps();

        // Process 16 sub-blocks (each 16 elements)
        for (int ib = 0; ib < QK_K/16; ++ib) {
            // Unpack 6-bit range and neg_min for this sub-block
            const int sbit0 = ib * 6;
            const int sbit1 = (ib + QK_K/16) * 6;
            const int sbyte0 = sbit0 / 8, soff0 = sbit0 % 8;
            const int sbyte1 = sbit1 / 8, soff1 = sbit1 % 8;

            uint8_t qrange = (scales[sbyte0] >> soff0) & 0x3F;
            if (soff0 > 2) { qrange |= (uint8_t)((scales[sbyte0+1] << (8 - soff0)) & 0x3F); }
            uint8_t qnmin = (scales[sbyte1] >> soff1) & 0x3F;
            if (soff1 > 2) { qnmin |= (uint8_t)((scales[sbyte1+1] << (8 - soff1)) & 0x3F); }

            const float range   = xd    * (float)qrange;
            const float sub_min = -xdmin * (float)qnmin;

            // Load bytes covering the 16 3-bit indices (48 bits = 6 bytes, but may need 7 due to alignment)
            const int idx_bit_start  = ib * 16 * 3;
            const int idx_byte_start = idx_bit_start / 8;
            const int idx_bit_off    = idx_bit_start % 8;

            // Load 8 bytes to cover all 16 indices plus alignment
            uint64_t idx_bytes = 0;
            memcpy(&idx_bytes, qs + idx_byte_start, 8);

            // Extract 16 3-bit indices
            uint32_t indices[16];
            for (int j = 0; j < 16; ++j) {
                int bit_off = idx_bit_off + j * 3;
                indices[j] = (uint32_t)((idx_bytes >> bit_off) & 0x7ULL);
            }

            // Process first 8 elements: permute levels, convert q8 to float, multiply-add
            __m256i idx_vec0 = _mm256_set_epi32(
                indices[7], indices[6], indices[5], indices[4],
                indices[3], indices[2], indices[1], indices[0]
            );
            __m256 lvls0 = _mm256_permutevar8x32_ps(levels_vec, idx_vec0);
            __m256i q8_0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q8 + ib*16)));
            __m256 q8f_0 = _mm256_cvtepi32_ps(q8_0);
            __m256 sum0 = _mm256_mul_ps(lvls0, q8f_0);

            // Process second 8 elements
            __m256i idx_vec1 = _mm256_set_epi32(
                indices[15], indices[14], indices[13], indices[12],
                indices[11], indices[10], indices[9], indices[8]
            );
            __m256 lvls1 = _mm256_permutevar8x32_ps(levels_vec, idx_vec1);
            __m256i q8_1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q8 + ib*16 + 8)));
            __m256 q8f_1 = _mm256_cvtepi32_ps(q8_1);
            __m256 sum1 = _mm256_mul_ps(lvls1, q8f_1);

            // Horizontal sum of sum0 + sum1
            __m256 sum_lq = _mm256_add_ps(sum0, sum1);
            sum_lq = _mm256_hadd_ps(sum_lq, sum_lq);
            sum_lq = _mm256_hadd_ps(sum_lq, sum_lq);
            float sum_lq_scalar = _mm256_cvtss_f32(_mm256_add_ps(sum_lq, _mm256_permute_ps(sum_lq, 0x01)));

            // Add contribution: sum_lq * range + sub_min * bsums[ib]
            __m256 contrib = _mm256_set1_ps(sum_lq_scalar * range + sub_min * (float)bsums[ib]);
            block_sum = _mm256_add_ps(block_sum, contrib);
        }

        // Sum the 8 lanes of block_sum (they're all the same value, but we need scalar)
        float block_sum_scalar = vd_hsum_float_8(block_sum);

        acc = _mm256_fmadd_ps(_mm256_set1_ps(yd), _mm256_set1_ps(block_sum_scalar), acc);
    }

    *s = vd_hsum_float_8(acc);
}

// ── AVX variant (verbatim copy of the #elif defined(__AVX__) branch) ──────
static void vd_avx(int n, float * GGML_RESTRICT s, size_t bs,
                   const void * GGML_RESTRICT vx, size_t bx,
                   const void * GGML_RESTRICT vy, size_t by, int nrc)
    __attribute__((target("avx")));
static void vd_avx(int n, float * GGML_RESTRICT s, size_t bs,
                   const void * GGML_RESTRICT vx, size_t bx,
                   const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;
    const vd_block_iq3_kl * GGML_RESTRICT x = (const vd_block_iq3_kl *)vx;
    const vd_block_q8_K   * GGML_RESTRICT y = (const vd_block_q8_K   *)vy;
    const int nb = n / QK_K;
    const float * levels = iq3kl_get_tensor_levels(vx);

    // AVX fallback: no permutevar8x32, use scalar index extraction with SIMD multiply
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float yd    = y[i].d;
        const float xd    = GGML_CPU_FP16_TO_FP32(x[i].d);
        const float xdmin = GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT scales = x[i].scales;
        const uint8_t * GGML_RESTRICT qs     = x[i].qs;
        const int8_t  * GGML_RESTRICT q8     = y[i].qs;
        const int16_t * GGML_RESTRICT bsums  = y[i].bsums;

        float block_sum = 0.0f;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const int sbit0 = ib * 6;
            const int sbit1 = (ib + QK_K/16) * 6;
            const int sbyte0 = sbit0 / 8, soff0 = sbit0 % 8;
            const int sbyte1 = sbit1 / 8, soff1 = sbit1 % 8;

            uint8_t qrange = (scales[sbyte0] >> soff0) & 0x3F;
            if (soff0 > 2) { qrange |= (uint8_t)((scales[sbyte0+1] << (8 - soff0)) & 0x3F); }
            uint8_t qnmin = (scales[sbyte1] >> soff1) & 0x3F;
            if (soff1 > 2) { qnmin |= (uint8_t)((scales[sbyte1+1] << (8 - soff1)) & 0x3F); }

            const float range   = xd    * (float)qrange;
            const float sub_min = -xdmin * (float)qnmin;

            const int idx_bit_start  = ib * 16 * 3;
            const int idx_byte_start = idx_bit_start / 8;
            const int idx_bit_off    = idx_bit_start % 8;

            uint64_t idx_bytes = 0;
            memcpy(&idx_bytes, qs + idx_byte_start, 8);

            // Scalar fallback for AVX (no permutevar8x32)
            float sum_lq = 0.0f;
            for (int j = 0; j < 16; ++j) {
                int bit_off = idx_bit_off + j * 3;
                uint8_t idx = (uint8_t)((idx_bytes >> bit_off) & 0x7ULL);
                sum_lq += levels[idx] * (float)q8[ib*16 + j];
            }

            block_sum += sum_lq * range + sub_min * (float)bsums[ib];
        }

        sumf += yd * block_sum;
    }

    *s = sumf;
}
#endif // __AVX2__ || __AVX__

typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

static bool test_vec_dot_correctness(std::mt19937 & rng) {
    // Register which variants to test
    struct Variant { const char * name; vec_dot_fn fn; };
    Variant variants[] = {
        { "generic", vd_generic },
#if defined(__AVX2__) || defined(__AVX__)
        { "avx2",    vd_avx2 },
        { "avx",     vd_avx  },
#endif
    };
    const int nvar = (int)(sizeof(variants) / sizeof(variants[0]));

    printf("vec_dot correctness: generic | avx2 | avx vs generic reference\n\n");
    // Header
    printf("%-10s  %12s", "n", "generic");
    for (int v = 1; v < nvar; ++v) { printf("  %12s  %10s", variants[v].name, "err"); }
    printf("  %6s\n", "pass");

    const int    sizes[]  = {256, 512, 1024, 2048, 4096};
    const int    n_trials = 4;
    bool         all_pass = true;
    std::normal_distribution<float> nd_x(0.f, 0.05f);
    std::normal_distribution<float> nd_y(0.f, 1.f);

    for (int n : sizes) {
        for (int trial = 0; trial < n_trials; ++trial) {
            std::vector<float> x(n), y(n);
            for (auto & v : x) v = nd_x(rng);
            for (auto & v : y) v = nd_y(rng);

            // Train + set levels (global fallback used by iq3kl_get_tensor_levels)
            float levels[IQ3KL_N_LEVELS];
            iq3kl_train_levels(x.data(), 1, n, nullptr, levels);
            iq3kl_set_levels(levels);

            // Quantize
            const size_t iq3kl_rs = ggml_row_size(GGML_TYPE_IQ3_KL, n);
            const size_t q8k_rs   = ggml_row_size(GGML_TYPE_Q8_K,   n);
            std::vector<uint8_t> qx(iq3kl_rs), qy(q8k_rs);
            quantize_iq3_kl(x.data(), qx.data(), 1, n, nullptr);
            quantize_row_q8_K_ref(y.data(), qy.data(), n);

            // Call all variants
            std::vector<float> results(nvar);
            for (int v = 0; v < nvar; ++v) {
                results[v] = 0.f;
                variants[v].fn(n, &results[v], sizeof(float),
                               qx.data(), iq3kl_rs, qy.data(), q8k_rs, 1);
            }

            // generic is the reference; compare all others against it
            const float ref = results[0];
            const float tol = 1e-4f * (std::abs(ref) + 1e-4f);
            bool ok = true;
            for (int v = 1; v < nvar; ++v) {
                if (std::abs(results[v] - ref) > tol) { ok = false; all_pass = false; }
            }

            if (trial == 0 || !ok) {
                printf("n=%-8d  %12.4f", n, ref);
                for (int v = 1; v < nvar; ++v) {
                    printf("  %12.4f  %10.3e", results[v], std::abs(results[v] - ref));
                }
                printf("  %s\n", ok ? "PASS" : "FAIL");
                fflush(stdout);
            }
        }
    }
    printf("\n");
    return all_pass;
}

// ---------------------------------------------------------------------------
// Scalar simulation: H3 affine Lloyd-Max — analytical ideal for IQ3_KL
// (float sub-block scales, no quantization noise on scale/min)
// ---------------------------------------------------------------------------
static float algo_h3_affine(const float * data, size_t nrow, size_t n_per_row) {
    const size_t n = nrow * n_per_row;

    // Collect affine-normalized values in [0, 1] from 16-element sub-blocks
    std::vector<float> norm_vals;
    norm_vals.reserve(n);
    for (size_t row = 0; row < nrow; ++row) {
        const float * xrow = data + row * n_per_row;
        for (size_t sb = 0; sb < n_per_row / 16; ++sb) {
            const float * xsb    = xrow + sb * 16;
            float         sb_min = xsb[0], sb_max = xsb[0];
            for (int j = 1; j < 16; ++j) {
                sb_min = std::min(sb_min, xsb[j]);
                sb_max = std::max(sb_max, xsb[j]);
            }
            float sb_range = sb_max - sb_min;
            for (int j = 0; j < 16; ++j) {
                norm_vals.push_back(sb_range > 1e-9f ? (xsb[j] - sb_min) / sb_range : 0.5f);
            }
        }
    }

    // Lloyd-Max on [0, 1] values, k=8
    float levels[8];
    for (int k = 0; k < 8; ++k) {
        levels[k] = (k + 0.5f) / 8.0f;
    }
    const size_t nn = norm_vals.size();
    for (int iter = 0; iter < 300; ++iter) {
        float sum[8] = {};
        int   cnt[8] = {};
        for (size_t i = 0; i < nn; ++i) {
            float v    = norm_vals[i];
            int   best = 0;
            float bd   = FLT_MAX;
            for (int k = 0; k < 8; ++k) {
                float d = std::abs(v - levels[k]);
                if (d < bd) {
                    bd   = d;
                    best = k;
                }
            }
            sum[best] += v;
            cnt[best]++;
        }
        bool changed = false;
        for (int k = 0; k < 8; ++k) {
            if (cnt[k] > 0) {
                float lk = sum[k] / cnt[k];
                if (std::abs(lk - levels[k]) > 1e-8f) {
                    changed = true;
                }
                levels[k] = lk;
            }
        }
        if (!changed) {
            break;
        }
    }
    std::sort(levels, levels + 8);

    // Simulate: per-16-element affine sub-block with float scale/min (no noise)
    double total_se = 0;
    for (size_t row = 0; row < nrow; ++row) {
        const float * xrow = data + row * n_per_row;
        for (size_t sb = 0; sb < n_per_row / 16; ++sb) {
            const float * xsb    = xrow + sb * 16;
            float         sb_min = xsb[0], sb_max = xsb[0];
            for (int j = 1; j < 16; ++j) {
                sb_min = std::min(sb_min, xsb[j]);
                sb_max = std::max(sb_max, xsb[j]);
            }
            float sb_range = sb_max - sb_min;
            for (int j = 0; j < 16; ++j) {
                float v_norm = sb_range > 1e-9f ? (xsb[j] - sb_min) / sb_range : 0.5f;
                int   best   = 0;
                float bd     = FLT_MAX;
                for (int k = 0; k < 8; ++k) {
                    float d = std::abs(v_norm - levels[k]);
                    if (d < bd) {
                        bd   = d;
                        best = k;
                    }
                }
                float  recon = levels[best] * sb_range + sb_min;
                double diff  = xsb[j] - recon;
                total_se += diff * diff;
            }
        }
    }
    return (float) std::sqrt(total_se / (double) n);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    bool scalar_only = (argc > 1 && std::string(argv[1]) == "--scalar");
    bool opt_test    = (argc > 1 && std::string(argv[1]) == "--opt-test");

    ggml_backend_load_all();
    ggml_quantize_init(GGML_TYPE_IQ3_XXS);
    ggml_quantize_init(GGML_TYPE_Q3_K);

    if (opt_test) {
        run_optimization_tests();
        return 0;
    }

    std::mt19937          rng(0xdeadbeef);
    std::vector<TestCase> cases;

    {
        TestCase tc;
        tc.name      = "Gaussian(0,0.02) 64x4096";
        tc.nrow      = 64;
        tc.n_per_row = 4096;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0, 0.02f);
        for (auto & v : tc.data) {
            v = nd(rng);
        }
        cases.push_back(std::move(tc));
    }
    {
        TestCase tc;
        tc.name      = "Gaussian(0,0.05) 32x8192";
        tc.nrow      = 32;
        tc.n_per_row = 8192;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0, 0.05f);
        for (auto & v : tc.data) {
            v = nd(rng);
        }
        cases.push_back(std::move(tc));
    }
    {
        TestCase tc;
        tc.name      = "Gaussian(0,0.01) 128x2048";
        tc.nrow      = 128;
        tc.n_per_row = 2048;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0, 0.01f);
        for (auto & v : tc.data) {
            v = nd(rng);
        }
        cases.push_back(std::move(tc));
    }
    {
        TestCase tc;
        tc.name      = "Laplace(0,0.01) 64x4096";
        tc.nrow      = 64;
        tc.n_per_row = 4096;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::exponential_distribution<float> ed(100.f);
        std::bernoulli_distribution          sgnd(0.5f);
        for (auto & v : tc.data) {
            v = ed(rng);
            if (sgnd(rng)) {
                v = -v;
            }
        }
        cases.push_back(std::move(tc));
    }

    if (scalar_only) {
        // Scalar simulation: H3 ideal vs baselines (no scale quantization noise)
        printf("H3 scalar simulation vs baselines (lower=better; 1.00=IQ3_XXS)\n\n");
        printf("%-28s  %7s  %7s  %7s\n", "Test", "IQ3_XXS", "Q3_K", "H3:sim");
        printf("%-28s  %7s  %7s  %7s\n", "----------------------------", "-------", "-------", "-------");
        int tc_idx = 0;
        for (auto & tc : cases) {
            fprintf(stderr, "[%d/%zu] %s... ", ++tc_idx, cases.size(), tc.name.c_str());
            fflush(stderr);
            float xxs = std_quant_rmse(GGML_TYPE_IQ3_XXS, tc.data.data(), tc.nrow, tc.n_per_row);
            float q3k = std_quant_rmse(GGML_TYPE_Q3_K, tc.data.data(), tc.nrow, tc.n_per_row);
            float rh3 = algo_h3_affine(tc.data.data(), tc.nrow, tc.n_per_row);
            fprintf(stderr, "done\n");
            printf("%-28s  %7.4f  %7.4f  %7.4f\n", tc.name.c_str(), 1.f, q3k / xxs, rh3 / xxs);
            fflush(stdout);
        }
        return 0;
    }

    // Default: actual IQ3_KL implementation vs baselines
    printf("IQ3_KL actual quantization vs baselines (lower=better; 1.00=IQ3_XXS)\n\n");
    printf("%-28s  %7s  %7s  %7s  %7s  %7s\n", "Test", "IQ3_XXS", "Q3_K", "IQ3_KL", "IQ3_KL_opt", "H3:sim");
    printf("%-28s  %7s  %7s  %7s  %7s  %7s\n", "----------------------------", "-------", "-------", "-------",
           "--------", "-------");
    int  tc_idx   = 0;
    bool any_fail = false;
    for (auto & tc : cases) {
        fprintf(stderr, "[%d/%zu] %s... ", ++tc_idx, cases.size(), tc.name.c_str());
        fflush(stderr);
        float xxs     = std_quant_rmse(GGML_TYPE_IQ3_XXS, tc.data.data(), tc.nrow, tc.n_per_row);
        float q3k     = std_quant_rmse(GGML_TYPE_Q3_K, tc.data.data(), tc.nrow, tc.n_per_row);
        float rkl     = iq3kl_rmse_actual(tc.data.data(), tc.nrow, tc.n_per_row);
        float rkl_opt = iq3kl_rmse_actual_opt(tc.data.data(), tc.nrow, tc.n_per_row);
        float rh3     = algo_h3_affine(tc.data.data(), tc.nrow, tc.n_per_row);
        fprintf(stderr, "done\n");
        // IQ3_KL should beat IQ3_XXS (allow 5% slack for scale-quantization overhead)
        bool ok = (rkl < xxs * 1.05f);
        if (!ok) {
            any_fail = true;
        }
        printf("%-28s  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f%s\n", tc.name.c_str(), 1.f, q3k / xxs, rkl / xxs,
               rkl_opt / xxs, rh3 / xxs, ok ? "" : "  FAIL");
        fflush(stdout);
    }
    if (any_fail) {
        fprintf(stderr, "\nFAIL: IQ3_KL did not beat IQ3_XXS on some test cases\n");
        return 1;
    }

    printf("\n");
    if (!test_vec_dot_correctness(rng)) {
        fprintf(stderr, "\nFAIL: vec_dot dispatch does not match inline reference\n");
        return 1;
    }

    printf("\nPASS\n");
    return 0;
}
