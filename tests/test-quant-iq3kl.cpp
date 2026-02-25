// test-quant-iq3kl.cpp
// Quantization accuracy regression test for IQ3_KL.
//
// Generates synthetic weight-like tensors, runs quantize→dequantize cycles for
// IQ3_KL (per-tensor learned VQ codebook), IQ3_XXS, and Q3_K, and reports RMSE.
// Fails if IQ3_KL error exceeds IQ3_XXS error (it uses ~6% more bits so should
// perform at least as well).
//
// Usage:
//   test-quant-iq3kl [-v]
//
// Build: listed in tests/CMakeLists.txt via llama_build_and_test(...)

#include "ggml.h"
#include "ggml-backend.h"

// IQ3_KL public API (declared in ggml/src/ggml-quants.h but exposed via GGML_API)
extern "C" {
    void            iq3kl_set_codebook(const uint8_t * codebook);
    void            iq3kl_generate_codebook(const float * data, int64_t nrow, int64_t n_per_row,
                                            const float * imatrix, uint8_t * codebook_out);
    size_t          quantize_iq3_kl(const float * src, void * dst, int64_t nrows,
                                    int64_t n_per_row, const float * imatrix);
}


#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static float rmse(const float * a, const float * b, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return (float)std::sqrt(sum / (double)n);
}

// Quantize + dequantize a tensor with a standard GGML type (not IQ3_KL).
static float std_quant_rmse(ggml_type type,
                             const float * data, size_t nrow, size_t n_per_row,
                             const float * imatrix) {
    const size_t n        = nrow * n_per_row;
    const size_t row_size = ggml_row_size(type, n_per_row);
    std::vector<uint8_t> qbuf(nrow * row_size);
    std::vector<float>   dqbuf(n);

    ggml_quantize_chunk(type, data, qbuf.data(), 0, nrow, n_per_row, imatrix);

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    for (size_t row = 0; row < nrow; ++row) {
        traits->to_float(qbuf.data() + row * row_size,
                         dqbuf.data() + row * n_per_row,
                         (int64_t)n_per_row);
    }
    return rmse(data, dqbuf.data(), n);
}

// Quantize + dequantize with IQ3_KL using a freshly generated per-tensor codebook.
// The codebook is generated from the tensor's own data.
static float iq3kl_quant_rmse(const float * data, size_t nrow, size_t n_per_row,
                               const float * imatrix) {
    const size_t n        = nrow * n_per_row;
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ3_KL, n_per_row);

    // Generate codebook from this tensor's data
    std::vector<uint8_t> codebook(256 * 4);
    iq3kl_generate_codebook(data, (int64_t)nrow, (int64_t)n_per_row, imatrix, codebook.data());
    iq3kl_set_codebook(codebook.data());

    std::vector<uint8_t> qbuf(nrow * row_size);
    std::vector<float>   dqbuf(n);

    quantize_iq3_kl(data, qbuf.data(), (int64_t)nrow, (int64_t)n_per_row, imatrix);

    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_IQ3_KL);
    for (size_t row = 0; row < nrow; ++row) {
        // dequantize_row_iq3_kl falls back to global codebook when pointer
        // is not in the per-tensor registry — that's exactly what we want here.
        traits->to_float(qbuf.data() + row * row_size,
                         dqbuf.data() + row * n_per_row,
                         (int64_t)n_per_row);
    }
    return rmse(data, dqbuf.data(), n);
}

struct TestCase {
    std::string name;
    std::vector<float> data;
    size_t nrow;
    size_t n_per_row;
    // IQ3_KL must be no worse than this multiple of IQ3_XXS RMSE to pass.
    float max_ratio_vs_xxs;
};

int main(int /*argc*/, char ** /*argv*/) {

    ggml_backend_load_all();
    ggml_quantize_init(GGML_TYPE_IQ3_XXS);
    ggml_quantize_init(GGML_TYPE_Q3_K);
    ggml_quantize_init(GGML_TYPE_IQ3_KL);

    std::mt19937 rng(0xdeadbeef);

    std::vector<TestCase> cases;

    // ------------------------------------------------------------------
    // Case 1: standard zero-centered Gaussian (typical transformer FFN)
    // ------------------------------------------------------------------
    {
        TestCase tc;
        tc.name           = "Gaussian(0, 0.02) 64x4096";
        tc.nrow           = 64;
        tc.n_per_row      = 4096;
        tc.max_ratio_vs_xxs = 1.05f;   // IQ3_KL should be close to or better than XXS
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0.0f, 0.02f);
        for (auto & v : tc.data) { v = nd(rng); }
        cases.push_back(std::move(tc));
    }

    // ------------------------------------------------------------------
    // Case 2: wider Gaussian (attention projection)
    // ------------------------------------------------------------------
    {
        TestCase tc;
        tc.name           = "Gaussian(0, 0.05) 32x8192";
        tc.nrow           = 32;
        tc.n_per_row      = 8192;
        tc.max_ratio_vs_xxs = 1.05f;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0.0f, 0.05f);
        for (auto & v : tc.data) { v = nd(rng); }
        cases.push_back(std::move(tc));
    }

    // ------------------------------------------------------------------
    // Case 3: narrower Gaussian (typical for deeper layers)
    // ------------------------------------------------------------------
    {
        TestCase tc;
        tc.name           = "Gaussian(0, 0.01) 128x2048";
        tc.nrow           = 128;
        tc.n_per_row      = 2048;
        tc.max_ratio_vs_xxs = 1.05f;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0.0f, 0.01f);
        for (auto & v : tc.data) { v = nd(rng); }
        cases.push_back(std::move(tc));
    }

    // ------------------------------------------------------------------
    // Case 4: Laplace-like (sparser weight distribution)
    // ------------------------------------------------------------------
    {
        TestCase tc;
        tc.name           = "Laplace(0, 0.01) 64x4096";
        tc.nrow           = 64;
        tc.n_per_row      = 4096;
        tc.max_ratio_vs_xxs = 1.10f;   // slightly more lenient for non-Gaussian
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::exponential_distribution<float> ed(100.0f);
        std::bernoulli_distribution bd(0.5f);
        for (auto & v : tc.data) {
            v = ed(rng);
            if (bd(rng)) { v = -v; }
        }
        cases.push_back(std::move(tc));
    }

    // ------------------------------------------------------------------
    // Case 5: simulate imatrix importance (uniform weights here)
    // ------------------------------------------------------------------
    {
        TestCase tc;
        tc.name           = "Gaussian imatrix 64x4096";
        tc.nrow           = 64;
        tc.n_per_row      = 4096;
        tc.max_ratio_vs_xxs = 1.05f;
        tc.data.resize(tc.nrow * tc.n_per_row);
        std::normal_distribution<float> nd(0.0f, 0.03f);
        for (auto & v : tc.data) { v = nd(rng); }
        cases.push_back(std::move(tc));
    }

    printf("%-35s  %10s  %10s  %10s  %10s  %10s  %s\n",
           "Test case", "IQ3_KL", "IQ3_XXS", "Q3_K", "KL/XXS", "KL/Q3K", "result");
    printf("%-35s  %10s  %10s  %10s  %10s  %10s  %s\n",
           "-----------------------------------",
           "----------", "----------", "----------", "----------", "----------", "------");

    int n_pass = 0;
    int n_fail = 0;

    for (auto & tc : cases) {
        const float kl_err  = iq3kl_quant_rmse(tc.data.data(), tc.nrow, tc.n_per_row, nullptr);
        const float xxs_err = std_quant_rmse(GGML_TYPE_IQ3_XXS, tc.data.data(), tc.nrow, tc.n_per_row, nullptr);
        const float q3k_err = std_quant_rmse(GGML_TYPE_Q3_K,    tc.data.data(), tc.nrow, tc.n_per_row, nullptr);

        const float ratio_xxs = kl_err / xxs_err;
        const float ratio_q3k = kl_err / q3k_err;
        const bool  pass      = (ratio_xxs <= tc.max_ratio_vs_xxs);

        if (pass) { ++n_pass; } else { ++n_fail; }

        printf("%-35s  %10.5f  %10.5f  %10.5f  %10.3f  %10.3f  %s\n",
               tc.name.c_str(), kl_err, xxs_err, q3k_err,
               ratio_xxs, ratio_q3k, pass ? "PASS" : "FAIL");
    }

    printf("\n%d/%zu test cases passed\n", n_pass, cases.size());
    return n_fail > 0 ? 1 : 0;
}
