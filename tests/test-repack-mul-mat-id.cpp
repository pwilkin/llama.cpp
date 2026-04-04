// Test REPACK MUL_MAT_ID kernels
// Build with cmake

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/repack.h"

static void init_tensor_uniform(ggml_tensor * t, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    if (t->type == GGML_TYPE_I32) {
        int32_t * data = (int32_t *) t->data;
        int64_t n_mats = t->ne[0];
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            data[i] = i % n_mats;  // simple pattern
        }
    } else if (t->type == GGML_TYPE_F32) {
        float * data = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            data[i] = dis(gen);
        }
    } else if (ggml_is_quantized(t->type)) {
        // Fill with random floats first, then quantize
        int64_t nels = ggml_nelements(t);
        std::vector<float> tmp(nels);
        for (int64_t i = 0; i < nels; i++) {
            tmp[i] = dis(gen);
        }
        ggml_quantize_chunk(t->type, tmp.data(), t->data, 0, 1, nels, nullptr);
    }
}

static float compute_nmse(const float * a, const float * b, int64_t n) {
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    
    for (int64_t i = 0; i < n; i++) {
        if (std::isnan(a[i]) || std::isnan(b[i])) {
            printf("NaN at index %lld: a=%f b=%f\n", (long long)i, a[i], b[i]);
            return INFINITY;
        }
        if (std::isinf(a[i]) || std::isinf(b[i])) {
            printf("Inf at index %lld: a=%f b=%f\n", (long long)i, a[i], b[i]);
            return INFINITY;
        }
        double diff = a[i] - b[i];
        sum_sq_diff += diff * diff;
        sum_sq_ref += (double)b[i] * b[i];
    }
    
    if (sum_sq_ref == 0.0) {
        return sum_sq_diff > 0 ? INFINITY : 0.0;
    }
    
    return sum_sq_diff / sum_sq_ref;
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;
    
    // Test parameters
    ggml_type type_a = GGML_TYPE_Q5_0;  // weight type
    int64_t n_mats = 8;   // number of experts
    int64_t n_used = 2;   // experts used per token
    int64_t m = 64;       // output dim
    int64_t n = 4;        // batch size
    int64_t k = 128;      // input dim

    printf("Testing MUL_MAT_ID with REPACK buffer\n");
    printf("  type_a: %s\n", ggml_type_name(type_a));
    printf("  n_mats: %lld\n", (long long)n_mats);
    printf("  n_used: %lld\n", (long long)n_used);
    printf("  m: %lld\n", (long long)m);
    printf("  n: %lld\n", (long long)n);
    printf("  k: %lld\n", (long long)k);
    printf("\n");

    // Initialize backends
    ggml_backend_load_all();
    
    // Find CPU device
    ggml_backend_dev_t cpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_dev = dev;
            break;
        }
    }
    
    if (!cpu_dev) {
        printf("Failed to find CPU device\n");
        return 1;
    }
    
    ggml_backend_t backend_cpu = ggml_backend_dev_init(cpu_dev, NULL);
    if (!backend_cpu) {
        printf("Failed to create CPU backend\n");
        return 1;
    }
    ggml_backend_buffer_type_t buft_cpu = ggml_backend_get_default_buffer_type(backend_cpu);
    ggml_backend_buffer_type_t buft_repack = ggml_backend_cpu_repack_buffer_type();

    // ========== Test 1: Reference implementation (regular CPU buffer) ==========
    printf("=== Test 1: Reference (CPU buffer) ===\n");
    
    ggml_init_params params = {
        /*.mem_size =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /*.mem_base =*/ NULL,
        /*.no_alloc =*/ true,
    };
    ggml_context * ctx1 = ggml_init(params);

    // Create tensors: as[k, m, n_mats], b[k, n_used, n], ids[n_used, n]
    ggml_tensor * as_ref = ggml_new_tensor_3d(ctx1, type_a, k, m, n_mats);
    ggml_set_name(as_ref, "as_ref");
    
    ggml_tensor * b_ref = ggml_new_tensor_3d(ctx1, GGML_TYPE_F32, k, n_used, n);
    ggml_set_name(b_ref, "b_ref");
    
    ggml_tensor * ids_ref = ggml_new_tensor_2d(ctx1, GGML_TYPE_I32, n_used, n);
    ggml_set_name(ids_ref, "ids_ref");
    
    ggml_tensor * out_ref = ggml_mul_mat_id(ctx1, as_ref, b_ref, ids_ref);
    ggml_set_name(out_ref, "out_ref");

    // Allocate on regular CPU buffer
    ggml_backend_buffer_t buf_ref = ggml_backend_alloc_ctx_tensors(ctx1, backend_cpu);
    if (!buf_ref) {
        printf("Failed to allocate reference tensors\n");
        return 1;
    }

    // Initialize tensors
    init_tensor_uniform(as_ref, -0.5f, 0.5f);
    init_tensor_uniform(b_ref, -1.0f, 1.0f);
    init_tensor_uniform(ids_ref);

    // Build and compute graph
    ggml_cgraph * gf_ref = ggml_new_graph(ctx1);
    ggml_build_forward_expand(gf_ref, out_ref);
    
    if (ggml_backend_graph_compute(backend_cpu, gf_ref) != GGML_STATUS_SUCCESS) {
        printf("Reference computation failed\n");
        return 1;
    }

    // Get reference output
    std::vector<float> ref_data(ggml_nelements(out_ref));
    ggml_backend_tensor_get(out_ref, ref_data.data(), 0, ggml_nbytes(out_ref));

    printf("Reference output sum: %f\n", 
        std::accumulate(ref_data.begin(), ref_data.end(), 0.0));
    printf("Reference output[0..3]: %f %f %f %f\n", 
        ref_data[0], ref_data[1], ref_data[2], ref_data[3]);

    // ========== Test 2: REPACK buffer ==========
    printf("\n=== Test 2: REPACK buffer ===\n");
    
    // Use single context, allocate all on REPACK buffer
    // REPACK buffer is just CPU buffer with extra tensor init handling
    params.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ggml_context * ctx2 = ggml_init(params);

    // Create tensors with same structure
    ggml_tensor * as_rep = ggml_new_tensor_3d(ctx2, type_a, k, m, n_mats);
    ggml_set_name(as_rep, "as_rep");
    
    ggml_tensor * b_rep = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32, k, n_used, n);
    ggml_set_name(b_rep, "b_rep");
    
    ggml_tensor * ids_rep = ggml_new_tensor_2d(ctx2, GGML_TYPE_I32, n_used, n);
    ggml_set_name(ids_rep, "ids_rep");
    
    ggml_tensor * out_rep = ggml_mul_mat_id(ctx2, as_rep, b_rep, ids_rep);
    ggml_set_name(out_rep, "out_rep");

    // Allocate all tensors on REPACK buffer
    ggml_backend_buffer_t buf_rep = ggml_backend_alloc_ctx_tensors_from_buft(ctx2, buft_repack);
    if (!buf_rep) {
        printf("Failed to allocate REPACK buffer\n");
        return 1;
    }

    // Copy data from reference
    printf("Copying as_rep data (size=%zu)\n", ggml_nbytes(as_ref));
    ggml_backend_tensor_set(as_rep, as_ref->data, 0, ggml_nbytes(as_ref));
    printf("Copying b_rep data (size=%zu)\n", ggml_nbytes(b_ref));
    ggml_backend_tensor_set(b_rep, b_ref->data, 0, ggml_nbytes(b_ref));
    printf("Copying ids_rep data (size=%zu)\n", ggml_nbytes(ids_ref));
    ggml_backend_tensor_set(ids_rep, ids_ref->data, 0, ggml_nbytes(ids_ref));
    printf("Data copy complete\n");

    // Build and compute graph
    ggml_cgraph * gf_rep = ggml_new_graph(ctx2);
    ggml_build_forward_expand(gf_rep, out_rep);
    
    if (ggml_backend_graph_compute(backend_cpu, gf_rep) != GGML_STATUS_SUCCESS) {
        printf("REPACK computation failed\n");
        return 1;
    }

    // Get REPACK output
    std::vector<float> rep_data(ggml_nelements(out_rep));
    ggml_backend_tensor_get(out_rep, rep_data.data(), 0, ggml_nbytes(out_rep));

    printf("REPACK output sum: %f\n", 
        std::accumulate(rep_data.begin(), rep_data.end(), 0.0));
    printf("REPACK output[0..3]: %f %f %f %f\n", 
        rep_data[0], rep_data[1], rep_data[2], rep_data[3]);

    // Compare results
    printf("\n=== Comparison ===\n");
    float nmse = compute_nmse(rep_data.data(), ref_data.data(), ggml_nelements(out_ref));
    printf("NMSE: %e\n", nmse);
    
    if (nmse < 1e-4) {
        printf("PASS: Results match within tolerance\n");
    } else {
        printf("FAIL: Results differ significantly\n");
        // Print first few differences
        for (int i = 0; i < std::min((int64_t)10, ggml_nelements(out_ref)); i++) {
            printf("  [%d] ref=%f rep=%f diff=%f\n", i, ref_data[i], rep_data[i], ref_data[i] - rep_data[i]);
        }
    }

    // Cleanup
    ggml_backend_buffer_free(buf_rep);
    ggml_backend_buffer_free(buf_ref);
    ggml_free(ctx2);
    ggml_free(ctx1);
    ggml_backend_free(backend_cpu);

    return nmse < 1e-4 ? 0 : 1;
}