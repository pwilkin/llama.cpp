#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "common.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

// Helper: fill tensors with deterministic random data
static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string>          hasher;
    std::mt19937                    gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);
    const int64_t                   ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    }
}

// Helper: create a minimal GGUF model for a given architecture
static gguf_context_ptr create_test_gguf(llm_arch arch) {
    gguf_context_ptr  ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t    n_ctx       = 32;
    const uint32_t    n_vocab     = 32;
    const uint32_t    n_embd      = 64;
    const uint32_t    n_head      = 2;
    const uint32_t    n_ff        = 128;
    const uint32_t    n_layer     = 2;
    const uint32_t    n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE, n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH, n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH, n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT, n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL, false);
    ms.add_kv(LLM_KV_LOGIT_SCALE, 1.0f);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT, n_embd_head);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");

    // Add tensors (F16 shapes)
    ggml_tensor t;
    memset(&t, 0, sizeof(t));
    t.type = GGML_TYPE_F16;

    // token embeddings
    ggml_format_name(&t, "token_embd.weight");
    t.ne[0] = n_embd;
    t.ne[1] = n_vocab;
    gguf_add_tensor(ms.gguf_ctx, &t);

    // output
    ggml_format_name(&t, "output.weight");
    t.ne[0] = n_embd;
    t.ne[1] = n_vocab;
    gguf_add_tensor(ms.gguf_ctx, &t);

    // per-layer tensors
    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_format_name(&t, "blk.%" PRIu32 ".attn_norm.weight", il);
        t.ne[0] = n_embd;
        t.ne[1] = 1;
        gguf_add_tensor(ms.gguf_ctx, &t);

        ggml_format_name(&t, "blk.%" PRIu32 ".attn_q.weight", il);
        t.ne[0] = n_embd;
        t.ne[1] = n_embd;
        gguf_add_tensor(ms.gguf_ctx, &t);

        ggml_format_name(&t, "blk.%" PRIu32 ".attn_k.weight", il);
        t.ne[0] = n_embd;
        t.ne[1] = n_embd;
        gguf_add_tensor(ms.gguf_ctx, &t);

        ggml_format_name(&t, "blk.%" PRIu32 ".attn_v.weight", il);
        t.ne[0] = n_embd;
        t.ne[1] = n_embd;
        gguf_add_tensor(ms.gguf_ctx, &t);

        ggml_format_name(&t, "blk.%" PRIu32 ".attn_output.weight", il);
        t.ne[0] = n_embd;
        t.ne[1] = n_embd;
        gguf_add_tensor(ms.gguf_ctx, &t);

        ggml_format_name(&t, "blk.%" PRIu32 ".ffn_norm.weight", il);
        t.ne[0] = n_embd;
        t.ne[1] = 1;
        gguf_add_tensor(ms.gguf_ctx, &t);

        ggml_format_name(&t, "blk.%" PRIu32 ".ffn_up.weight", il);
        t.ne[0] = n_ff;
        t.ne[1] = n_embd;
        gguf_add_tensor(ms.gguf_ctx, &t);

        ggml_format_name(&t, "blk.%" PRIu32 ".ffn_down.weight", il);
        t.ne[0] = n_embd;
        t.ne[1] = n_ff;
        gguf_add_tensor(ms.gguf_ctx, &t);
    }

    return ret;
}

// Save a model from a gguf context to a file
static bool save_test_model(const char * path, gguf_context * gguf_ctx, size_t seed) {
    llama_model_params mparams = llama_model_default_params();
    size_t             tmp     = seed;
    llama_model_ptr    model(llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, mparams));
    if (!model) {
        return false;
    }
    llama_model_save_to_file(model.get(), path);
    return true;
}

// Load output GGUF and return map of tensor_name -> type
static std::map<std::string, ggml_type> load_tensor_types(const char * path) {
    std::map<std::string, ggml_type> result;
    struct gguf_init_params          params = { /*.no_alloc = */ true, /*.ctx = */ nullptr };
    struct gguf_context *            ctx    = gguf_init_from_file(path, params);
    if (!ctx) {
        return result;
    }
    int n = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n; i++) {
        result[gguf_get_tensor_name(ctx, i)] = gguf_get_tensor_type(ctx, i);
    }
    gguf_free(ctx);
    return result;
}

// --- Test callbacks ---

// Test 1: Override output.weight to F16, everything else to Q4_0
static ggml_type test1_resolver(const char *, int, const int64_t *, int, const char * category, void *) {
    if (strcmp(category, "OUTPUT") == 0) {
        return GGML_TYPE_F16;
    }
    if (strcmp(category, "TOKEN_EMBD") == 0) {
        return GGML_TYPE_F16;
    }
    return GGML_TYPE_Q4_0;
}

// Test 2: Layer 0 gets Q8_0, other layers get Q4_0, output stays F16
static ggml_type test2_resolver(const char *, int, const int64_t *, int layer, const char * category, void *) {
    if (strcmp(category, "OUTPUT") == 0) {
        return GGML_TYPE_F16;
    }
    if (strcmp(category, "TOKEN_EMBD") == 0) {
        return GGML_TYPE_F16;
    }
    if (layer == 0) {
        return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_Q4_0;
}

// Test 3: Everything goes to F16 (no quantization effectively)
static ggml_type test3_resolver(const char *, int, const int64_t *, int, const char *, void *) {
    return GGML_TYPE_F16;
}

// Test 4: Return GGML_TYPE_COUNT for everything -> should fall through to standard logic
static ggml_type test4_resolver(const char *, int, const int64_t *, int, const char *, void *) {
    return GGML_TYPE_COUNT;
}

static bool run_test(const char *                             test_name,
                     llm_arch                                 arch,
                     llama_ftype                              ftype,
                     llama_tensor_type_resolver               resolver,
                     void *                                   ud,
                     const std::map<std::string, ggml_type> & expected) {
    printf("  Running: %s ... ", test_name);
    fflush(stdout);

    // Create and save source model
    char src_path[64], out_path[64];
    snprintf(src_path, sizeof(src_path), "/tmp/test_resolver_src_%s.gguf", test_name);
    snprintf(out_path, sizeof(out_path), "/tmp/test_resolver_out_%s.gguf", test_name);

    gguf_context_ptr gguf_ctx = create_test_gguf(arch);
    if (!save_test_model(src_path, gguf_ctx.get(), 42)) {
        printf("FAIL (save model)\n");
        return false;
    }

    // Quantize with callback
    llama_model_quantize_params qparams = llama_model_quantize_default_params();
    qparams.ftype                       = ftype;
    qparams.tensor_type_resolver        = resolver;
    qparams.tensor_type_resolver_ud     = ud;

    uint32_t ret = llama_model_quantize(src_path, out_path, &qparams);
    if (ret != 0) {
        printf("FAIL (quantize returned %u)\n", ret);
        return false;
    }

    // Verify tensor types
    auto types = load_tensor_types(out_path);
    bool pass  = true;
    for (const auto & [name, expected_type] : expected) {
        auto it = types.find(name);
        if (it == types.end()) {
            printf("FAIL (tensor '%s' not found)\n", name.c_str());
            pass = false;
            break;
        }
        if (it->second != expected_type) {
            printf("FAIL (tensor '%s': expected %s, got %s)\n", name.c_str(), ggml_type_name(expected_type),
                   ggml_type_name(it->second));
            pass = false;
            break;
        }
    }
    if (pass) {
        printf("PASS\n");
    }

    // Cleanup
    std::remove(src_path);
    std::remove(out_path);

    return pass;
}

int main() {
    common_init();

    const llm_arch arch = LLM_ARCH_LLAMA;

    printf("Testing llama_tensor_type_resolver callback\n");

    bool all_pass = true;

    // Test 1: Category-based override
    {
        std::map<std::string, ggml_type> expected;
        expected["token_embd.weight"]     = GGML_TYPE_F16;
        expected["output.weight"]         = GGML_TYPE_F16;
        expected["blk.0.attn_q.weight"]   = GGML_TYPE_Q4_0;
        expected["blk.1.attn_q.weight"]   = GGML_TYPE_Q4_0;
        expected["blk.0.ffn_down.weight"] = GGML_TYPE_Q4_0;
        expected["blk.1.ffn_down.weight"] = GGML_TYPE_Q4_0;
        if (!run_test("category_override", arch, LLAMA_FTYPE_MOSTLY_Q5_1, test1_resolver, nullptr, expected)) {
            all_pass = false;
        }
    }

    // Test 2: Layer-based override
    {
        std::map<std::string, ggml_type> expected;
        expected["token_embd.weight"]     = GGML_TYPE_F16;
        expected["output.weight"]         = GGML_TYPE_F16;
        expected["blk.0.attn_q.weight"]   = GGML_TYPE_Q8_0;
        expected["blk.0.ffn_down.weight"] = GGML_TYPE_Q8_0;
        expected["blk.1.attn_q.weight"]   = GGML_TYPE_Q4_0;
        expected["blk.1.ffn_down.weight"] = GGML_TYPE_Q4_0;
        if (!run_test("layer_override", arch, LLAMA_FTYPE_MOSTLY_Q5_1, test2_resolver, nullptr, expected)) {
            all_pass = false;
        }
    }

    // Test 3: Everything F16
    {
        std::map<std::string, ggml_type> expected;
        expected["token_embd.weight"]     = GGML_TYPE_F16;
        expected["output.weight"]         = GGML_TYPE_F16;
        expected["blk.0.attn_q.weight"]   = GGML_TYPE_F16;
        expected["blk.0.ffn_down.weight"] = GGML_TYPE_F16;
        if (!run_test("all_f16", arch, LLAMA_FTYPE_MOSTLY_Q5_1, test3_resolver, nullptr, expected)) {
            all_pass = false;
        }
    }

    // Test 4: Return COUNT -> fall through to standard logic
    {
        // With standard logic and Q5_1, most tensors become Q5_1
        // output.weight stays original if quantize_output_tensor is true
        std::map<std::string, ggml_type> expected;
        expected["token_embd.weight"]     = GGML_TYPE_Q5_1;
        expected["blk.0.attn_q.weight"]   = GGML_TYPE_Q5_1;
        expected["blk.0.ffn_down.weight"] = GGML_TYPE_Q5_1;
        if (!run_test("fallback", arch, LLAMA_FTYPE_MOSTLY_Q5_1, test4_resolver, nullptr, expected)) {
            all_pass = false;
        }
    }

    printf("\n%s: %s\n", __func__, all_pass ? "ALL PASSED" : "SOME TESTS FAILED");

    return all_pass ? 0 : 1;
}
