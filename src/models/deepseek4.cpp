#include "models.h"

#include "../llama-kv-cache.h"

void llama_model_deepseek4::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // MLA-style attention: single low-rank Q (wq_a/wq_b) and a single head-sized KV
    // latent (wkv) that serves as both K and V; grouped low-rank output projection.
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,  hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_O_LORA_RANK,  hparams.n_lora_o);
    ml.get_key(LLM_KV_ATTENTION_O_GROUPS,     hparams.n_o_groups);
    // custom sliding window (do NOT use hparams.n_swa: that triggers llama's SWA cache machinery)
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_window, false);

    // per-layer KV compression ratios (0 = pure sliding-window attention)
    std::fill(hparams.compress_ratios.begin(), hparams.compress_ratios.end(), 0);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.compress_ratios, hparams.n_layer, false);

    // sparse-attention indexer
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head,    false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k,     false);

    // MoE
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm,  false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func,   false);
    ml.get_key(LLM_KV_HASH_LAYER_COUNT,           hparams.n_hash_layers,        false);
    ml.get_key(LLM_KV_SWIGLU_LIMIT,               hparams.f_swiglu_limit,       false);

    // hyper-connections
    ml.get_key(LLM_KV_HC_MULT,           hparams.hc_mult);
    ml.get_key(LLM_KV_HC_SINKHORN_ITERS, hparams.hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HC_EPS,            hparams.hc_eps, false);

    // rope: KV-compressed layers use a separate (larger) base + YaRN frequency interpolation.
    // factor / n_ctx_orig_yarn are read by the common loader; beta_fast/slow are not, so read here.
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_COMPRESS,      hparams.rope_freq_base_compress, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST,  hparams.yarn_beta_fast,          false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,  hparams.yarn_beta_slow,          false);

    // Decode/KV-cache (incremental): the graph caches, per token cell, the per-layer attention
    // input x AND the derived compressed-KV block + indexer-KV block that "complete" at that cell,
    // so decode reads cached blocks instead of recomputing the compressor over all past tokens.
    // We repurpose the V cache as one wide, non-transposed KV head (n_head_kv=1) holding the cell
    //   [ x (n_embd) | compressed block (head_dim) | indexer block (indexer_head_size) ].
    // Region offsets are fixed: x@0, comp@n_embd, idx@(n_embd+head_dim). (K side is unused;
    // n_embd_head_k stays the attention head dim.)
    const int64_t hd_attn = hparams.n_embd_head_k_full; // attention head dim (= compressed latent size)
    bool any_comp = false, any_idx = false;
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        if (hparams.compress_ratios[i] > 0) any_comp = true;
        if (hparams.compress_ratios[i] == 4) any_idx = true;
    }
    const int64_t v_width = (int64_t) hparams.n_embd
                          + (any_comp ? hd_attn : 0)
                          + (any_idx  ? (int64_t) hparams.indexer_head_size : 0);
    hparams.n_head_kv_arr.fill(1);
    hparams.n_embd_head_v_full = v_width;
    hparams.n_embd_head_v_swa  = v_width;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_deepseek4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t hd              = n_embd_head_k;          // attention head dim (= key/value length)
    const int64_t n_lora_q        = hparams.n_lora_q;
    const int64_t n_lora_o        = hparams.n_lora_o;
    const int64_t n_groups        = hparams.n_o_groups;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t hc              = hparams.hc_mult;
    const int64_t mix_hc          = (2 + hc) * hc;
    const int64_t hc_dim          = hc * n_embd;
    const int64_t idx_nh          = hparams.indexer_n_head;
    const int64_t idx_hd          = hparams.indexer_head_size;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // final hyper-connection head reduction (hc_mult streams -> 1)
    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN),    {hc_dim, hc}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE),  {hc}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE), {1}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t cr            = hparams.compress_ratios[i];
        const bool    has_compressor = cr > 0;
        const bool    overlap        = (cr == 4);
        const bool    has_indexer    = (cr == 4);
        const int64_t coff           = overlap ? 2 : 1;
        const bool    is_hash        = (uint32_t) i < hparams.n_hash_layers;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);

        // hyper-connection mixing params (attn + ffn sublayers)
        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    i), {hc_dim, mix_hc}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  i), {mix_hc}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     i), {hc_dim, mix_hc}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   i), {mix_hc}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  i), {3}, 0);

        // attention
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, n_lora_q}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {n_lora_q}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {n_lora_q, n_head * hd}, 0);
        layer.wkv_a_mqa     = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, hd}, 0);
        layer.attn_kv_a_norm= create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM,"weight", i), {hd}, 0);
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_O_A,      "weight", i), {n_head * hd / n_groups, n_groups * n_lora_o}, 0);
        layer.wo            = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {n_groups * n_lora_o, n_embd}, 0);

        if (has_compressor) {
            layer.compressor_kv   = create_tensor(tn(LLM_TENSOR_COMPRESSOR_KV,   "weight", i), {n_embd, coff * hd}, 0);
            layer.compressor_gate = create_tensor(tn(LLM_TENSOR_COMPRESSOR_GATE, "weight", i), {n_embd, coff * hd}, 0);
            layer.compressor_norm = create_tensor(tn(LLM_TENSOR_COMPRESSOR_NORM, "weight", i), {hd}, 0);
            layer.compressor_ape  = create_tensor(tn(LLM_TENSOR_COMPRESSOR_APE,  i), {coff * hd, cr}, 0);
        }
        if (has_indexer) {
            layer.indexer_attn_q_b    = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B,        "weight", i), {n_lora_q, idx_nh * idx_hd}, 0);
            layer.indexer_proj        = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,            "weight", i), {n_embd, idx_nh}, 0);
            layer.idx_compressor_kv   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_KV,   "weight", i), {n_embd, 2 * idx_hd}, 0);
            layer.idx_compressor_gate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_GATE, "weight", i), {n_embd, 2 * idx_hd}, 0);
            layer.idx_compressor_norm = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM, "weight", i), {idx_hd}, 0);
            layer.idx_compressor_ape  = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,  i), {2 * idx_hd, cr}, 0);
        }

        // MoE: hash routing (leading n_hash_layers) vs score routing (rest)
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        if (is_hash) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, i), {n_expert_used, n_vocab}, 0);
        } else {
            layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        }
        layer.ffn_gate_exps  = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps  = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps    = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// ---------------------------------------------------------------------------
// DS4 sparse-attention mask input. Keys = [ raw window cells (n_win) ++ compressed blocks (n_comp) ];
// shape [n_keys, n_seq_tokens, 1, n_stream] (the 1 broadcasts over q heads). Tokens are stream-major
// in the ubatch: token (stream s, seq-token t) is at flat index s*n_seq_tokens + t.
//   window c (cache cell = win_base + c): 0 if 0 <= pos - (win_base + c) < window  else -inf
//   comp   b:                            0 if b < (pos + 1) / ratio                else -inf
// Per stream, cache cell == absolute position (each stream is its own contiguous in-order buffer).
// (n_comp == 0 -> pure sliding-window; ratio is then unused.)
// ---------------------------------------------------------------------------
namespace {
struct llm_graph_input_ds4_mask : public llm_graph_input_i {
    llm_graph_input_ds4_mask(uint32_t window, int64_t ratio, int64_t n_comp, int64_t win_base,
                             int64_t n_seq_tokens, int64_t n_stream)
        : window(window), ratio(ratio), n_comp(n_comp), win_base(win_base),
          n_seq_tokens(n_seq_tokens), n_stream(n_stream) {}
    void set_input(const llama_ubatch * ubatch) override {
        if (!mask) {
            return;
        }
        const int64_t n_keys = mask->ne[0];           // window + compressed keys
        const int64_t n_win  = n_keys - n_comp;       // cached raw window cells
        std::vector<float> buf((size_t) n_keys * n_seq_tokens * n_stream);
        for (int64_t s = 0; s < n_stream; ++s) {
            for (int64_t t = 0; t < n_seq_tokens; ++t) {
                const llama_pos pi = ubatch->pos[s * n_seq_tokens + t]; // query position
                float * row = &buf[((size_t) s * n_seq_tokens + t) * n_keys];
                for (int64_t c = 0; c < n_win; ++c) {
                    const int64_t d = pi - (win_base + c);             // window key position == cell index
                    row[c] = (d >= 0 && d < (int64_t) window) ? 0.0f : -INFINITY;
                }
                for (int64_t b = 0; b < n_comp; ++b) {
                    row[n_win + b] = (b < (pi + 1) / ratio) ? 0.0f : -INFINITY; // block-causal
                }
            }
        }
        ggml_backend_tensor_set(mask, buf.data(), 0, buf.size() * sizeof(float));
    }
    ggml_tensor * mask = nullptr; // F32 [n_win+n_comp, n_seq_tokens, 1, n_stream]
    uint32_t window;
    int64_t  ratio;
    int64_t  n_comp;
    int64_t  win_base;
    int64_t  n_seq_tokens;
    int64_t  n_stream;
};

// block-cell store indices: for each stream s, the (re)written blocks go to cache cells
// [bases[s], bases[s] + n). bases[s] = stream s's global cell base + its first written block index.
// The block values tensor is [out_dim, n, n_stream] flattened to [out_dim, n*n_stream] (column
// j = i + s*n) so idx[s*n + i] = bases[s] + i.
struct llm_graph_input_ds4_blkidx : public llm_graph_input_i {
    llm_graph_input_ds4_blkidx(std::vector<int64_t> bases, int64_t n) : bases(std::move(bases)), n(n) {}
    void set_input(const llama_ubatch * /*ubatch*/) override {
        if (!idxs) {
            return;
        }
        std::vector<int64_t> buf((size_t) n * (int64_t) bases.size());
        for (size_t s = 0; s < bases.size(); ++s) {
            for (int64_t i = 0; i < n; ++i) {
                buf[(size_t) s * n + i] = bases[s] + i;
            }
        }
        ggml_backend_tensor_set(idxs, buf.data(), 0, buf.size() * sizeof(int64_t));
    }
    ggml_tensor * idxs = nullptr; // I64 [n * n_stream] global cell indices
    std::vector<int64_t> bases;
    int64_t n;
};

// hash-routing dedup mask: for each (slot, token), 1.0 unless a later slot routes to the
// same expert (then 0.0). This replicates model.py's `y[idx] += ...` last-write-wins add,
// which keeps only the last occurrence of a duplicated expert.
struct llm_graph_input_ds4_hash : public llm_graph_input_i {
    explicit llm_graph_input_ds4_hash(ggml_tensor * tid2eid) : tid2eid(tid2eid) {}
    void set_input(const llama_ubatch * ubatch) override {
        if (!mask) {
            return;
        }
        const int64_t n_act    = mask->ne[0];
        const int64_t n_tokens = ubatch->n_tokens;
        const int64_t vocab    = tid2eid->ne[1];
        if (table.empty()) {
            table.resize(ggml_nelements(tid2eid));
            ggml_backend_tensor_get(tid2eid, table.data(), 0, ggml_nbytes(tid2eid));
        }
        std::vector<float> buf((size_t) n_act * n_tokens, 1.0f);
        for (int64_t t = 0; t < n_tokens; ++t) {
            int64_t id = ubatch->token ? ubatch->token[t] : 0;
            if (id < 0 || id >= vocab) {
                id = 0;
            }
            const float * e = &table[id * n_act];
            for (int64_t s = 0; s < n_act; ++s) {
                bool last = true;
                for (int64_t s2 = s + 1; s2 < n_act; ++s2) {
                    if ((int) e[s2] == (int) e[s]) { last = false; break; }
                }
                buf[(size_t) t * n_act + s] = last ? 1.0f : 0.0f;
            }
        }
        ggml_backend_tensor_set(mask, buf.data(), 0, buf.size() * sizeof(float));
    }
    ggml_tensor * tid2eid = nullptr;
    ggml_tensor * mask = nullptr; // F32 [n_act, n_tokens]
    std::vector<float> table;     // cached tid2eid contents
};

// minimal KV-cache input: only the V-slot store indices (we cache the per-layer attention input x
// in V and read it back manually, so we do not use the standard k_idxs / kq_mask machinery).
struct llm_graph_input_ds4_vstore : public llm_graph_input_i {
    explicit llm_graph_input_ds4_vstore(const llama_kv_cache_context * mctx) : mctx(mctx) {}
    void set_input(const llama_ubatch * ubatch) override {
        if (v_idxs) {
            mctx->set_input_v_idxs(v_idxs, ubatch);
        }
    }
    const llama_kv_cache_context * mctx;
    ggml_tensor * v_idxs = nullptr; // I64 store indices for the current ubatch
};
} // namespace

// rotate the trailing rope_head_dim dims with adjacent-pair complex rope.
// x: [head_dim, n_head, n_tokens]; cos/sin: [rope_head_dim/2, n_tokens]; inverse de-rotates.
ggml_tensor * llama_model_deepseek4::graph::rope_tail(ggml_tensor * x, ggml_tensor * rcos, ggml_tensor * rsin, bool inverse) {
    const int64_t hd  = x->ne[0];
    const int64_t nh  = x->ne[1];
    const int64_t nt  = x->ne[2];
    const int64_t rd  = hparams.n_rot();
    const int64_t rd2 = rd / 2;

    // split [nope | pe]
    ggml_tensor * nope = ggml_cont(ctx0, ggml_view_3d(ctx0, x, hd - rd, nh, nt, x->nb[1], x->nb[2], 0));
    ggml_tensor * pe   = ggml_cont(ctx0, ggml_view_3d(ctx0, x, rd, nh, nt, x->nb[1], x->nb[2], (hd - rd) * x->nb[0]));

    // pairs: [2, rd2, nh, nt]; even/odd are ne0 slices
    ggml_tensor * pairs = ggml_reshape_4d(ctx0, pe, 2, rd2, nh, nt);
    ggml_tensor * even  = ggml_view_4d(ctx0, pairs, 1, rd2, nh, nt, pairs->nb[1], pairs->nb[2], pairs->nb[3], 0);
    ggml_tensor * odd   = ggml_view_4d(ctx0, pairs, 1, rd2, nh, nt, pairs->nb[1], pairs->nb[2], pairs->nb[3], pairs->nb[0]);

    ggml_tensor * c = ggml_reshape_4d(ctx0, rcos, 1, rd2, 1, nt); // broadcast over nh
    ggml_tensor * s = ggml_reshape_4d(ctx0, rsin, 1, rd2, 1, nt);

    ggml_tensor * ec = ggml_mul(ctx0, even, c);
    ggml_tensor * es = ggml_mul(ctx0, even, s);
    ggml_tensor * oc = ggml_mul(ctx0, odd,  c);
    ggml_tensor * os = ggml_mul(ctx0, odd,  s);

    ggml_tensor * out_even = inverse ? ggml_add(ctx0, ec, os) : ggml_sub(ctx0, ec, os);
    ggml_tensor * out_odd  = inverse ? ggml_sub(ctx0, oc, es) : ggml_add(ctx0, es, oc);

    ggml_tensor * roped = ggml_concat(ctx0, out_even, out_odd, 0);     // [2, rd2, nh, nt]
    roped = ggml_reshape_3d(ctx0, ggml_cont(ctx0, roped), rd, nh, nt); // [rd, nh, nt]

    return ggml_concat(ctx0, nope, roped, 0); // [hd, nh, nt]
}

// learned gated KV compression (prefill). For each block of `ratio` consecutive tokens:
//   kv = wkv(x), score = wgate(x) + ape; pooled = sum_w(kv * softmax_w(score)); RMS-norm; rope.
// Non-overlap (coff=1): pools the `ratio` tokens of the block. Overlap (coff=2, ratio==4 layers):
// wkv/wgate emit 2*head_dim; each block pools over 2*ratio positions = the previous block's
// first-half ("a") projection followed by the current block's second-half ("b") projection
// (the reference `overlap_transform`), with the missing previous block (block 0) masked out.
// x: [n_embd, n_tokens]; comp_rcos/comp_rsin: [rope_head_dim/2, n_comp] (block-start rope angles).
ggml_tensor * llama_model_deepseek4::graph::build_compressor(ggml_tensor * x, ggml_tensor * w_kv,
        ggml_tensor * w_gate, ggml_tensor * norm_w, ggml_tensor * ape, int64_t ratio,
        ggml_tensor * comp_rcos, ggml_tensor * comp_rsin) {
    const int64_t hd      = norm_w->ne[0];     // head_dim (compressed latent size)
    const int64_t hd_proj = ape->ne[0];        // coff*head_dim (wkv/wgate output size)
    const bool    overlap = hd_proj == 2 * hd; // ratio==4 layers use overlapping windows
    const int64_t n_comp  = comp_rcos->ne[1];  // number of compressed blocks
    const int64_t cutoff  = n_comp * ratio;    // tokens belonging to full blocks (prefill drops the remainder)

    ggml_tensor * kv = ggml_mul_mat(ctx0, w_kv,   x); // [hd_proj, n_tokens]
    ggml_tensor * sc = ggml_mul_mat(ctx0, w_gate, x); // [hd_proj, n_tokens]

    // keep only the first `cutoff` tokens, then group consecutive `ratio` tokens into a block axis
    if (cutoff != kv->ne[1]) {
        kv = ggml_view_2d(ctx0, kv, hd_proj, cutoff, kv->nb[1], 0);
        sc = ggml_view_2d(ctx0, sc, hd_proj, cutoff, sc->nb[1], 0);
    }
    kv = ggml_reshape_3d(ctx0, ggml_cont(ctx0, kv), hd_proj, ratio, n_comp); // [hd_proj, within, block]
    sc = ggml_reshape_3d(ctx0, ggml_cont(ctx0, sc), hd_proj, ratio, n_comp);

    // score += ape (ape: [hd_proj, within] broadcast over blocks)
    sc = ggml_add(ctx0, sc, ggml_reshape_3d(ctx0, ape, hd_proj, ratio, 1));

    int64_t       win    = ratio; // positions pooled per block
    ggml_tensor * kv_blk = kv;
    ggml_tensor * sc_blk = sc;
    if (overlap) {
        // split the 2*head_dim projection into the "a" (previous-block) and "b" (current-block) halves
        ggml_tensor * kv_a = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, hd, ratio, n_comp, kv->nb[1], kv->nb[2], 0));
        ggml_tensor * kv_b = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, hd, ratio, n_comp, kv->nb[1], kv->nb[2], hd * kv->nb[0]));
        ggml_tensor * sc_a = ggml_cont(ctx0, ggml_view_3d(ctx0, sc, hd, ratio, n_comp, sc->nb[1], sc->nb[2], 0));
        ggml_tensor * sc_b = ggml_cont(ctx0, ggml_view_3d(ctx0, sc, hd, ratio, n_comp, sc->nb[1], sc->nb[2], hd * sc->nb[0]));

        // shift the "a" halves one block forward (block n uses block n-1); block 0 gets a masked fill
        ggml_tensor * kv_zero = ggml_scale(ctx0, ggml_view_3d(ctx0, kv_a, hd, ratio, 1, kv_a->nb[1], kv_a->nb[2], 0), 0.0f);
        ggml_tensor * sc_ninf = ggml_scale_bias(ctx0, kv_zero, 0.0f, -INFINITY); // [hd, ratio, 1] all -inf
        ggml_tensor * kv_a_sh = kv_zero;
        ggml_tensor * sc_a_sh = sc_ninf;
        if (n_comp > 1) {
            ggml_tensor * kv_a_prev = ggml_cont(ctx0, ggml_view_3d(ctx0, kv_a, hd, ratio, n_comp - 1, kv_a->nb[1], kv_a->nb[2], 0));
            ggml_tensor * sc_a_prev = ggml_cont(ctx0, ggml_view_3d(ctx0, sc_a, hd, ratio, n_comp - 1, sc_a->nb[1], sc_a->nb[2], 0));
            kv_a_sh = ggml_concat(ctx0, kv_zero, kv_a_prev, 2); // [hd, ratio, n_comp]
            sc_a_sh = ggml_concat(ctx0, sc_ninf, sc_a_prev, 2);
        }

        // concat along the within-axis: [a-half (prev block) | b-half (current block)] -> 2*ratio
        kv_blk = ggml_concat(ctx0, kv_a_sh, kv_b, 1); // [hd, 2*ratio, n_comp]
        sc_blk = ggml_concat(ctx0, sc_a_sh, sc_b, 1);
        win = 2 * ratio;
    }

    // softmax over the within axis, then weighted sum of kv over it (ggml_soft_max/sum_rows act on ne0)
    ggml_tensor * sc_p = ggml_cont(ctx0, ggml_permute(ctx0, sc_blk, 1, 0, 2, 3)); // [win, hd, block]
    ggml_tensor * kv_p = ggml_cont(ctx0, ggml_permute(ctx0, kv_blk, 1, 0, 2, 3)); // [win, hd, block]
    ggml_tensor * w    = ggml_soft_max(ctx0, sc_p);                               // softmax over `win`
    ggml_tensor * pooled = ggml_sum_rows(ctx0, ggml_mul(ctx0, kv_p, w));          // [1, hd, block]
    pooled = ggml_reshape_2d(ctx0, pooled, hd, n_comp);                           // [hd, block]
    GGML_UNUSED(win);

    // RMS-norm over head_dim, then rope on the trailing rope dims at block-start positions
    pooled = ggml_rms_norm(ctx0, pooled, hparams.f_norm_rms_eps);
    pooled = ggml_mul(ctx0, pooled, norm_w);
    pooled = ggml_reshape_3d(ctx0, pooled, hd, 1, n_comp);                         // [hd, 1, block]
    pooled = rope_tail(pooled, comp_rcos, comp_rsin, false);                       // [hd, 1, block]
    return ggml_reshape_2d(ctx0, pooled, hd, n_comp);                              // [hd, block]
}

// sparse-attention indexer top-k selection (stream-batched). Builds index_score[block, seq_token, stream]
// from the cached indexer-KV blocks (kvc [idx_hd, n_comp, n_stream]), causally masks it, takes the
// per-query top-k blocks, and returns an additive [n_win+n_comp, n_seq_tokens, 1, n_stream] mask
// (0 over window + selected blocks, -inf elsewhere). Hadamard rotation omitted (orthonormal -> cancels).
// host_mask is [n_keys, n_seq_tokens, 1, n_stream]; n_seq_tokens/n_stream are read from it.
ggml_tensor * llama_model_deepseek4::graph::build_indexer_mask(ggml_tensor * qr, ggml_tensor * x_query,
        ggml_tensor * kvc, const llama_layer & layer, int64_t n_comp, int64_t n_win,
        int64_t topk, ggml_tensor * rcos_tok, ggml_tensor * rsin_tok, ggml_tensor * host_mask) {
    const int64_t idx_nh = hparams.indexer_n_head;
    const int64_t idx_hd = hparams.indexer_head_size;
    const int64_t n_st   = host_mask->ne[1];   // n_seq_tokens
    const int64_t n_str  = host_mask->ne[3];   // n_stream
    const float   wscale = (1.0f / sqrtf((float) idx_hd)) * (1.0f / sqrtf((float) idx_nh));

    // q: wq_b_idx(qr) -> [idx_hd, idx_nh, n_tokens] roped (flat), then split per stream:
    // [idx_hd, idx_nh*n_seq_tokens, n_stream] (column m = h + t*idx_nh; stream-major tokens)
    ggml_tensor * q = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);     // [idx_nh*idx_hd, n_tokens]
    q = ggml_reshape_3d(ctx0, q, idx_hd, idx_nh, n_tokens);
    q = rope_tail(q, rcos_tok, rsin_tok, false);                         // [idx_hd, idx_nh, n_tokens]
    q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q), idx_hd, idx_nh * n_st, n_str);

    // index_score[b,h,t,s] = relu(<kvc[:,b,s], q[:,h,t,s]>) * weight[h,t,s], summed over heads
    ggml_tensor * score = ggml_relu(ctx0, ggml_mul_mat(ctx0, kvc, q));   // [n_comp, idx_nh*n_st, n_str]
    score = ggml_reshape_4d(ctx0, score, n_comp, idx_nh, n_st, n_str);
    ggml_tensor * w = ggml_scale(ctx0, ggml_mul_mat(ctx0, layer.indexer_proj, x_query), wscale); // [idx_nh, n_tokens]
    w = ggml_reshape_4d(ctx0, w, 1, idx_nh, n_st, n_str);
    score = ggml_mul(ctx0, score, w);                                   // [n_comp, idx_nh, n_st, n_str]
    ggml_tensor * idx_score = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3))); // [1, n_comp, n_st, n_str]
    idx_score = ggml_reshape_3d(ctx0, idx_score, n_comp, n_st, n_str);  // [n_comp, n_st, n_str]

    // causally pre-mask blocks (reuse host_mask's compressed-block region) before top-k
    ggml_tensor * causal = ggml_cont(ctx0, ggml_view_3d(ctx0, host_mask, n_comp, n_st, n_str,
                                          host_mask->nb[1], host_mask->nb[3], n_win * host_mask->nb[0]));
    idx_score = ggml_add(ctx0, idx_score, causal);                      // [n_comp, n_st, n_str]

    // deterministic top-k tie-break: lower block index wins among equal scores
    ggml_tensor * ramp = ggml_scale(ctx0, ggml_arange(ctx0, 0.0f, (float) n_comp, 1.0f), -1e-6f);
    idx_score = ggml_add(ctx0, idx_score, ggml_reshape_3d(ctx0, ramp, n_comp, 1, 1)); // broadcast over t,s

    // top-k per (seq_token, stream); scatter 0 (selected) into a -inf [n_comp, n_st, n_str] tensor
    ggml_tensor * top  = ggml_top_k(ctx0, idx_score, topk);             // [topk, n_st, n_str] I32
    ggml_tensor * neg  = ggml_fill(ctx0, ggml_reshape_4d(ctx0, idx_score, 1, n_comp, n_st, n_str), -INFINITY);
    ggml_tensor * zero = ggml_scale(ctx0, ggml_cast(ctx0, top, GGML_TYPE_F32), 0.0f); // [topk, n_st, n_str]
    zero = ggml_reshape_4d(ctx0, zero, 1, topk, n_st, n_str);           // [1, topk, n_st, n_str]
    ggml_tensor * sel  = ggml_set_rows(ctx0, neg, zero, top);           // [1, n_comp, n_st, n_str]
    sel = ggml_reshape_3d(ctx0, sel, n_comp, n_st, n_str);             // [n_comp, n_st, n_str]

    // window region never indexer-masked (0); concat -> [n_keys, n_st, n_str], then reshape to host_mask layout
    ggml_tensor * win_zero = ggml_fill(ctx0, ggml_cont(ctx0, ggml_view_3d(ctx0, host_mask, n_win, n_st, n_str,
                                          host_mask->nb[1], host_mask->nb[3], 0)), 0.0f);
    ggml_tensor * full = ggml_concat(ctx0, win_zero, sel, 0);           // [n_keys, n_st, n_str]
    return ggml_reshape_4d(ctx0, full, n_win + n_comp, n_st, 1, n_str); // match host_mask
}

// reduce hc_mult residual streams -> 1, emitting post/comb (Sinkhorn-normalized) for the expand.
// streams: [d, hc, n]; fn: [hc*d, mix_hc]; base: [mix_hc]; scale: [3].
ggml_tensor * llama_model_deepseek4::graph::hc_reduce(ggml_tensor * streams, ggml_tensor * fn, ggml_tensor * base,
        ggml_tensor * scale, ggml_tensor ** post_out, ggml_tensor ** comb_out, int il) {
    GGML_UNUSED(il);
    const int64_t d   = streams->ne[0];
    const int64_t hc  = streams->ne[1];
    const int64_t n   = streams->ne[2];
    const float   eps = hparams.hc_eps;

    ggml_tensor * eps_c = ggml_fill(ctx0, ggml_view_1d(ctx0, scale, 1, 0), eps); // [1] = hc_eps

    ggml_tensor * x_flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, streams), hc * d, n);
    ggml_tensor * x_norm = ggml_rms_norm(ctx0, x_flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes  = ggml_mul_mat(ctx0, fn, x_norm); // [mix_hc, n]

    const size_t nb0 = mixes->nb[0];
    ggml_tensor * m_pre  = ggml_view_2d(ctx0, mixes, hc,      n, mixes->nb[1], 0);
    ggml_tensor * m_post = ggml_view_2d(ctx0, mixes, hc,      n, mixes->nb[1], hc * nb0);
    ggml_tensor * m_comb = ggml_view_2d(ctx0, mixes, hc * hc, n, mixes->nb[1], 2 * hc * nb0);

    ggml_tensor * s0 = ggml_view_1d(ctx0, scale, 1, 0);
    ggml_tensor * s1 = ggml_view_1d(ctx0, scale, 1, scale->nb[0]);
    ggml_tensor * s2 = ggml_view_1d(ctx0, scale, 1, 2 * scale->nb[0]);
    ggml_tensor * b_pre  = ggml_view_1d(ctx0, base, hc,      0);
    ggml_tensor * b_post = ggml_view_1d(ctx0, base, hc,      hc * base->nb[0]);
    ggml_tensor * b_comb = ggml_view_1d(ctx0, base, hc * hc, 2 * hc * base->nb[0]);

    // pre = sigmoid(m_pre*s0 + b_pre) + eps
    ggml_tensor * pre = ggml_add(ctx0, ggml_mul(ctx0, m_pre, s0), b_pre);
    pre = ggml_add(ctx0, ggml_sigmoid(ctx0, pre), eps_c);

    // post = 2*sigmoid(m_post*s1 + b_post)
    ggml_tensor * post = ggml_add(ctx0, ggml_mul(ctx0, m_post, s1), b_post);
    post = ggml_scale(ctx0, ggml_sigmoid(ctx0, post), 2.0f);

    // comb[k,j,n] = m_comb*s2 + b_comb  (k = inner = softmax/col axis)
    ggml_tensor * comb = ggml_add(ctx0, ggml_mul(ctx0, m_comb, s2), b_comb);
    comb = ggml_reshape_3d(ctx0, comb, hc, hc, n); // [k, j, n]
    comb = ggml_add(ctx0, ggml_soft_max(ctx0, comb), eps_c); // softmax over k + eps
    // col-normalize: divide by sum over j (ne1)
    {
        ggml_tensor * cj  = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3)); // [j, k, n]
        ggml_tensor * sj  = ggml_add(ctx0, ggml_sum_rows(ctx0, cj), eps_c);        // [1, k, n]
        sj = ggml_cont(ctx0, ggml_permute(ctx0, sj, 1, 0, 2, 3));                  // [k, 1, n]
        comb = ggml_div(ctx0, comb, sj);
    }
    for (uint32_t it = 1; it < hparams.hc_sinkhorn_iters; ++it) {
        // row-normalize: sum over k (ne0)
        ggml_tensor * sk = ggml_add(ctx0, ggml_sum_rows(ctx0, comb), eps_c);       // [1, j, n]
        comb = ggml_div(ctx0, comb, sk);
        // col-normalize: sum over j (ne1)
        ggml_tensor * cj = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));  // [j, k, n]
        ggml_tensor * sj = ggml_add(ctx0, ggml_sum_rows(ctx0, cj), eps_c);         // [1, k, n]
        sj = ggml_cont(ctx0, ggml_permute(ctx0, sj, 1, 0, 2, 3));                  // [k, 1, n]
        comb = ggml_div(ctx0, comb, sj);
    }

    // y = sum_hc(pre[None,:,:] * streams)
    ggml_tensor * pre_r = ggml_reshape_3d(ctx0, pre, 1, hc, n);
    ggml_tensor * w     = ggml_mul(ctx0, streams, pre_r);                     // [d, hc, n]
    ggml_tensor * y     = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, w, 1, 0, 2, 3))); // [1, d, n]
    y = ggml_reshape_2d(ctx0, y, d, n);

    *post_out = post;
    *comb_out = comb;
    return y;
}

// expand sub-layer output back to hc_mult streams.
// sub: [d, n]; streams(residual): [d, hc, n]; post: [hc, n]; comb: [k, j, n].
ggml_tensor * llama_model_deepseek4::graph::hc_expand(ggml_tensor * sub, ggml_tensor * streams,
        ggml_tensor * post, ggml_tensor * comb) {
    const int64_t d  = streams->ne[0];
    const int64_t hc = streams->ne[1];
    const int64_t n  = streams->ne[2];

    // term1[d,k,n] = post[k,n] * sub[d,n]
    ggml_tensor * sub_big = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, sub, d, 1, n), d, hc, n, 1);
    ggml_tensor * term1   = ggml_mul(ctx0, sub_big, ggml_reshape_3d(ctx0, post, 1, hc, n)); // [d, hc, n]

    // term2[d,k,n] = sum_j comb[k,j,n] * streams[d,j,n]
    ggml_tensor * streams_t = ggml_cont(ctx0, ggml_permute(ctx0, streams, 1, 0, 2, 3)); // [j, d, n]
    ggml_tensor * comb_t    = ggml_cont(ctx0, ggml_permute(ctx0, comb,    1, 0, 2, 3)); // [j, k, n]
    ggml_tensor * term2     = ggml_mul_mat(ctx0, streams_t, comb_t);                    // [d, k, n]

    return ggml_add(ctx0, term1, term2); // [d, hc, n]
}

// final head reduction (sigmoid-gated, no Sinkhorn): hc_mult streams -> 1.
ggml_tensor * llama_model_deepseek4::graph::hc_head(ggml_tensor * streams, ggml_tensor * fn,
        ggml_tensor * base, ggml_tensor * scale) {
    const int64_t d  = streams->ne[0];
    const int64_t hc = streams->ne[1];
    const int64_t n  = streams->ne[2];

    ggml_tensor * eps_c  = ggml_fill(ctx0, ggml_view_1d(ctx0, scale, 1, 0), hparams.hc_eps);
    ggml_tensor * x_flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, streams), hc * d, n);
    ggml_tensor * x_norm = ggml_rms_norm(ctx0, x_flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes  = ggml_mul_mat(ctx0, fn, x_norm); // [hc, n]

    ggml_tensor * pre = ggml_add(ctx0, ggml_mul(ctx0, mixes, scale), base);
    pre = ggml_add(ctx0, ggml_sigmoid(ctx0, pre), eps_c); // [hc, n]

    ggml_tensor * w = ggml_mul(ctx0, streams, ggml_reshape_3d(ctx0, pre, 1, hc, n)); // [d, hc, n]
    ggml_tensor * y = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, w, 1, 0, 2, 3))); // [1, d, n]
    return ggml_reshape_2d(ctx0, y, d, n);
}

llama_model_deepseek4::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_head  = hparams.n_head();
    const int64_t hd      = hparams.n_embd_head_k();   // attention head dim (= rope/key/value length)
    const int64_t rd      = hparams.n_rot();           // rope_head_dim
    const int64_t rd2     = rd / 2;
    const int64_t hc      = hparams.hc_mult;
    const int64_t n_groups = hparams.n_o_groups;
    const int64_t n_lora_o = hparams.n_lora_o;
    const int64_t per_grp  = n_head * hd / n_groups;
    const float   kq_scale = 1.0f / sqrtf((float) hd);

    // Unified prefill + decode with an INCREMENTAL compressed-KV cache. Each token cell stores
    //   [ x (n_embd) | compressed block (head_dim) | indexer block (indexer_head_size) ].
    // x is written every step; the compressed/indexer blocks are computed once when each block
    // "completes" and stored, so decode reads cached blocks instead of recomputing the compressor
    // over all history. Keys = [ last `window` raw cells ++ all completed compressed blocks ].
    // Single-sequence: cache cell == absolute position.
    //
    // Reserve-safety: the worst-case (reserve) graph is built with all-zero positions, n_tokens =
    // n_ubatch and n_kv = kv_size. So tensor SHAPES must be maximal at reserve: window/block-read
    // are sized by n_kv (n_win = n_kv - w_start shrinks during decode), and the block COMPUTE/WRITE
    // is sized by n_tokens (n_blk_max). Runtime offsets/indices are set at set_input time.
    ggml_tensor * inpL    = build_inp_embd(model.tok_embd);     // [d, n_tokens]
    ggml_tensor * inp_pos = build_inp_pos();                    // [n_tokens] i32 (query positions)
    ggml_tensor * inp_tokens = res->get_inp_tokens();           // [n_tokens] i32 (for hash routing)
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx); // typed KV-cache context
    const int64_t n_kv = (int64_t) mctx_cur->get_n_kv();        // padded cache length (= kv_size at reserve)
    ggml_tensor * v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);  // store cells for current tokens
    {
        auto vinp = std::make_unique<llm_graph_input_ds4_vstore>(mctx_cur);
        vinp->v_idxs = v_idxs;
        res->add_input(std::move(vinp));
    }
    cb(inpL, "ds4_embed", -1);

    // expand embeddings to hc parallel residual streams: [d, hc, n_tokens]
    ggml_tensor * streams = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
                                           n_embd, hc, n_tokens, 1);

    // sliding-window key range: cells [w_start, n_kv). w_start is a runtime offset (0 at reserve, so
    // n_win = n_kv at reserve and shrinks to ~window during decode).
    llama_pos p0 = ubatch.pos[0];
    for (int64_t i = 1; i < n_tokens; ++i) {
        p0 = std::min(p0, ubatch.pos[i]);    // global earliest query position (over all streams)
    }
    const int64_t w_start = std::max((int64_t) 0, (int64_t) p0 - (int64_t) hparams.n_window + 1);
    const int64_t n_win   = n_kv - w_start;                                  // window keys (n_kv-based shape)

    // multi-sequence: each sequence is its own cache stream (non-unified). The attention carries a
    // stream axis (ne3); HC/FFN stay flat over n_tokens. Tokens are stream-major in the ubatch:
    // token (stream s, seq-token t) at flat index s*n_seq_tokens + t.
    const int64_t n_stream     = cparams.kv_unified ? 1 : std::max<int64_t>(1, (int64_t) ubatch.n_seqs_unq);
    const int64_t n_seq_tokens = n_tokens / n_stream;
    const int64_t kv_size      = (int64_t) cparams.n_ctx_seq;                // per-stream cell count (cache stride)
    const int64_t s0           = (int64_t) mctx_cur->get_s0();               // first cache stream of this ubatch

    ggml_tensor * pos_q = ggml_cast(ctx0, inp_pos, GGML_TYPE_F32);           // query positions [n_tokens]
    ggml_tensor * pos_k = ggml_arange(ctx0, (float) w_start, (float) n_kv, 1.0f); // window cell positions [n_win]

    // outer(freqs[rd2], positions[np]) -> angles[rd2, np] -> (cos, sin)
    auto make_cossin = [&](ggml_tensor * fr, ggml_tensor * pos, int64_t np,
                           ggml_tensor ** oc, ggml_tensor ** os) {
        ggml_tensor * ang = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, fr, 1, rd2),
                                               ggml_reshape_2d(ctx0, pos, 1, np)); // [rd2, np]
        *oc = ggml_cos(ctx0, ang);
        *os = ggml_sin(ctx0, ang);
    };

    // window-layer rope (theta = freq_base, no YaRN): q at query positions, k at window cell positions
    ggml_tensor * ar    = ggml_arange(ctx0, 0.0f, (float) rd, 2.0f);                        // [rd2]
    ggml_tensor * freqs = ggml_exp(ctx0, ggml_scale(ctx0, ar, -logf((float) freq_base) / (float) rd));
    ggml_tensor * rcos_q = nullptr, * rsin_q = nullptr, * rcos_k = nullptr, * rsin_k = nullptr;
    make_cossin(freqs, pos_q, n_tokens, &rcos_q, &rsin_q);
    make_cossin(freqs, pos_k, n_win,    &rcos_k, &rsin_k);

    // KV-compressed-layer rope (theta = freq_base_compress + YaRN). `freqs_c_yarn` is the per-dim
    // frequency vector; compressed-block rope angles (block-start positions) are built per-layer.
    bool any_compress = false;
    for (int il = 0; il < n_layer; ++il) {
        if (hparams.compress_ratios[il] > 0) { any_compress = true; break; }
    }
    ggml_tensor * freqs_c_yarn = nullptr;
    ggml_tensor * rcos_qc = nullptr, * rsin_qc = nullptr, * rcos_kc = nullptr, * rsin_kc = nullptr;
    if (any_compress) {
        constexpr float PI     = 3.14159265358979323846f;
        const float base_c = hparams.rope_freq_base_compress;
        const float factor = hparams.rope_freq_scale_train != 0.0f ? 1.0f / hparams.rope_freq_scale_train : 1.0f;
        const float orig   = (float) hparams.n_ctx_orig_yarn;
        auto corr_dim = [&](float num_rot) {
            return (float) rd * logf(orig / (num_rot * 2.0f * PI)) / (2.0f * logf(base_c));
        };
        float low  = floorf(corr_dim(hparams.yarn_beta_fast));
        float high = ceilf (corr_dim(hparams.yarn_beta_slow));
        low  = low  < 0.0f          ? 0.0f          : low;
        high = high > (float)(rd-1) ? (float)(rd-1) : high;
        if (low == high) { high += 0.001f; }

        // smooth(k) = 1 - clamp((k - low)/(high - low), 0, 1);  mult = smooth + (1-smooth)/factor
        ggml_tensor * freqs_c = ggml_exp(ctx0, ggml_scale(ctx0, ar, -logf(base_c) / (float) rd)); // [rd2]
        ggml_tensor * kidx    = ggml_arange(ctx0, 0.0f, (float) rd2, 1.0f);                        // [rd2]
        ggml_tensor * ramp    = ggml_clamp(ctx0, ggml_scale_bias(ctx0, kidx, 1.0f/(high-low), -low/(high-low)), 0.0f, 1.0f);
        ggml_tensor * smooth  = ggml_scale_bias(ctx0, ramp, -1.0f, 1.0f);                          // 1 - ramp
        ggml_tensor * mult    = ggml_scale_bias(ctx0, smooth, 1.0f - 1.0f/factor, 1.0f/factor);
        freqs_c_yarn = ggml_mul(ctx0, freqs_c, mult);                                              // [rd2]
        make_cossin(freqs_c_yarn, pos_q, n_tokens, &rcos_qc, &rsin_qc);
        make_cossin(freqs_c_yarn, pos_k, n_win,    &rcos_kc, &rsin_kc);
    }

    const int64_t idx_hd = hparams.indexer_head_size;

    // Compute the `n_blk_max` compressed/indexer blocks for ONE stream, from that stream's cached x
    // (xcs [n_embd, n_kv]), ending at block (bstart + n_blk_max). Returns [out_dim, n_blk_max] (pooled,
    // normed, roped). Emits one extra block (a carry predecessor for overlap, or a trailing overshoot
    // at the start) which is then dropped. build_compressor detects overlap from the ape width.
    auto compute_blocks = [&](ggml_tensor * xcs, int64_t cr, ggml_tensor * w_kv, ggml_tensor * w_gate,
                              ggml_tensor * norm_w, ggml_tensor * ape, int64_t out_dim,
                              int64_t n_blk_max, int64_t bstart) -> ggml_tensor * {
        const int64_t slice_blk  = n_blk_max + 1;
        const int64_t off_blk    = std::max((int64_t) 0, bstart - 1);
        const int64_t drop_first = bstart > 0 ? 1 : 0;                  // drop carry (front) or overshoot (back)
        const int64_t slice_len  = slice_blk * cr;
        ggml_tensor * xs = ggml_cont(ctx0, ggml_view_2d(ctx0, xcs, n_embd, slice_len,
                                                        xcs->nb[1], off_blk * cr * xcs->nb[1]));
        ggml_tensor * bpos = ggml_arange(ctx0, (float) (off_blk * cr), (float) ((off_blk + slice_blk) * cr), (float) cr);
        ggml_tensor * brc = nullptr, * brs = nullptr;
        make_cossin(freqs_c_yarn, bpos, slice_blk, &brc, &brs);
        ggml_tensor * blk = build_compressor(xs, w_kv, w_gate, norm_w, ape, cr, brc, brs); // [out_dim, slice_blk]
        return ggml_cont(ctx0, ggml_view_2d(ctx0, blk, out_dim, n_blk_max, blk->nb[1], drop_first * blk->nb[1]));
    };

    // Per-stream block compute + ONE batched write into the V cells. Returns the post-write region
    // tensor [out_dim, kv_size*n_stream] (or the committed region if n_blk_max<=0). For each stream s
    // we (re)write the last n_blk_max blocks ending at its completed-block count (re-writes a few
    // already-stored blocks idempotently plus the genuinely new ones). Reserve-safe: n_blk_max is
    // n_tokens-based (max at reserve, where positions are 0); it is emitted unconditionally (not gated
    // on the runtime block count) so the reserve graph is never smaller than a real decode graph.
    auto store_blocks_ms = [&](ggml_tensor * vb, ggml_tensor * x_cache_3d, int64_t cr,
                               ggml_tensor * w_kv, ggml_tensor * w_gate, ggml_tensor * norm_w, ggml_tensor * ape,
                               int64_t out_dim, int64_t region_off, int64_t n_blk_max) -> ggml_tensor * {
        const int64_t ncells = vb->ne[1];
        ggml_tensor * region = ggml_view_2d(ctx0, vb, out_dim, ncells, vb->nb[1], region_off * vb->nb[0]);
        if (n_blk_max <= 0) {
            return region; // cache too small to hold any block (read is fully masked)
        }
        std::vector<int64_t> bases;
        ggml_tensor * blk_all = nullptr;
        for (int64_t s = 0; s < n_stream; ++s) {
            llama_pos pmax_s = ubatch.pos[s * n_seq_tokens];
            for (int64_t t = 1; t < n_seq_tokens; ++t) {
                pmax_s = std::max(pmax_s, ubatch.pos[s * n_seq_tokens + t]);
            }
            const int64_t n_comp_act_s = ((int64_t) pmax_s + 1) / cr;  // stream s's completed blocks
            const int64_t bstart_s     = std::max((int64_t) 0, n_comp_act_s - n_blk_max);
            ggml_tensor * xcs   = ggml_view_2d(ctx0, x_cache_3d, n_embd, n_kv, x_cache_3d->nb[1], s * x_cache_3d->nb[2]);
            ggml_tensor * blk_s = compute_blocks(xcs, cr, w_kv, w_gate, norm_w, ape, out_dim, n_blk_max, bstart_s);
            blk_s = ggml_reshape_3d(ctx0, blk_s, out_dim, n_blk_max, 1);
            blk_all = blk_all ? ggml_concat(ctx0, blk_all, blk_s, 2) : blk_s; // [out_dim, n_blk_max, n_stream]
            bases.push_back((s0 + s) * kv_size + bstart_s);                   // global cell base for cache stream (s0+s)
        }
        blk_all = ggml_reshape_2d(ctx0, ggml_cont(ctx0, blk_all), out_dim, n_blk_max * n_stream);
        ggml_tensor * idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_blk_max * n_stream);
        ggml_set_input(idxs);
        {
            auto bi = std::make_unique<llm_graph_input_ds4_blkidx>(std::move(bases), n_blk_max);
            bi->idxs = idxs;
            res->add_input(std::move(bi));
        }
        ggml_tensor * w = ggml_set_rows(ctx0, region, blk_all, idxs); // [out_dim, ncells] post-write
        ggml_build_forward_expand(gf, w);
        return w;
    };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        const int64_t cr          = hparams.compress_ratios[il];
        const bool    compressed  = cr > 0;
        const bool    has_indexer = (cr == 4);
        const bool    is_hash     = (uint32_t) il < hparams.n_hash_layers;
        // compressed layers rope q/kv with the YaRN'd compressed-theta angles
        ggml_tensor * rc_q = compressed ? rcos_qc : rcos_q;
        ggml_tensor * rs_q = compressed ? rsin_qc : rsin_q;
        ggml_tensor * rc_k = compressed ? rcos_kc : rcos_k;
        ggml_tensor * rs_k = compressed ? rsin_kc : rsin_k;

        // ---- attention sublayer ----
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;
        ggml_tensor * residual = streams;
        ggml_tensor * x = hc_reduce(streams, layer.hc_attn_fn, layer.hc_attn_base, layer.hc_attn_scale, &post, &comb, il);
        x = build_norm(x, layer.attn_norm, NULL, LLM_NORM_RMS, il); // x = current tokens' attn input [d, n_tokens]

        // store x into the cache x-region (cells = v_idxs, global), then read it back per stream as
        // [n_embd, n_kv, n_stream]. Reads from the set_rows result so the read sees this write (data dep).
        ggml_tensor * vb       = mctx_cur->get_v_base(ctx0, il);                 // [v_width, kv_size*n_stream]
        ggml_tensor * x_region = ggml_view_2d(ctx0, vb, n_embd, vb->ne[1], vb->nb[1], 0);
        ggml_tensor * write_x  = ggml_set_rows(ctx0, x_region, x, v_idxs);       // [n_embd, ncells] post-write
        ggml_build_forward_expand(gf, write_x);
        ggml_tensor * x_cache  = ggml_view_3d(ctx0, write_x, n_embd, n_kv, n_stream,
                                              write_x->nb[1], kv_size * write_x->nb[1],
                                              s0 * kv_size * write_x->nb[1]); // [n_embd, n_kv, n_stream] @ stream s0

        ggml_tensor * attn_out;
        {
            // Q (current tokens): wq_a -> q_a_norm -> wq_b -> [hd, n_head, n_tokens] -> per-head RMS -> rope (flat),
            // then split into streams -> [hd, n_seq_tokens, n_head, n_stream]
            ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, x);
            q = build_norm(q, layer.attn_q_a_norm, NULL, LLM_NORM_RMS, il);
            ggml_tensor * qr = q;                                           // low-rank q (shared with indexer)
            q = ggml_mul_mat(ctx0, layer.wq_b, q);                          // [n_head*hd, n_tokens]
            q = ggml_reshape_3d(ctx0, q, hd, n_head, n_tokens);
            q = ggml_rms_norm(ctx0, q, hparams.f_norm_rms_eps);            // per-head RMS (no weight)
            q = rope_tail(q, rc_q, rs_q, false);                           // [hd, n_head, n_tokens]
            q = ggml_reshape_4d(ctx0, q, hd, n_head, n_seq_tokens, n_stream);
            ggml_tensor * q_p = ggml_permute(ctx0, q, 0, 2, 1, 3);         // [hd, n_seq_tokens, n_head, n_stream]

            // window (raw) KV latent per stream from cells [w_start, n_kv): [hd, n_win, n_stream]
            ggml_tensor * x_win = ggml_cont(ctx0, ggml_view_3d(ctx0, x_cache, n_embd, n_win, n_stream,
                                              x_cache->nb[1], x_cache->nb[2], w_start * x_cache->nb[1]));
            ggml_tensor * kvw = ggml_mul_mat(ctx0, layer.wkv_a_mqa, x_win); // [hd, n_win, n_stream]
            kvw = build_norm(kvw, layer.attn_kv_a_norm, NULL, LLM_NORM_RMS, il);
            // rope by window cell position (shared across streams): arrange [hd, n_stream, n_win] so rope indexes n_win
            kvw = ggml_cont(ctx0, ggml_permute(ctx0, kvw, 0, 2, 1, 3));     // [hd, n_stream, n_win]
            kvw = rope_tail(kvw, rc_k, rs_k, false);                       // [hd, n_stream, n_win]
            kvw = ggml_cont(ctx0, ggml_permute(ctx0, kvw, 0, 2, 1, 3));     // [hd, n_win, n_stream]

            ggml_tensor * mask = nullptr;
            ggml_tensor * kv;                                              // [hd, n_keys, n_stream]
            const int64_t n_comp = compressed ? n_kv / cr : 0;             // blocks read (n_kv-based; masked)
            if (n_comp > 0) {
                const int64_t n_blk_max = std::min(n_tokens / cr + 2, n_comp - 1); // compute/write shape (n_tokens-based)
                ggml_tensor * comp_region = store_blocks_ms(vb, x_cache, cr,
                        layer.compressor_kv, layer.compressor_gate, layer.compressor_norm, layer.compressor_ape,
                        hd, n_embd, n_blk_max);
                ggml_tensor * comp_kv = ggml_cont(ctx0, ggml_view_3d(ctx0, comp_region, hd, n_comp, n_stream,
                                                  comp_region->nb[1], kv_size * comp_region->nb[1],
                                                  s0 * kv_size * comp_region->nb[1])); // [hd, n_comp, n_stream] @ stream s0
                kv = ggml_concat(ctx0, kvw, comp_kv, 1);                  // [hd, n_win+n_comp, n_stream]

                auto inp = std::make_unique<llm_graph_input_ds4_mask>(hparams.n_window, cr, n_comp, w_start, n_seq_tokens, n_stream);
                inp->mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_win + n_comp, n_seq_tokens, 1, n_stream);
                ggml_set_input(inp->mask);
                mask = inp->mask;
                res->add_input(std::move(inp));

                if (has_indexer) {
                    const int64_t topk = (int64_t) hparams.indexer_top_k < n_comp ? (int64_t) hparams.indexer_top_k : n_comp;
                    ggml_tensor * idx_region = store_blocks_ms(vb, x_cache, cr,
                            layer.idx_compressor_kv, layer.idx_compressor_gate, layer.idx_compressor_norm,
                            layer.idx_compressor_ape, idx_hd, n_embd + hd, n_blk_max);
                    ggml_tensor * idx_kv = ggml_cont(ctx0, ggml_view_3d(ctx0, idx_region, idx_hd, n_comp, n_stream,
                                                     idx_region->nb[1], kv_size * idx_region->nb[1],
                                                     s0 * kv_size * idx_region->nb[1])); // [idx_hd, n_comp, n_stream] @ stream s0
                    ggml_tensor * sel = build_indexer_mask(qr, x, idx_kv, layer, n_comp, n_win, topk, rc_q, rs_q, mask);
                    mask = ggml_add(ctx0, mask, sel);
                }
            } else {
                kv = kvw;                                                 // pure sliding-window attention
                auto inp = std::make_unique<llm_graph_input_ds4_mask>(hparams.n_window, 0, 0, w_start, n_seq_tokens, n_stream);
                inp->mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_win, n_seq_tokens, 1, n_stream);
                ggml_set_input(inp->mask);
                mask = inp->mask;
                res->add_input(std::move(inp));
            }

            // batched attention over streams (ne3). k = [hd, 1, n_keys, n_stream] (single KV head)
            const int64_t n_keys = kv->ne[1];
            ggml_tensor * k   = ggml_reshape_4d(ctx0, kv, hd, 1, n_keys, n_stream);
            ggml_tensor * k_p = ggml_permute(ctx0, k, 0, 2, 1, 3);        // [hd, n_keys, 1, n_stream]
            ggml_tensor * kq  = ggml_mul_mat(ctx0, k_p, q_p);            // [n_keys, n_seq_tokens, n_head, n_stream]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx0, kq, mask, kq_scale, 0.0f);
            ggml_soft_max_add_sinks(kq, layer.attn_sinks);

            ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, k_p)); // [n_keys, hd, 1, n_stream]
            ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);              // [hd, n_seq_tokens, n_head, n_stream]
            ggml_tensor * o = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3)); // [hd, n_head, n_seq_tokens, n_stream]
            o = ggml_reshape_3d(ctx0, o, hd, n_head, n_tokens);          // back to flat tokens (stream-major)
            o = rope_tail(o, rc_q, rs_q, true);                          // inverse rope on output (query positions)

            // grouped low-rank output projection (flat over n_tokens)
            o = ggml_reshape_3d(ctx0, ggml_reshape_2d(ctx0, o, hd * n_head, n_tokens), per_grp, n_groups, n_tokens);
            o = ggml_cont(ctx0, ggml_permute(ctx0, o, 0, 2, 1, 3));       // [per_grp, n_tokens, n_groups]
            ggml_tensor * wo_a = ggml_reshape_3d(ctx0, layer.wo_a, per_grp, n_lora_o, n_groups);
            ggml_tensor * oa = ggml_mul_mat(ctx0, wo_a, o);              // [o_lora, n_tokens, n_groups]
            oa = ggml_cont(ctx0, ggml_permute(ctx0, oa, 0, 2, 1, 3));    // [o_lora, n_groups, n_tokens]
            oa = ggml_reshape_2d(ctx0, oa, n_lora_o * n_groups, n_tokens);
            attn_out = ggml_mul_mat(ctx0, layer.wo, oa);                 // [d, n_tokens]
        }
        cb(attn_out, "ds4_attn", il);
        streams = hc_expand(attn_out, residual, post, comb);             // [d, hc, n_tokens]

        // ---- ffn sublayer (score-routed MoE) ----
        residual = streams;
        x = hc_reduce(streams, layer.hc_ffn_fn, layer.hc_ffn_base, layer.hc_ffn_scale, &post, &comb, il);
        x = build_norm(x, layer.ffn_norm, NULL, LLM_NORM_RMS, il);

        ggml_tensor * ffn_out;
        {
            // hash layers: experts chosen by a token->expert table; weights still from the gate scores
            ggml_tensor * sel  = nullptr;
            ggml_tensor * wmul = nullptr;
            ggml_tensor * exp_b = layer.ffn_exp_probs_b;
            if (is_hash) {
                ggml_tensor * t2e = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens); // [n_used, n] f32
                sel   = ggml_cast(ctx0, t2e, GGML_TYPE_I32);
                exp_b = nullptr; // hash gating has no selection bias
                // dedup mask (drop all but the last occurrence of a duplicated expert)
                auto inp = std::make_unique<llm_graph_input_ds4_hash>(layer.ffn_gate_tid2eid);
                inp->mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_expert_used, n_tokens);
                ggml_set_input(inp->mask);
                wmul = inp->mask;
                res->add_input(std::move(inp));
            }
            ggml_tensor * moe = build_moe_ffn(x,
                layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
                exp_b, n_expert, n_expert_used, LLM_FFN_SILU,
                hparams.expert_weights_norm, hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func, il,
                /*probs_in*/ nullptr, /*gate_up_exps*/ nullptr,
                /*up_exps_s*/ nullptr, /*gate_exps_s*/ nullptr, /*down_exps_s*/ nullptr,
                /*selected_experts_in*/ sel, /*weights_mul*/ wmul);

            // shared expert: clamped SwiGLU (silu(clamp(gate,max=L)) * clamp(up,-L,L))
            const float L = hparams.f_swiglu_limit;
            ggml_tensor * g = ggml_mul_mat(ctx0, layer.ffn_gate_shexp, x);
            ggml_tensor * u = ggml_mul_mat(ctx0, layer.ffn_up_shexp,   x);
            if (L > 0.0f) {
                g = ggml_clamp(ctx0, g, -INFINITY, L);
                u = ggml_clamp(ctx0, u, -L, L);
            }
            ggml_tensor * sh = ggml_mul(ctx0, ggml_silu(ctx0, g), u);
            sh = ggml_mul_mat(ctx0, layer.ffn_down_shexp, sh);
            ffn_out = ggml_add(ctx0, moe, sh);
        }
        cb(ffn_out, "ds4_ffn", il);
        streams = hc_expand(ffn_out, residual, post, comb);
        cb(streams, "ds4_l_out", il);
    }

    // final hyper-connection head reduction + output norm + lm_head
    ggml_tensor * cur = hc_head(streams, model.hc_head_fn, model.hc_head_base, model.hc_head_scale); // [d, n]
    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
