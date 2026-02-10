#include "models.h"

llm_build_deepseek2::llm_build_deepseek2(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const bool is_mla = hparams.is_mla();

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot;
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    // And also: https://github.com/ggml-org/llama.cpp/pull/17945 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]

    // first cancel the adjustment from llama_hparams::yarn_attn_factor_adjust to get the original attn_factor
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    // use the original attn_factor to pre-scale the kq_scale
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    // (optional) temperature tuning - used by mistral-large
    ggml_tensor * inp_attn_scale = nullptr;
    if (hparams.f_attn_temp_scale != 0.0f) {
        inp_attn_scale = build_inp_attn_scale();
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn_kv = !is_mla ? build_attn_inp_kv() : nullptr;
    auto * inp_attn_k  = is_mla ? build_attn_inp_k() : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * q = NULL;

            const bool is_lite = model.layers[il].wq;

            if (!is_lite) {
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q, "q", il);

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
            }
            // split into {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            // and {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // and {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            if (is_mla) {
                // {n_embd_head_qk_nope, n_tokens, n_head}
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                // {kv_lora_rank, n_head, n_tokens}
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
                // note: rope must go first for in-place context shifting in build_rope_shift()
                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                cb(Kcur, "Kcur", il);

                // {kv_lora_rank, 1, n_tokens}
                ggml_tensor * Vcur = kv_cmpr;
                cb(Vcur, "Vcur", il);

                if (inp_attn_scale) {
                    // apply llama 4 temperature scaling
                    Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                    cb(Qcur, "Qcur_attn_temp_scaled", il);
                }

                // DeepSeek Sparse Attention (DSA) - experimental
                // When DSA is enabled, use sparse attention with top-k token selection
                if (params.cparams.use_dsa && model.layers[il].wq_idx_a) {
                    // ========================================
                    // Step 1: Compute Indexer Queries
                    // ========================================
                    // compressed_q = wq_idx_a @ cur (hidden states)
                    ggml_tensor * compressed_q = ggml_mul_mat(ctx0, model.layers[il].wq_idx_a, cur);
                    cb(compressed_q, "compressed_q", il);

                    // Apply layer norm
                    compressed_q = build_norm(compressed_q, model.layers[il].dsa_idx_norm, nullptr, LLM_NORM_RMS, il);
                    cb(compressed_q, "compressed_q_norm", il);

                    // Project to indexer heads
                    ggml_tensor * q_idx                  = ggml_mul_mat(ctx0, model.layers[il].wq_idx_b, compressed_q);
                    // Shape: [n_tokens, n_dsa_indexer_heads * n_dsa_indexer_head_dim]
                    const int64_t n_dsa_indexer_heads    = hparams.n_dsa_indexer_heads;
                    const int64_t n_dsa_indexer_head_dim = hparams.n_dsa_indexer_head_dim;
                    q_idx = ggml_reshape_3d(ctx0, q_idx, n_dsa_indexer_head_dim, n_dsa_indexer_heads, n_tokens);
                    cb(q_idx, "q_idx", il);

                    // ========================================
                    // Step 2: Compute Attention Scores (using existing KV cache)
                    // ========================================
                    // NOTE: For proper DSA, the indexer should access a FULL KV cache
                    // with all past tokens in context, not just current batch tokens.
                    // This prototype uses the existing K_cur from inp_attn_k for simplicity.
                    // To implement full DSA, you would need a separate KV cache that persists
                    // across batches and is accessible to the indexer.

                    // Use K_cur from inp_attn_k (contains current batch KV)
                    // For demonstration, we treat this as the indexer cache
                    ggml_tensor * kv_indexer = Kcur;
                    cb(kv_indexer, "kv_indexer", il);

                    // Reshape for matmul: [kv_lora_rank, n_tokens]
                    ggml_tensor * kv_indexer_2d = ggml_reshape_2d(ctx0, kv_indexer, kv_lora_rank, n_tokens);
                    cb(kv_indexer_2d, "kv_indexer_2d", il);

                    // Transpose q_idx for matmul: [n_tokens, n_dsa_indexer_heads * n_dsa_indexer_head_dim]
                    ggml_tensor * q_idx_perm = ggml_permute(ctx0, q_idx, 0, 2, 1, 3);
                    cb(q_idx_perm, "q_idx_perm", il);

                    // Compute logits: [n_dsa_indexer_heads, n_tokens, n_tokens]
                    ggml_tensor * logits = ggml_mul_mat(ctx0, kv_indexer_2d, q_idx_perm);
                    cb(logits, "idx_logits", il);

                    // Apply ReLU activation
                    logits = ggml_relu(ctx0, logits);
                    cb(logits, "idx_logits_relu", il);

                    // ========================================
                    // Step 3: Apply Head Weights and Sum
                    // ========================================
                    ggml_tensor * idx_weights_expanded =
                        ggml_reshape_3d(ctx0, model.layers[il].dsa_idx_weights, 1, 1, n_dsa_indexer_heads);
                    ggml_tensor * idx_weights_perm = ggml_permute(ctx0, idx_weights_expanded, 2, 0, 1, 3);
                    cb(idx_weights_perm, "idx_weights_expanded", il);

                    logits = ggml_mul(ctx0, logits, idx_weights_perm);
                    cb(logits, "idx_logits_weighted", il);

                    // Sum over indexer heads
                    logits = ggml_sum_rows(ctx0, logits);
                    // Result is [1, n_tokens, n_tokens], need to squeeze
                    logits = ggml_reshape_2d(ctx0, logits, n_tokens, n_tokens);
                    cb(logits, "idx_logits_sum", il);

                    // ========================================
                    // Step 4: Top-K Selection
                    // ========================================
                    const int top_k        = hparams.n_dsa_topk;
                    const int actual_top_k = std::min(top_k, (int) n_tokens);

                    ggml_tensor * topk_indices = ggml_top_k(ctx0, logits, actual_top_k);
                    cb(topk_indices, "topk_indices", il);
                    // topk_indices shape: [actual_top_k, n_tokens] (int32)

                    // ========================================
                    // Step 5: Gather Selected KV
                    // ========================================
                    // Use V_cur from inp_attn_k for values
                    // K_cur shape: [n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens]
                    // We need to reshape to [n_embd_head, n_tokens] for get_rows
                    ggml_tensor * k_cache_2d = ggml_reshape_2d(ctx0, Kcur, Kcur->ne[0], n_tokens);
                    cb(k_cache_2d, "k_cache_2d", il);

                    ggml_tensor * v_cache_2d = ggml_reshape_2d(ctx0, Vcur, Vcur->ne[0], n_tokens);
                    cb(v_cache_2d, "v_cache_2d", il);

                    ggml_tensor * k_selected = ggml_get_rows(ctx0, k_cache_2d, topk_indices);
                    cb(k_selected, "k_selected", il);
                    // k_selected shape: [head_dim, actual_top_k, n_tokens]

                    ggml_tensor * v_selected = ggml_get_rows(ctx0, v_cache_2d, topk_indices);
                    cb(v_selected, "v_selected", il);
                    // v_selected shape: [kv_lora_rank, actual_top_k, n_tokens]

                    // ========================================
                    // Step 6: Sparse Attention
                    // ========================================
                    // Use build_attn_sparse to compute attention with selected tokens
                    cur = build_attn_sparse(inp_attn_k, model.layers[il].wo, NULL, Qcur, k_selected, v_selected,
                                            nullptr, model.layers[il].wv_b, kq_scale, il, actual_top_k);

                } else {
                    // Standard MLA attention (no DSA)
                    // note: MLA with the absorption optimzation converts into MQA (ie: GQA with 1 group)
                    cur = build_attn(inp_attn_k, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr,
                                     model.layers[il].wv_b, kq_scale, il);
                }
            } else {
                ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_cmpr);
                cb(kv, "kv", il);

                // split into {n_embd_head_qk_nope, n_head, n_tokens}
                ggml_tensor * k_nope =
                    ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head, 0);
                cb(k_nope, "k_nope_view", il);

                // and {n_embd_head_v, n_head, n_tokens}
                ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v, n_head, n_tokens,
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head,
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope));
                cb(Vcur, "Vcur_view", il);

                Vcur = ggml_cont(ctx0, Vcur);
                cb(Vcur, "Vcur_cont", il);

                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(Kcur, "Kcur", il);

                if (inp_attn_scale) {
                    // apply llama 4 temperature scaling
                    Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                    cb(Qcur, "Qcur_attn_temp_scaled", il);
                }

                // note: MLA without the absorption optimization converts into MHA (ie: GQA with full n_head groups)
                cur = build_attn(inp_attn_kv, model.layers[il].wo, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                                 kq_scale, il);
            }
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL,
                            model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out = build_moe_ffn(
                cur, model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps, model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps, model.layers[il].ffn_exp_probs_b, n_expert, n_expert_used, LLM_FFN_SILU,
                hparams.expert_weights_norm, hparams.expert_weights_scale, hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func, il);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp =
                    build_ffn(cur, model.layers[il].ffn_up_shexp, NULL, NULL, model.layers[il].ffn_gate_shexp, NULL,
                              NULL, model.layers[il].ffn_down_shexp, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
