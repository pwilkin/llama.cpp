#pragma once

#include "common.cuh"

void ggml_cuda_launch_mm_ids_helper(
        ggml_backend_cuda_context & ctx,
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse);
