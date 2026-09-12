#include "ggml.h"
static ggml_tensor * qsa_pack_keys(ggml_context * ctx, ggml_tensor * keys) {
    GGML_ASSERT(keys->type == GGML_TYPE_F16 && keys->ne[0] == 256 && keys->ne[1] % 4 == 0 && keys->ne[3] == 1);
    auto * contiguous = ggml_is_contiguous(keys) ? keys : ggml_cont(ctx, keys);
    auto * blocks = ggml_reshape_4d(ctx, contiguous, 16, 16, 4, keys->ne[1]/4*keys->ne[2]);
    auto * packed = ggml_cont(ctx, ggml_permute(ctx, blocks, 0, 2, 1, 3));
    ggml_set_name(packed, "qsa_packed_keys");
    return packed;
}

static ggml_tensor * qsa_pack_values(ggml_context * ctx, ggml_tensor * values) {
    GGML_ASSERT(values->type == GGML_TYPE_F16 && values->ne[0] == 256 && values->ne[1] % 4 == 0 && values->ne[3] == 1);
    auto * contiguous = ggml_is_contiguous(values) ? values : ggml_cont(ctx, values);
    auto * blocks = ggml_reshape_3d(ctx, contiguous, 256, 4, values->ne[1]/4*values->ne[2]);
    auto * packed = ggml_cont(ctx, ggml_permute(ctx, blocks, 1, 0, 2, 3));
    ggml_set_name(packed, "qsa_packed_values");
    return packed;
}
