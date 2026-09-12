static ggml_tensor * qwen4exp_apply_compact_visibility(ggml_context *ctx,ggml_tensor *score,
        ggml_tensor *limits,int64_t blocks,int64_t first,int64_t queries) {
    GGML_ASSERT(limits->type==GGML_TYPE_I32 && score->ne[0]<=blocks && score->ne[1]==queries && score->ne[2]==1);
    auto *starts=ggml_cast(ctx,ggml_view_1d(ctx,limits,score->ne[0],0),GGML_TYPE_F32);
    auto *tails=ggml_cast(ctx,ggml_view_1d(ctx,limits,queries,(blocks+first)*sizeof(int32_t)),GGML_TYPE_F32);
    tails=ggml_reshape_2d(ctx,tails,1,queries);
    auto *visible=ggml_step(ctx,ggml_sub(ctx,ggml_repeat(ctx,tails,score),starts));
    return ggml_add(ctx,score,ggml_log(ctx,visible));
}

static ggml_tensor * qwen4exp_select_complete_blocks(ggml_context *ctx0, ggml_tensor *score,
        ggml_tensor *block_cells, ggml_tensor *tail, int64_t block_budget, int64_t r) {
    const int64_t n_blocks=score->ne[0], n_query=score->ne[1], n_stream=score->ne[2];
    GGML_ASSERT(score->type==GGML_TYPE_F32 && ggml_is_contiguous(score));
    GGML_ASSERT(block_cells->type==GGML_TYPE_I32 && block_cells->ne[0]==r*n_blocks && block_cells->ne[1]==n_stream);
    GGML_ASSERT(tail->type==GGML_TYPE_I32 && tail->ne[0]==r-1 && tail->ne[1]==n_query && tail->ne[3]==n_stream);
            GGML_ASSERT(n_blocks>=block_budget);
            ggml_tensor * blocks=ggml_cont(ctx0,ggml_top_k(ctx0,score,block_budget));
            ggml_tensor * order=ggml_argsort(ctx0,ggml_cast(ctx0,blocks,GGML_TYPE_F32),GGML_SORT_ORDER_ASC);
            ggml_tensor * blocks_view=ggml_view_4d(ctx0,blocks,1,block_budget,n_query,n_stream,
                    blocks->nb[0],blocks->nb[1],blocks->nb[2],0);
            blocks=ggml_reshape_3d(ctx0,ggml_get_rows(ctx0,blocks_view,order),block_budget,n_query,n_stream);
            ggml_tensor * score_view=ggml_view_4d(ctx0,score,1,n_blocks,n_query,n_stream,
                    score->nb[0],score->nb[1],score->nb[2],0);
            ggml_tensor * picked_score=ggml_get_rows(ctx0,score_view,blocks);
            ggml_tensor * valid=ggml_step(ctx0,ggml_scale_bias(ctx0,picked_score,1.0f,1.0f));
            ggml_tensor * cells_view=ggml_reshape_3d(ctx0,block_cells,r,n_blocks,n_stream);
            ggml_tensor * flat_blocks=ggml_reshape_2d(ctx0,blocks,block_budget*n_query,n_stream);
            ggml_tensor * cells=ggml_get_rows(ctx0,cells_view,flat_blocks);
            cells=ggml_reshape_4d(ctx0,cells,r,block_budget,n_query,n_stream);
            cells=ggml_scale_bias(ctx0,ggml_cast(ctx0,cells,GGML_TYPE_F32),1.0f,1.0f);
            cells=ggml_mul(ctx0,cells,valid);
            cells=ggml_cast(ctx0,ggml_scale_bias(ctx0,cells,1.0f,-1.0f),GGML_TYPE_I32);
            cells=ggml_reshape_4d(ctx0,cells,block_budget*r,n_query,1,n_stream);
            return ggml_concat(ctx0,cells,tail,0);
}
