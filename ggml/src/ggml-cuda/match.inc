static int ggml_cuda_match_qsa_expand(const ggml_cgraph * graph, int first, ggml_cuda_qsa_expand_args & args) {
    const char * enabled = getenv("LLAMA_QSA_FUSE_EXPAND");
    if (!enabled || atoi(enabled) == 0 || first + 19 >= graph->n_nodes) return 0;
    if (graph->nodes[first]->op != GGML_OP_CONT || !graph->nodes[first]->src[0] || graph->nodes[first]->src[0]->op != GGML_OP_TOP_K) return 0;
    const bool debug = getenv("QSA_EXPAND_MATCH_DEBUG") != nullptr;
    auto reject = [&](int line) { if (debug) fprintf(stderr,"QSA_EXPAND_REJECT first=%d line=%d\n",first,line);return 0; };
    if (debug) {
        fprintf(stderr,"QSA_EXPAND_CANDIDATE first=%d:",first);
        for(int j=0;j<21 && first+j<graph->n_nodes;++j)fprintf(stderr," %s",ggml_op_name(graph->nodes[first+j]->op));
        fputc('\n',stderr);
    }
    const ggml_op expected[] = {GGML_OP_CONT, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_ARGSORT,
        GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
        GGML_OP_CPY, GGML_OP_SCALE, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE, GGML_OP_UNARY,
        GGML_OP_MUL, GGML_OP_SCALE, GGML_OP_CPY, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_CONCAT};
    const ggml_tensor * n[21];
    for (int j = 0; j < 19; ++j) {
        n[j] = graph->nodes[first+j];
        if (n[j]->op != expected[j]) return reject(__LINE__);
    }
    const bool tail_view = graph->nodes[first+19]->op == GGML_OP_VIEW;
    const int node_count = tail_view ? 21 : 20;
    if (first + node_count > graph->n_nodes) return reject(__LINE__);
    n[20] = graph->nodes[first+node_count-1];
    if (n[20]->op != GGML_OP_CONCAT) return reject(__LINE__);
    n[19] = tail_view ? graph->nodes[first+19] : n[20]->src[1];
    if (!n[19]) return reject(__LINE__);
    auto source = [&](int node, int src, int producer) { return n[node]->src[src] == n[producer]; };
    for (const auto & edge : {std::array<int,3>{1,0,0}, {2,0,0}, {3,0,2}, {4,0,1}, {4,1,3},
            {5,0,4}, {6,0,5}, {7,1,6}, {8,0,7}, {9,0,8}, {10,0,9}, {12,0,11}, {12,1,5},
            {13,0,12}, {14,0,13}, {15,0,10}, {15,1,14}, {16,0,15}, {17,0,16}, {18,0,17},
            {20,0,18}, {20,1,19}}) {
        if (!source(edge[0], edge[1], edge[2])) return reject(__LINE__);
    }
    const auto * blocks = n[0]->src[0];
    const auto * scores = n[11]->src[0];
    const auto * cells = n[7]->src[0];
    const auto * tail = n[19];
    if (!blocks || !scores || !cells || blocks->op != GGML_OP_TOP_K || blocks->src[0] != scores ||
            blocks->type != GGML_TYPE_I32 || scores->type != GGML_TYPE_F32 || cells->type != GGML_TYPE_I32 ||
            tail->type != GGML_TYPE_I32 || n[20]->type != GGML_TYPE_I32 ||
            ggml_get_op_params_i32(n[3],0) != GGML_SORT_ORDER_ASC ||
            ggml_get_unary_op(n[14]) != GGML_UNARY_OP_STEP || ggml_get_op_params_i32(n[20],0) != 0) return reject(__LINE__);
    auto shape = [](const ggml_tensor * t, int64_t a, int64_t b, int64_t c=1, int64_t d=1) {
        return t->ne[0]==a && t->ne[1]==b && t->ne[2]==c && t->ne[3]==d;
    };
    const int64_t nq = blocks->ne[1], nb = scores->ne[0];
    if (nq < 1 || nq > INT_MAX || nb < 512 || nb > 16777216 ||
            !shape(blocks,512,nq) || !shape(scores,nb,nq) || !shape(cells,4,nb) || !shape(tail,3,nq) ||
            !shape(n[0],512,nq) || !shape(n[1],1,512,nq) || !shape(n[2],512,nq) ||
            !shape(n[3],512,nq) || !shape(n[4],1,512,nq) || !shape(n[5],512,nq) ||
            !shape(n[6],512*nq,1) || !shape(n[7],4,512*nq) || !shape(n[8],4,512,nq) ||
            !shape(n[9],4,512,nq) || !shape(n[10],4,512,nq) || !shape(n[11],1,nb,nq) ||
            !shape(n[12],1,512,nq) || !shape(n[13],1,512,nq) || !shape(n[14],1,512,nq) ||
            !shape(n[15],4,512,nq) || !shape(n[16],4,512,nq) || !shape(n[17],4,512,nq) ||
            !shape(n[18],2048,nq) || !shape(n[20],2051,nq)) return reject(__LINE__);
    for (int j : {2,9,10,11,12,13,14,15,16}) if (n[j]->type != GGML_TYPE_F32) return reject(__LINE__);
    for (int j : {0,1,3,4,5,6,7,8,17,18,19,20}) if (n[j]->type != GGML_TYPE_I32) return reject(__LINE__);
    for (int j : {10,13,16}) {
        if (ggml_get_op_params_f32(n[j],0) != 1.0f || ggml_get_op_params_f32(n[j],1) != (j==16 ? -1.0f : 1.0f)) return reject(__LINE__);
    }
    for (const auto * t : {blocks,scores,cells,tail,n[20]}) {
        if (!ggml_is_contiguous(t) || t->nb[0] != 4) return reject(__LINE__);
    }
    if (n[1]->view_offs != 0 || n[11]->view_offs != 0 ||
            n[1]->nb[1] != 4 || n[1]->nb[2] != size_t(512)*4 ||
            n[11]->nb[1] != 4 || n[11]->nb[2] != size_t(nb)*4) return reject(__LINE__);
    int indices[21], count=0; ggml_op ops[21];
    for (int j=0;j<node_count;++j) {
        if (j==11 || (tail_view && j==19)) continue;
        indices[count]=first+j;ops[count++]=graph->nodes[first+j]->op;
    }
    const int outputs[] = {first+2,first+9,first+17,first+18,first+node_count-1};
    if (!ggml_can_fuse_subgraph_ext(graph, indices, count, ops, outputs, 5)) return reject(__LINE__);
    args = {blocks,scores,cells,tail,graph->nodes[first+2],graph->nodes[first+9],graph->nodes[first+17],graph->nodes[first+node_count-1]};
    return node_count-1;
}

static bool ggml_cuda_qsa_expand_alias_ok(const ggml_cuda_qsa_expand_args & args) {
    const ggml_tensor * outputs[] = {args.cast_blocks,args.cast_cells,args.expanded,args.output};
    for (int i=0;i<4;++i) {
        for (const auto * input : {args.blocks,args.scores,args.cells,args.tail}) {
            if (ggml_cuda_mean4_ranges_overlap(input,outputs[i])) return false;
        }
        for (int j=0;j<i;++j) if (ggml_cuda_mean4_ranges_overlap(outputs[i],outputs[j])) return false;
    }
    return true;
}
