// Algebraic-identity tests for the DeepSeek-V4 mHC (manifold-constrained hyper-connection) ops:
//   - build_hc_weighted_sum  (fused as one ggml_mul_mat)
//   - build_hc_post          (fused as one ggml_mul_mat)
//   - ggml_sinkhorn          (fused Sinkhorn-Knopp op, vs the old explicit ggml sequence)
// Each fused result is compared, on the CPU backend, against an independent reference computed the
// other way (plain loops for the first two; the pre-fusion ggml op sequence for Sinkhorn), on random
// inputs with deliberately odd sizes. See src/models/deepseek-v4.cpp.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <initializer_list>

static uint32_t s_rng = 12345;
static float frand() { s_rng = s_rng*1664525u + 1013904223u; return ((s_rng>>8)&0xFFFFFF)/8388608.0f - 1.0f; }

static float at2(const float * d, int ne0, int i0, int i1) { return d[i0 + ne0*i1]; }
static float at3(const float * d, int ne0, int ne1, int i0, int i1, int i2) { return d[i0 + ne0*(i1 + ne1*i2)]; }

int main() {
    const int n_embd = 13, hc = 4, nt = 7;   // odd sizes to catch stride bugs
    const float EPS = 1e-6f; const int ITERS = 20;

    ggml_init_params ip = { 256ull*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x3   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);  // weighted_sum x
    ggml_tensor * w    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt);          // weighted_sum weights
    ggml_tensor * x2   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, nt);      // post x (sublayer out)
    ggml_tensor * res  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);  // post residual
    ggml_tensor * post = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt);          // post gates
    ggml_tensor * comb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, nt);      // post/sinkhorn comb [dst,src,t]
    for (ggml_tensor * t : { x3, w, x2, res, post, comb }) {
        float * d = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = frand();
    }

    // ---- weighted_sum: out[e,t] = sum_ih x[e,ih,t]*w[ih,t] ----
    ggml_tensor * ws_n; {                                            // fused
        ggml_tensor * xT = ggml_cont(ctx, ggml_permute(ctx, x3, 1, 0, 2, 3));
        ggml_tensor * wr = ggml_reshape_3d(ctx, w, hc, 1, nt);
        ws_n = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, xT, wr), n_embd, nt);
    }

    // ---- hc_post: out[e,dst,t] = x[e,t]*post[dst,t] + sum_src res[e,src,t]*comb[dst,src,t] ----
    ggml_tensor * po_n; {                                            // fused
        ggml_tensor * x_row    = ggml_reshape_3d(ctx, x2, n_embd, 1, nt);
        ggml_tensor * rp       = ggml_concat(ctx, res, x_row, 1);
        ggml_tensor * post_row = ggml_reshape_3d(ctx, post, hc, 1, nt);
        ggml_tensor * mix      = ggml_concat(ctx, comb, post_row, 1);
        ggml_tensor * rpT      = ggml_cont(ctx, ggml_permute(ctx, rp,  1, 0, 2, 3));
        ggml_tensor * mixT     = ggml_cont(ctx, ggml_permute(ctx, mix, 1, 0, 2, 3));
        po_n = ggml_mul_mat(ctx, rpT, mixT);
    }

    // ---- sinkhorn: fused op vs the old explicit ggml sequence (both on softmax(comb)) ----
    ggml_tensor * x0 = ggml_soft_max(ctx, comb);
    ggml_tensor * sink_o; {                                          // old sequence
        ggml_tensor * eps_t = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), EPS);
        ggml_tensor * c = ggml_add(ctx, x0, eps_t);
        auto ncol = [&]() { ggml_tensor * cd = ggml_cont(ctx, ggml_permute(ctx, c, 1, 0, 2, 3)); ggml_tensor * cs = ggml_sum_rows(ctx, cd); cs = ggml_add(ctx, cs, eps_t); cs = ggml_permute(ctx, cs, 1, 0, 2, 3); c = ggml_div(ctx, c, cs); };
        auto nrow = [&]() { ggml_tensor * rs = ggml_sum_rows(ctx, c); rs = ggml_add(ctx, rs, eps_t); c = ggml_div(ctx, c, rs); };
        ncol(); for (int i = 1; i < ITERS; ++i) { nrow(); ncol(); }
        sink_o = c;
    }
    ggml_tensor * sink_n = ggml_sinkhorn(ctx, x0, EPS, ITERS);       // fused op

    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : { ws_n, po_n, sink_o, sink_n }) ggml_build_forward_expand(gf, t);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const float *X3=(float*)x3->data,*W=(float*)w->data,*X2=(float*)x2->data,*R=(float*)res->data,*P=(float*)post->data,*C=(float*)comb->data;
    const float *WSN=(float*)ws_n->data,*PN=(float*)po_n->data,*SO=(float*)sink_o->data,*SN=(float*)sink_n->data;

    double mws=0, mpo=0, msk=0, dsr=0, dsc=0;
    for (int t=0;t<nt;++t) for (int e=0;e<n_embd;++e) {
        double ref=0; for (int ih=0;ih<hc;++ih) ref += at3(X3,n_embd,hc,e,ih,t)*at2(W,hc,ih,t);
        mws = fmax(mws, fabs(at2(WSN,n_embd,e,t) - ref));
    }
    for (int t=0;t<nt;++t) for (int dst=0;dst<hc;++dst) for (int e=0;e<n_embd;++e) {
        double ref = at2(X2,n_embd,e,t)*at2(P,hc,dst,t);
        for (int src=0;src<hc;++src) ref += at3(R,n_embd,hc,e,src,t)*at3(C,hc,hc,dst,src,t);
        mpo = fmax(mpo, fabs(at3(PN,n_embd,hc,e,dst,t) - ref));
    }
    for (int64_t e=0;e<(int64_t)hc*hc*nt;++e) msk = fmax(msk, fabs((double)SO[e]-SN[e]));
    for (int t=0;t<nt;++t) {
        for (int i=0;i<hc;++i){ double r=0; for(int j=0;j<hc;++j) r+=at3(SN,hc,hc,i,j,t); dsr=fmax(dsr,fabs(r-1.0)); }
        for (int j=0;j<hc;++j){ double c=0; for(int i=0;i<hc;++i) c+=at3(SN,hc,hc,i,j,t); dsc=fmax(dsc,fabs(c-1.0)); }
    }

    printf("weighted_sum |fused-ref|=%.2e\n", mws);
    printf("hc_post      |fused-ref|=%.2e\n", mpo);
    printf("sinkhorn     |fused-old|=%.2e  doubly-stochastic max|rowsum-1|=%.2e max|colsum-1|=%.2e\n", msk, dsr, dsc);

    ggml_free(ctx);

    const bool ok = mws < 1e-4 && mpo < 1e-4 && msk < 1e-4 && dsr < 1e-3 && dsc < 1e-3;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
