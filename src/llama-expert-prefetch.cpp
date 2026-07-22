#include "llama-expert-prefetch.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__))
#   include <sys/mman.h>
#   include <unistd.h>
#   include <cerrno>
#   define LLAMA_EXPERT_PREFETCH_POSIX 1
#endif

#ifdef LLAMA_EXPERT_PREFETCH_POSIX

// number of worker threads faulting expert rows in; 0 disables the prefetcher entirely
static int llama_expert_prefetch_n_threads() {
    const char * env = getenv("LLAMA_EXPERT_PREFETCH_THREADS");
    if (env) {
        const int n = atoi(env);
        return n < 0 ? 0 : n;
    }
    return 8;
}

struct llama_expert_prefetcher::impl {
    struct region {
        const char * addr;
        size_t       size;
    };

    // one MoE layer's routed expert weight tensors (gate/up/down, or the fused gate_up + down)
    struct group {
        const ggml_tensor * t[4] = { nullptr, nullptr, nullptr, nullptr };
        int                 n    = 0;
        uint64_t            last_ids = 0;   // fingerprint of the routing this group was last queued for
        bool                seeded   = false;
        int                 il       = -1;  // layer index (prediction study only)
        uint64_t            last_obs = 0;   // separate fingerprint for the prediction study
        bool                obs_seeded = false;
    };

    // ---- prediction study (LLAMA_EXPERT_PREDICT_STATS=1) ------------------------------------
    // Tests whether layer L+1's routing can be computed early from layer L's activations -- the
    // residual stream changes only incrementally between layers, so running L+1's *real* router on
    // a stale input may pick mostly the same experts. If it does, the prefetch can be issued a full
    // layer ahead (covering the GPU-attention gap) instead of having ~no lead time.
    //
    // Two figures are collected per layer:
    //   self : route(gate_inp[L], act[L])   vs actual ids[L]  -- CONTROL. Must be ~100%, otherwise
    //                                                            the router reimplementation below
    //                                                            is wrong and `pred` means nothing.
    //   pred : route(gate_inp[L], act[L-1]) vs actual ids[L]  -- the hypothesis.
    struct predict_study {
        bool     enabled   = false;   // routers loaded and prediction machinery usable
        bool     stats     = false;   // also collect/report the self/pred accuracy figures
        bool     prefetch  = false;   // actually issue prefetches from the prediction
        int      min_layer = 0;       // only predict layers at or beyond this depth
        std::atomic<uint64_t> n_predicted{0};
        int64_t  n_embd = 0, n_expert = 0, n_expert_used = 0;
        uint32_t gating_op = 0, n_expert_groups = 0, n_group_used = 0;

        std::vector<std::vector<float>> gate_w;    // [il][n_expert*n_embd] dequantised router
        std::vector<std::vector<float>> gate_b;    // [il][n_expert] pre-gating bias (may be empty)
        std::vector<std::vector<float>> probs_b;   // [il][n_expert] DeepSeek-V3 selection bias

        std::vector<float> prev_act;               // activations of the previously seen MoE layer
        bool               prev_valid = false;

        struct stat { uint64_t n = 0, self_hit = 0, pred_hit = 0, slots = 0; };
        std::vector<stat> per_layer;
        uint64_t          n_since_dump = 0;

        // scratch (single-threaded: only ever touched from the ith==0 hook)
        std::vector<float>   logits, sel;
        std::vector<int32_t> pick;
    } ps;

    // route() -- faithful CPU reimplementation of build_moe_ffn()'s expert selection:
    //   logits -> (+gate_inp_b) -> gating_op -> (+exp_probs_b) -> group mask -> top_k
    void route(const float * act, int il, std::vector<int32_t> & out) {
        const int64_t ne = ps.n_expert;
        const int64_t nx = ps.n_embd;

        ps.logits.assign(ne, 0.0f);
        const float * W = ps.gate_w[il].data();
        for (int64_t e = 0; e < ne; ++e) {
            // four independent accumulators: a plain float reduction is not reassociable, so this
            // is what lets the compiler vectorise it. This runs once per predicted layer per token
            // on a compute thread, so it is on the critical path.
            const float * w = W + e*nx;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int64_t i = 0;
            for (; i + 4 <= nx; i += 4) {
                s0 += w[i+0]*act[i+0];
                s1 += w[i+1]*act[i+1];
                s2 += w[i+2]*act[i+2];
                s3 += w[i+3]*act[i+3];
            }
            float s = s0 + s1 + s2 + s3;
            for (; i < nx; ++i) {
                s += w[i]*act[i];
            }
            ps.logits[e] = s + (ps.gate_b[il].empty() ? 0.0f : ps.gate_b[il][e]);
        }

        ps.sel.assign(ne, 0.0f);
        switch (ps.gating_op) {
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: {
                float m = *std::max_element(ps.logits.begin(), ps.logits.end());
                float sum = 0.0f;
                for (int64_t e = 0; e < ne; ++e) { ps.sel[e] = expf(ps.logits[e] - m); sum += ps.sel[e]; }
                for (int64_t e = 0; e < ne; ++e) { ps.sel[e] /= sum; }
            } break;
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
                for (int64_t e = 0; e < ne; ++e) { ps.sel[e] = 1.0f/(1.0f + expf(-ps.logits[e])); }
                break;
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
                for (int64_t e = 0; e < ne; ++e) { ps.sel[e] = sqrtf(log1pf(expf(ps.logits[e]))); }
                break;
            default: // SOFTMAX_WEIGHT / NONE: selection is on the raw logits
                ps.sel = ps.logits;
                break;
        }

        if (!ps.probs_b[il].empty()) {
            for (int64_t e = 0; e < ne; ++e) { ps.sel[e] += ps.probs_b[il][e]; }
        }

        // DeepSeek-V3 group masking: score each group by the sum of its top 2, keep n_group_used
        if (ps.n_expert_groups > 1) {
            const int64_t per = ne / (int64_t) ps.n_expert_groups;
            std::vector<std::pair<float,int64_t>> gs;
            gs.reserve(ps.n_expert_groups);
            for (uint32_t g = 0; g < ps.n_expert_groups; ++g) {
                float a = -INFINITY, b = -INFINITY;
                for (int64_t j = 0; j < per; ++j) {
                    const float v = ps.sel[g*per + j];
                    if (v > a) { b = a; a = v; } else if (v > b) { b = v; }
                }
                gs.emplace_back(a + b, (int64_t) g);
            }
            std::partial_sort(gs.begin(), gs.begin() + std::min<size_t>(ps.n_group_used, gs.size()), gs.end(),
                    [](const auto & x, const auto & y) { return x.first > y.first; });
            std::vector<bool> keep(ps.n_expert_groups, false);
            for (uint32_t k = 0; k < ps.n_group_used && k < gs.size(); ++k) { keep[gs[k].second] = true; }
            for (uint32_t g = 0; g < ps.n_expert_groups; ++g) {
                if (!keep[g]) {
                    for (int64_t j = 0; j < per; ++j) { ps.sel[g*per + j] = -INFINITY; }
                }
            }
        }

        ps.pick.resize(ne);
        std::iota(ps.pick.begin(), ps.pick.end(), 0);
        const size_t k = std::min<size_t>(ps.n_expert_used, (size_t) ne);
        std::partial_sort(ps.pick.begin(), ps.pick.begin() + k, ps.pick.end(),
                [this](int32_t a, int32_t b) { return ps.sel[a] > ps.sel[b]; });
        out.assign(ps.pick.begin(), ps.pick.begin() + k);
    }

    struct range {
        char * addr;
        size_t len;
    };

    std::vector<region>                                regions;
    std::vector<group>                                 groups;
    std::unordered_map<const ggml_tensor *, size_t>    by_tensor;   // tensor -> index into groups

    std::vector<std::thread> workers;
    std::deque<range>        queue;
    std::mutex               mtx;
    std::condition_variable  cv;
    bool                     stop      = false;
    // Lead, in queued rows. Deliberately shallow: a deep queue lets the pool fall behind and keep
    // populating a layer the model has already moved past, which on a working set larger than the
    // page cache evicts rows that are still wanted -- the same way an unbounded WILLNEED does.
    // ~2-3 layers (24 rows each) is enough to cover the gap and bounds how stale a request can get.
    size_t                   max_queue = 64;
    size_t                   page      = 4096;

    std::atomic<uint64_t> n_enqueued{0};
    std::atomic<uint64_t> n_dropped{0};
    std::atomic<uint64_t> n_done{0};
    std::atomic<uint64_t> n_failed{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<bool>     use_populate{true};

    ~impl() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
        }
        cv.notify_all();
        for (auto & w : workers) {
            if (w.joinable()) {
                w.join();
            }
        }
    }

    // Snap a row to page boundaries (outward) and reject anything not wholly inside a mapping.
    // madvise() requires a page-aligned start, and expert rows are not aligned -- an unaligned call
    // fails with EINVAL and is silently dropped.
    bool make_range(const char * start, size_t len, range & out) const {
        for (const auto & r : regions) {
            if (start < r.addr || start + len > r.addr + r.size) {
                continue;
            }
            const uintptr_t base = (uintptr_t) r.addr;
            const uintptr_t end  = base + r.size;

            uintptr_t lo = ((uintptr_t) start)       & ~(uintptr_t)(page - 1);
            uintptr_t hi = ((uintptr_t) start + len + page - 1) & ~(uintptr_t)(page - 1);

            if (lo < base) { lo = base; }
            if (hi > end)  { hi = end;  }
            if (hi <= lo)  { return false; }

            out.addr = (char *) lo;
            out.len  = (size_t) (hi - lo);
            return true;
        }
        return false;
    }

    void push(const range & r) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            // Never block a compute thread: if the pool is already behind, drop the oldest request.
            // Whatever is dropped simply faults in the normal way when the expert is reached.
            if (queue.size() >= max_queue) {
                queue.pop_front();
                n_dropped.fetch_add(1, std::memory_order_relaxed);
            }
            queue.push_back(r);
        }
        n_enqueued.fetch_add(1, std::memory_order_relaxed);
        cv.notify_one();
    }

    void populate(const range & r) {
#ifdef MADV_POPULATE_READ
        if (use_populate.load(std::memory_order_relaxed)) {
            if (madvise(r.addr, r.len, MADV_POPULATE_READ) == 0) {
                bytes.fetch_add(r.len, std::memory_order_relaxed);
                n_done.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            if (errno == EINVAL) {
                // kernel predates MADV_POPULATE_READ (< 5.14): fall back for the rest of the run
                use_populate.store(false, std::memory_order_relaxed);
            } else {
                n_failed.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
#endif
        // Fallback: touch one byte per page. Same effect that matters here -- the pages are faulted
        // in by *this* thread and enter as accessed, so they are not reclaimed before first use.
        const volatile char * p = r.addr;
        char sink = 0;
        for (size_t off = 0; off < r.len; off += page) {
            sink = (char) (sink ^ p[off]);
        }
        (void) sink;
        bytes.fetch_add(r.len, std::memory_order_relaxed);
        n_done.fetch_add(1, std::memory_order_relaxed);
    }

    void worker() {
        for (;;) {
            range r;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [this] { return stop || !queue.empty(); });
                if (stop) {
                    return;
                }
                r = queue.front();
                queue.pop_front();
            }
            populate(r);
        }
    }

    // Called from on_node() once per MoE layer, with that layer's router input (the gate/up node's
    // src1 -- the down node's src1 is the post-SwiGLU activation, a different thing) and its actual
    // routing. Compares the real router run on this layer's input (control) and on the previous
    // layer's input (the hypothesis) against what the model actually selected.
    void observe_prediction(int il, const float * act, const int32_t * actual, int n_used) {
        if (il < 0 || il >= (int) ps.per_layer.size() || ps.gate_w[il].empty()) {
            return;
        }
        auto & st = ps.per_layer[il];

        std::vector<int32_t> got;
        const auto overlap = [&](const std::vector<int32_t> & p) {
            uint64_t h = 0;
            for (int i = 0; i < n_used; ++i) {
                if (std::find(p.begin(), p.end(), actual[i]) != p.end()) { h++; }
            }
            return h;
        };

        route(act, il, got);
        st.self_hit += overlap(got);

        if (ps.prev_valid) {
            route(ps.prev_act.data(), il, got);
            st.pred_hit += overlap(got);
            st.n++;
            st.slots += n_used;
        }

        ps.prev_act.assign(act, act + ps.n_embd);
        ps.prev_valid = true;

        // The server can deadlock on shutdown with a >RAM mmap'd model, so the destructor is not a
        // reliable place to report from -- dump periodically instead.
        if (++ps.n_since_dump >= 4000) {
            ps.n_since_dump = 0;
            dump_prediction_stats();
        }
    }

    void dump_prediction_stats() const {
        uint64_t tot_n = 0, tot_self = 0, tot_pred = 0, tot_slots = 0;
        for (size_t il = 0; il < ps.per_layer.size(); ++il) {
            const auto & s = ps.per_layer[il];
            if (s.n == 0) {
                continue;
            }
            LLAMA_LOG_INFO("PREDICT layer %3zu: n=%6" PRIu64 "  self=%5.1f%%  pred=%5.1f%%\n",
                    il, s.n, 100.0*s.self_hit/s.slots, 100.0*s.pred_hit/s.slots);
            tot_n += s.n; tot_self += s.self_hit; tot_pred += s.pred_hit; tot_slots += s.slots;
        }
        if (tot_slots) {
            LLAMA_LOG_INFO("PREDICT TOTAL: n=%" PRIu64 "  self=%.1f%%  pred=%.1f%%  (self is the "
                           "control: if it is not ~100%% the router reimplementation is wrong)\n",
                    tot_n, 100.0*tot_self/tot_slots, 100.0*tot_pred/tot_slots);
        }
    }

    // queue every row this group needs for the given expert ids
    void enqueue_experts(const group & g, const int32_t * experts, int n) {
        for (int i = 0; i < n; ++i) {
            const int32_t e = experts[i];
            for (int k = 0; k < g.n; ++k) {
                const ggml_tensor * t = g.t[k];
                if (e < 0 || e >= t->ne[2]) {
                    continue;
                }
                range r;
                if (make_range((const char *) t->data + (size_t) e * t->nb[2], t->nb[2], r)) {
                    push(r);
                }
            }
        }
    }

    // The point of the whole exercise: run the NEXT MoE layer's real router on THIS layer's
    // activations and start paging in the experts it selects. Those reads then have a full layer of
    // lead -- this layer's expert compute plus the next layer's attention on the GPU, which is where
    // the disk otherwise sits idle. Measured overlap with the true routing is ~83% beyond layer ~30
    // and chance-level in the first few MoE layers, hence min_layer.
    void predict_next(size_t idx, const float * act) {
        const size_t nxt = idx + 1;
        if (nxt >= groups.size()) {
            return;
        }
        const group & g = groups[nxt];
        if (g.il < ps.min_layer || g.il >= (int) ps.gate_w.size() || ps.gate_w[g.il].empty()) {
            return;
        }
        std::vector<int32_t> pred;
        route(act, g.il, pred);
        enqueue_experts(g, pred.data(), (int) pred.size());
        ps.n_predicted.fetch_add(1, std::memory_order_relaxed);
    }

    void on_node(const ggml_tensor * dst) {
        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];
        const ggml_tensor * ids  = dst->src[2];
        if (!src0 || !src1 || !ids) {
            return;
        }
        const auto it = by_tensor.find(src0);
        if (it == by_tensor.end()) {
            return;
        }

        const int64_t n_used   = ids->ne[0];
        const int64_t n_tokens = ids->ne[1];

        // Decode-shaped batches only. A large batch's routed set approaches "every expert", and
        // prefetching that is exactly the pattern that doubles bytes/token on a working set larger
        // than the page cache. Big batches also already saturate the queue by themselves.
        if (n_tokens > 8 || n_used <= 0 || ids->data == nullptr) {
            return;
        }

        group & g = groups[it->second];

        // gate/up/down are three separate MUL_MAT_ID nodes sharing one routing, so the first of them
        // queues the whole group and the other two are skipped. Fingerprint the ids to detect that.
        // (If a later token routes identically the group is skipped again, which is harmless: those
        // rows were just read and are the most-recently-used pages in the cache.)
        uint64_t fp = 1469598103934665603ull;
        for (int64_t i1 = 0; i1 < n_tokens; ++i1) {
            for (int64_t i0 = 0; i0 < n_used; ++i0) {
                const int32_t e = *(const int32_t *)
                    ((const char *) ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
                fp = (fp ^ (uint64_t) (uint32_t) e) * 1099511628211ull;
            }
        }
        // Runs off the gate/up node, whose src1 IS the router input; the down node's src1 is the
        // post-SwiGLU activation (n_ff wide), so it is skipped. Tracked with its own fingerprint so
        // it is unaffected by the enqueue dedup below.
        if (ps.enabled && n_tokens == 1 && src1->type == GGML_TYPE_F32 &&
            src1->ne[0] == ps.n_embd && src1->data != nullptr &&
            !(g.obs_seeded && g.last_obs == fp)) {
            g.last_obs   = fp;
            g.obs_seeded = true;
            const float * act = (const float *) src1->data;

            if (ps.prefetch) {
                predict_next(it->second, act);
            }
            if (ps.stats) {
                observe_prediction(g.il, act, (const int32_t *) ids->data, (int) n_used);
            }
        }

        if (g.seeded && g.last_ids == fp) {
            return;
        }
        g.last_ids = fp;
        g.seeded   = true;

        // n_tokens is capped at 8 above, so this holds the whole routed union for any sane
        // n_expert_used; past it dedup degrades to redundant (harmless) re-queues.
        int32_t seen[256];
        int     n_seen = 0;

        for (int64_t i1 = 0; i1 < n_tokens; ++i1) {
            for (int64_t i0 = 0; i0 < n_used; ++i0) {
                const int32_t e = *(const int32_t *)
                    ((const char *) ids->data + i1*ids->nb[1] + i0*ids->nb[0]);

                bool dup = false;
                for (int k = 0; k < n_seen; ++k) {
                    if (seen[k] == e) { dup = true; break; }
                }
                if (dup) {
                    continue;
                }
                if (n_seen < (int) (sizeof(seen)/sizeof(seen[0]))) {
                    seen[n_seen++] = e;
                }

                for (int k = 0; k < g.n; ++k) {
                    const ggml_tensor * t = g.t[k];
                    if (e < 0 || e >= t->ne[2]) {
                        continue;
                    }
                    range r;
                    if (make_range((const char *) t->data + (size_t) e * t->nb[2], t->nb[2], r)) {
                        push(r);
                    }
                }
            }
        }
    }
};

static llama_expert_prefetcher::impl * g_installed = nullptr;

static void llama_expert_prefetch_hook(const ggml_tensor * dst, void * user_data) {
    auto * p = (llama_expert_prefetcher::impl *) user_data;
    if (p) {
        p->on_node(dst);
    }
}

// Pull a (possibly quantised, possibly device-resident) tensor into a host F32 buffer.
static bool llama_load_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) {
        return false;
    }
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];

    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());

    out.assign((size_t) ne0*ne1, 0.0f);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), raw.data(), out.size()*sizeof(float));
        return true;
    }
    const auto * tt = ggml_get_type_traits(t->type);
    if (!tt || !tt->to_float) {
        return false;
    }
    for (int64_t r = 0; r < ne1; ++r) {
        tt->to_float(raw.data() + r*t->nb[1], out.data() + r*ne0, ne0);
    }
    return true;
}

llama_expert_prefetcher::llama_expert_prefetcher() : pimpl(new impl()) {}

llama_expert_prefetcher::~llama_expert_prefetcher() {
    if (g_installed == pimpl.get()) {
        uninstall();
    }
    print_stats();
}

std::unique_ptr<llama_expert_prefetcher> llama_expert_prefetcher::create(
        const llama_model &                            model,
        const std::vector<std::pair<void *, size_t>> & mapped_regions) {
    const int n_threads = llama_expert_prefetch_n_threads();
    if (n_threads == 0 || mapped_regions.empty()) {
        return nullptr;
    }

    std::unique_ptr<llama_expert_prefetcher> res(new llama_expert_prefetcher());
    impl & p = *res->pimpl;

    const long ps = sysconf(_SC_PAGESIZE);
    p.page = ps > 0 ? (size_t) ps : 4096;

    for (const auto & r : mapped_regions) {
        if (r.first && r.second) {
            p.regions.push_back({ (const char *) r.first, r.second });
        }
    }

    p.groups.reserve(model.layers.size());

    size_t n_tensors = 0;
    for (const auto & layer : model.layers) {
        const ggml_tensor * cand[4] = {
            layer.ffn_gate_exps,
            layer.ffn_up_exps,
            layer.ffn_down_exps,
            layer.ffn_gate_up_exps,
        };

        impl::group g;
        for (const ggml_tensor * t : cand) {
            // ne[2] is the expert axis; a row is nb[2] bytes. Only take tensors that really live in
            // one of the mappings -- offloaded or malloc'd experts must never be advised.
            if (!t || t->data == nullptr || t->ne[2] <= 1) {
                continue;
            }
            impl::range r;
            if (!p.make_range((const char *) t->data, t->nb[2], r)) {
                continue;
            }
            if (g.n < 4) {
                g.t[g.n++] = t;
            }
        }
        if (g.n == 0) {
            continue;
        }

        const size_t idx = p.groups.size();
        g.il = (int) (&layer - model.layers.data());
        p.groups.push_back(g);
        for (int k = 0; k < g.n; ++k) {
            p.by_tensor[g.t[k]] = idx;
            n_tensors++;
        }
    }

    if (p.groups.empty()) {
        return nullptr;
    }

    // Speculative cross-layer prefetch: OFF by default (LLAMA_EXPERT_PREDICT=1 enables), because on
    // a working set larger than RAM it measured net NEGATIVE -- see below. Kept because it is a
    // clean win for any placement where the extra reads are free rather than displacing cache.
    //
    // The prediction itself works: running layer L+1's real router on layer L's activations picks
    // 76.8% of the true experts overall, ~83% beyond layer 30. But throughput = disk_bandwidth /
    // bytes_per_token when the model does not fit, so the ~23% mispredictions are pure cost:
    //   min_layer=15: 1.740 t/s (-6.8% vs 1.867), read +22.0%
    //   min_layer=30: 1.820 t/s (-2.5% vs 1.867), read +16.5%
    // Restricting to the high-accuracy layers cuts both the waste and the loss, monotonically -- but
    // the line does not cross zero. Break-even needs accuracy near 100%, which staleness cannot give.
    {
        const char * e_pred  = getenv("LLAMA_EXPERT_PREDICT");
        const char * e_min   = getenv("LLAMA_EXPERT_PREDICT_MIN_LAYER");
        const bool   want_pf = e_pred && atoi(e_pred) != 0;
        const bool   want_st = getenv("LLAMA_EXPERT_PREDICT_STATS") != nullptr;

        auto & ps = p.ps;
        const auto & hp = model.hparams;
        ps.prefetch        = want_pf;
        ps.stats           = want_st;
        ps.min_layer       = e_min ? atoi(e_min) : 15;
        ps.enabled         = want_pf || want_st;
        ps.n_embd          = hp.n_embd;
        ps.n_expert        = hp.n_expert;
        ps.n_expert_used   = hp.n_expert_used;
        ps.gating_op       = hp.expert_gating_func;
        ps.n_expert_groups = hp.n_expert_groups;
        ps.n_group_used    = hp.n_group_used;

        const size_t n_layers = model.layers.size();
        ps.gate_w.resize(n_layers);
        ps.gate_b.resize(n_layers);
        ps.probs_b.resize(n_layers);
        ps.per_layer.resize(n_layers);

        size_t loaded = 0, bytes = 0;
        for (size_t il = 0; il < n_layers; ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_gate_inp) {
                continue;
            }
            // The stats mode needs every layer to report the depth profile; prefetching only ever
            // predicts layers >= min_layer, and each router is n_expert*n_embd floats (6 MiB here),
            // so skipping the shallow ones keeps that much RAM in the page cache instead.
            if (!ps.stats && (int) il < ps.min_layer) {
                continue;
            }
            if (llama_load_f32(l.ffn_gate_inp, ps.gate_w[il])) {
                loaded++;
                bytes += ps.gate_w[il].size()*sizeof(float);
            } else {
                ps.gate_w[il].clear();
            }
            llama_load_f32(l.ffn_gate_inp_b,  ps.gate_b[il]);
            llama_load_f32(l.ffn_exp_probs_b, ps.probs_b[il]);
        }
        ps.prev_act.resize(ps.n_embd);
        if (loaded == 0) {
            ps.enabled = ps.prefetch = ps.stats = false;
        }

        LLAMA_LOG_INFO("%s: expert predict: %s%s, %zu routers (%.0f MiB), min_layer=%d, "
                       "n_embd=%" PRId64 " n_expert=%" PRId64 " n_used=%" PRId64
                       " gating=%u groups=%u/%u\n",
                __func__, ps.prefetch ? "prefetch" : "off", ps.stats ? "+stats" : "",
                loaded, bytes/(1024.0*1024.0), ps.min_layer, ps.n_embd, ps.n_expert,
                ps.n_expert_used, ps.gating_op, ps.n_group_used, ps.n_expert_groups);
    }

    for (int i = 0; i < n_threads; ++i) {
        p.workers.emplace_back([ptr = &p] { ptr->worker(); });
    }

    LLAMA_LOG_INFO("%s: expert prefetch: %zu MoE layers, %zu tensors, %d workers\n",
            __func__, p.groups.size(), n_tensors, n_threads);

    return res;
}

void llama_expert_prefetcher::install() {
    g_installed = pimpl.get();
    ggml_cpu_set_mul_mat_id_prefetch_hook(llama_expert_prefetch_hook, pimpl.get());
}

void llama_expert_prefetcher::uninstall() {
    ggml_cpu_set_mul_mat_id_prefetch_hook(nullptr, nullptr);
    g_installed = nullptr;
}

void llama_expert_prefetcher::print_stats() const {
    const impl & p = *pimpl;

    if (p.ps.enabled) {
        uint64_t tot_n = 0, tot_self = 0, tot_pred = 0, tot_slots = 0;
        LLAMA_LOG_INFO("expert predict study: per-layer overlap of routing computed from the "
                       "PREVIOUS layer's activations vs actual (self = control, must be ~100%%)\n");
        for (size_t il = 0; il < p.ps.per_layer.size(); ++il) {
            const auto & s = p.ps.per_layer[il];
            if (s.n == 0) {
                continue;
            }
            LLAMA_LOG_INFO("  layer %3zu: n=%6" PRIu64 "  self=%5.1f%%  pred=%5.1f%%\n",
                    il, s.n, 100.0*s.self_hit/s.slots, 100.0*s.pred_hit/s.slots);
            tot_n += s.n; tot_self += s.self_hit; tot_pred += s.pred_hit; tot_slots += s.slots;
        }
        if (tot_slots) {
            LLAMA_LOG_INFO("expert predict study: TOTAL n=%" PRIu64 "  self=%.1f%%  pred=%.1f%%\n",
                    tot_n, 100.0*tot_self/tot_slots, 100.0*tot_pred/tot_slots);
        }
    }

    const uint64_t enq = p.n_enqueued.load();
    if (enq == 0) {
        return;
    }
    LLAMA_LOG_INFO("%s: expert prefetch: %" PRIu64 " queued, %" PRIu64 " completed, %" PRIu64
                   " dropped, %" PRIu64 " failed, %.1f GiB populated, %" PRIu64 " layers predicted%s\n",
            __func__, enq, p.n_done.load(), p.n_dropped.load(), p.n_failed.load(),
            p.bytes.load() / (1024.0*1024.0*1024.0), p.ps.n_predicted.load(),
            p.use_populate.load() ? "" : " (MADV_POPULATE_READ unavailable, used page touch)");
}

#else // !LLAMA_EXPERT_PREFETCH_POSIX

struct llama_expert_prefetcher::impl {};

llama_expert_prefetcher::llama_expert_prefetcher() : pimpl(new impl()) {}
llama_expert_prefetcher::~llama_expert_prefetcher() = default;

std::unique_ptr<llama_expert_prefetcher> llama_expert_prefetcher::create(
        const llama_model &                            model,
        const std::vector<std::pair<void *, size_t>> & mapped_regions) {
    GGML_UNUSED(model);
    GGML_UNUSED(mapped_regions);
    return nullptr;
}

void llama_expert_prefetcher::install()      {}
void llama_expert_prefetcher::uninstall()    {}
void llama_expert_prefetcher::print_stats() const {}

#endif // LLAMA_EXPERT_PREFETCH_POSIX
