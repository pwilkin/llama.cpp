#include "llama-expert-prefetch.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpu.h"
#include "ggml.h"

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
    };

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

    void on_node(const ggml_tensor * src0, const ggml_tensor * ids) {
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

static void llama_expert_prefetch_hook(const ggml_tensor * src0, const ggml_tensor * ids, void * user_data) {
    auto * p = (llama_expert_prefetcher::impl *) user_data;
    if (p) {
        p->on_node(src0, ids);
    }
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
        p.groups.push_back(g);
        for (int k = 0; k < g.n; ++k) {
            p.by_tensor[g.t[k]] = idx;
            n_tensors++;
        }
    }

    if (p.groups.empty()) {
        return nullptr;
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
    const uint64_t enq = p.n_enqueued.load();
    if (enq == 0) {
        return;
    }
    LLAMA_LOG_INFO("%s: expert prefetch: %" PRIu64 " queued, %" PRIu64 " completed, %" PRIu64
                   " dropped, %" PRIu64 " failed, %.1f GiB populated%s\n",
            __func__, enq, p.n_done.load(), p.n_dropped.load(), p.n_failed.load(),
            p.bytes.load() / (1024.0*1024.0*1024.0),
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
