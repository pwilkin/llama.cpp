#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

struct ggml_tensor;
struct llama_model;

// Fire-and-forget prefetcher for routed MoE expert rows on an mmap'd model (--lazy-experts).
//
// Why this exists: MUL_MAT_ID walks its experts strictly one at a time (the cur_a loop in
// ggml-cpu.c), and the compute threads split rows *within* one expert. So on a model whose experts
// do not fit in RAM, expert e+1's pages are not requested until expert e has finished computing,
// and reads never overlap compute. Measured on a 213 GB GLM-5.2 that leaves the NVMe idle ~67% of
// wall time at an average queue depth of ~6, with read latency flat at 0.16 ms -- i.e. the device
// has headroom and is simply not being asked. Correcting for the duty cycle, the bursts themselves
// already run at the device's capability, so the lever is starting the reads earlier, not deeper.
//
// This object is handed each MUL_MAT_ID node as it begins (see ggml_cpu_set_mul_mat_id_prefetch_hook)
// and pushes that layer's routed expert rows onto a small worker pool, which faults them in with
// MADV_POPULATE_READ while the calling node computes. Requests are never waited on: anything not
// resident by the time a compute thread reaches it just faults in the usual way.
//
// Deliberately NOT MADV_WILLNEED: WILLNEED pages land on the inactive LRU list as not-yet-accessed
// and, under a full and churning page cache, are reclaimed before first touch and then read again
// (measured: bytes/token doubled, 1.573 -> 0.679 t/s). MADV_POPULATE_READ faults synchronously on
// the calling thread, so pages enter as accessed and stay.
//
// POSIX/mmap only; create() returns nullptr when there is nothing it can manage.
struct llama_expert_prefetcher {
    // Register the model's routed expert tensors that live inside one of its file mappings.
    // `mapped_regions` are the model's mmap (base, size) pairs -- a tensor is only ever advised if
    // its rows fall wholly inside one of them, so this can never touch a malloc'd or device buffer.
    // Returns nullptr if there are no routed experts in a mapping, or on unsupported platforms.
    static std::unique_ptr<llama_expert_prefetcher> create(
            const llama_model &                            model,
            const std::vector<std::pair<void *, size_t>> & mapped_regions);

    ~llama_expert_prefetcher();

    llama_expert_prefetcher(const llama_expert_prefetcher &)             = delete;
    llama_expert_prefetcher & operator=(const llama_expert_prefetcher &) = delete;

    // Install/remove this prefetcher as the process-wide MUL_MAT_ID hook.
    void install();
    static void uninstall();

    void print_stats() const;

    struct impl;

private:
    llama_expert_prefetcher();

    std::unique_ptr<impl> pimpl;
};
