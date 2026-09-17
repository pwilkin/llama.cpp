#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    // token slots per warp lane group, padded to a power of 2 so a warp divides evenly
    constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

// ---------------------------------------------------------------------------
// Stable counting sort of the (token, slot) pairs by expert.
//
// The scan implementation above gives one warp to each expert and walks every
// token with it, so its cost grows as n_experts*n_tokens no matter how few
// pairs there are. A model with 512 experts and 2048 tokens spends a million
// warp steps to bucket twenty thousand pairs.
//
// Counting sort costs O(n_pairs + n_experts*n_tiles) instead, and produces the
// same order: pairs ascending by token, and by slot within a token.
//   1. mm_ids_count     per tile histogram, and the total per expert
//   2. mm_ids_bounds    exclusive scan of the totals -> expert_bounds
//   3. mm_ids_starts    per expert, exclusive scan over tiles -> tile start
//   4. mm_ids_scatter   each tile writes its pairs, in order
// ---------------------------------------------------------------------------

#define MM_IDS_TILE 256 // pairs per tile

// exclusive scan of val across the block, returns this thread's prefix; *total gets the sum
template <int block_size>
static __device__ __forceinline__ int mm_ids_scan_block(int val, int * total, int * tmp) {
    const int tid = threadIdx.x;
    tmp[tid] = val;
    __syncthreads();
    for (int off = 1; off < block_size; off <<= 1) {
        const int add = tid >= off ? tmp[tid - off] : 0;
        __syncthreads();
        tmp[tid] += add;
        __syncthreads();
    }
    const int inclusive = tmp[tid];
    *total = tmp[block_size - 1];
    __syncthreads();
    return inclusive - val;
}

static __global__ void mm_ids_count(
        const int32_t * __restrict__ ids, int32_t * __restrict__ tile_hist, int32_t * __restrict__ counts,
        const int n_pairs, const int n_experts, const int si1, const uint3 neu_fd) {
    extern __shared__ int32_t hist[];

    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        hist[e] = 0;
    }
    __syncthreads();

    const int p0 = blockIdx.x * MM_IDS_TILE;
    for (int p = p0 + threadIdx.x; p < min(p0 + MM_IDS_TILE, n_pairs); p += blockDim.x) {
        const uint32_t it  = fastdiv(p, neu_fd);
        const uint32_t iex = p - it*neu_fd.z;
        atomicAdd(&hist[ids[it*si1 + iex]], 1);
    }
    __syncthreads();

    int32_t * dst = tile_hist + (size_t) blockIdx.x * n_experts;
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        const int c = hist[e];
        dst[e] = c;
        if (c) {
            atomicAdd(&counts[e], c);
        }
    }
}

template <int block_size>
static __global__ void mm_ids_bounds(
        const int32_t * __restrict__ counts, int32_t * __restrict__ expert_bounds, const int n_experts) {
    __shared__ int tmp[block_size];

    int base = 0;
    for (int e0 = 0; e0 < n_experts; e0 += block_size) {
        const int e = e0 + threadIdx.x;
        const int c = e < n_experts ? counts[e] : 0;

        int total;
        const int pref = mm_ids_scan_block<block_size>(c, &total, tmp);

        if (e < n_experts) {
            expert_bounds[e] = base + pref;
        }
        base += total;
    }

    if (threadIdx.x == 0) {
        expert_bounds[n_experts] = base;
    }
}

// one block per expert: turn its per tile counts into the position each tile starts writing at
template <int block_size>
static __global__ void mm_ids_starts(
        const int32_t * __restrict__ tile_hist, const int32_t * __restrict__ expert_bounds,
        int32_t * __restrict__ tile_start, const int n_tiles, const int n_experts) {
    __shared__ int tmp[block_size];

    const int e = blockIdx.x;
    int base = expert_bounds[e];

    for (int t0 = 0; t0 < n_tiles; t0 += block_size) {
        const int t = t0 + threadIdx.x;
        const int c = t < n_tiles ? tile_hist[(size_t) t*n_experts + e] : 0;

        int total;
        const int pref = mm_ids_scan_block<block_size>(c, &total, tmp);

        if (t < n_tiles) {
            tile_start[(size_t) t*n_experts + e] = base + pref;
        }
        base += total;
    }
}

// one warp per tile, walking the tile in lane order so equal experts keep their pair order
static __global__ __launch_bounds__(32) void mm_ids_scatter(
        const int32_t * __restrict__ ids, const int32_t * __restrict__ tile_start,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int n_pairs, const int n_experts, const int n_expert_used, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse, const uint3 neu_fd) {
    extern __shared__ int32_t cur[];

    const int lane = threadIdx.x;

    const int32_t * start = tile_start + (size_t) blockIdx.x * n_experts;
    for (int e = lane; e < n_experts; e += 32) {
        cur[e] = start[e];
    }
    __syncwarp();

    const int p0 = blockIdx.x * MM_IDS_TILE;
    const int p1 = min(p0 + MM_IDS_TILE, n_pairs);

    for (int pc = p0; pc < p1; pc += 32) {
        const int p = pc + lane;
        const bool active = p < p1;

        uint32_t it = 0, iex = 0;
        int e = -1;
        if (active) {
            const uint2 dm = fast_div_modulo(p, neu_fd);
            it  = dm.x;
            iex = dm.y;
            e   = ids[it*si1 + iex];
        }

        // rank among the lanes of this chunk that hold the same expert, and whether this is the
        // last of them - the last lane is the one that advances the shared cursor
        int  rank    = 0;
        bool is_last = true;
        for (int l = 0; l < 32; ++l) {
            const int el = __shfl_sync(0xFFFFFFFF, e, l, 32);
            if (e >= 0 && el == e) {
                rank    += l < lane;
                is_last &= l <= lane;
            }
        }

        const int pos = active ? cur[e] + rank : 0;
        __syncwarp(); // every lane has read its cursor before any lane advances one

        if (active) {
            ids_dst[pos] = it*n_expert_used + iex;
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex] = pos;
            } else {
                ids_src1[pos] = it*sis1 + iex % nchannels_y;
            }
            if (is_last) {
                cur[e] = pos + 1;
            }
        }
        __syncwarp();
    }
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

// counting sort of n_pairs pairs into n_experts buckets; see the kernels above
static void launch_mm_ids_sort(
        ggml_backend_cuda_context & ctx,
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse) {
    cudaStream_t stream = ctx.stream();

    const int n_pairs = n_tokens*n_expert_used;
    const int n_tiles = (n_pairs + MM_IDS_TILE - 1) / MM_IDS_TILE;

    const uint3  neu_fd      = init_fastdiv_values(n_expert_used);
    const size_t nbytes_hist = (size_t) n_experts*sizeof(int32_t);

    ggml_cuda_pool_alloc<int32_t> tile_hist (ctx.pool(), (size_t) n_tiles*n_experts);
    ggml_cuda_pool_alloc<int32_t> tile_start(ctx.pool(), (size_t) n_tiles*n_experts);
    ggml_cuda_pool_alloc<int32_t> counts    (ctx.pool(), (size_t) n_experts);

    CUDA_CHECK(cudaMemsetAsync(counts.get(), 0, nbytes_hist, stream));

    if (n_tiles > 0) {
        mm_ids_count<<<n_tiles, 256, nbytes_hist, stream>>>
            (ids, tile_hist.get(), counts.get(), n_pairs, n_experts, si1, neu_fd);
    }

    mm_ids_bounds<256><<<1, 256, 0, stream>>>
        (counts.get(), expert_bounds, n_experts);

    mm_ids_starts<256><<<n_experts, 256, 0, stream>>>
        (tile_hist.get(), expert_bounds, tile_start.get(), n_tiles, n_experts);

    if (n_tiles > 0) {
        mm_ids_scatter<<<n_tiles, 32, nbytes_hist, stream>>>
            (ids, tile_start.get(), ids_src1, ids_dst, n_pairs, n_experts, n_expert_used, nchannels_y,
             si1, sis1, write_inverse, neu_fd);
    }
}

void ggml_cuda_launch_mm_ids_helper(
        ggml_backend_cuda_context & ctx,
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    cudaStream_t stream = ctx.stream();

    // the sort holds one histogram per block in shared memory, so a model with more experts than
    // fit there keeps the scan below; measured, its extra launches cost decode nothing, so there
    // is no size threshold beyond that
    const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    static const bool no_sort = getenv("GGML_CUDA_NO_MM_IDS_SORT") != nullptr;

    if (!no_sort && (size_t) n_experts*sizeof(int32_t) <= smpbo) {
        launch_mm_ids_sort(ctx, ids, ids_src1, ids_dst, expert_bounds,
            n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
        return;
    }

    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
