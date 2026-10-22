#pragma once

#include "ggml-moe-single.h"
#include "moe-single-publish.cuh"

#if defined(GGML_CUDA_MOE_SINGLE)

static void ggml_moe_single_publish_marker(ggml_tensor *, int, int, void *) {
    GGML_ABORT("single-graph MoE publication requires a supported GPU backend");
}

static void ggml_moe_single_merge_marker(ggml_tensor *, int, int, void *) {
    GGML_ABORT("single-graph MoE merge requires a supported GPU backend");
}

static bool ggml_moe_single_is_op(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) return false;
    ggml_custom_op_params params;
    memcpy(&params, op->op_params, sizeof(params));
    return params.fun == ggml_moe_single_publish_marker || params.fun == ggml_moe_single_merge_marker;
}

static __device__ uint64_t ggml_moe_single_clock() {
#if defined(GGML_USE_HIP)
    return wall_clock64();
#else
    uint64_t ticks;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ticks));
    return ticks;
#endif
}

template <bool parallel>
static __global__ void ggml_moe_single_publish_kernel(const int32_t * ids, const float * input, int32_t * slots,
        const int32_t * map, ggml_moe_single_control * control, int32_t * host_ids, int32_t * trace_ids, float * host_input,
        int n_expert, int n_used, int n_embd, int zero_slot, uint32_t * abort) {
    ggml_moe_single_publish_device<parallel>(ids, input, slots, map, control, host_ids, trace_ids, host_input,
        n_expert, n_used, n_embd, zero_slot, abort, threadIdx.x, blockDim.x);
}

static __global__ void ggml_moe_single_merge_kernel(const float * gpu, const int32_t * slots, float * dst,
        ggml_moe_single_control * control, const float * host_output, int n_embd, int zero_slot, uint64_t timeout_ticks, uint32_t * abort) {
    const int row = blockIdx.x;
    __shared__ bool failed;
    if (threadIdx.x == 0) {
        failed = ggml_moe_single_load(&control->cancel.value) || ggml_moe_single_load(abort) || ggml_moe_single_load(&control->error.value);
        if (!failed && slots[row] == zero_slot) {
            const uint32_t request = ggml_moe_single_load(&control->request.value);
            const uint64_t start = ggml_moe_single_clock();
            while (ggml_moe_single_load(&control->completed.value) != request) {
                if (ggml_moe_single_load(&control->cancel.value) || ggml_moe_single_load(abort) || ggml_moe_single_load(&control->error.value)) {
                    failed = true;
                    break;
                }
                if (ggml_moe_single_clock() - start > timeout_ticks) {
                    ggml_moe_single_fail(control, abort, GGML_MOE_SINGLE_TIMEOUT);
                    failed = true;
                    break;
                }
#if defined(GGML_USE_HIP)
                __builtin_amdgcn_s_sleep(1);
#elif __CUDA_ARCH__ >= 700
                __nanosleep(64);
#endif
            }
            failed |= ggml_moe_single_load(abort) != 0 || ggml_moe_single_load(&control->error.value) != 0;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const int index = row*n_embd + i;
        dst[index] = failed ? NAN : gpu[index] + (slots[row] == zero_slot ? host_output[index] : 0.0f);
    }
}

static void ggml_moe_single_compute(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_custom_op_params params;
    memcpy(&params, dst->op_params, sizeof(params));
    auto * layer = static_cast<ggml_moe_single_layer *>(params.userdata);
    GGML_ASSERT(layer && layer->device_id == ctx.device);
    if (params.fun == ggml_moe_single_publish_marker) {
        ggml_moe_single_publish_kernel<false><<<1, 256, 0, ctx.stream()>>>(
                static_cast<const int32_t *>(dst->src[0]->data), static_cast<const float *>(dst->src[1]->data),
                static_cast<int32_t *>(dst->data), layer->device_map, layer->device, layer->device_ids,
                layer->device_trace_ids, layer->device_input, layer->n_expert, layer->n_used, layer->n_embd, layer->zero_slot, layer->device_abort);
    } else {
        GGML_ASSERT(params.fun == ggml_moe_single_merge_marker);
        ggml_moe_single_merge_kernel<<<layer->n_used, 256, 0, ctx.stream()>>>(
            static_cast<const float *>(dst->src[0]->data), static_cast<const int32_t *>(dst->src[1]->data),
            static_cast<float *>(dst->data), layer->device, layer->device_output, layer->n_embd, layer->zero_slot, layer->timeout_ticks, layer->device_abort);
    }
}

static void ggml_moe_single_destroy(ggml_moe_single_layer * layer) {
    if (!layer) return;
    ggml_cuda_set_device(layer->device_id);
    if (layer->device_map) {
        ggml_cuda_storage_changed();
        CUDA_CHECK(cudaFree(layer->device_map));
    }
    if (layer->host) {
        ggml_cuda_storage_changed();
        CUDA_CHECK(cudaFreeHost(layer->host));
    }
    delete layer;
}

static ggml_moe_single_layer * ggml_moe_single_create(ggml_backend_t backend, int n_expert, int n_used, int n_embd,
        int zero_slot, const int32_t * map) {
    if (!ggml_backend_is_cuda(backend) || n_expert < 1 || n_expert > 65536 || n_used < 1 || n_used > 512 ||
            n_used > n_expert || n_embd < 1 || n_embd > 65536 || zero_slot < 0 || !map) return nullptr;
    for (int i = 0; i < n_expert; ++i) if (map[i] < -1 || map[i] >= zero_slot) return nullptr;
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx->device);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));
    if (!prop.canMapHostMemory) return nullptr;
    if (prop.major < 7 && !GGML_CUDA_CC_IS_AMD(ggml_cuda_info().devices[ctx->device].cc)) return nullptr;
#if defined(GGML_USE_HIP)
    int wall_clock_khz = 0;
    if (hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, ggml_cuda_get_physical_device(ctx->device)) != cudaSuccess || wall_clock_khz <= 0) {
        GGML_LOG_ERROR("single-graph MoE requires a constant-frequency HIP wall clock\n");
        return nullptr;
    }
    const uint64_t timeout_ticks = uint64_t(wall_clock_khz)*1000*30;
#else
    const uint64_t timeout_ticks = UINT64_C(30000000000);
#endif
    auto * layer = new ggml_moe_single_layer{};
    static std::atomic<uint64_t> next_cookie{1};
    layer->creation_cookie = next_cookie.fetch_add(1, std::memory_order_relaxed);
    GGML_ASSERT(layer->creation_cookie != 0);
    layer->device_id = ctx->device;
    layer->n_expert = n_expert;
    layer->n_used = n_used;
    layer->n_embd = n_embd;
    layer->zero_slot = zero_slot;
    layer->timeout_ticks = timeout_ticks;
    const size_t ids_size = n_used*sizeof(int32_t);
    const size_t input_size = n_embd*sizeof(float);
    const size_t bytes = sizeof(ggml_moe_single_control) + 2*ids_size + input_size + size_t(n_embd)*n_used*sizeof(float);
    void * host = nullptr;
    void * device = nullptr;
    unsigned int alloc_flags = cudaHostAllocMapped;
#if defined(GGML_USE_HIP)
    alloc_flags |= hipHostMallocCoherent;
#endif
    if (cudaHostAlloc(&host, bytes, alloc_flags) != cudaSuccess) {
        delete layer;
        return nullptr;
    }
    ggml_cuda_storage_changed();
    memset(host, 0, bytes);
    layer->host = static_cast<ggml_moe_single_control *>(host);
    bool coherent = true;
#if defined(GGML_USE_HIP)
    unsigned int flags = 0;
    coherent = hipHostGetFlags(&flags, host) == cudaSuccess && (flags & hipHostMallocCoherent);
#endif
    if (!coherent || cudaHostGetDevicePointer(&device, host, 0) != cudaSuccess) {
        ggml_moe_single_destroy(layer);
        return nullptr;
    }
    layer->device = static_cast<ggml_moe_single_control *>(device);
    layer->host_abort = &layer->host->abort.value;
    layer->device_abort = &layer->device->abort.value;
    layer->ids = reinterpret_cast<int32_t *>(layer->host + 1);
    layer->trace_ids = layer->ids + n_used;
    layer->input = reinterpret_cast<float *>(layer->trace_ids + n_used);
    layer->output = layer->input + n_embd;
    layer->device_ids = reinterpret_cast<int32_t *>(layer->device + 1);
    layer->device_trace_ids = layer->device_ids + n_used;
    layer->device_input = reinterpret_cast<float *>(layer->device_trace_ids + n_used);
    layer->device_output = layer->device_input + n_embd;
    if (cudaMalloc(reinterpret_cast<void **>(&layer->device_map), n_expert*sizeof(int32_t)) != cudaSuccess) {
        ggml_moe_single_destroy(layer);
        return nullptr;
    }
    ggml_cuda_storage_changed();
    CUDA_CHECK(cudaMemcpy(layer->device_map, map, n_expert*sizeof(int32_t), cudaMemcpyHostToDevice));
    return layer;
}

static ggml_tensor * ggml_moe_single_publish(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * input, ggml_moe_single_layer * layer) {
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_nelements(ids) == layer->n_used && ggml_is_contiguous(ids));
    GGML_ASSERT(input->type == GGML_TYPE_F32 && ggml_nelements(input) == layer->n_embd && ggml_is_contiguous(input));
    ggml_tensor * args[] = { ids, input };
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_I32, layer->n_used, 1, 1, 1, args, 2, ggml_moe_single_publish_marker, 1, layer);
    static_assert(sizeof(ggml_custom_op_params) + sizeof(layer->creation_cookie) <= GGML_MAX_OP_PARAMS, "custom parameters overflow");
    memcpy(reinterpret_cast<char *>(result->op_params) + sizeof(ggml_custom_op_params), &layer->creation_cookie, sizeof(layer->creation_cookie));
    return result;
}

static ggml_tensor * ggml_moe_single_merge(ggml_context * ctx, ggml_tensor * gpu, ggml_tensor * slots, ggml_moe_single_layer * layer) {
    GGML_ASSERT(gpu->type == GGML_TYPE_F32 && gpu->ne[0] == layer->n_embd && gpu->ne[1] == layer->n_used && gpu->ne[2] == 1 && gpu->ne[3] == 1 && ggml_is_contiguous(gpu));
    GGML_ASSERT(slots->type == GGML_TYPE_I32 && ggml_nelements(slots) == layer->n_used && ggml_is_contiguous(slots));
    ggml_tensor * args[] = { gpu, slots };
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32, layer->n_embd, layer->n_used, 1, 1, args, 2, ggml_moe_single_merge_marker, 1, layer);
    const uint64_t captured[] = { layer->creation_cookie, layer->timeout_ticks };
    static_assert(sizeof(ggml_custom_op_params) + sizeof(captured) <= GGML_MAX_OP_PARAMS, "custom parameters overflow");
    memcpy(reinterpret_cast<char *>(result->op_params) + sizeof(ggml_custom_op_params), captured, sizeof(captured));
    return result;
}

static const ggml_moe_single_api * ggml_moe_single_get_api() {
    static const ggml_moe_single_api api = { ggml_moe_single_create, ggml_moe_single_destroy, ggml_moe_single_publish, ggml_moe_single_merge };
    return &api;
}

static bool ggml_moe_single_update(ggml_backend_t backend, const ggml_moe_single_bank_write * banks, size_t n_banks,
        const ggml_moe_single_map_write * maps, size_t n_maps, bool * mutation_started) {
    if (!mutation_started) return false;
    *mutation_started = false;
    if (!ggml_backend_is_cuda(backend) || !banks || !n_banks || !maps || !n_maps) return false;
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (cudaSetDevice(ggml_cuda_get_physical_device(ctx->device)) != cudaSuccess) return false;
    const auto stream = ctx->streams[ctx->device][ctx->curr_stream_no];
#if defined(GGML_USE_HIP)
    hipStreamCaptureStatus capture;
    if (!stream || hipStreamIsCapturing(stream, &capture) != cudaSuccess || capture != hipStreamCaptureStatusNone) return false;
#else
    cudaStreamCaptureStatus capture;
    if (!stream || cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone) return false;
#endif
    for (size_t i = 0; i < n_banks; ++i) {
        const auto & w = banks[i];
        if (!w.bank || !w.bank->data || !w.bank->buffer || w.bank->view_src || !w.data || !w.size ||
                w.bank->buffer->buft != ggml_backend_cuda_buffer_type(ctx->device) ||
                w.offset > ggml_nbytes(w.bank) || w.size > ggml_nbytes(w.bank) - w.offset) return false;
    }
    for (size_t i = 0; i < n_maps; ++i) {
        const auto & w = maps[i];
        if (!w.layer || !w.map || !w.layer->device_map || w.layer->device_id != ctx->device ||
                w.n_slots < 1 || w.n_slots != w.layer->zero_slot - w.layer->n_used) return false;
        for (int32_t e = 0; e < w.layer->n_expert; ++e) if (w.map[e] < -1 || w.map[e] >= w.n_slots) return false;
    }
    *mutation_started = true;
    bool success = true;
    for (size_t i = 0; success && i < n_banks; ++i) {
        const auto & w = banks[i];
        success = cudaMemcpyAsync(static_cast<uint8_t *>(w.bank->data) + w.offset, w.data, w.size, cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }
    success = cudaStreamSynchronize(stream) == cudaSuccess && success;
    for (size_t i = 0; success && i < n_maps; ++i) {
        const auto & w = maps[i];
        success = cudaMemcpyAsync(w.layer->device_map, w.map, size_t(w.layer->n_expert)*sizeof(int32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }
    return cudaStreamSynchronize(stream) == cudaSuccess && success;
}

static const ggml_moe_single_update_api_v1 * ggml_moe_single_get_update_api_v1() {
    static const ggml_moe_single_update_api_v1 api = { ggml_moe_single_update };
    return &api;
}

#endif
