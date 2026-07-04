#include "ggml-xdna.h"
#include "xdna-device.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#    include <unistd.h>
#endif

// ============================================================================
// AMD XDNA (Ryzen AI NPU) backend — initial scaffold.
//
// This first cut wires the backend into ggml (device registration, op support
// query, graph dispatch) and implements a handful of elementwise unary ops so
// the plumbing can be validated end-to-end (e.g. `test-backend-ops -b XDNA0`).
//
// The unary kernels here run on the host for now: the NPU shares system memory
// with the CPU, so the tensor data is directly addressable and this lets us
// confirm the scheduler dispatches ops to this backend and that results are
// correct. Real on-NPU execution (XRT buffer objects + AIE `.xclbin` kernels,
// enabled by GGML_XDNA_HAS_XRT) will replace these host loops op by op.
// ============================================================================

struct ggml_backend_xdna_context {
    // reserved for XRT device/kernel handles once on-NPU execution lands
};

// ---- unary op kernels (host placeholder) -----------------------------------

static const float XDNA_GELU_COEF_A     = 0.044715f;
static const float XDNA_SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

static bool ggml_backend_xdna_supports_unary(enum ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_SGN:
        case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_ELU:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_HARDSIGMOID:
        case GGML_UNARY_OP_HARDSWISH:
        case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_SILU:
            return true;
        default:
            return false;
    }
}

static float ggml_backend_xdna_unary_f32(enum ggml_unary_op op, float x) {
    switch (op) {
        case GGML_UNARY_OP_ABS:         return fabsf(x);
        case GGML_UNARY_OP_SGN:         return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
        case GGML_UNARY_OP_NEG:         return -x;
        case GGML_UNARY_OP_STEP:        return (x > 0.f) ? 1.f : 0.f;
        case GGML_UNARY_OP_TANH:        return tanhf(x);
        case GGML_UNARY_OP_ELU:         return (x > 0.f) ? x : expm1f(x);
        case GGML_UNARY_OP_RELU:        return (x > 0.f) ? x : 0.f;
        case GGML_UNARY_OP_SIGMOID:     return 1.f / (1.f + expf(-x));
        case GGML_UNARY_OP_HARDSIGMOID: return fminf(1.f, fmaxf(0.f, (x + 3.f) / 6.f));
        case GGML_UNARY_OP_HARDSWISH:   return x * fminf(1.f, fmaxf(0.f, (x + 3.f) / 6.f));
        case GGML_UNARY_OP_EXP:         return expf(x);
        case GGML_UNARY_OP_GELU:        return 0.5f * x * (1.f + tanhf(XDNA_SQRT_2_OVER_PI * x * (1.f + XDNA_GELU_COEF_A * x * x)));
        case GGML_UNARY_OP_GELU_QUICK:  return x * (1.f / (1.f + expf(-1.702f * x)));
        case GGML_UNARY_OP_SILU:        return x / (1.f + expf(-x));
        default:                        return x; // unreachable: guarded by supports_op
    }
}

static void ggml_backend_xdna_compute_unary(ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const enum ggml_unary_op op = ggml_get_unary_op(dst);

    const float * s = (const float *) src->data;
    float *       d = (float *)       dst->data;
    const int64_t n = ggml_nelements(dst);

    // try the NPU (XRT + .xclbin) first; fall back to the host kernel otherwise
    if (ggml_xdna_npu_try_unary(op, s, d, n)) {
        return;
    }

    for (int64_t i = 0; i < n; i++) {
        d[i] = ggml_backend_xdna_unary_f32(op, s[i]);
    }
}

// dst = mul_mat(a, b): dst[n,m] = sum_k a(k,n) * b(k,m). Try the NPU; if there
// is no xclbin for this shape, fall back to a host matmul so the op is always
// correct (a is dequantized to f32 for the fallback).
static void ggml_backend_xdna_compute_mul_mat(ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];

    if (ggml_xdna_npu_try_mul_mat(a, b, dst)) {
        return; // ran on the NPU
    }

    // host fallback
    const int64_t K = a->ne[0];
    const int64_t N = a->ne[1];
    const int64_t M = b->ne[1];

    const float * bf = (const float *) b->data;
    float *       df = (float *)       dst->data;

    const float * af;
    if (a->type == GGML_TYPE_F32) {
        af = (const float *) a->data;
    } else {
        // Weights are constant across calls, so dequantize once and cache by data
        // pointer. This is only the host-fallback path (shapes without an xclbin);
        // it keeps whole-model runs on XDNA tolerable while ops migrate to the NPU.
        static std::unordered_map<const void *, std::vector<float>> cache;
        static std::mutex cache_mtx;
        std::vector<float> * slot;
        {
            std::lock_guard<std::mutex> lk(cache_mtx);
            auto it = cache.find(a->data);
            if (it == cache.end()) {
                it = cache.emplace(a->data, std::vector<float>((size_t) K * N)).first;
                ggml_get_type_traits(a->type)->to_float(a->data, it->second.data(), (int64_t) K * N);
            }
            slot = &it->second; // node storage is stable; safe to use after unlock
        }
        af = slot->data();
    }

    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                acc += af[k + n * K] * bf[k + m * K];
            }
            df[n + m * N] = acc;
        }
    }
}

// RMS norm over each row of ne[0]: y = x / sqrt(mean(x^2) + eps). NPU kernel
// bakes eps = 1e-6, so only that eps uses the NPU; otherwise host fallback.
static void ggml_backend_xdna_compute_rms_norm(ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    float eps;
    std::memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne0   = src->ne[0];
    const int64_t nrows = ggml_nrows(src);
    const float * s = (const float *) src->data;
    float *       d = (float *)       dst->data;

    if (eps == 1e-6f && ggml_xdna_npu_try_rows("rms_norm", (int) ne0, s, d, ne0 * nrows)) {
        return;
    }
    for (int64_t r = 0; r < nrows; r++) {
        const float * x = s + r * ne0;
        float *       y = d + r * ne0;
        double ss = 0.0;
        for (int64_t i = 0; i < ne0; i++) ss += (double) x[i] * x[i];
        const float sc = 1.0f / sqrtf((float) (ss / ne0) + eps);
        for (int64_t i = 0; i < ne0; i++) y[i] = x[i] * sc;
    }
}

// Plain softmax over each row of ne[0] (no mask, scale 1) — the eligible case.
static void ggml_backend_xdna_compute_soft_max(ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const int64_t ne0   = src->ne[0];
    const int64_t nrows = ggml_nrows(src);
    const float * s = (const float *) src->data;
    float *       d = (float *)       dst->data;

    if (ggml_xdna_npu_try_rows("softmax", (int) ne0, s, d, ne0 * nrows)) {
        return;
    }
    for (int64_t r = 0; r < nrows; r++) {
        const float * x = s + r * ne0;
        float *       y = d + r * ne0;
        float mx = -INFINITY;
        for (int64_t i = 0; i < ne0; i++) mx = std::max(mx, x[i]);
        double sum = 0.0;
        for (int64_t i = 0; i < ne0; i++) { y[i] = expf(x[i] - mx); sum += y[i]; }
        const float inv = 1.0f / (float) sum;
        for (int64_t i = 0; i < ne0; i++) y[i] *= inv;
    }
}

// dst = a (op) b, elementwise with ggml broadcast of b. Same-shape contiguous
// runs on the NPU; broadcast cases use a host fallback.
static void ggml_backend_xdna_compute_binary(ggml_tensor * dst, const char * op, bool is_mul) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const float * af = (const float *) a->data;
    const float * bf = (const float *) b->data;
    float *       df = (float *)       dst->data;

    if (ggml_are_same_shape(a, b) && ggml_is_contiguous(a) && ggml_is_contiguous(b) &&
        ggml_xdna_npu_try_binary(op, af, bf, df, ggml_nelements(dst))) {
        return;
    }

    // host fallback with broadcast (contiguous)
    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    for (int64_t i3 = 0; i3 < ne3; i3++)
    for (int64_t i2 = 0; i2 < ne2; i2++)
    for (int64_t i1 = 0; i1 < ne1; i1++)
    for (int64_t i0 = 0; i0 < ne0; i0++) {
        const int64_t di = i0 + ne0 * (i1 + ne1 * (i2 + ne2 * i3));
        const int64_t bi = (i0 % b->ne[0]) + b->ne[0] * ((i1 % b->ne[1]) +
                            b->ne[1] * ((i2 % b->ne[2]) + b->ne[2] * (i3 % b->ne[3])));
        df[di] = is_mul ? af[di] * bf[bi] : af[di] + bf[bi];
    }
}

// GET_ROWS: gather rows of the (possibly quantized) table `a` at indices `ids`
// into f32 dst. A pure gather — no NPU compute benefit — done on the host, but
// it keeps the op on this backend and exercises the all-quants dequantizer.
static void ggml_backend_xdna_compute_get_rows(ggml_tensor * dst) {
    const ggml_tensor * a   = dst->src[0]; // table [ne0, n_rows, ...]
    const ggml_tensor * ids = dst->src[1]; // i32 indices
    const int64_t ne0  = a->ne[0];
    const int64_t nids = ggml_nelements(ids);
    const struct ggml_type_traits * tt = ggml_get_type_traits(a->type);
    float * d = (float *) dst->data;

    for (int64_t i = 0; i < nids; i++) {
        const int32_t row = ((const int32_t *) ids->data)[i];
        const void *  src = (const char *) a->data + row * a->nb[1];
        float *       out = d + i * ne0;
        if (a->type == GGML_TYPE_F32) {
            std::memcpy(out, src, ne0 * sizeof(float));
        } else {
            tt->to_float(src, out, ne0);
        }
    }
}

// FLASH_ATTN_EXT: dst[dv,h,iq,i3] = softmax(scale*Q·K^T + mask)·V, per head with
// GQA. Correct host implementation (the eligible case: no ALiBi bias, no sinks;
// softcap and f16/quantized K/V handled). Serves as the always-correct base;
// the NPU per-head path (below) accelerates the common head sizes.
static void ggml_backend_xdna_compute_flash_attn_ext(ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f, max_bias = 0.0f, softcap = 0.0f;
    std::memcpy(&scale,   (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias,(const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (softcap != 0.0f) scale /= softcap;

    const int64_t DK  = q->ne[0], DV = v->ne[0];
    const int64_t NQ  = q->ne[1], NH = q->ne[2], NB = q->ne[3];
    const int64_t NKV = k->ne[1];
    const int64_t rk2 = NH / k->ne[2], rk3 = NB / k->ne[3];
    const int64_t rv2 = NH / v->ne[2], rv3 = NB / v->ne[3];

    // try the NPU streaming flash kernel first (falls back to host below)
    if (softcap == 0.0f && ggml_xdna_npu_try_flash(q, k, v, mask, dst, scale)) {
        return;
    }

    const struct ggml_type_traits * ttk = ggml_get_type_traits(k->type);
    const struct ggml_type_traits * ttv = ggml_get_type_traits(v->type);
    float * d = (float *) dst->data;

    std::vector<float> krow(DK), vrow(DV), s(NKV);
    auto deq = [](const struct ggml_type_traits * tt, ggml_type t, const void * src, float * out, int64_t n) {
        if (t == GGML_TYPE_F32) std::memcpy(out, src, n * sizeof(float));
        else                    tt->to_float(src, out, n);
    };

    for (int64_t i3 = 0; i3 < NB; i3++)
    for (int64_t h  = 0; h  < NH; h++)
    for (int64_t iq = 0; iq < NQ; iq++) {
        const float * qr = (const float *) ((const char *) q->data + iq * q->nb[1] + h * q->nb[2] + i3 * q->nb[3]);
        const ggml_fp16_t * mr = mask ? (const ggml_fp16_t *) ((const char *) mask->data +
                                     iq * mask->nb[1] + (i3 % mask->ne[3]) * mask->nb[3]) : nullptr;
        float mx = -INFINITY;
        for (int64_t ic = 0; ic < NKV; ic++) {
            const void * kp = (const char *) k->data + ic * k->nb[1] + (h / rk2) * k->nb[2] + (i3 / rk3) * k->nb[3];
            deq(ttk, k->type, kp, krow.data(), DK);
            float acc = 0.0f;
            for (int64_t dd = 0; dd < DK; dd++) acc += qr[dd] * krow[dd];
            acc *= scale;
            if (softcap != 0.0f) acc = softcap * tanhf(acc);
            if (mr) acc += ggml_fp16_to_fp32(mr[ic]);
            s[ic] = acc;
            mx = std::max(mx, acc);
        }
        double sum = 0.0;
        for (int64_t ic = 0; ic < NKV; ic++) { s[ic] = expf(s[ic] - mx); sum += s[ic]; }
        const float inv = 1.0f / (float) sum;

        float * out = d + DV * (h + NH * (iq + NQ * i3));
        for (int64_t dd = 0; dd < DV; dd++) out[dd] = 0.0f;
        for (int64_t ic = 0; ic < NKV; ic++) {
            const void * vp = (const char *) v->data + ic * v->nb[1] + (h / rv2) * v->nb[2] + (i3 / rv3) * v->nb[3];
            deq(ttv, v->type, vp, vrow.data(), DV);
            const float w = s[ic] * inv;
            for (int64_t dd = 0; dd < DV; dd++) out[dd] += w * vrow[dd];
        }
    }
}

// ---- backend (stream) ------------------------------------------------------

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    return "XDNA";

    GGML_UNUSED(backend);
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    delete (ggml_backend_xdna_context *) backend->context;
    delete backend;
}

static ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_UNARY:
                ggml_backend_xdna_compute_unary(node);
                break;

            case GGML_OP_MUL_MAT:
                ggml_backend_xdna_compute_mul_mat(node);
                break;

            case GGML_OP_RMS_NORM:
                ggml_backend_xdna_compute_rms_norm(node);
                break;

            case GGML_OP_SOFT_MAX:
                ggml_backend_xdna_compute_soft_max(node);
                break;

            case GGML_OP_MUL:
                ggml_backend_xdna_compute_binary(node, "mul", true);
                break;

            case GGML_OP_ADD:
                ggml_backend_xdna_compute_binary(node, "add", false);
                break;

            case GGML_OP_GET_ROWS:
                ggml_backend_xdna_compute_get_rows(node);
                break;

            case GGML_OP_FLASH_ATTN_EXT:
                ggml_backend_xdna_compute_flash_attn_ext(node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i ggml_backend_xdna_i = {
    /* .get_name                = */ ggml_backend_xdna_get_name,
    /* .free                    = */ ggml_backend_xdna_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xdna_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static const char * guid_str = "AMD-XDNA-NPU-0001";
    return reinterpret_cast<ggml_guid_t>(const_cast<char *>(guid_str));
}

ggml_backend_t ggml_backend_xdna_init(void) {
    ggml_backend_xdna_context * ctx = new ggml_backend_xdna_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ ggml_backend_xdna_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_xdna(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xdna_guid());
}

// ---- device ----------------------------------------------------------------

// The NPU is present iff its accel device node exists (amdxdna driver loaded).
static bool ggml_backend_xdna_device_present(void) {
#ifndef _WIN32
    return access("/dev/accel/accel0", F_OK) == 0;
#else
    return false; // TODO: Windows detection via XRT
#endif
}

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    return "XDNA";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    return "AMD XDNA (Ryzen AI NPU)";

    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // shared system memory; nothing dedicated to report
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description  = ggml_backend_xdna_device_get_description(dev);
    props->type         = ggml_backend_xdna_device_get_type(dev);
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_xdna_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_xdna_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    // shared-memory accelerator: operate on host tensors
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_xdna_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_UNARY:
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op) &&
                   ggml_backend_xdna_supports_unary(ggml_get_unary_op(op));

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * a = op->src[0]; // weights [K,N]
            const struct ggml_tensor * b = op->src[1]; // activations [K,M]
            if (op->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
                return false;
            }
            if (a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1) {
                return false; // 2D only for now
            }
            if (b->ne[0] != a->ne[0] || !ggml_is_contiguous(b)) {
                return false;
            }
            // inner tile 32, and the ping-pong runtime needs M_div_m even, so
            // require M (b->ne[1]) a multiple of 64 and K,N multiples of 32.
            if (b->ne[1] % 64 || a->ne[0] % 32 || a->ne[1] % 32) {
                return false;
            }
            const struct ggml_type_traits * tt = ggml_get_type_traits(a->type);
            return a->type == GGML_TYPE_F32 || (tt && tt->to_float);
        }

        case GGML_OP_MUL:
        case GGML_OP_ADD:
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
                   ggml_is_contiguous(op) && ggml_can_repeat(op->src[1], op->src[0]);

        case GGML_OP_GET_ROWS: {
            const struct ggml_tensor * a   = op->src[0];
            const struct ggml_tensor * ids = op->src[1];
            const struct ggml_type_traits * tt = ggml_get_type_traits(a->type);
            return op->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 &&
                   ids->ne[1] == 1 && ids->ne[2] == 1 && // 1-D index list
                   (a->type == GGML_TYPE_F32 || (tt && tt->to_float));
        }

        case GGML_OP_FLASH_ATTN_EXT: {
            const struct ggml_tensor * q    = op->src[0];
            const struct ggml_tensor * k    = op->src[1];
            const struct ggml_tensor * v    = op->src[2];
            const struct ggml_tensor * mask = op->src[3];
            float max_bias = 0.0f;
            std::memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            const struct ggml_type_traits * ttk = ggml_get_type_traits(k->type);
            const struct ggml_type_traits * ttv = ggml_get_type_traits(v->type);
            return op->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 &&
                   max_bias == 0.0f &&               // no ALiBi
                   op->src[4] == nullptr &&          // no attention sinks
                   (mask == nullptr || mask->type == GGML_TYPE_F16) &&
                   (k->type == GGML_TYPE_F32 || (ttk && ttk->to_float)) &&
                   (v->type == GGML_TYPE_F32 || (ttv && ttv->to_float));
        }

        case GGML_OP_RMS_NORM:
            // any eps handled (NPU for 1e-6, host fallback otherwise)
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op);

        case GGML_OP_SOFT_MAX: {
            // only the plain case (no mask, unit scale, no ALiBi bias)
            float scale = 1.0f, max_bias = 0.0f;
            std::memcpy(&scale,    (const float *) op->op_params + 0, sizeof(float));
            std::memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1] == nullptr && op->src[2] == nullptr && // no mask, no sinks
                   scale == 1.0f && max_bias == 0.0f &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op);
        }

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name               = */ ggml_backend_xdna_device_get_name,
    /* .get_description        = */ ggml_backend_xdna_device_get_description,
    /* .get_memory             = */ ggml_backend_xdna_device_get_memory,
    /* .get_type               = */ ggml_backend_xdna_device_get_type,
    /* .get_props              = */ ggml_backend_xdna_device_get_props,
    /* .init_backend           = */ ggml_backend_xdna_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type   = */ NULL,
    /* .buffer_from_host_ptr   = */ ggml_backend_xdna_device_buffer_from_host_ptr,
    /* .supports_op            = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft          = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op             = */ NULL,
    /* .event_new              = */ NULL,
    /* .event_free             = */ NULL,
    /* .event_synchronize      = */ NULL,
};

// ---- reg -------------------------------------------------------------------

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    return "XDNA";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_xdna_device_present() ? 1 : 0;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_xdna_device = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_xdna_device;

    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static struct ggml_backend_reg ggml_backend_xdna_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xdna_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
