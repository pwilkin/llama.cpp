#include "ggml-xdna.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstring>

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

    for (int64_t i = 0; i < n; i++) {
        d[i] = ggml_backend_xdna_unary_f32(op, s[i]);
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
