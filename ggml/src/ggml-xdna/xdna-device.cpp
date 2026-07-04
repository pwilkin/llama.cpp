// On-NPU execution for the XDNA backend via XRT + AIE .xclbin kernels.
//
// Compiled only with real XRT support (GGML_XDNA_HAS_XRT). Without it, the
// entry point is a stub returning false so the backend uses its host kernels.

#include "xdna-device.h"

#ifdef GGML_XDNA_HAS_XRT

#include "ggml-impl.h" // ggml_fp32_to_bf16_row / ggml_bf16_to_fp32_row

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/deprecated/xrt.h> // XCL_BO_SYNC_BO_{TO,FROM}_DEVICE

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

// Must match the length the kernels/ xclbins were built for (Makefile LENGTH).
constexpr int      KERNEL_LEN   = 4096;
constexpr uint64_t RUN_OPCODE   = 3; // "execute instruction sequence"
constexpr char     KERNEL_NAME[] = "MLIR_AIE";

const char * op_name(enum ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_RELU:       return "relu";
        case GGML_UNARY_OP_NEG:        return "neg";
        case GGML_UNARY_OP_ABS:        return "abs";
        case GGML_UNARY_OP_EXP:        return "exp";
        case GGML_UNARY_OP_TANH:       return "tanh";
        case GGML_UNARY_OP_SIGMOID:    return "sigmoid";
        case GGML_UNARY_OP_SILU:       return "silu";
        case GGML_UNARY_OP_GELU:       return "gelu";
        case GGML_UNARY_OP_GELU_QUICK: return "gelu_quick";
        default:                       return nullptr;
    }
}

std::vector<uint32_t> read_u32(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint32_t> v(n / sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(v.data()), n);
    return v;
}

// One loaded op: kernel + instruction BO (uploaded once) + persistent I/O
// buffers reused across calls (BO allocation pins memory, so we do it once).
// The AIE DPU kernel signature is (opcode, instr, instr_len, bo3..bo7) — 5
// buffer slots; this unary design uses bo3=in, bo4=out, the rest are dummies.
struct op_kernel {
    xrt::hw_context ctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    unsigned        instr_words = 0;
    xrt::bo         bo_in;
    xrt::bo         bo_out;
    ggml_bf16_t *   in_map  = nullptr;
    ggml_bf16_t *   out_map = nullptr;
    xrt::bo         dummy[3]; // slots 5,6,7
};

struct npu_state {
    xrt::device            device;
    std::string            kdir;
    bool                   ok = false;
    std::map<int, op_kernel> ops;
    std::mutex             mtx;

    npu_state() {
        const char * d = std::getenv("GGML_XDNA_KERNELS");
        if (!d || !*d) {
            return; // no kernel directory -> NPU path disabled
        }
        kdir = d;
        try {
            device = xrt::device(0);
            ok     = true;
        } catch (...) {
            ok = false;
        }
    }

    // load + cache the kernel for `op`, or nullptr if unavailable
    op_kernel * get(enum ggml_unary_op op) {
        const char * name = op_name(op);
        if (!name) {
            return nullptr;
        }
        auto it = ops.find((int) op);
        if (it != ops.end()) {
            return &it->second;
        }

        const std::vector<uint32_t> instr = read_u32(kdir + "/" + name + "_insts.bin");
        if (instr.empty()) {
            return nullptr;
        }

        try {
            xrt::xclbin xclbin(kdir + "/" + name + ".xclbin");
            device.register_xclbin(xclbin);

            op_kernel k;
            k.ctx         = xrt::hw_context(device, xclbin.get_uuid());
            k.kernel      = xrt::kernel(k.ctx, KERNEL_NAME);
            k.instr_words = (unsigned) instr.size();

            k.bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t),
                                 xrt::bo::flags::cacheable, k.kernel.group_id(1));
            std::memcpy(k.bo_instr.map<void *>(), instr.data(), instr.size() * sizeof(uint32_t));
            k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // persistent I/O buffers (reused across calls)
            k.bo_in  = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN,
                               xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.bo_out = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN,
                               xrt::bo::flags::host_only, k.kernel.group_id(4));
            k.in_map  = k.bo_in.map<ggml_bf16_t *>();
            k.out_map = k.bo_out.map<ggml_bf16_t *>();

            for (int i = 0; i < 3; i++) {
                k.dummy[i] = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN,
                                     xrt::bo::flags::host_only, k.kernel.group_id(5 + i));
            }

            auto res = ops.emplace((int) op, std::move(k));
            return &res.first->second;
        } catch (...) {
            return nullptr;
        }
    }
};

npu_state & npu() {
    static npu_state inst;
    return inst;
}

} // namespace

bool ggml_xdna_npu_try_unary(enum ggml_unary_op op, const float * src, float * dst, int64_t n) {
    npu_state & N = npu();
    if (!N.ok) {
        return false;
    }

    std::lock_guard<std::mutex> lock(N.mtx);

    op_kernel * k = N.get(op);
    if (!k) {
        return false;
    }

    try {
        for (int64_t off = 0; off < n; off += KERNEL_LEN) {
            const int64_t chunk = std::min<int64_t>(KERNEL_LEN, n - off);
            const size_t  bytes = (size_t) chunk * sizeof(ggml_bf16_t);

            // only the live [0,chunk) region matters; the kernel reads/writes the
            // full tile but we sync and read back just what we filled.
            ggml_fp32_to_bf16_row(src + off, k->in_map, chunk);
            k->bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);

            auto run = k->kernel(RUN_OPCODE, k->bo_instr, k->instr_words,
                                 k->bo_in, k->bo_out, k->dummy[0], k->dummy[1], k->dummy[2]);
            run.wait();

            k->bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
            ggml_bf16_to_fp32_row(k->out_map, dst + off, chunk);
        }
        if (std::getenv("GGML_XDNA_DEBUG")) {
            fprintf(stderr, "[xdna] ran %s on NPU (n=%lld)\n", op_name(op), (long long) n);
        }
        return true;
    } catch (const std::exception & e) {
        if (std::getenv("GGML_XDNA_DEBUG")) {
            fprintf(stderr, "[xdna] NPU run failed for %s: %s\n", op_name(op), e.what());
        }
        return false;
    } catch (...) {
        return false;
    }
}

#else // !GGML_XDNA_HAS_XRT

bool ggml_xdna_npu_try_unary(enum ggml_unary_op, const float *, float *, int64_t) {
    return false;
}

#endif
