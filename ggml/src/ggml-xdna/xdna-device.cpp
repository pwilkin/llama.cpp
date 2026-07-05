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
constexpr int      FLASH_BLK    = 16; // KV block size the flash xclbins use

// dequantize one row of `n` elements of type `t` to f32
inline void deq_row(const struct ggml_type_traits * tt, ggml_type t, const void * src, float * out, int64_t n) {
    if (t == GGML_TYPE_F32) std::memcpy(out, src, n * sizeof(float));
    else                    tt->to_float(src, out, n);
}

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

// A loaded matmul, keyed by (M,K,N). A[M,K] and B[K,N] bf16 inputs, C[M,N] f32.
struct mm_kernel {
    xrt::hw_context ctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    unsigned        instr_words = 0;
    xrt::bo         boA, boB, boC;
    ggml_bf16_t *   A_map = nullptr;
    ggml_bf16_t *   B_map = nullptr;
    float *         C_map = nullptr;
    xrt::bo         dummy[2]; // slots 6,7
};

// A loaded streaming-flash kernel for a fixed (NQ, DK, NKV). Q @3 (bf16),
// KV blocks @4 (bf16), STATE/output @5 (fp32).
struct flash_kernel {
    xrt::hw_context ctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    unsigned        instr_words = 0;
    int             NQ = 0, DK = 0, NKV = 0, NBLK = 0;
    xrt::bo         boQ, boKV, boO;
    ggml_bf16_t *   Q_map  = nullptr;
    ggml_bf16_t *   KV_map = nullptr;
    float *         O_map  = nullptr;
    xrt::bo         dummy[2];
};

// A loaded gemv kernel for a fixed (K, N). x @3 (bf16), W @4 (bf16, cached per
// weight tensor), y @5 (fp32). Used for M=1 (decode) matmuls.
struct gemv_kernel {
    xrt::hw_context ctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    unsigned        instr_words = 0;
    int             K = 0, N = 0;
    xrt::bo         x_bo, y_bo;
    ggml_bf16_t *   x_map = nullptr;
    float *         y_map = nullptr;
    xrt::bo         dummy[2];
};

struct npu_state {
    xrt::device            device;
    std::string            kdir;
    bool                   ok = false;
    std::map<int, op_kernel>          ops;
    std::map<std::string, op_kernel>  named; // reductions etc., keyed by xclbin base name
    std::map<std::string, mm_kernel>  mms;
    std::map<std::string, mm_kernel>  bins;  // 2-in/1-out binary, keyed by op name
    std::map<std::string, flash_kernel> flashes;
    std::map<std::string, gemv_kernel>  gemvs;
    std::map<const void *, xrt::bo>     wcache; // per-weight bf16 W bo (gemv)
    std::mutex             mtx;

    // load + cache gemv_<K>_<N>.xclbin
    gemv_kernel * get_gemv(int K, int Nn) {
        const std::string key = std::to_string(K) + "_" + std::to_string(Nn);
        auto it = gemvs.find(key);
        if (it != gemvs.end()) return &it->second;
        const std::string base = kdir + "/gemv_" + key;
        const std::vector<uint32_t> instr = read_u32(base + "_insts.bin");
        if (instr.empty()) return nullptr;
        try {
            xrt::xclbin xclbin(base + ".xclbin");
            device.register_xclbin(xclbin);
            gemv_kernel k;
            k.K = K; k.N = Nn;
            k.ctx         = xrt::hw_context(device, xclbin.get_uuid());
            k.kernel      = xrt::kernel(k.ctx, KERNEL_NAME);
            k.instr_words = (unsigned) instr.size();
            k.bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, k.kernel.group_id(1));
            std::memcpy(k.bo_instr.map<void *>(), instr.data(), instr.size() * sizeof(uint32_t));
            k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k.x_bo = xrt::bo(device, (size_t) K  * sizeof(uint16_t), xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.y_bo = xrt::bo(device, (size_t) Nn * sizeof(float),    xrt::bo::flags::host_only, k.kernel.group_id(5));
            k.x_map = k.x_bo.map<ggml_bf16_t *>();
            k.y_map = k.y_bo.map<float *>();
            for (int i = 0; i < 2; i++) k.dummy[i] = xrt::bo(device, 64, xrt::bo::flags::host_only, k.kernel.group_id(6 + i));
            auto res = gemvs.emplace(key, std::move(k));
            return &res.first->second;
        } catch (...) { return nullptr; }
    }

    // load + cache flash_<NQ>_<DK>_<NKV>.xclbin
    flash_kernel * get_flash(int NQ, int DK, int NKV) {
        const std::string key = std::to_string(NQ) + "_" + std::to_string(DK) + "_" + std::to_string(NKV);
        auto it = flashes.find(key);
        if (it != flashes.end()) return &it->second;
        const std::string base = kdir + "/flash_" + key;
        const std::vector<uint32_t> instr = read_u32(base + "_insts.bin");
        if (instr.empty()) return nullptr;
        try {
            xrt::xclbin xclbin(base + ".xclbin");
            device.register_xclbin(xclbin);
            flash_kernel k;
            k.NQ = NQ; k.DK = DK; k.NKV = NKV; k.NBLK = NKV / FLASH_BLK;
            k.ctx         = xrt::hw_context(device, xclbin.get_uuid());
            k.kernel      = xrt::kernel(k.ctx, KERNEL_NAME);
            k.instr_words = (unsigned) instr.size();
            k.bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, k.kernel.group_id(1));
            std::memcpy(k.bo_instr.map<void *>(), instr.data(), instr.size() * sizeof(uint32_t));
            k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            const int blk_elems = 2 * FLASH_BLK * DK + NQ * FLASH_BLK;
            k.boQ  = xrt::bo(device, (size_t) NQ * DK * sizeof(uint16_t),          xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.boKV = xrt::bo(device, (size_t) k.NBLK * blk_elems * sizeof(uint16_t), xrt::bo::flags::host_only, k.kernel.group_id(4));
            k.boO  = xrt::bo(device, (size_t) (NQ * DK + 2 * NQ) * sizeof(float),  xrt::bo::flags::host_only, k.kernel.group_id(5));
            k.Q_map  = k.boQ.map<ggml_bf16_t *>();
            k.KV_map = k.boKV.map<ggml_bf16_t *>();
            k.O_map  = k.boO.map<float *>();
            for (int i = 0; i < 2; i++) k.dummy[i] = xrt::bo(device, 64, xrt::bo::flags::host_only, k.kernel.group_id(6 + i));
            auto res = flashes.emplace(key, std::move(k));
            return &res.first->second;
        } catch (...) { return nullptr; }
    }

    // load + cache a 2-in/1-out binary kernel by name ("add"/"mul"). Reuses the
    // mm_kernel struct (A=in0 @3, B=in1 @4, C=out @5), sized KERNEL_LEN.
    mm_kernel * get_binary(const std::string & name) {
        auto it = bins.find(name);
        if (it != bins.end()) {
            return &it->second;
        }
        const std::vector<uint32_t> instr = read_u32(kdir + "/" + name + "_insts.bin");
        if (instr.empty()) {
            return nullptr;
        }
        try {
            xrt::xclbin xclbin(kdir + "/" + name + ".xclbin");
            device.register_xclbin(xclbin);
            mm_kernel k;
            k.ctx         = xrt::hw_context(device, xclbin.get_uuid());
            k.kernel      = xrt::kernel(k.ctx, KERNEL_NAME);
            k.instr_words = (unsigned) instr.size();
            k.bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, k.kernel.group_id(1));
            std::memcpy(k.bo_instr.map<void *>(), instr.data(), instr.size() * sizeof(uint32_t));
            k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k.boA = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN, xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.boB = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN, xrt::bo::flags::host_only, k.kernel.group_id(4));
            k.boC = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN, xrt::bo::flags::host_only, k.kernel.group_id(5));
            k.A_map = k.boA.map<ggml_bf16_t *>();
            k.B_map = k.boB.map<ggml_bf16_t *>();
            // C is bf16 for binary; remapped as ggml_bf16_t* in try_binary
            for (int i = 0; i < 2; i++) {
                k.dummy[i] = xrt::bo(device, 64, xrt::bo::flags::host_only, k.kernel.group_id(6 + i));
            }
            auto res = bins.emplace(name, std::move(k));
            return &res.first->second;
        } catch (...) {
            return nullptr;
        }
    }

    // load + cache a 1-in/1-out kernel by xclbin base name (e.g. "rms_norm_1024")
    op_kernel * get_named(const std::string & name) {
        auto it = named.find(name);
        if (it != named.end()) {
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
            k.bo_in  = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN, xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.bo_out = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN, xrt::bo::flags::host_only, k.kernel.group_id(4));
            k.in_map  = k.bo_in.map<ggml_bf16_t *>();
            k.out_map = k.bo_out.map<ggml_bf16_t *>();
            for (int i = 0; i < 3; i++) {
                k.dummy[i] = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN, xrt::bo::flags::host_only, k.kernel.group_id(5 + i));
            }
            auto res = named.emplace(name, std::move(k));
            return &res.first->second;
        } catch (...) {
            return nullptr;
        }
    }

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

    // load + cache the matmul xclbin for shape (M,K,N), or nullptr if not found
    mm_kernel * get_matmul(int M, int K, int Nn) {
        const std::string key = std::to_string(M) + "_" + std::to_string(K) + "_" + std::to_string(Nn);
        auto it = mms.find(key);
        if (it != mms.end()) {
            return &it->second;
        }
        const std::string base = kdir + "/matmul_" + key;
        const std::vector<uint32_t> instr = read_u32(base + "_insts.bin");
        if (instr.empty()) {
            return nullptr;
        }
        try {
            xrt::xclbin xclbin(base + ".xclbin");
            device.register_xclbin(xclbin);

            mm_kernel k;
            k.ctx         = xrt::hw_context(device, xclbin.get_uuid());
            k.kernel      = xrt::kernel(k.ctx, KERNEL_NAME);
            k.instr_words = (unsigned) instr.size();

            k.bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t),
                                 xrt::bo::flags::cacheable, k.kernel.group_id(1));
            std::memcpy(k.bo_instr.map<void *>(), instr.data(), instr.size() * sizeof(uint32_t));
            k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            k.boA = xrt::bo(device, (size_t) M * K * sizeof(uint16_t),  xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.boB = xrt::bo(device, (size_t) K * Nn * sizeof(uint16_t), xrt::bo::flags::host_only, k.kernel.group_id(4));
            k.boC = xrt::bo(device, (size_t) M * Nn * sizeof(float),    xrt::bo::flags::host_only, k.kernel.group_id(5));
            k.A_map = k.boA.map<ggml_bf16_t *>();
            k.B_map = k.boB.map<ggml_bf16_t *>();
            k.C_map = k.boC.map<float *>();
            for (int i = 0; i < 2; i++) {
                k.dummy[i] = xrt::bo(device, 64, xrt::bo::flags::host_only, k.kernel.group_id(6 + i));
            }

            auto res = mms.emplace(key, std::move(k));
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

bool ggml_xdna_npu_try_mul_mat(const struct ggml_tensor * a, const struct ggml_tensor * b, struct ggml_tensor * dst) {
    npu_state & N = npu();
    if (!N.ok) {
        return false;
    }
    // 2D only for now (no batched heads), f32 activations/output
    if (a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1) {
        return false;
    }
    if (b->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || b->ne[0] != a->ne[0]) {
        return false;
    }
    const int K  = (int) a->ne[0];
    const int Nn = (int) a->ne[1];
    const int M  = (int) b->ne[1];

    std::lock_guard<std::mutex> lock(N.mtx);

    // M=1 (decode) is gemv — the tiled matmul kernel needs M%64, so use the
    // dedicated gemv kernel with a per-weight cached bf16 W buffer.
    if (M == 1) {
        gemv_kernel * g = N.get_gemv(K, Nn);
        if (!g) {
            return false;
        }
        try {
            auto wit = N.wcache.find(a->data);
            if (wit == N.wcache.end()) {
                xrt::bo wbo(N.device, (size_t) Nn * K * sizeof(uint16_t), xrt::bo::flags::host_only, g->kernel.group_id(4));
                ggml_bf16_t * wm = wbo.map<ggml_bf16_t *>();
                const auto * tt = ggml_get_type_traits(a->type);
                if (a->type == GGML_TYPE_F32) {
                    ggml_fp32_to_bf16_row((const float *) a->data, wm, (int64_t) K * Nn);
                } else if (tt && tt->to_float) {
                    std::vector<float> adeq((size_t) K * Nn);
                    tt->to_float(a->data, adeq.data(), (int64_t) K * Nn);
                    ggml_fp32_to_bf16_row(adeq.data(), wm, (int64_t) K * Nn);
                } else {
                    return false;
                }
                wbo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                wit = N.wcache.emplace(a->data, std::move(wbo)).first;
            }
            ggml_fp32_to_bf16_row((const float *) b->data, g->x_map, (int64_t) K);
            g->x_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto run = g->kernel(RUN_OPCODE, g->bo_instr, g->instr_words,
                                 g->x_bo, wit->second, g->y_bo, g->dummy[0], g->dummy[1]);
            run.wait();
            g->y_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            std::memcpy(dst->data, g->y_map, (size_t) Nn * sizeof(float));
            if (std::getenv("GGML_XDNA_DEBUG")) {
                fprintf(stderr, "[xdna] ran gemv %dx%d on NPU\n", K, Nn);
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    mm_kernel * k = N.get_matmul(M, K, Nn);
    if (!k) {
        return false;
    }

    try {
        // A[M,K] = activations b (b is ggml [K,M]; b_data[m*K+k] == A[m*K+k])
        ggml_fp32_to_bf16_row((const float *) b->data, k->A_map, (int64_t) M * K);

        // B[K,N] (col-major N x K) = weights a, dequantized to f32 then bf16.
        // Dequantized a is [K,N] row-major (a_f32[k+n*K]); that is exactly the
        // N x K col-major layout the kernel wants (B[n*K+k]).
        const auto * tt = ggml_get_type_traits(a->type);
        if (a->type == GGML_TYPE_F32) {
            ggml_fp32_to_bf16_row((const float *) a->data, k->B_map, (int64_t) K * Nn);
        } else if (tt && tt->to_float) {
            std::vector<float> adeq((size_t) K * Nn);
            tt->to_float(a->data, adeq.data(), (int64_t) K * Nn);
            ggml_fp32_to_bf16_row(adeq.data(), k->B_map, (int64_t) K * Nn);
        } else {
            return false;
        }

        k->boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        k->boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = k->kernel(RUN_OPCODE, k->bo_instr, k->instr_words,
                             k->boA, k->boB, k->boC, k->dummy[0], k->dummy[1]);
        run.wait();

        k->boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        // C[M,N] f32 and dst [N,M] f32 share the linear layout: C[m*N+n] == dst[n+m*N]
        std::memcpy(dst->data, k->C_map, (size_t) M * Nn * sizeof(float));

        if (std::getenv("GGML_XDNA_DEBUG")) {
            fprintf(stderr, "[xdna] ran mul_mat %dx%dx%d on NPU\n", M, K, Nn);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool ggml_xdna_npu_try_rows(const char * op, int row_len, const float * src, float * dst, int64_t n) {
    npu_state & N = npu();
    if (!N.ok) {
        return false;
    }
    // rows must align to the kernel's tiles within a KERNEL_LEN run
    if (row_len <= 0 || KERNEL_LEN % row_len != 0 || n % row_len != 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(N.mtx);
    op_kernel * k = N.get_named(std::string(op) + "_" + std::to_string(row_len));
    if (!k) {
        return false;
    }

    try {
        for (int64_t off = 0; off < n; off += KERNEL_LEN) {
            const int64_t chunk = std::min<int64_t>(KERNEL_LEN, n - off);
            const size_t  bytes = (size_t) chunk * sizeof(ggml_bf16_t);
            ggml_fp32_to_bf16_row(src + off, k->in_map, chunk);
            k->bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
            auto run = k->kernel(RUN_OPCODE, k->bo_instr, k->instr_words,
                                 k->bo_in, k->bo_out, k->dummy[0], k->dummy[1], k->dummy[2]);
            run.wait();
            k->bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
            ggml_bf16_to_fp32_row(k->out_map, dst + off, chunk);
        }
        if (std::getenv("GGML_XDNA_DEBUG")) {
            fprintf(stderr, "[xdna] ran %s row=%d on NPU (n=%lld)\n", op, row_len, (long long) n);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool ggml_xdna_npu_try_binary(const char * op, const float * a, const float * b, float * c, int64_t n) {
    npu_state & N = npu();
    if (!N.ok) {
        return false;
    }
    std::lock_guard<std::mutex> lock(N.mtx);
    mm_kernel * k = N.get_binary(op);
    if (!k) {
        return false;
    }
    try {
        ggml_bf16_t * c_map = k->boC.map<ggml_bf16_t *>();
        for (int64_t off = 0; off < n; off += KERNEL_LEN) {
            const int64_t chunk = std::min<int64_t>(KERNEL_LEN, n - off);
            const size_t  bytes = (size_t) chunk * sizeof(ggml_bf16_t);
            ggml_fp32_to_bf16_row(a + off, k->A_map, chunk);
            ggml_fp32_to_bf16_row(b + off, k->B_map, chunk);
            k->boA.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
            k->boB.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
            auto run = k->kernel(RUN_OPCODE, k->bo_instr, k->instr_words,
                                 k->boA, k->boB, k->boC, k->dummy[0], k->dummy[1]);
            run.wait();
            k->boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
            ggml_bf16_to_fp32_row(c_map, c + off, chunk);
        }
        if (std::getenv("GGML_XDNA_DEBUG")) {
            fprintf(stderr, "[xdna] ran %s (binary) on NPU (n=%lld)\n", op, (long long) n);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool ggml_xdna_npu_try_flash(const struct ggml_tensor * q, const struct ggml_tensor * k,
                             const struct ggml_tensor * v, const struct ggml_tensor * mask,
                             struct ggml_tensor * dst, float scale) {
    npu_state & N = npu();
    if (!N.ok) {
        return false;
    }
    const int DK  = (int) q->ne[0];
    const int DV  = (int) v->ne[0];
    const int NQ  = (int) q->ne[1];
    const int NH  = (int) q->ne[2];
    const int NB  = (int) q->ne[3];
    const int NKV = (int) k->ne[1];
    if (DK != DV || NKV % FLASH_BLK != 0) {
        return false;
    }
    const int rk2 = NH / (int) k->ne[2], rk3 = NB / (int) k->ne[3];
    const int rv2 = NH / (int) v->ne[2], rv3 = NB / (int) v->ne[3];

    std::lock_guard<std::mutex> lock(N.mtx);
    flash_kernel * fk = N.get_flash(NQ, DK, NKV);
    if (!fk) {
        return false;
    }

    const int blk_elems = 2 * FLASH_BLK * DK + NQ * FLASH_BLK;
    const struct ggml_type_traits * ttk = ggml_get_type_traits(k->type);
    const struct ggml_type_traits * ttv = ggml_get_type_traits(v->type);
    std::vector<float> krow(DK), vrow(DV);
    float * d = (float *) dst->data;

    try {
        for (int i3 = 0; i3 < NB; i3++)
        for (int h  = 0; h  < NH; h++) {
            // Q pre-scaled (the flash xclbin bakes scale=1)
            for (int iq = 0; iq < NQ; iq++) {
                const float * qr = (const float *) ((const char *) q->data + iq * q->nb[1] + h * q->nb[2] + i3 * q->nb[3]);
                for (int dd = 0; dd < DK; dd++) fk->Q_map[iq * DK + dd] = ggml_fp32_to_bf16(scale * qr[dd]);
            }
            // K/V/mask packed per block
            for (int b = 0; b < fk->NBLK; b++) {
                ggml_bf16_t * blk = fk->KV_map + (size_t) b * blk_elems;
                for (int ic = 0; ic < FLASH_BLK; ic++) {
                    const int kv = b * FLASH_BLK + ic;
                    const void * kp = (const char *) k->data + kv * k->nb[1] + (h / rk2) * k->nb[2] + (i3 / rk3) * k->nb[3];
                    const void * vp = (const char *) v->data + kv * v->nb[1] + (h / rv2) * v->nb[2] + (i3 / rv3) * v->nb[3];
                    deq_row(ttk, k->type, kp, krow.data(), DK);
                    deq_row(ttv, v->type, vp, vrow.data(), DV);
                    for (int dd = 0; dd < DK; dd++) blk[ic * DK + dd]               = ggml_fp32_to_bf16(krow[dd]);
                    for (int dd = 0; dd < DV; dd++) blk[FLASH_BLK * DK + ic * DV + dd] = ggml_fp32_to_bf16(vrow[dd]);
                }
                ggml_bf16_t * mblk = blk + 2 * FLASH_BLK * DK;
                for (int iq = 0; iq < NQ; iq++) {
                    if (mask) {
                        const ggml_fp16_t * mr = (const ggml_fp16_t *) ((const char *) mask->data +
                                                 iq * mask->nb[1] + (i3 % mask->ne[3]) * mask->nb[3]);
                        for (int ic = 0; ic < FLASH_BLK; ic++) {
                            mblk[iq * FLASH_BLK + ic] = ggml_fp32_to_bf16(ggml_fp16_to_fp32(mr[b * FLASH_BLK + ic]));
                        }
                    } else {
                        for (int ic = 0; ic < FLASH_BLK; ic++) mblk[iq * FLASH_BLK + ic] = ggml_fp32_to_bf16(0.0f);
                    }
                }
            }
            fk->boQ.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            fk->boKV.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto run = fk->kernel(RUN_OPCODE, fk->bo_instr, fk->instr_words,
                                  fk->boQ, fk->boKV, fk->boO, fk->dummy[0], fk->dummy[1]);
            run.wait();
            fk->boO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            // O_map[iq*DV+dv] -> dst[dv, h, iq, i3]  (dst = [DV, NH, NQ, NB])
            for (int iq = 0; iq < NQ; iq++)
                for (int dv = 0; dv < DV; dv++)
                    d[dv + DV * (h + NH * (iq + NQ * i3))] = fk->O_map[iq * DV + dv];
        }
        if (std::getenv("GGML_XDNA_DEBUG")) {
            fprintf(stderr, "[xdna] ran flash_attn %dx%dx%d (heads=%d) on NPU\n", NQ, DK, NKV, NH);
        }
        return true;
    } catch (...) {
        return false;
    }
}

#else // !GGML_XDNA_HAS_XRT

bool ggml_xdna_npu_try_unary(enum ggml_unary_op, const float *, float *, int64_t) {
    return false;
}

bool ggml_xdna_npu_try_flash(const struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *,
                             const struct ggml_tensor *, struct ggml_tensor *, float) {
    return false;
}

bool ggml_xdna_npu_try_mul_mat(const struct ggml_tensor *, const struct ggml_tensor *, struct ggml_tensor *) {
    return false;
}

bool ggml_xdna_npu_try_rows(const char *, int, const float *, float *, int64_t) {
    return false;
}

bool ggml_xdna_npu_try_binary(const char *, const float *, const float *, float *, int64_t) {
    return false;
}

#endif
