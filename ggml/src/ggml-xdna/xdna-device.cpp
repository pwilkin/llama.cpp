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
constexpr int      GDN_R        = 16; // state rows per block the gdn xclbins use

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

// Multi-op unary kernel (one hw-context for several unary ops). op-code @3 (int32),
// in @4 / out @5 (bf16). Consolidates contexts against the NPU's context cap.
struct umulti_kernel {
    xrt::hw_context ctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    unsigned        instr_words = 0;
    xrt::bo         bo_op, bo_in, bo_out;
    int32_t *       op_map  = nullptr;
    ggml_bf16_t *   in_map  = nullptr;
    ggml_bf16_t *   out_map = nullptr;
    xrt::bo         dummy[2]; // slots 6,7
};

// Multi-op binary kernel (one hw-context for mul/add). op-code @3, a @4, b @5,
// c @6 (bf16).
struct bmulti_kernel {
    xrt::hw_context ctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    unsigned        instr_words = 0;
    xrt::bo         bo_op, boAB, boC; // ab packed [a|b] per tile
    int32_t *       op_map = nullptr;
    ggml_bf16_t *   AB_map = nullptr;
    ggml_bf16_t *   C_map = nullptr;
    xrt::bo         dummy[2]; // slots 6,7
};

// op-code for the multi-op binary xclbin (mul=0, add=1), or -1.
inline int binary_multi_code(const char * op) {
    if (std::strcmp(op, "mul") == 0) return 0;
    if (std::strcmp(op, "add") == 0) return 1;
    return -1;
}

// op-code for the multi-op unary xclbin, or -1 if not consolidated (falls back to
// a per-op xclbin). Codes must match xdna_unary_multi in unary.cc.
inline int unary_multi_code(enum ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_SILU:    return 0;
        case GGML_UNARY_OP_SIGMOID: return 1;
        case GGML_UNARY_OP_EXP:     return 2;
        case GGML_UNARY_OP_NEG:     return 3;
        case GGML_UNARY_OP_RELU:    return 4;
        case GGML_UNARY_OP_TANH:    return 5;
        default:                    return -1;
    }
}

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

// A loaded NEOX rope kernel for (head_dim HD, rotary dims NDIMS). cos/sin @3
// (bf16, host-precomputed), src head-rows @4, out @5. Processes NR rows/call.
struct rope_kernel {
    xrt::hw_context ctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    unsigned        instr_words = 0;
    int             HD = 0, NDIMS = 0, NR = 0;
    xrt::bo         bo_cs, bo_src, bo_out;
    ggml_bf16_t *   cs_map  = nullptr;
    ggml_bf16_t *   src_map = nullptr;
    ggml_bf16_t *   out_map = nullptr;
    xrt::bo         dummy[2];
};

// A loaded gated-delta-net kernel for a fixed head dim SV. params @3 (bf16),
// state row-blocks @4 in / @5 out (bf16). One recurrent step per (head, seq).
struct gdn_kernel {
    xrt::hw_context ctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    unsigned        instr_words = 0;
    int             SV = 0, R = 0, NB = 0;
    xrt::bo         boP, boI, boO;
    ggml_bf16_t *   P_map = nullptr;
    ggml_bf16_t *   I_map = nullptr;
    ggml_bf16_t *   O_map = nullptr;
    xrt::bo         dummy[2];
};

// A loaded ssm_conv kernel for a fixed channel count NC (KW=4). in @3 (bf16
// channel blocks), out @4 (fp32).
struct conv_kernel {
    xrt::hw_context ctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    unsigned        instr_words = 0;
    int             NC = 0, KW = 0, R = 0, NB = 0;
    xrt::bo         boI, boO;
    ggml_bf16_t *   I_map = nullptr;
    float *         O_map = nullptr;
    xrt::bo         dummy[3];
};

// A loaded swiglu (GLU) kernel for a fixed row width NC. Input @3 is [gate|up]
// packed per TILE block (bf16); output @4 is [NC] bf16.
struct glu_kernel {
    xrt::hw_context ctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    unsigned        instr_words = 0;
    int             NC = 0, TILE = 0, NB = 0;
    xrt::bo         boI, boO;
    ggml_bf16_t *   I_map = nullptr;
    ggml_bf16_t *   O_map = nullptr;
    xrt::bo         dummy[3];
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
    std::map<int, gdn_kernel>           gdns;   // keyed by SV (head dim)
    std::map<int, conv_kernel>          convs;  // keyed by NC (channels)
    std::map<int, glu_kernel>           glus;   // keyed by NC (row width)
    std::map<int, umulti_kernel>        umultis; // multi-op unary (key 0)
    std::map<int, bmulti_kernel>        bmultis; // multi-op binary (key 0)
    std::map<std::string, rope_kernel>  ropes;   // keyed by "HD_NDIMS"
    std::mutex             mtx;

    // load + cache rope_<HD>_<NDIMS>.xclbin (NR=4)
    rope_kernel * get_rope(int HD, int NDIMS) {
        const std::string key = std::to_string(HD) + "_" + std::to_string(NDIMS);
        auto it = ropes.find(key);
        if (it != ropes.end()) return &it->second;
        const std::string base = kdir + "/rope_" + key;
        const std::vector<uint32_t> instr = read_u32(base + "_insts.bin");
        if (instr.empty()) return nullptr;
        try {
            xrt::xclbin xclbin(base + ".xclbin");
            device.register_xclbin(xclbin);
            rope_kernel k;
            k.HD = HD; k.NDIMS = NDIMS; k.NR = 4;
            k.ctx         = xrt::hw_context(device, xclbin.get_uuid());
            k.kernel      = xrt::kernel(k.ctx, KERNEL_NAME);
            k.instr_words = (unsigned) instr.size();
            k.bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, k.kernel.group_id(1));
            std::memcpy(k.bo_instr.map<void *>(), instr.data(), instr.size() * sizeof(uint32_t));
            k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k.bo_cs  = xrt::bo(device, (size_t) NDIMS * sizeof(uint16_t),      xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.bo_src = xrt::bo(device, (size_t) k.NR * HD * sizeof(uint16_t),  xrt::bo::flags::host_only, k.kernel.group_id(4));
            k.bo_out = xrt::bo(device, (size_t) k.NR * HD * sizeof(uint16_t),  xrt::bo::flags::host_only, k.kernel.group_id(5));
            k.cs_map  = k.bo_cs.map<ggml_bf16_t *>();
            k.src_map = k.bo_src.map<ggml_bf16_t *>();
            k.out_map = k.bo_out.map<ggml_bf16_t *>();
            for (int i = 0; i < 2; i++) k.dummy[i] = xrt::bo(device, 64, xrt::bo::flags::host_only, k.kernel.group_id(6 + i));
            auto res = ropes.emplace(key, std::move(k));
            return &res.first->second;
        } catch (...) { return nullptr; }
    }

    // load + cache binary_multi.xclbin
    bmulti_kernel * get_bmulti() {
        auto it = bmultis.find(0);
        if (it != bmultis.end()) return &it->second;
        const std::string base = kdir + "/binary_multi";
        const std::vector<uint32_t> instr = read_u32(base + "_insts.bin");
        if (instr.empty()) return nullptr;
        try {
            xrt::xclbin xclbin(base + ".xclbin");
            device.register_xclbin(xclbin);
            bmulti_kernel k;
            k.ctx         = xrt::hw_context(device, xclbin.get_uuid());
            k.kernel      = xrt::kernel(k.ctx, KERNEL_NAME);
            k.instr_words = (unsigned) instr.size();
            k.bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, k.kernel.group_id(1));
            std::memcpy(k.bo_instr.map<void *>(), instr.data(), instr.size() * sizeof(uint32_t));
            k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k.bo_op = xrt::bo(device, 16 * sizeof(int32_t),              xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.boAB  = xrt::bo(device, sizeof(uint16_t) * 2 * KERNEL_LEN, xrt::bo::flags::host_only, k.kernel.group_id(4));
            k.boC   = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN,     xrt::bo::flags::host_only, k.kernel.group_id(5));
            k.op_map = k.bo_op.map<int32_t *>();
            k.AB_map = k.boAB.map<ggml_bf16_t *>();
            k.C_map  = k.boC.map<ggml_bf16_t *>();
            for (int i = 0; i < 2; i++) k.dummy[i] = xrt::bo(device, 64, xrt::bo::flags::host_only, k.kernel.group_id(6 + i));
            auto res = bmultis.emplace(0, std::move(k));
            return &res.first->second;
        } catch (...) { return nullptr; }
    }

    // load + cache unary_multi.xclbin (LENGTH=4096)
    umulti_kernel * get_umulti() {
        auto it = umultis.find(0);
        if (it != umultis.end()) return &it->second;
        const std::string base = kdir + "/unary_multi";
        const std::vector<uint32_t> instr = read_u32(base + "_insts.bin");
        if (instr.empty()) return nullptr;
        try {
            xrt::xclbin xclbin(base + ".xclbin");
            device.register_xclbin(xclbin);
            umulti_kernel k;
            k.ctx         = xrt::hw_context(device, xclbin.get_uuid());
            k.kernel      = xrt::kernel(k.ctx, KERNEL_NAME);
            k.instr_words = (unsigned) instr.size();
            k.bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, k.kernel.group_id(1));
            std::memcpy(k.bo_instr.map<void *>(), instr.data(), instr.size() * sizeof(uint32_t));
            k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k.bo_op  = xrt::bo(device, 16 * sizeof(int32_t),              xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.bo_in  = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN,     xrt::bo::flags::host_only, k.kernel.group_id(4));
            k.bo_out = xrt::bo(device, sizeof(uint16_t) * KERNEL_LEN,     xrt::bo::flags::host_only, k.kernel.group_id(5));
            k.op_map  = k.bo_op.map<int32_t *>();
            k.in_map  = k.bo_in.map<ggml_bf16_t *>();
            k.out_map = k.bo_out.map<ggml_bf16_t *>();
            for (int i = 0; i < 2; i++) k.dummy[i] = xrt::bo(device, 64, xrt::bo::flags::host_only, k.kernel.group_id(6 + i));
            auto res = umultis.emplace(0, std::move(k));
            return &res.first->second;
        } catch (...) { return nullptr; }
    }

    // load + cache swiglu_<NC>.xclbin (TILE=1024)
    glu_kernel * get_glu(int NC) {
        auto it = glus.find(NC);
        if (it != glus.end()) return &it->second;
        const std::string base = kdir + "/swiglu_" + std::to_string(NC);
        const std::vector<uint32_t> instr = read_u32(base + "_insts.bin");
        if (instr.empty()) return nullptr;
        try {
            xrt::xclbin xclbin(base + ".xclbin");
            device.register_xclbin(xclbin);
            glu_kernel k;
            k.NC = NC; k.TILE = 1024; k.NB = NC / 1024;
            k.ctx         = xrt::hw_context(device, xclbin.get_uuid());
            k.kernel      = xrt::kernel(k.ctx, KERNEL_NAME);
            k.instr_words = (unsigned) instr.size();
            k.bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, k.kernel.group_id(1));
            std::memcpy(k.bo_instr.map<void *>(), instr.data(), instr.size() * sizeof(uint32_t));
            k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k.boI = xrt::bo(device, (size_t) NC * 2 * sizeof(uint16_t), xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.boO = xrt::bo(device, (size_t) NC * sizeof(uint16_t),     xrt::bo::flags::host_only, k.kernel.group_id(4));
            k.I_map = k.boI.map<ggml_bf16_t *>();
            k.O_map = k.boO.map<ggml_bf16_t *>();
            for (int i = 0; i < 3; i++) k.dummy[i] = xrt::bo(device, 64, xrt::bo::flags::host_only, k.kernel.group_id(5 + i));
            auto res = glus.emplace(NC, std::move(k));
            return &res.first->second;
        } catch (...) { return nullptr; }
    }

    // load + cache ssm_conv_<NC>.xclbin (KW=4, R=256)
    conv_kernel * get_ssm_conv(int NC) {
        auto it = convs.find(NC);
        if (it != convs.end()) return &it->second;
        const std::string base = kdir + "/ssm_conv_" + std::to_string(NC);
        const std::vector<uint32_t> instr = read_u32(base + "_insts.bin");
        if (instr.empty()) return nullptr;
        try {
            xrt::xclbin xclbin(base + ".xclbin");
            device.register_xclbin(xclbin);
            conv_kernel k;
            k.NC = NC; k.KW = 4; k.R = 256; k.NB = NC / 256;
            const int blk = 2 * k.KW * k.R;
            k.ctx         = xrt::hw_context(device, xclbin.get_uuid());
            k.kernel      = xrt::kernel(k.ctx, KERNEL_NAME);
            k.instr_words = (unsigned) instr.size();
            k.bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, k.kernel.group_id(1));
            std::memcpy(k.bo_instr.map<void *>(), instr.data(), instr.size() * sizeof(uint32_t));
            k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k.boI = xrt::bo(device, (size_t) k.NB * blk * sizeof(uint16_t), xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.boO = xrt::bo(device, (size_t) NC * sizeof(float),            xrt::bo::flags::host_only, k.kernel.group_id(4));
            k.I_map = k.boI.map<ggml_bf16_t *>();
            k.O_map = k.boO.map<float *>();
            for (int i = 0; i < 3; i++) k.dummy[i] = xrt::bo(device, 64, xrt::bo::flags::host_only, k.kernel.group_id(5 + i));
            auto res = convs.emplace(NC, std::move(k));
            return &res.first->second;
        } catch (...) { return nullptr; }
    }

    // load + cache gdn_<SV>.xclbin
    gdn_kernel * get_gdn(int SV) {
        auto it = gdns.find(SV);
        if (it != gdns.end()) return &it->second;
        const std::string base = kdir + "/gdn_" + std::to_string(SV);
        const std::vector<uint32_t> instr = read_u32(base + "_insts.bin");
        if (instr.empty()) return nullptr;
        try {
            xrt::xclbin xclbin(base + ".xclbin");
            device.register_xclbin(xclbin);
            gdn_kernel k;
            k.SV = SV; k.R = GDN_R; k.NB = SV / GDN_R;
            const int blk = k.R * SV + k.R;
            k.ctx         = xrt::hw_context(device, xclbin.get_uuid());
            k.kernel      = xrt::kernel(k.ctx, KERNEL_NAME);
            k.instr_words = (unsigned) instr.size();
            k.bo_instr = xrt::bo(device, instr.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, k.kernel.group_id(1));
            std::memcpy(k.bo_instr.map<void *>(), instr.data(), instr.size() * sizeof(uint32_t));
            k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            k.boP = xrt::bo(device, (size_t) (3 * SV + 16) * sizeof(uint16_t), xrt::bo::flags::host_only, k.kernel.group_id(3));
            k.boI = xrt::bo(device, (size_t) k.NB * blk * sizeof(uint16_t),     xrt::bo::flags::host_only, k.kernel.group_id(4));
            k.boO = xrt::bo(device, (size_t) k.NB * blk * sizeof(uint16_t),     xrt::bo::flags::host_only, k.kernel.group_id(5));
            k.P_map = k.boP.map<ggml_bf16_t *>();
            k.I_map = k.boI.map<ggml_bf16_t *>();
            k.O_map = k.boO.map<ggml_bf16_t *>();
            for (int i = 0; i < 2; i++) k.dummy[i] = xrt::bo(device, 64, xrt::bo::flags::host_only, k.kernel.group_id(6 + i));
            auto res = gdns.emplace(SV, std::move(k));
            return &res.first->second;
        } catch (...) { return nullptr; }
    }

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
        } catch (const std::exception & e) {
            if (std::getenv("GGML_XDNA_FLASHDBG")) fprintf(stderr, "[xdna] get_flash threw: %s\n", e.what());
            return nullptr;
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

    // consolidated multi-op unary path (one hw-context for several ops)
    const int code = unary_multi_code(op);
    if (code >= 0) {
        umulti_kernel * u = N.get_umulti();
        if (u) {
            try {
                u->op_map[0] = code;
                u->bo_op.sync(XCL_BO_SYNC_BO_TO_DEVICE, sizeof(int32_t), 0);
                for (int64_t off = 0; off < n; off += KERNEL_LEN) {
                    const int64_t chunk = std::min<int64_t>(KERNEL_LEN, n - off);
                    const size_t  bytes = (size_t) chunk * sizeof(ggml_bf16_t);
                    ggml_fp32_to_bf16_row(src + off, u->in_map, chunk);
                    u->bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
                    auto run = u->kernel(RUN_OPCODE, u->bo_instr, u->instr_words,
                                         u->bo_op, u->bo_in, u->bo_out, u->dummy[0], u->dummy[1]);
                    run.wait();
                    u->bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
                    ggml_bf16_to_fp32_row(u->out_map, dst + off, chunk);
                }
                if (std::getenv("GGML_XDNA_DEBUG")) {
                    fprintf(stderr, "[xdna] ran %s on NPU (n=%lld, multi)\n", op_name(op), (long long) n);
                }
                return true;
            } catch (...) { /* fall through to the per-op xclbin */ }
        }
    }

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

    // consolidated multi-op binary path (one hw-context for mul/add)
    const int bcode = binary_multi_code(op);
    if (bcode >= 0) {
        bmulti_kernel * u = N.get_bmulti();
        if (u) {
            try {
                u->op_map[0] = bcode;
                u->bo_op.sync(XCL_BO_SYNC_BO_TO_DEVICE, sizeof(int32_t), 0);
                for (int64_t off = 0; off < n; off += KERNEL_LEN) {
                    const int64_t chunk = std::min<int64_t>(KERNEL_LEN, n - off);
                    const size_t  bytes = (size_t) chunk * sizeof(ggml_bf16_t);
                    // ab = [ a(KERNEL_LEN) | b(KERNEL_LEN) ] (one tile per run)
                    ggml_fp32_to_bf16_row(a + off, u->AB_map, chunk);
                    ggml_fp32_to_bf16_row(b + off, u->AB_map + KERNEL_LEN, chunk);
                    u->boAB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    auto run = u->kernel(RUN_OPCODE, u->bo_instr, u->instr_words,
                                         u->bo_op, u->boAB, u->boC, u->dummy[0], u->dummy[1]);
                    run.wait();
                    u->boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
                    ggml_bf16_to_fp32_row(u->C_map, c + off, chunk);
                }
                if (std::getenv("GGML_XDNA_DEBUG")) {
                    fprintf(stderr, "[xdna] ran %s (binary) on NPU (n=%lld, multi)\n", op, (long long) n);
                }
                return true;
            } catch (...) { /* fall through to the per-op xclbin */ }
        }
    }

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
    if (std::getenv("GGML_XDNA_FLASHDBG")) {
        fprintf(stderr, "[xdna] flash req NQ=%d DK=%d DV=%d NKV=%d NH=%d kt=%d vt=%d\n",
                NQ, DK, DV, NKV, NH, k->type, v->type);
    }
    if (DK != DV || NKV % FLASH_BLK != 0) {
        return false;
    }
    const int rk2 = NH / (int) k->ne[2], rk3 = NB / (int) k->ne[3];
    const int rv2 = NH / (int) v->ne[2], rv3 = NB / (int) v->ne[3];

    std::lock_guard<std::mutex> lock(N.mtx);
    flash_kernel * fk = N.get_flash(NQ, DK, NKV);
    if (!fk) {
        if (std::getenv("GGML_XDNA_FLASHDBG")) fprintf(stderr, "[xdna] flash get_flash(%d,%d,%d) NULL\n", NQ, DK, NKV);
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
    } catch (const std::exception & e) {
        if (std::getenv("GGML_XDNA_FLASHDBG")) fprintf(stderr, "[xdna] flash run threw: %s\n", e.what());
        return false;
    } catch (...) {
        if (std::getenv("GGML_XDNA_FLASHDBG")) fprintf(stderr, "[xdna] flash run threw unknown\n");
        return false;
    }
}

bool ggml_xdna_npu_try_gdn(struct ggml_tensor * dst) {
    npu_state & N = npu();
    if (!N.ok) {
        return false;
    }
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int S_v      = (int) v->ne[0];
    const int H        = (int) v->ne[1];
    const int n_tokens = (int) v->ne[2];
    const int n_seqs   = (int) v->ne[3];

    // decode step only for now; head dim must match a built gdn_<SV>.xclbin; the
    // gate is scalar (g.ne0==1) or per-dim (kda, ==S_v); all inputs f32.
    if (n_tokens != 1 || S_v != 128 || (g->ne[0] != 1 && g->ne[0] != S_v)) {
        return false;
    }
    const bool kda = (g->ne[0] == S_v);
    if (q->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32) {
        return false;
    }

    std::lock_guard<std::mutex> lock(N.mtx);
    gdn_kernel * gk = N.get_gdn(S_v);
    if (!gk) {
        return false;
    }
    const int   R   = gk->R, NB = gk->NB, blk = R * S_v + R;
    const float scale = 1.0f / std::sqrt((float) S_v);
    const int64_t state_seq_stride = state->nb[3] / sizeof(float);
    const int64_t rq3 = v->ne[3] / q->ne[3], rk3 = v->ne[3] / k->ne[3];
    const int64_t attn_score_elems = (int64_t) S_v * H * n_tokens * n_seqs;
    float * attn_out_base  = (float *) dst->data;
    float * state_out_base = (float *) dst->data + attn_score_elems;
    const float * state_in_base = (const float *) state->data;

    try {
        for (int iv3 = 0; iv3 < n_seqs; iv3++)
        for (int iv1 = 0; iv1 < H; iv1++) {
            const int iq1 = iv1 % (int) q->ne[1];
            const int ik1 = iv1 % (int) k->ne[1];
            const int iq3 = iv3 / rq3;
            const int ik3 = iv3 / rk3;
            const float * q_d = (const float *) ((const char *) q->data + iq3 * q->nb[3] + iq1 * q->nb[1]);
            const float * k_d = (const float *) ((const char *) k->data + ik3 * k->nb[3] + ik1 * k->nb[1]);
            const float * v_d = (const float *) ((const char *) v->data + iv3 * v->nb[3] + iv1 * v->nb[1]);
            const float * g_d = (const float *) ((const char *) g->data + iv3 * g->nb[3] + iv1 * g->nb[1]);
            const float   bv  = *(const float *) ((const char *) beta->data + iv3 * beta->nb[3] + iv1 * beta->nb[1]);

            // gate decay: scalar gate broadcasts exp(g[0]); kda uses exp(g[i])
            for (int i = 0; i < S_v; i++) gk->P_map[i]          = ggml_fp32_to_bf16(std::exp(kda ? g_d[i] : g_d[0]));
            for (int i = 0; i < S_v; i++) gk->P_map[S_v + i]     = ggml_fp32_to_bf16(k_d[i]);
            for (int i = 0; i < S_v; i++) gk->P_map[2 * S_v + i] = ggml_fp32_to_bf16(q_d[i]);
            gk->P_map[3 * S_v]     = ggml_fp32_to_bf16(bv);
            gk->P_map[3 * S_v + 1] = ggml_fp32_to_bf16(scale);

            const float * s_in = state_in_base + iv3 * state_seq_stride + (int64_t) iv1 * S_v * S_v;
            for (int b = 0; b < NB; b++) {
                ggml_bf16_t * bin = gk->I_map + (size_t) b * blk;
                for (int r = 0; r < R; r++) {
                    const int j = b * R + r;
                    for (int i = 0; i < S_v; i++) bin[r * S_v + i] = ggml_fp32_to_bf16(s_in[j * S_v + i]);
                    bin[R * S_v + r] = ggml_fp32_to_bf16(v_d[j]);
                }
            }

            gk->boP.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            gk->boI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto run = gk->kernel(RUN_OPCODE, gk->bo_instr, gk->instr_words,
                                  gk->boP, gk->boI, gk->boO, gk->dummy[0], gk->dummy[1]);
            run.wait();
            gk->boO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            float * attn_data = attn_out_base  + (int64_t) (iv3 * H + iv1) * S_v;
            float * s_out     = state_out_base + (int64_t) (iv3 * H + iv1) * S_v * S_v; // slot 0
            for (int b = 0; b < NB; b++) {
                const ggml_bf16_t * bo = gk->O_map + (size_t) b * blk;
                for (int r = 0; r < R; r++) {
                    const int j = b * R + r;
                    for (int i = 0; i < S_v; i++) s_out[j * S_v + i] = ggml_bf16_to_fp32(bo[r * S_v + i]);
                    attn_data[j] = ggml_bf16_to_fp32(bo[R * S_v + r]);
                }
            }
        }
        if (std::getenv("GGML_XDNA_DEBUG")) {
            fprintf(stderr, "[xdna] ran gated_delta_net H=%d seqs=%d on NPU\n", H, n_seqs);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool ggml_xdna_npu_try_ssm_conv(struct ggml_tensor * dst) {
    npu_state & N = npu();
    if (!N.ok) {
        return false;
    }
    const ggml_tensor * ci = dst->src[0]; // conv_input [KW-1+n_tokens, NC]
    const ggml_tensor * w  = dst->src[1]; // weight [KW, NC]
    const int NC = (int) ci->ne[1];
    const int KW = (int) w->ne[0];
    const int n_tokens = (int) dst->ne[1];
    if (n_tokens != 1 || KW != 4 || (int) ci->ne[0] != KW || (NC % 256) != 0) {
        return false;
    }
    if (ci->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32) {
        return false;
    }
    std::lock_guard<std::mutex> lock(N.mtx);
    conv_kernel * ck = N.get_ssm_conv(NC);
    if (!ck) {
        return false;
    }
    const int R = ck->R, NB = ck->NB, blk = 2 * KW * R;
    try {
        for (int b = 0; b < NB; b++) {
            ggml_bf16_t * bin = ck->I_map + (size_t) b * blk;
            for (int t = 0; t < KW; t++) {
                for (int c = 0; c < R; c++) {
                    const int cc = b * R + c;
                    const float xv = *(const float *) ((const char *) ci->data + t * ci->nb[0] + cc * ci->nb[1]);
                    const float wv = *(const float *) ((const char *) w->data  + t * w->nb[0]  + cc * w->nb[1]);
                    bin[t * R + c]        = ggml_fp32_to_bf16(xv);
                    bin[(KW + t) * R + c] = ggml_fp32_to_bf16(wv);
                }
            }
        }
        ck->boI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = ck->kernel(RUN_OPCODE, ck->bo_instr, ck->instr_words,
                              ck->boI, ck->boO, ck->dummy[0], ck->dummy[1], ck->dummy[2]);
        run.wait();
        ck->boO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(dst->data, ck->O_map, (size_t) NC * sizeof(float));
        if (std::getenv("GGML_XDNA_DEBUG")) {
            fprintf(stderr, "[xdna] ran ssm_conv NC=%d on NPU\n", NC);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool ggml_xdna_npu_try_glu(struct ggml_tensor * dst) {
    npu_state & N = npu();
    if (!N.ok) {
        return false;
    }
    const ggml_tensor * gate = dst->src[0];
    const ggml_tensor * up   = dst->src[1];
    if (up == nullptr) {
        return false; // 2-input GLU only (no single-input split)
    }
    const int NC = (int) gate->ne[0];
    if ((NC % 1024) != 0 || gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t nrows = ggml_nrows(gate);
    std::lock_guard<std::mutex> lock(N.mtx);
    glu_kernel * gk = N.get_glu(NC);
    if (!gk) {
        return false;
    }
    const int TILE = gk->TILE, NB = gk->NB;
    try {
        for (int64_t r = 0; r < nrows; r++) {
            const float * gr = (const float *) ((const char *) gate->data + r * gate->nb[1]);
            const float * ur = (const float *) ((const char *) up->data   + r * up->nb[1]);
            for (int b = 0; b < NB; b++) {
                ggml_bf16_t * blk = gk->I_map + (size_t) b * 2 * TILE;
                for (int i = 0; i < TILE; i++) {
                    blk[i]        = ggml_fp32_to_bf16(gr[b * TILE + i]);
                    blk[TILE + i] = ggml_fp32_to_bf16(ur[b * TILE + i]);
                }
            }
            gk->boI.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto run = gk->kernel(RUN_OPCODE, gk->bo_instr, gk->instr_words,
                                  gk->boI, gk->boO, gk->dummy[0], gk->dummy[1], gk->dummy[2]);
            run.wait();
            gk->boO.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            float * dr = (float *) ((char *) dst->data + r * dst->nb[1]);
            for (int i = 0; i < NC; i++) dr[i] = ggml_bf16_to_fp32(gk->O_map[i]);
        }
        if (std::getenv("GGML_XDNA_DEBUG")) {
            fprintf(stderr, "[xdna] ran swiglu NC=%d rows=%ld on NPU\n", NC, (long) nrows);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool ggml_xdna_npu_try_rope(struct ggml_tensor * dst) {
    npu_state & N = npu();
    if (!N.ok) {
        return false;
    }
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * pos = dst->src[1];
    const int head_dim = (int) src->ne[0];
    const int n_head   = (int) src->ne[1];
    const int n_tokens = (int) src->ne[2];
    const int n_dims   = ggml_get_op_params_i32(dst, 1);
    float freq_base = 0.0f, freq_scale = 1.0f, ext_factor = 0.0f, attn_factor = 1.0f;
    std::memcpy(&freq_base,   (const char *) dst->op_params + 5 * 4, 4);
    std::memcpy(&freq_scale,  (const char *) dst->op_params + 6 * 4, 4);
    std::memcpy(&ext_factor,  (const char *) dst->op_params + 7 * 4, 4);
    std::memcpy(&attn_factor, (const char *) dst->op_params + 8 * 4, 4);

    // decode step, head dim 256, n_rot 64, no yarn — matches rope_256_64.xclbin.
    if (n_tokens != 1 || head_dim != 256 || n_dims != 64 || ext_factor != 0.0f) {
        return false;
    }
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        pos == nullptr || pos->type != GGML_TYPE_I32) {
        return false;
    }

    std::lock_guard<std::mutex> lock(N.mtx);
    rope_kernel * rk = N.get_rope(head_dim, n_dims);
    if (!rk) {
        return false;
    }
    const int NR = rk->NR, nd2 = n_dims / 2;
    try {
        // host-precompute cos/sin for this token (imrope == NEOX for text)
        const int32_t p = ((const int32_t *) pos->data)[0];
        const float theta_scale = std::pow(freq_base, -2.0f / n_dims);
        for (int j = 0; j < nd2; j++) {
            const float th = freq_scale * (float) p * std::pow(theta_scale, (float) j);
            rk->cs_map[j]       = ggml_fp32_to_bf16(std::cos(th) * attn_factor);
            rk->cs_map[nd2 + j] = ggml_fp32_to_bf16(std::sin(th) * attn_factor);
        }
        rk->bo_cs.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        for (int hb = 0; hb < n_head; hb += NR) {
            const int nrows = std::min(NR, n_head - hb);
            for (int r = 0; r < NR; r++) {
                if (r < nrows) {
                    const float * s = (const float *) ((const char *) src->data + (hb + r) * src->nb[1]);
                    for (int i = 0; i < head_dim; i++) rk->src_map[r * head_dim + i] = ggml_fp32_to_bf16(s[i]);
                } else {
                    for (int i = 0; i < head_dim; i++) rk->src_map[r * head_dim + i] = ggml_fp32_to_bf16(0.0f);
                }
            }
            rk->bo_src.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto run = rk->kernel(RUN_OPCODE, rk->bo_instr, rk->instr_words,
                                  rk->bo_cs, rk->bo_src, rk->bo_out, rk->dummy[0], rk->dummy[1]);
            run.wait();
            rk->bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            for (int r = 0; r < nrows; r++) {
                float * d = (float *) ((char *) dst->data + (hb + r) * dst->nb[1]);
                for (int i = 0; i < head_dim; i++) d[i] = ggml_bf16_to_fp32(rk->out_map[r * head_dim + i]);
            }
        }
        if (std::getenv("GGML_XDNA_DEBUG")) {
            fprintf(stderr, "[xdna] ran rope hd=%d nrot=%d heads=%d on NPU\n", head_dim, n_dims, n_head);
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

bool ggml_xdna_npu_try_gdn(struct ggml_tensor *) {
    return false;
}

bool ggml_xdna_npu_try_rope(struct ggml_tensor *) {
    return false;
}

bool ggml_xdna_npu_try_ssm_conv(struct ggml_tensor *) {
    return false;
}

bool ggml_xdna_npu_try_glu(struct ggml_tensor *) {
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
