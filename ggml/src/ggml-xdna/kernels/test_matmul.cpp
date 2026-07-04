// Standalone XRT test: run the 32x32x32 bf16->f32 matmul xclbin on the NPU and
// compare against a CPU reference. Layout: A[M,K] row-major, B stored N x K
// (b_col_maj), C[M,N] row-major.
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/deprecated/xrt.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

static uint16_t f2bf(float f) { uint32_t b; std::memcpy(&b, &f, 4); return (uint16_t)((b + 0x8000u) >> 16); }
static float    bf2f(uint16_t h){ uint32_t b = (uint32_t)h << 16; float f; std::memcpy(&f, &b, 4); return f; }

int main() {
    const int M = 32, K = 32, N = 32;
    std::vector<uint32_t> instr;
    { std::ifstream f("mm32_insts.bin", std::ios::binary|std::ios::ate);
      auto n=f.tellg(); f.seekg(0); instr.resize(n/4); f.read((char*)instr.data(), n); }

    xrt::device dev(0);
    xrt::xclbin xclbin(std::string("mm32.xclbin"));
    dev.register_xclbin(xclbin);
    xrt::hw_context ctx(dev, xclbin.get_uuid());
    xrt::kernel k(ctx, "MLIR_AIE");

    auto bo_instr = xrt::bo(dev, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
    auto bo_a = xrt::bo(dev, M*K*sizeof(uint16_t), XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
    auto bo_b = xrt::bo(dev, K*N*sizeof(uint16_t), XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
    auto bo_c = xrt::bo(dev, M*N*sizeof(float),    XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));
    auto bo_d6= xrt::bo(dev, 64, XRT_BO_FLAGS_HOST_ONLY, k.group_id(6));
    auto bo_d7= xrt::bo(dev, 64, XRT_BO_FLAGS_HOST_ONLY, k.group_id(7));

    std::mt19937 rng(123); std::uniform_real_distribution<float> U(-1.f,1.f);
    std::vector<float> A(M*K), B(N*K); // B stored N x K
    for (auto& x : A) x = U(rng);
    for (auto& x : B) x = U(rng);

    auto* am = bo_a.map<uint16_t*>(); for (int i=0;i<M*K;i++) am[i]=f2bf(A[i]);
    auto* bm = bo_b.map<uint16_t*>(); for (int i=0;i<N*K;i++) bm[i]=f2bf(B[i]);
    std::memcpy(bo_instr.map<void*>(), instr.data(), instr.size()*4);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = k(3u, bo_instr, (uint32_t)instr.size(), bo_a, bo_b, bo_c, bo_d6, bo_d7);
    run.wait();
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* cm = bo_c.map<float*>();

    // reference from the bf16-rounded inputs: C[m,n] = sum_k A[m,k]*B[n,k]
    double max_rel = 0, max_abs = 0; int nbad = 0;
    for (int m=0;m<M;m++) for (int n=0;n<N;n++) {
        double ref = 0;
        for (int kk=0;kk<K;kk++) ref += (double)bf2f(am[m*K+kk]) * (double)bf2f(bm[n*K+kk]);
        double got = cm[m*N+n];
        double ad = std::fabs(got-ref); double rd = ad/(std::fabs(ref)+1e-6);
        max_abs=std::max(max_abs,ad); max_rel=std::max(max_rel,rd);
        if (rd > 0.05 && ad > 0.05) nbad++;
    }
    printf("C[0,0]=%.4f  ref=%.4f\n", cm[0], ({double r=0; for(int kk=0;kk<K;kk++) r+=(double)bf2f(am[kk])*(double)bf2f(bm[kk]); r;}));
    printf("max_abs=%.4f  max_rel=%.4f  bad(>5%%)=%d/%d\n", max_abs, max_rel, nbad, M*N);
    printf("%s\n", nbad==0 ? "MATMUL ON NPU: CORRECT" : "MATMUL MISMATCH");
    return nbad==0 ? 0 : 1;
}
