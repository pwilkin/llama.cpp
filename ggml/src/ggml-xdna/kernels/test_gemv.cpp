#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/deprecated/xrt.h>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
static uint16_t f2bf(float f){uint32_t b;std::memcpy(&b,&f,4);return (uint16_t)((b+0x8000u)>>16);}
static float    bf2f(uint16_t h){uint32_t b=(uint32_t)h<<16;float f;std::memcpy(&f,&b,4);return f;}
int main(int argc,char**argv){
    const int K=argc>1?atoi(argv[1]):2560, N=argc>2?atoi(argv[2]):256;
    std::vector<uint32_t> instr;{std::ifstream f("gemv_insts.bin",std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);instr.resize(n/4);f.read((char*)instr.data(),n);}
    xrt::device dev(0);xrt::xclbin xc(std::string("gemv.xclbin"));dev.register_xclbin(xc);
    xrt::hw_context ctx(dev,xc.get_uuid());xrt::kernel k(ctx,"MLIR_AIE");
    auto bi=xrt::bo(dev,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto bx=xrt::bo(dev,K*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bw=xrt::bo(dev,(size_t)N*K*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto by=xrt::bo(dev,N*4,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    auto b6=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(6));auto b7=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(7));
    std::mt19937 rng(1);std::uniform_real_distribution<float> U(-1.f,1.f);
    std::vector<float> x(K),W((size_t)N*K);
    for(auto&v:x)v=U(rng);for(auto&v:W)v=U(rng);
    auto*xm=bx.map<uint16_t*>();for(int i=0;i<K;i++)xm[i]=f2bf(x[i]);
    auto*wm=bw.map<uint16_t*>();for(size_t i=0;i<(size_t)N*K;i++)wm[i]=f2bf(W[i]);
    std::memcpy(bi.map<void*>(),instr.data(),instr.size()*4);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);bx.sync(XCL_BO_SYNC_BO_TO_DEVICE);bw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k(3u,bi,(uint32_t)instr.size(),bx,bw,by,b6,b7);run.wait();by.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto*ym=by.map<float*>();
    double maxe=0;int bad=0;
    for(int n=0;n<N;n++){double a=0;for(int kk=0;kk<K;kk++)a+=(double)bf2f(f2bf(x[kk]))*bf2f(f2bf(W[(size_t)n*K+kk]));double e=std::fabs(ym[n]-a);maxe=std::max(maxe,e);if(e>0.5)bad++;}
    printf("GEMV K=%d N=%d: max_err=%.4f bad=%d/%d %s\n",K,N,maxe,bad,N,bad==0?"CORRECT":"MISMATCH");
    return bad==0?0:1;
}
