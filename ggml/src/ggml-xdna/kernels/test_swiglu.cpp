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
static float bf(float x){return bf2f(f2bf(x));}
int main(){
    const int TILE=1024,NB=9,N=TILE*NB;
    std::vector<uint32_t> instr;{std::ifstream f("swiglu_insts.bin",std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);instr.resize(n/4);f.read((char*)instr.data(),n);}
    xrt::device dev(0);xrt::xclbin xc(std::string("swiglu.xclbin"));dev.register_xclbin(xc);
    xrt::hw_context ctx(dev,xc.get_uuid());xrt::kernel k(ctx,"MLIR_AIE");
    auto bi=xrt::bo(dev,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto bin=xrt::bo(dev,(size_t)N*2*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bout=xrt::bo(dev,(size_t)N*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto b5=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));auto b6=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(6));auto b7=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(7));
    std::mt19937 rng(4);std::uniform_real_distribution<float> U(-3.f,3.f);
    std::vector<float> g(N),u(N);for(auto&z:g)z=U(rng);for(auto&z:u)z=U(rng);
    auto*im=bin.map<uint16_t*>();
    for(int b=0;b<NB;b++){int off=b*2*TILE;for(int i=0;i<TILE;i++){im[off+i]=f2bf(g[b*TILE+i]);im[off+TILE+i]=f2bf(u[b*TILE+i]);}}
    std::memcpy(bi.map<void*>(),instr.data(),instr.size()*4);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);bin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k(3u,bi,(uint32_t)instr.size(),bin,bout,b5,b6,b7);run.wait();bout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto*om=bout.map<uint16_t*>();
    double me=0;int bad=0;
    for(int i=0;i<N;i++){float gg=bf(g[i]);float silu=gg/(1.f+std::exp(-gg));float ref=silu*bf(u[i]);double e=std::fabs(bf2f(om[i])-ref);me=std::max(me,e);if(e>0.1)bad++;}
    printf("SWIGLU N=%d: max_err=%.4f bad=%d/%d %s\n",N,me,bad,N,bad<N/50?"CORRECT":"MISMATCH");
    return bad<N/50?0:1;
}
