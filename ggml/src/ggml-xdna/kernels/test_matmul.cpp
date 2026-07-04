// Generalized XRT matmul test: ./test_mm M K N xclbin insts
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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>
static uint16_t f2bf(float f){uint32_t b;std::memcpy(&b,&f,4);return (uint16_t)((b+0x8000u)>>16);}
static float    bf2f(uint16_t h){uint32_t b=(uint32_t)h<<16;float f;std::memcpy(&f,&b,4);return f;}
int main(int argc,char**argv){
    int M=atoi(argv[1]),K=atoi(argv[2]),N=atoi(argv[3]);
    std::string xc=argv[4], ip=argv[5];
    std::vector<uint32_t> instr;
    {std::ifstream f(ip,std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);instr.resize(n/4);f.read((char*)instr.data(),n);}
    xrt::device dev(0); xrt::xclbin xclbin(xc); dev.register_xclbin(xclbin);
    xrt::hw_context ctx(dev,xclbin.get_uuid()); xrt::kernel k(ctx,"MLIR_AIE");
    auto bi=xrt::bo(dev,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto ba=xrt::bo(dev,(size_t)M*K*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bb=xrt::bo(dev,(size_t)K*N*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto bc=xrt::bo(dev,(size_t)M*N*4,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    auto b6=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(6));
    auto b7=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(7));
    std::mt19937 rng(7); std::uniform_real_distribution<float> U(-1.f,1.f);
    auto*am=ba.map<uint16_t*>(); for(int i=0;i<M*K;i++)am[i]=f2bf(U(rng));
    auto*bm=bb.map<uint16_t*>(); for(int i=0;i<K*N;i++)bm[i]=f2bf(U(rng)); // B stored NxK
    std::memcpy(bi.map<void*>(),instr.data(),instr.size()*4);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE); ba.sync(XCL_BO_SYNC_BO_TO_DEVICE); bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k(3u,bi,(uint32_t)instr.size(),ba,bb,bc,b6,b7); run.wait();
    bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE); auto*cm=bc.map<float*>();
    double maxrel=0,maxabs=0;int bad=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        double ref=0; for(int kk=0;kk<K;kk++) ref+=(double)bf2f(am[m*K+kk])*(double)bf2f(bm[n*K+kk]);
        double got=cm[m*N+n],ad=std::fabs(got-ref),rd=ad/(std::fabs(ref)+1e-6);
        maxabs=std::max(maxabs,ad);maxrel=std::max(maxrel,rd); if(rd>0.05&&ad>0.05)bad++;
    }
    printf("%dx%dx%d  max_abs=%.4f max_rel=%.4f bad=%d/%d  %s\n",M,K,N,maxabs,maxrel,bad,M*N,bad==0?"CORRECT":"MISMATCH");
    return bad==0?0:1;
}
