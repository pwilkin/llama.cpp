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
    const int HD=256,NDIMS=64,ND2=32,NR=4; const float freq_base=1e7f; const int pos=5;
    const float theta_scale=std::pow(freq_base,-2.0f/NDIMS);
    std::vector<uint32_t> instr;{std::ifstream f("rope_insts.bin",std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);instr.resize(n/4);f.read((char*)instr.data(),n);}
    xrt::device dev(0);xrt::xclbin xc(std::string("rope.xclbin"));dev.register_xclbin(xc);
    xrt::hw_context ctx(dev,xc.get_uuid());xrt::kernel k(ctx,"MLIR_AIE");
    auto bi=xrt::bo(dev,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto bcs=xrt::bo(dev,2*ND2*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bin=xrt::bo(dev,(size_t)NR*HD*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto bout=xrt::bo(dev,(size_t)NR*HD*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    auto b6=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(6));auto b7=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(7));
    std::mt19937 rng(9);std::uniform_real_distribution<float> U(-1.f,1.f);
    std::vector<float> src(NR*HD);for(auto&x:src)x=U(rng);
    std::vector<float> cs(ND2),sn(ND2);
    for(int j=0;j<ND2;j++){float th=pos*std::pow(theta_scale,(float)j);cs[j]=std::cos(th);sn[j]=std::sin(th);}
    auto*csm=bcs.map<uint16_t*>();for(int j=0;j<ND2;j++){csm[j]=f2bf(cs[j]);csm[ND2+j]=f2bf(sn[j]);}
    auto*im=bin.map<uint16_t*>();for(int i=0;i<NR*HD;i++)im[i]=f2bf(src[i]);
    std::memcpy(bi.map<void*>(),instr.data(),instr.size()*4);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);bcs.sync(XCL_BO_SYNC_BO_TO_DEVICE);bin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k(3u,bi,(uint32_t)instr.size(),bcs,bin,bout,b6,b7);run.wait();bout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto*om=bout.map<uint16_t*>();
    // CPU NEOX reference
    double me=0;int bad=0;
    for(int r=0;r<NR;r++){const float*s=src.data()+r*HD;
      std::vector<float> o(HD);
      for(int j=0;j<ND2;j++){float x0=bf(s[j]),x1=bf(s[j+ND2]);o[j]=x0*bf(cs[j])-x1*bf(sn[j]);o[j+ND2]=x0*bf(sn[j])+x1*bf(cs[j]);}
      for(int i=NDIMS;i<HD;i++)o[i]=bf(s[i]);
      for(int i=0;i<HD;i++){double e=std::fabs(bf2f(om[r*HD+i])-o[i]);me=std::max(me,e);if(e>0.03)bad++;}
    }
    printf("ROPE HD=%d NDIMS=%d NR=%d: max_err=%.4f bad=%d/%d %s\n",HD,NDIMS,NR,me,bad,NR*HD,bad==0?"CORRECT":"MISMATCH");
    return bad==0?0:1;
}
