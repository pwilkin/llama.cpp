#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/deprecated/xrt.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>
static uint16_t f2bf(float f){uint32_t b;std::memcpy(&b,&f,4);return (uint16_t)((b+0x8000u)>>16);}
static float    bf2f(uint16_t h){uint32_t b=(uint32_t)h<<16;float f;std::memcpy(&f,&b,4);return f;}
int main(){
    const int NQ=16,DH=64,BLK=16,NBLK=2,KV=NBLK*BLK; const float scale=1.0f/std::sqrt((float)DH);
    const int IN=NQ*DH+2*KV*DH+NQ*KV;
    std::vector<uint32_t> instr;{std::ifstream f("flash_insts.bin",std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);instr.resize(n/4);f.read((char*)instr.data(),n);}
    xrt::device dev(0);xrt::xclbin xc(std::string("flash.xclbin"));dev.register_xclbin(xc);
    xrt::hw_context ctx(dev,xc.get_uuid());xrt::kernel k(ctx,"MLIR_AIE");
    auto bi=xrt::bo(dev,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto ba=xrt::bo(dev,IN*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bc=xrt::bo(dev,NQ*DH*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto b5=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));auto b6=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(6));auto b7=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(7));
    std::mt19937 rng(3);std::uniform_real_distribution<float> U(-1.f,1.f);
    auto*am=ba.map<uint16_t*>();
    std::vector<float> Q(NQ*DH),Kk(KV*DH),Vv(KV*DH),Mm(NQ*KV);
    for(auto&x:Q)x=U(rng);for(auto&x:Kk)x=U(rng);for(auto&x:Vv)x=U(rng);
    for(int iq=0;iq<NQ;iq++)for(int j=0;j<KV;j++)Mm[iq*KV+j]=(j<=iq+8)?0.f:-1e30f; // causal-ish
    int o=0;for(float x:Q)am[o++]=f2bf(x);for(float x:Kk)am[o++]=f2bf(x);for(float x:Vv)am[o++]=f2bf(x);for(float x:Mm)am[o++]=f2bf(x);
    std::memcpy(bi.map<void*>(),instr.data(),instr.size()*4);bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k(3u,bi,(uint32_t)instr.size(),ba,bc,b5,b6,b7);run.wait();bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);auto*cm=bc.map<uint16_t*>();
    double maxerr=0;int bad=0;
    for(int iq=0;iq<NQ;iq++){
        std::vector<double> s(KV);double mx=-1e30;
        for(int j=0;j<KV;j++){double a=0;for(int d=0;d<DH;d++)a+=(double)bf2f(f2bf(Q[iq*DH+d]))*bf2f(f2bf(Kk[j*DH+d]));a=a*scale+bf2f(f2bf(Mm[iq*KV+j]));s[j]=a;mx=std::max(mx,a);}
        double sum=0;for(int j=0;j<KV;j++){s[j]=std::exp(s[j]-mx);sum+=s[j];}
        for(int d=0;d<DH;d++){double a=0;for(int j=0;j<KV;j++)a+=s[j]/sum*bf2f(f2bf(Vv[j*DH+d]));double got=bf2f(cm[iq*DH+d]),e=std::fabs(got-a);maxerr=std::max(maxerr,e);if(e>0.05)bad++;}
    }
    printf("flash NQ=%d KV=%d DH=%d: max_abs_err=%.5f bad(>0.05)=%d/%d  %s\n",NQ,KV,DH,maxerr,bad,NQ*DH,bad==0?"CORRECT":"MISMATCH");
    return bad==0?0:1;
}
