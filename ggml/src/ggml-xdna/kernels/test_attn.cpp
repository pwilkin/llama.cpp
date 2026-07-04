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
    const int NQ=32,NKV=32,DH=32; const float scale=0.17677669529663687f;
    const int QKV=(NQ+2*NKV)*DH;
    std::vector<uint32_t> instr;
    {std::ifstream f("attn_insts.bin",std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);instr.resize(n/4);f.read((char*)instr.data(),n);}
    xrt::device dev(0); xrt::xclbin xc(std::string("attn.xclbin")); dev.register_xclbin(xc);
    xrt::hw_context ctx(dev,xc.get_uuid()); xrt::kernel k(ctx,"MLIR_AIE");
    auto bi=xrt::bo(dev,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto ba=xrt::bo(dev,QKV*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bc=xrt::bo(dev,NQ*DH*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto b5=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    auto b6=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(6));
    auto b7=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(7));
    std::mt19937 rng(9); std::uniform_real_distribution<float> U(-1.f,1.f);
    auto*am=ba.map<uint16_t*>(); for(int i=0;i<QKV;i++) am[i]=f2bf(U(rng));
    float* Q=(float*)malloc(NQ*DH*4);float* K=(float*)malloc(NKV*DH*4);float* Vv=(float*)malloc(NKV*DH*4);
    for(int i=0;i<NQ*DH;i++)Q[i]=bf2f(am[i]);
    for(int i=0;i<NKV*DH;i++)K[i]=bf2f(am[NQ*DH+i]);
    for(int i=0;i<NKV*DH;i++)Vv[i]=bf2f(am[NQ*DH+NKV*DH+i]);
    std::memcpy(bi.map<void*>(),instr.data(),instr.size()*4);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE); ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k(3u,bi,(uint32_t)instr.size(),ba,bc,b5,b6,b7); run.wait();
    bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE); auto*cm=bc.map<uint16_t*>();
    // CPU ref
    double maxerr=0;int bad=0;
    for(int iq=0;iq<NQ;iq++){
        std::vector<double> s(NKV); double mx=-1e30;
        for(int ik=0;ik<NKV;ik++){double a=0;for(int d=0;d<DH;d++)a+=(double)Q[iq*DH+d]*K[ik*DH+d];a*=scale;s[ik]=a;mx=std::max(mx,a);}
        double sum=0;for(int ik=0;ik<NKV;ik++){s[ik]=std::exp(s[ik]-mx);sum+=s[ik];}
        for(int d=0;d<DH;d++){double a=0;for(int ik=0;ik<NKV;ik++)a+=s[ik]/sum*Vv[ik*DH+d];
            double got=bf2f(cm[iq*DH+d]),e=std::fabs(got-a);maxerr=std::max(maxerr,e);if(e>0.03)bad++;}
    }
    printf("attention %dx%dx%d: max_abs_err=%.5f bad(>0.03)=%d/%d  %s\n",NQ,NKV,DH,maxerr,bad,NQ*DH,bad==0?"CORRECT":"MISMATCH");
    return bad==0?0:1;
}
