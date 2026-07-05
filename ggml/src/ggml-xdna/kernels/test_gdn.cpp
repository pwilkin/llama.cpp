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
    const int SV=128,R=16,NB=SV/R; const float scale=1.0f/std::sqrt((float)SV);
    const int BLK=R*SV+R;
    std::vector<uint32_t> instr;{std::ifstream f("gdn_insts.bin",std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);instr.resize(n/4);f.read((char*)instr.data(),n);}
    xrt::device dev(0);xrt::xclbin xc(std::string("gdn.xclbin"));dev.register_xclbin(xc);
    xrt::hw_context ctx(dev,xc.get_uuid());xrt::kernel k(ctx,"MLIR_AIE");
    auto bi=xrt::bo(dev,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto bp=xrt::bo(dev,(3*SV+16)*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bin=xrt::bo(dev,(size_t)NB*BLK*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto bout=xrt::bo(dev,(size_t)NB*BLK*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    auto b6=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(6));auto b7=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(7));
    std::mt19937 rng(7);std::uniform_real_distribution<float> U(-1.f,1.f);
    std::vector<float> S(SV*SV),q(SV),kk(SV),v(SV),g(SV);float beta=0.6f;
    for(auto&x:S)x=U(rng);for(auto&x:q)x=U(rng);for(auto&x:kk)x=U(rng);for(auto&x:v)x=U(rng);for(auto&x:g)x=U(rng)*0.3f;
    // params
    auto*pm=bp.map<uint16_t*>();
    for(int i=0;i<SV;i++){pm[i]=f2bf(std::exp(g[i]));pm[SV+i]=f2bf(kk[i]);pm[2*SV+i]=f2bf(q[i]);}
    pm[3*SV]=f2bf(beta);pm[3*SV+1]=f2bf(scale);
    // block_in: M rows (transposed state) + v
    auto*im=bin.map<uint16_t*>();
    for(int b=0;b<NB;b++){int off=b*BLK;
      for(int r=0;r<R;r++){int j=b*R+r;for(int i=0;i<SV;i++)im[off+r*SV+i]=f2bf(S[j*SV+i]);im[off+R*SV+r]=f2bf(v[j]);}
    }
    std::memcpy(bi.map<void*>(),instr.data(),instr.size()*4);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);bp.sync(XCL_BO_SYNC_BO_TO_DEVICE);bin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k(3u,bi,(uint32_t)instr.size(),bp,bin,bout,b6,b7);run.wait();bout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto*om=bout.map<uint16_t*>();
    // CPU reference (bf16-rounded to match)
    std::vector<float> M(SV*SV),attn(SV);
    for(int j=0;j<SV;j++)for(int i=0;i<SV;i++)M[j*SV+i]=bf(S[j*SV+i]);
    std::vector<float> expg(SV);for(int i=0;i<SV;i++)expg[i]=bf(std::exp2((float)(bf(g[i])*bf(1.44269504f))));
    double me_s=0,me_a=0;
    for(int j=0;j<SV;j++){
      for(int i=0;i<SV;i++)M[j*SV+i]=bf(M[j*SV+i]*expg[i]);
      float dk=0;for(int i=0;i<SV;i++)dk+=M[j*SV+i]*bf(kk[i]);
      float delta=(bf(v[j])-dk)*bf(beta);
      float db=bf(delta);
      float dq=0;for(int i=0;i<SV;i++){M[j*SV+i]=bf(M[j*SV+i]+bf(db*bf(kk[i])));dq+=M[j*SV+i]*bf(q[i]);}
      attn[j]=dq*scale;
    }
    // compare
    for(int b=0;b<NB;b++)for(int r=0;r<R;r++){int j=b*R+r;
      for(int i=0;i<SV;i++)me_s=std::max(me_s,(double)std::fabs(bf2f(om[b*BLK+r*SV+i])-M[j*SV+i]));
      me_a=std::max(me_a,(double)std::fabs(bf2f(om[b*BLK+R*SV+r])-attn[j]));
    }
    // fp32 reference (bf16 inputs, fp32 intermediates) to gauge the true bf16 cost
    std::vector<float> Mf(SV*SV),attnf(SV);
    for(int j=0;j<SV;j++)for(int i=0;i<SV;i++)Mf[j*SV+i]=bf(S[j*SV+i]);
    for(int j=0;j<SV;j++){
      for(int i=0;i<SV;i++)Mf[j*SV+i]*=std::exp2(bf(g[i])*1.44269504f);
      float dk=0;for(int i=0;i<SV;i++)dk+=Mf[j*SV+i]*bf(kk[i]);
      float delta=(bf(v[j])-dk)*bf(beta);
      float dq=0;for(int i=0;i<SV;i++){Mf[j*SV+i]+=delta*bf(kk[i]);dq+=Mf[j*SV+i]*bf(q[i]);}
      attnf[j]=dq*scale;
    }
    double me_sf=0,me_af=0;int badc=0;
    for(int b=0;b<NB;b++)for(int r=0;r<R;r++){int j=b*R+r;
      for(int i=0;i<SV;i++){double e=std::fabs(bf2f(om[b*BLK+r*SV+i])-Mf[j*SV+i]);me_sf=std::max(me_sf,e);if(e>0.1)badc++;}
      me_af=std::max(me_af,(double)std::fabs(bf2f(om[b*BLK+R*SV+r])-attnf[j]));
    }
    printf("GDN SV=%d: vs-bf16 state=%.3f attn=%.3f | vs-fp32 state=%.3f attn=%.3f bad=%d/%d\n",SV,me_s,me_a,me_sf,me_af,badc,SV*SV);
    return 0;
}
