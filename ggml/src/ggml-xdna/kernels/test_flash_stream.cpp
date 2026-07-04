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
int main(){
    const int NQ=16,DH=64,BLK=16,NBLK=4,KV=NBLK*BLK; const float scale=0.125f;
    const int blk=2*BLK*DH+NQ*BLK, KVLEN=NBLK*blk, STATE=NQ*DH+2*NQ;
    std::vector<uint32_t> instr;{std::ifstream f("flash_insts.bin",std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);instr.resize(n/4);f.read((char*)instr.data(),n);}
    xrt::device dev(0);xrt::xclbin xc(std::string("flash.xclbin"));dev.register_xclbin(xc);
    xrt::hw_context ctx(dev,xc.get_uuid());xrt::kernel k(ctx,"MLIR_AIE");
    auto bi=xrt::bo(dev,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto bq=xrt::bo(dev,NQ*DH*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bkv=xrt::bo(dev,KVLEN*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto bo_=xrt::bo(dev,STATE*4,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    auto b6=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(6));auto b7=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(7));
    std::mt19937 rng(3);std::uniform_real_distribution<float> U(-1.f,1.f);
    std::vector<float> Q(NQ*DH),Kk(KV*DH),Vv(KV*DH),Mm(NQ*KV);
    for(auto&x:Q)x=U(rng);for(auto&x:Kk)x=U(rng);for(auto&x:Vv)x=U(rng);
    for(int iq=0;iq<NQ;iq++)for(int j=0;j<KV;j++)Mm[iq*KV+j]=(j<=iq+20)?0.f:-1e30f;
    auto*qm=bq.map<uint16_t*>();for(int i=0;i<NQ*DH;i++)qm[i]=f2bf(Q[i]);
    auto*km=bkv.map<uint16_t*>();
    for(int b=0;b<NBLK;b++){int off=b*blk;
      for(int ic=0;ic<BLK;ic++)for(int d=0;d<DH;d++)km[off+ic*DH+d]=f2bf(Kk[(b*BLK+ic)*DH+d]);
      for(int ic=0;ic<BLK;ic++)for(int d=0;d<DH;d++)km[off+BLK*DH+ic*DH+d]=f2bf(Vv[(b*BLK+ic)*DH+d]);
      for(int iq=0;iq<NQ;iq++)for(int ic=0;ic<BLK;ic++)km[off+2*BLK*DH+iq*BLK+ic]=f2bf(Mm[iq*KV+b*BLK+ic]);
    }
    std::memcpy(bi.map<void*>(),instr.data(),instr.size()*4);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);bq.sync(XCL_BO_SYNC_BO_TO_DEVICE);bkv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k(3u,bi,(uint32_t)instr.size(),bq,bkv,bo_,b6,b7);run.wait();bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto*om=bo_.map<float*>();
    double maxerr=0;int bad=0;
    for(int iq=0;iq<NQ;iq++){std::vector<double> s(KV);double mx=-1e30;
      for(int j=0;j<KV;j++){double a=0;for(int d=0;d<DH;d++)a+=(double)bf2f(f2bf(Q[iq*DH+d]))*bf2f(f2bf(Kk[j*DH+d]));a=a*scale+bf2f(f2bf(Mm[iq*KV+j]));s[j]=a;mx=std::max(mx,a);}
      double sum=0;for(int j=0;j<KV;j++){s[j]=std::exp(s[j]-mx);sum+=s[j];}
      for(int d=0;d<DH;d++){double a=0;for(int j=0;j<KV;j++)a+=s[j]/sum*bf2f(f2bf(Vv[j*DH+d]));double e=std::fabs(om[iq*DH+d]-a);maxerr=std::max(maxerr,e);if(e>0.05)bad++;}
    }
    printf("STREAMING flash NQ=%d KV=%d DH=%d NBLK=%d: max_err=%.5f bad=%d/%d  %s\n",NQ,KV,DH,NBLK,maxerr,bad,NQ*DH,bad==0?"CORRECT":"MISMATCH");
    return bad==0?0:1;
}
