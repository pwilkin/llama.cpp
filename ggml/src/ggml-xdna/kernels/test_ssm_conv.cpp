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
    const int NC=8192,KW=4,R=256,NB=NC/R,blk=2*KW*R;
    std::vector<uint32_t> instr;{std::ifstream f("ssm_conv_insts.bin",std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);instr.resize(n/4);f.read((char*)instr.data(),n);}
    xrt::device dev(0);xrt::xclbin xc(std::string("ssm_conv.xclbin"));dev.register_xclbin(xc);
    xrt::hw_context ctx(dev,xc.get_uuid());xrt::kernel k(ctx,"MLIR_AIE");
    auto bi=xrt::bo(dev,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto bin=xrt::bo(dev,(size_t)NB*blk*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bout=xrt::bo(dev,(size_t)NC*4,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto b5=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));auto b6=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(6));auto b7=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(7));
    std::mt19937 rng(2);std::uniform_real_distribution<float> U(-1.f,1.f);
    // conv input x[i,c] and weight w[i,c], i=0..KW-1, c=0..NC-1
    std::vector<float> x(KW*NC),w(KW*NC);
    for(auto&z:x)z=U(rng);for(auto&z:w)z=U(rng);
    // pack blocks [x0(R)|x1|x2|x3|w0|w1|w2|w3]
    auto*im=bin.map<uint16_t*>();
    for(int b=0;b<NB;b++){int off=b*blk;
      for(int t=0;t<KW;t++)for(int c=0;c<R;c++){int cc=b*R+c;im[off+t*R+c]=f2bf(x[t*NC+cc]);im[off+(KW+t)*R+c]=f2bf(w[t*NC+cc]);}
    }
    std::memcpy(bi.map<void*>(),instr.data(),instr.size()*4);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);bin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k(3u,bi,(uint32_t)instr.size(),bin,bout,b5,b6,b7);run.wait();bout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto*om=bout.map<float*>();
    double me=0;int bad=0;
    for(int c=0;c<NC;c++){double a=0;for(int t=0;t<KW;t++)a+=(double)bf(x[t*NC+c])*bf(w[t*NC+c]);double e=std::fabs(om[c]-a);me=std::max(me,e);if(e>0.02)bad++;}
    printf("SSM_CONV NC=%d KW=%d: max_err=%.4f bad=%d/%d %s\n",NC,KW,me,bad,NC,bad==0?"CORRECT":"MISMATCH");
    return bad==0?0:1;
}
