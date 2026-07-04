// Validate softmax xclbin on NPU: LENGTH elems = LENGTH/ROW rows, each softmaxed.
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
    const int ROW=256, LEN=4096, ROWS=LEN/ROW;
    std::vector<uint32_t> instr;
    {std::ifstream f("softmax_insts.bin",std::ios::binary|std::ios::ate);auto n=f.tellg();f.seekg(0);instr.resize(n/4);f.read((char*)instr.data(),n);}
    xrt::device dev(0); xrt::xclbin xc(std::string("softmax.xclbin")); dev.register_xclbin(xc);
    xrt::hw_context ctx(dev,xc.get_uuid()); xrt::kernel k(ctx,"MLIR_AIE");
    auto bi=xrt::bo(dev,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,k.group_id(1));
    auto ba=xrt::bo(dev,LEN*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(3));
    auto bc=xrt::bo(dev,LEN*2,XRT_BO_FLAGS_HOST_ONLY,k.group_id(4));
    auto b5=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(5));
    auto b6=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(6));
    auto b7=xrt::bo(dev,64,XRT_BO_FLAGS_HOST_ONLY,k.group_id(7));
    std::mt19937 rng(5); std::uniform_real_distribution<float> U(-4.f,4.f);
    std::vector<float> in(LEN); auto*am=ba.map<uint16_t*>();
    for(int i=0;i<LEN;i++){in[i]=U(rng);am[i]=f2bf(in[i]);}
    std::memcpy(bi.map<void*>(),instr.data(),instr.size()*4);
    bi.sync(XCL_BO_SYNC_BO_TO_DEVICE); ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run=k(3u,bi,(uint32_t)instr.size(),ba,bc,b5,b6,b7); run.wait();
    bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE); auto*cm=bc.map<uint16_t*>();
    double maxerr=0; int bad=0;
    for(int r=0;r<ROWS;r++){
        double mx=-1e9; for(int j=0;j<ROW;j++) mx=std::max(mx,(double)bf2f(am[r*ROW+j]));
        double sum=0; std::vector<double> e(ROW);
        for(int j=0;j<ROW;j++){e[j]=std::exp((double)bf2f(am[r*ROW+j])-mx);sum+=e[j];}
        for(int j=0;j<ROW;j++){double ref=e[j]/sum,got=bf2f(cm[r*ROW+j]),d=std::fabs(got-ref);maxerr=std::max(maxerr,d);if(d>0.02)bad++;}
    }
    printf("softmax %d rows x %d: max_abs_err=%.5f bad(>0.02)=%d/%d  %s\n",ROWS,ROW,maxerr,bad,LEN,bad==0?"CORRECT":"MISMATCH");
    return bad==0?0:1;
}
