# XDNA2 (Ryzen AI NPU / AIE2P) kernels

AIE compute kernels for the `ggml-xdna` backend, plus the toolchain to compile
them to loadable `.xclbin` binaries. This is the on-NPU execution path that
replaces the host-side placeholder kernels in `../ggml-xdna.cpp` (gated by
`GGML_XDNA_HAS_XRT`).

## Contents

| file | what |
|------|------|
| `unary.cc` | Vectorized elementwise unary activations for one AIE2P compute tile, in bfloat16 (the NPU's native type): `relu, neg, abs, exp, tanh, sigmoid, silu, gelu, gelu_quick`. One `extern "C"` entry point per op. |
| `build_unary.py` | IRON (`aie.iron`) program: builds a single-core `NPU2` dataflow (shim-DMA in → compute tile → shim-DMA out) that binds one op kernel, and lowers it to MLIR. |
| `Makefile` | Compiles `unary.cc` with Peano and drives `aiecc.py` to produce `build/<op>.xclbin` + `build/<op>_insts.bin` for every op. |
| `setup-toolchain.sh` | Installs the open MLIR-AIE + Peano toolchain into an isolated Python 3.12 venv. |

## Build

```bash
./setup-toolchain.sh                 # one-time: mlir-aie + llvm-aie (Peano)
export PATH=/opt/xilinx/xrt/bin:$PATH # xclbinutil, to package the xclbin
make                                 # -> build/<op>.xclbin + build/<op>_insts.bin
```

Everything uses the **open** toolchain — Peano (LLVM-AIE), no proprietary
Vitis/xchesscc. Target device is `NPU2` / `aie2p-none-unknown-elf` (XDNA2, Strix
Halo). Verified building all 9 ops on Ubuntu 26.04 / Ryzen AI MAX+ 395.

## Kernel notes

- Data is **bfloat16** at 16 lanes/vector (the fp32-accumulator activation path).
  `aie::mul/add/sub` return an `aie::accum`; `.to_vector<T>()` (or assignment to
  an `aie::vector`) narrows back to bf16.
- `exp` uses `2^(log2e·x)` via `aie::exp2`; `tanh`/`sigmoid`/`silu`/`gelu` use the
  hardware `aie::tanh`. Idioms follow the AMD reference kernels in
  `mlir-aie/include/aie_kernels/aie2p/{silu,gelu,bf16_exp}.cc`.
- `TILE_N` (default 1024) is the element count per invocation; keep it a multiple
  of the vector width.

## Wiring into the backend (next step)

`ggml-xdna.cpp` currently computes unary ops on the host. Under
`GGML_XDNA_HAS_XRT` the plan is to load these xclbins once and dispatch via XRT.
The runtime sequence (from the mlir-aie test hosts) is:

```cpp
// once, at backend init (per op / per xclbin):
xrt::device dev(0);
auto xclbin = xrt::xclbin("relu.xclbin");
dev.register_xclbin(xclbin);
xrt::hw_context ctx(dev, xclbin.get_uuid());
xrt::kernel k(ctx, "MLIR_AIE");                 // kernel name in the xclbin
auto instr = load_instr_binary("relu_insts.bin");

// per op invocation:
auto bo_instr = xrt::bo(dev, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
auto bo_in    = xrt::bo(dev, n*sizeof(bf16), XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
auto bo_out   = xrt::bo(dev, n*sizeof(bf16), XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
memcpy(bo_instr.map(), instr.data(), instr.size()*4);
/* f32 src -> bf16 into bo_in.map() */         bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
auto run = k(/*opcode*/3, bo_instr, instr.size(), bo_in, bo_out);
run.wait();
bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);        /* bf16 -> f32 dst */
```

Open items for that step: f32↔bf16 conversion at the buffer edge, tiling the
tensor into `TILE_N` chunks, batching multiple ops into one design to amortize
DMA, and — the actually-motivating case — running small models (Whisper /
embeddings) on the NPU concurrently with the iGPU.
