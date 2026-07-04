# Build XDNA2 (NPU2 / AIE2P) elementwise-unary xclbins from unary.cc + build_unary.py.
#
# Prerequisites:
#   1. ./setup-toolchain.sh          (creates .aie-venv with mlir-aie + Peano)
#   2. xclbinutil on PATH            (from an XRT install; used to package the xclbin)
#      e.g.  export PATH=/opt/xilinx/xrt/bin:$PATH
#
# Then:  make            # builds build/<op>.xclbin + build/<op>_insts.bin for every op
#        make relu       # just one op
#        make clean

OPS    := relu neg abs exp tanh sigmoid silu gelu gelu_quick
LENGTH ?= 4096
TILE   ?= 1024

# Toolchain locations — override on the command line if your venv differs.
VENV     ?= $(CURDIR)/.aie-venv
SP        = $(VENV)/lib/python3.12/site-packages
PEANO    ?= $(SP)/llvm-aie
MLIR_AIE ?= $(SP)/mlir_aie
PYTHON   ?= $(VENV)/bin/python
AIECC    ?= $(VENV)/bin/aiecc.py

CXX      = $(PEANO)/bin/clang++
CXXFLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG \
           -I $(MLIR_AIE)/include -D__AIE_API_AIE_ADF_HPP__ -DTILE_N=$(TILE)

BUILD   := build
XCLBINS := $(OPS:%=$(BUILD)/%.xclbin)

all: $(XCLBINS)

# every op links the same object (unary.cc holds them all); the AIE design's
# `link_with` expects a file named after the entry point, so we symlink per op.
$(BUILD)/unary.o: unary.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.xclbin: build_unary.py $(BUILD)/unary.o
	$(PYTHON) build_unary.py --op $* --length $(LENGTH) --tile $(TILE) --mlir $(BUILD)/$*.mlir
	ln -sf unary.o $(BUILD)/xdna_$*_bf16.o
	cd $(BUILD) && PATH="$(VENV)/bin:$$PATH" $(AIECC) --aie-generate-xclbin --no-compile-host \
	    --xclbin-name=$*.xclbin --aie-generate-npu-insts --npu-insts-name=$*_insts.bin \
	    --no-xchesscc --no-xbridge --peano $(PEANO) $*.mlir

$(BUILD):
	mkdir -p $(BUILD)

# convenience: `make relu` etc.
$(OPS): %: $(BUILD)/%.xclbin

clean:
	rm -rf $(BUILD)

.PHONY: all clean $(OPS)
