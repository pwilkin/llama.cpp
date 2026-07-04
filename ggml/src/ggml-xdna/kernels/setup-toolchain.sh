#!/usr/bin/env bash
# Set up the open MLIR-AIE / IRON + Peano (LLVM-AIE) toolchain used to compile
# the XDNA2 (Ryzen AI NPU / AIE2P) kernels in this directory into .xclbin.
#
# Verified on Ubuntu 26.04 / Strix Halo (AMD Ryzen AI MAX+ 395), 2026-07.
# Needs Python 3.12 (mlir-aie wheels don't build for 3.13/3.14 yet) — we make an
# isolated venv with `uv` so the system Python is untouched.
#
# Usage:  ./setup-toolchain.sh [VENV_DIR]   (default: ./.aie-venv)
set -euo pipefail

VENV="${1:-$(cd "$(dirname "$0")" && pwd)/.aie-venv}"

command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }

uv venv --python 3.12 "$VENV"
# mlir-aie: the IRON front end + aiecc.py compiler driver
uv pip install --python "$VENV" mlir_aie \
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-2
# llvm-aie ("Peano"): the LLVM-based AIE backend compiler
uv pip install --python "$VENV" llvm-aie \
    -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

PEANO="$VENV/lib/python3.12/site-packages/llvm-aie"

cat <<EOF

Toolchain ready.
  activate:  source "$VENV/bin/activate"
  peano:     $PEANO           (export PEANO_INSTALL_DIR=$PEANO)
  target:    NPU2  (from aie.iron.device — this is XDNA2 / Strix Halo)
  compiler:  aiecc.py         (in $VENV/bin)

Build a kernel:  make -C "\$(dirname "$0")"   (see the Makefile)
EOF
