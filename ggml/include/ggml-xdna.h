#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// AMD XDNA (Ryzen AI NPU) backend.
//
// Targets the AIE-ML / XDNA2 NPU found in AMD "Strix"/"Strix Halo" APUs via the
// amdxdna driver and XRT. The NPU shares system memory with the CPU/iGPU, so this
// backend is a DEVICE_TYPE_ACCEL that operates on host tensors and does not need
// its own buffer type.

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_xdna_init(void);

GGML_BACKEND_API bool ggml_backend_is_xdna(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna_reg(void);

#ifdef __cplusplus
}
#endif
