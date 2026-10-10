#pragma once

// Tile geometry and gates for the mmw kernels on RDNA3.5 (gfx1151); an architecture without a header here keeps its matmuls on MMQ. The routed tiles were tuned with llama-bench prompt processing on Qwen3.8-Flash-Next (512 experts, top-10) and GLM-5.3-Flash (288, top-8).
namespace mmw_cfg_rdna3_5 {

// dense MUL_MAT tile: weight rows x tokens per block, split over 8 waves
constexpr int dense_bm  = 128;
constexpr int dense_bn  = 128;
constexpr int dense_wtm = 32;
constexpr int dense_wtn = 64;

// routed MUL_MAT_ID tile for experts with at least thresh rows in the ubatch
constexpr int routed_bm  = 128;
constexpr int routed_bn  = 128;
constexpr int routed_wtm = 32;
constexpr int routed_wtn = 64;

// routed tile for the remaining experts: a narrow token tile wastes less of the block. Its 32 x 32 wave tiles leave 4 of the 8 waves to multiply (waves 0..3, one per SIMD): fewer shared-memory operand loads
constexpr int routed_small_bm  = 128;
constexpr int routed_small_bn  = 32;
constexpr int routed_small_wtm = 32;
constexpr int routed_small_wtn = 32;
constexpr int routed_thresh    = 32;

// smallest token count the kernels take; below it MMQ is faster
constexpr int min_t = 512;

} // namespace mmw_cfg_rdna3_5
