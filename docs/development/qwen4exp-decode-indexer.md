# Qwen4Exp HIP decode indexer ablation

HIP's current small-query flash-attention fallback does not consume the QSA
selected-cell indices. Computing scores and selecting cells during decode therefore
rebuilds work over the full indexer history without changing the attention result.

For at most eight input tokens, one cache stream, flash attention with KQV offload,
no ALiBi and no attention soft cap, the HIP build now stores the raw indexer keys
without computing selection. Attention keeps its ordinary visibility mask. Other
cases keep the previous graph. The raw keys remain available for subsequent prefill.

This is a temporary optimization of the existing dense decode path. Revisit this
guard when adding a sparse decode kernel that actually consumes selected indices.

## Ablation

The optimization is enabled by default. Set `LLAMA_QSA_DECODE_INDEXER=1` before
starting the process to restore the previous indexer graph. The variable is tested
for presence and cached: unset it to enable the optimization again; `=0` also
restores the old graph.

With the installed launcher:

```sh
# Default fix
env -u LLAMA_QSA_DECODE_INDEXER qwen3.8-strix-halo-server

# Original indexer work
LLAMA_QSA_DECODE_INDEXER=1 qwen3.8-strix-halo-server
```

Run each server separately. Do not start another model until the previous process
has exited. These measurements used a 115 GiB memory limit, no scope swap, and a
100 GiB MemAvailable startup check on a 128 GB Strix Halo machine.

## Measurements, 2026-09-13

Baseline source: `be905cf7ded784466c2d0068fa490e0c8a2d0ac5`.
gfx1151, Qwen3.8-Flash-Next IQ4_NL-PROJFIX, F16 K/V, full GPU offload,
batch and ubatch 16384, 16 threads, retained PM4 graph replay enabled.
The initial experiment used the same relinked model library for both MTP arms.

| Workload | Original t/s | Skip t/s | Change |
|---|---:|---:|---:|
| Serial, depth 0, 128 tokens, mean of 3 | 29.844 | 30.525 | +2.3% |
| Serial, depth 40000, 128 tokens, mean of 3 | 22.697 | 26.545 | +17.0% |
| MTP width 3, warm short prompt, 128 tokens | 48.833 | 48.342 | -1.0% |
| MTP width 3, 40680-token prompt, first run | 31.818 | 37.075 | +16.5% |
| MTP width 3, 40680-token prompt, repeat | 32.890 | 36.173 | +10.0% |

The long natural prompt is ggml-backend-meta.cpp with relevant GGML headers and
the instruction to summarize the file. All four paired requests (two short, two
long) produced identical target token IDs: 512/512. Draft and acceptance counts
also matched per request. These are sequential measurements, not a thermally
matched ABBA experiment; the first short request had a startup transient.

Separate traces captured four serial calls per depth. At 40k, all 48 indexer
selection nodes disappeared and all 48 raw-key nodes remained. Indexer pooling
disappeared; RoPE fell from 1.577 to 0.080 GPU ms/token. Attention still uses the
dense masked kernel. Aggregate profiler timings are not native throughput:
quantized matrix-vector kernels ran slower in the candidate trace.

No new kernel arithmetic is introduced. These output and graph checks do not
constitute a full perplexity, multi-sequence or 100k-context qualification.
The proper sparse decode kernel and incremental compressed-key cache remain
separate work.

## Shipping-build check

Rebuilt llama-server and llama-bench with the default-on fix and the ablation
switch, then repeated both server arms against that exact library. All 512 paired
target tokens matched again, including the initial experimental output, and all
paired MTP draft/acceptance counts matched. Both workers exited cleanly without a
memory-guard trigger; minimum MemAvailable was 23.4 GiB.

The rebuilt pair measured 31.182 -> 33.699 and 32.957 -> 35.654 t/s on the two long
requests (+8.1% and +8.2%). Warm short generation was 48.434 -> 47.395 t/s (-2.1%).
The long-context improvement reproduced, but its size varied between sequential
runs; do not treat the initial 16.5% result as a guaranteed improvement.
