# Qwen4Exp sparse decode and incremental indexer

The Strix Halo build enables sparse target and MTP decode and incremental indexer
state automatically. No experiment environment variables are required. This
supersedes the temporary dense-decode optimization that skipped unused indexer
selection: sparse attention consumes the selected indices, so selection now runs.

## Dispatch and fallback

The gfx1151 sparse decode kernels read selected cells directly from existing F16
K/V caches. They support D=256, one stream and 1-8 query tokens, with F32 queries
and output, up to 262144 cached cells and 2560 selected indices. Sinks, ALiBi and
attention soft caps use the existing fallback. The WMMA path requires 12 query
heads per KV head and aligned keys; other supported layouts use the SIMT path.

MTP small-query selection requires HIP, flash attention with KQV offload, one
stream, no ALiBi and no attention soft cap. Short histories below the selection
width retain the existing dense attention behavior.

The HIP indexer retains F32 pooled, normalized and rotated keys for compression
ratio 4. Contiguous scalar positions owned by one sequence use incremental
updates; other layouts retain full preparation. A decode step updates only the
affected groups, with distinct padding groups to keep the graph shape reusable.

Suffix rollback preserves unaffected groups. Full state restore and sequence
copy invalidate derived state; an eligible single-owner prefix is reconstructed
from the raw cache and reseeded once. Shared, holey or shifted layouts fall back.
Derived keys are not serialized, so the raw state format is unchanged. Allocation
and memory reporting include the cache: 104 MiB for target plus MTP at 65536
context, or approximately 416 MiB at full 262144 context for this model.

## Qualification, 2026-09-14

Qwen3.8-Flash-Next IQ4_NL-PROJFIX on gfx1151, F16 K/V, full GPU offload,
retained PM4 graph replay, 16 threads. Natural requests use batch and ubatch
16384, MTP width 3, and ggml-backend-meta.cpp plus relevant headers (40680 tokens).

Same-build sparse recomputation versus incremental state:

| Workload | Recompute t/s | Incremental t/s |
|---|---:|---:|
| Serial, depth 40000, 128 tokens, mean of 3 | 25.847 | 28.816 |
| MTP, 40680-token prompt, first request | 31.17 | 35.57 |
| MTP, 40680-token prompt, repeated request | 32.69 | 39.10 |

All 512 paired target tokens and MTP draft/acceptance counts matched. Against the
previous shipping dense path, a separate same-build pair measured 34.47 -> 35.41
and 36.87 -> 38.43 t/s for the long requests. Warm short means were 46.51 and
46.64 t/s. These are sequential measurements, not a guaranteed speedup.

Kernel tests used a scalar FP64 attention oracle, including masked and maskless
inputs, masked NaN poisoning, interleaved layouts and full-capacity boundaries.
SIMT passed 18/18 cases and WMMA passed 49/49 at NMSE <= 1e-5. Live incremental
cache checks matched full recomputation exactly, including metadata. Lifecycle
checks covered append, rollback, restore, copy, keep and clear, with 1,986,560
bit-identical logits. A 20000-operation randomized prefix-state test passed.

A two-context WikiText-2 screen with 8192-token contexts and ubatch 8 exercised
the decode path: perplexity was 2.0244 +/- 0.03685 both before and after caching.
This is a limited screen, not a full-corpus or 100k-prefill qualification. A full
40680-token source summary completed coherently to EOS.

For reproduction of the old ablations, use the pre-promotion source and archived
experiment patches. The obsolete decode-indexer bypass and sparse/incremental
experiment switches, including reference-graph instrumentation, have been removed.

## Default-build verification

After removing the switches, the rebuilt library passed 18 masked/maskless
selected-attention cases with dispatch counters confirming sparse execution,
the 20000-operation prefix-state test, and the eight-stage lifecycle comparison
(1,986,560 bit-identical logits plus identical serialized state).

All 512 natural-request tokens and MTP draft/acceptance counts matched the
qualified opt-in implementation. The two 40680-token requests measured 1164.30
and 1174.60 prefill t/s, and 35.83 and 39.01 generation t/s. Warm short generation
was 50.28 t/s. This validates the default dispatch; it is not a new matched A/B
performance claim. Minimum MemAvailable was 22.89 GiB, with no memory-guard trigger.
