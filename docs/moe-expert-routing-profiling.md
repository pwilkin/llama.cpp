# Profiling MoE expert routing predictability

On a MoE model whose routed experts do not fit in RAM, decode is disk-bound: `throughput ≈
disk_bandwidth / bytes_per_token`. The obvious way to buy back time is to start reading an expert's
weights before the model asks for them — but routing is data-dependent, so layer L+1's expert ids do
not exist until layer L has finished computing. There is no lead time to prefetch into.

The way around the data dependency is not to *predict* the routing but to *compute the real router on
a stale input*: run layer L+1's own `ffn_gate_inp` on layer L's activations. The residual stream
changes only incrementally between layers, so the same experts may well come out. If the overlap with
the true routing is high enough, the prefetch can be issued a full layer early.

Whether it is high enough is an empirical question, and it is a property of the *model*, not of
llama.cpp. This document describes the harness that measures it.

> **The measured answer for one model is already in the tree** and is net negative — see
> [Interpreting the result](#interpreting-the-result). Read that before building anything on top of a
> good-looking accuracy number.

---

## What is measured

The harness hangs off the CPU backend's `MUL_MAT_ID` hook, which fires once per MoE node before any
expert row is read. At that point both things it needs are already host-resident: the routed ids
(`src[2]`) and the router input (the gate/up node's `src[1]`).

For every MoE layer it reimplements `build_moe_ffn()`'s selection on the CPU — `logits` →
`+ffn_gate_inp_b` → gating op (softmax/sigmoid) → `+ffn_exp_probs_b` → expert-group mask → top-k —
and reports two overlap figures against the routing the model actually used:

| figure | what it runs | meaning |
|---|---|---|
| `self` | layer L's router on **layer L's** activations | **control.** Must be ~100%. |
| `pred` | layer L's router on **layer L-1's** activations | the hypothesis. |

**`self` is not optional and not decoration.** It is what separates "stale activations are a poor
predictor" from "the router reimplementation is wrong". If `self` is not ~100% on your model, `pred`
is meaningless and the selection path above is missing something your architecture does (a gating
function, a bias, an expert-group scheme). Fix that first.

Overlap is counted per slot: of the `n_expert_used` experts the model actually selected, how many
appear in the predicted top-k. So the floor is chance, `n_expert_used / n_expert` — for 8-of-256 that
is 3.1%, and a layer scoring near it is telling you the prediction carries no information at all.

---

## Requirements

The harness lives inside the `--lazy-experts` prefetcher, so it only exists when that prefetcher does:

- **POSIX** (Linux/macOS). Compiled out elsewhere.
- **mmap'd model** — the default. Not `--no-mmap`, not direct I/O.
- **`--lazy-experts`**, which is what constructs the prefetcher at all.
- **Experts on the CPU** (`--cpu-moe` / `-ncmoe`). The hook is a CPU-backend hook, and expert rows
  are only ever advised when they fall inside one of the model's mappings — offloaded experts live in
  device buffers, are skipped, and if *every* layer is skipped the prefetcher is never created.
- **`LLAMA_EXPERT_PREFETCH_THREADS` not set to 0** — 0 disables the prefetcher, and the harness with
  it.

Note that the model does *not* have to exceed RAM to be profiled. Routing predictability is a
property of the weights; measure it on whatever fits, then decide whether the I/O economics work.

---

## Running it

```bash
LLAMA_EXPERT_PREDICT_STATS=1 \
llama-server -m model.gguf --lazy-experts --cpu-moe -ngl 99 --no-warmup
```

Then generate at least a few hundred tokens. Two things about that:

- **Only decode is sampled.** Batches wider than 8 tokens are ignored, because a prompt-processing
  batch routes to nearly every expert and would swamp the statistic. Prompt processing contributes
  nothing, so a run that only ingests a prompt reports nothing.
- **Accuracy converges slowly per layer.** One measured model moved 76.8% → 76.2% between n=3999 and
  n=7999, i.e. the third digit was still settling well past a thousand tokens. Treat a hundred-token
  run as a smoke test.

Confirm at startup that the routers were actually loaded:

```
expert predict: off+stats, 75 routers (450 MiB), min_layer=15, n_embd=6144 n_expert=256 n_used=8 gating=2 groups=1/1
```

`0 routers` means the layers had no `ffn_gate_inp` the loader could reach, and nothing will be
reported.

### Output

Stats are dumped every 4000 observations rather than only at exit — a `>RAM` mmap'd model can deadlock
the server on shutdown, and losing the whole run to that is annoying:

```
PREDICT layer  30: n=  1000  self=100.0%  pred= 82.4%
PREDICT layer  31: n=  1000  self=100.0%  pred= 81.9%
...
PREDICT TOTAL: n=  3999  self=100.0%  pred= 76.8%  (self is the control: if it is not ~100% the router reimplementation is wrong)
```

The interesting structure is per-layer, not the total. Expect a strong depth gradient, and expect the
first MoE layer after a dense stack to be the worst layer in the model by a wide margin — that is
where the residual is least incremental, so a stale input predicts nothing.

### Costs of measuring

- **Two router evaluations per MoE layer per token**, on a compute thread. On the reference model that
  cost ~6% of decode throughput (1.71 vs 1.82 t/s). It is measurement overhead, not the feature's
  cost — the prefetch path runs one evaluation, not two.
- **Stats mode dequantises every layer's router to F32 and holds it in RAM**, ignoring `min_layer`,
  because the depth profile is the whole point. That is `n_layer × n_expert × n_embd × 4` bytes —
  450 MiB on the reference model. On a working set already larger than RAM that displaces page
  cache, so profile and benchmark in separate runs; do not read throughput off a stats run.

---

## Acting on the result

The same code can issue prefetches from the prediction instead of just scoring it:

| variable | default | meaning |
|---|---|---|
| `LLAMA_EXPERT_PREDICT_STATS` | unset | collect and report the accuracy profile |
| `LLAMA_EXPERT_PREDICT` | `0` (off) | prefetch layer L+1's predicted experts while layer L computes |
| `LLAMA_EXPERT_PREDICT_MIN_LAYER` | `15` | only predict at or beyond this depth |
| `LLAMA_EXPERT_PREFETCH_THREADS` | `8` | worker pool faulting rows in; `0` disables everything |

`min_layer` exists because of the depth gradient: the shallow layers are where accuracy is worst, so
predicting them burns bandwidth and evicts cache for nothing. Set it from your own profile — the
default is tuned to one model and is not a universal constant.

### Interpreting the result

The reference measurement (GLM-5.2 IQ1_M, 213 GB, 123 GB RAM, `n_embd` 6144, 256 experts, 8 used,
sigmoid gating, no expert groups) came out at `self` = 100.0% exactly and `pred` = 76.8%, with a clear
depth gradient:

| layers | mean overlap |
|---|---|
| 0–14 | 59.4% |
| 15–29 | 73.1% |
| 30–44 | 82.0% |
| 45–59 | 83.1% |
| 60–74 | 83.5% (best single layer 87.3%) |

Layer 3 — the first MoE layer after the dense stack — scored 3.2%, which is exactly chance for
8-of-256.

**And the prefetch built on it still lost.** Every intended mechanical effect showed up: queue depth
6.70 → 10.49, device utilisation 35.2 → 44.1%, mean request 56.8 → 65.8 KB, reads genuinely issued a
layer early. Throughput fell anyway:

| config | decode | vs base | bytes read |
|---|---|---|---|
| baseline | 1.867 t/s | — | — |
| `min_layer=30` (~83% accuracy band) | 1.820 t/s | −2.5% | +16.5% |
| `min_layer=15` (mixed) | 1.740 t/s | −6.8% | +22.0% |

Restricting to the accurate layers cut waste and loss together, monotonically — but the line never
crossed zero. That is the lesson worth carrying: **on a working set larger than RAM, a wasted byte
costs in direct proportion, so break-even needs accuracy near 100%, not near 70%.** A ~77% predictor
is genuinely informative and still not good enough, and any go/no-go threshold has to be derived from
the byte economics rather than picked as a round number.

This does **not** generalise to every deployment. The cost here is entirely "extra reads displace page
cache the rest of the model depends on". Where expert weights are not competing for cache — they fit,
or they come from somewhere with spare bandwidth — the same prediction at the same accuracy is a
straight win. That is why the code is kept and defaulted off rather than removed.
