# VRAMER comparative benchmarks

An honest, reproducible comparison of four ways to run the **same** model on the
**same** GPU, with an identical prompt, greedy decode, and token budget. The goal
is to make VRAMER's value legible: where it wins, where it ties the PCIe wall,
and where it simply loses — stated with the exact numbers, not rounded up.

> **Read this first — the one-paragraph verdict.** For a **single-stream,
> latency-bound decode** on this hardware, VRAMER does **not** win on throughput.
> It is bandwidth-bound and pays a per-token pager overhead on top of the H2D
> traffic, so it is slower than llama.cpp's own CPU-offload (`-ot …=CPU`) of the
> same tensors, and far slower than fully resident. What VRAMER *does* deliver,
> measurably, is: **(1) byte-identical output** to the resident baseline (greedy),
> **(2) an explicit, tunable VRAM budget** — you name the ceiling and trade it
> against hit-rate — where native offload gives you no such knob, and **(3) for
> MoE, real selectivity**: it pages only the active-expert slices, streaming
> **7.5 %** of what a full expert-stack copy would move. VRAMER's throughput
> advantage is reserved for a regime these two models on a 15 GiB card do not
> exercise: a paged set that will not fit resident at all, under a hard VRAM cap.

## Hardware & caveats

- **GPU:** 1× NVIDIA A2 (SM86, 15356 MiB), `CUDA_VISIBLE_DEVICES=0`. (The box has
  two A2s; all runs pinned to GPU 0.)
- **Host:** QEMU/KVM VM, 24 vCPU, 78 GiB RAM.
- **PCIe is the dominant caveat.** Measured host→device bandwidth on this VM is
  **~5.7 GiB/s — roughly half of nominal x8 Gen4** (virtualized passthrough).
  Every paging/offload row below is bounded by that number, so absolute tok/s for
  the offloaded rows would roughly double on bare metal; the *ordering* of the
  rows would not change.
- **Peak VRAM** is sampled from `nvidia-smi --query-gpu=memory.used` at 10 Hz
  during each run (idle baseline on GPU 0 is 14 MiB, so peak ≈ footprint). This
  build of llama.cpp suppresses the per-buffer model-load log, so we report the
  honest device-wide peak, which *includes* transient compute/offload buffers,
  not just resident weights.

## Builds

- **Baselines** (`resident`, `ngl=K`, `cpu-offload`) use **upstream llama.cpp**
  at `/tmp/llama.cpp` (commit `6050b46`), built with `GGML_CUDA=ON`,
  `CMAKE_CUDA_ARCHITECTURES=86`.
- **VRAMER** rows use the fork built by
  `examples/llamacpp/adapter/build_fork.sh /tmp/llama.cpp /tmp/llama-vramer-bench`
  (the adapter + VRAMER static lib; **no adapter or library source was changed
  for this benchmark**).

## The four configurations

| row | what it is |
|---|---|
| **resident** | full GPU offload (`-ngl 99`), every weight in VRAM. The 100 % baseline. |
| **ngl=K** | llama.cpp **native partial offload**: K transformer layers on GPU, the rest computed on CPU. |
| **cpu-offload** | the **same** tensor set kept on host with `-ot …=CPU`, uploaded H2D by **llama.cpp's own** per-token copy (`GGML_OP_OFFLOAD_MIN_BATCH=1` forces the GPU op during single-token decode). **VRAMER is not involved.** This is VRAMER's true apples-to-apples rival: same tensors, same GPU compute, llama's `memcpy` vs VRAMER's pager. |
| **vramer/⟨B⟩** | the **same** `-ot …=CPU` tensor set, but every H2D upload is served by the VRAMER pager through a fixed VRAM budget **⟨B⟩ that is smaller than the paged set** (4 MiB pages, MRU, one-ahead prefetch). Three budget points per model. |

## Settings (identical across every row)

- prompt `"The capital of France is"`, greedy decode (`--temp 0`), `--seed 1`,
  `-no-cnv`, `-n 64`.
- warm run: one short generation is discarded before the measured run.
- decode tok/s is llama.cpp's own `eval time … tokens per second` (the **decode**
  line, not prompt eval); the dense model decodes 63 tokens, the MoE model emits
  EOS after 22 — **identical for every row of that model**, so the comparison is
  fair.

---

## Dense — Qwen2.5-0.5B-Instruct Q4_K_M

Paged/offloaded tensor set: the 72 FFN weights, `ffn_(down|gate|up).weight`,
**206.12 MiB**. Budgets 16/32/64 MiB are all **smaller** than that set.
63 decode tokens, greedy, seed 1.

| config | decode tok/s | peak VRAM (MiB) | real H2D (MiB) | pager hit-rate | byte-identical |
|---|--:|--:|--:|--:|:--:|
| resident (`-ngl 99`) | **208.32** | 1180 | — | — | baseline |
| native `-ngl 12` | 29.05 | 810 | — | — | n/a |
| native `-ngl 6` | 19.67 | 656 | — | — | n/a |
| cpu-offload (llama H2D) | 15.92 | 980 | — | — | n/a |
| vramer / 16 MiB | 8.25 | 996 | 22244 | 49.3 % | ✅ |
| vramer / 32 MiB | 9.24 | 1012 | 20940 | 50.4 % | ✅ |
| vramer / 64 MiB | 10.97 | 1044 | 17824 | 55.1 % | ✅ |

Notes:
- **Peak VRAM tracks the budget almost exactly**: 996 → 1012 → 1044 as the budget
  doubles 16 → 32 → 64. The offloaded baseline (cpu-offload) sits at 980; VRAMER
  is essentially "that baseline **plus** its budget". So VRAMER spends *more* peak
  VRAM than native cpu-offload, by exactly the budget you asked for.
- **Bigger budget → higher hit-rate → fewer bytes re-paged → faster** (49 % → 55 %
  hit, 22.2 → 17.8 GiB H2D, 8.3 → 11.0 tok/s): the pager behaves as designed.
- The overlap change only affects the MoE expert path; the dense whole-tensor path
  is behaviorally identical, so these rows match the pre-overlap run within variance.
- All three VRAMER rows are **byte-identical** to the resident baseline.

## MoE — Qwen1.5-MoE-A2.7B-Chat Q4_K_M (60 experts/layer, 4 active)

Paged/offloaded tensor set: the per-layer expert stacks,
`ffn_(down|gate|up)_exps.weight`, **7920 MiB** (72 tensors). Budgets
256/512/1024 MiB are all **smaller** than that set.
22 decode tokens, greedy, seed 1.

VRAMER rows use the **overlap-enabled** adapter (current `master` — the active
expert-slice transfers are pipelined behind a single sync per tensor).

| config | decode tok/s | peak VRAM (MiB) | real H2D (MiB) | pager hit-rate | byte-identical |
|---|--:|--:|--:|--:|:--:|
| resident (`-ngl 99`) | **62.44** | 13946 | — | — | baseline |
| native `-ngl 12` | 18.54 | 7598 | — | — | n/a |
| native `-ngl 6` | 10.24 | 3964 | — | — | n/a |
| cpu-offload (llama H2D) | 6.98 | 7568 | — | — | n/a |
| vramer / 256 MiB | 2.70 | 7824 | 40312 | 34.9 % | ✅ |
| vramer / 512 MiB | 2.88 | 8080 | 39156 | 36.1 % | ✅ |
| vramer / 1024 MiB | 3.11 | 8592 | 36648 | 38.8 % | ✅ |

Notes:
- **Selectivity (the MoE thesis, confirmed):** across the 22-token run VRAMER
  paged only the active experts — **14263 MiB streamed vs 190080 MiB a full
  expert-stack copy would have moved = 7.5 %**. Only the routed 4-of-60 experts
  per layer are brought in. This is the property a dense model does **not** have.
- **Expert-transfer overlap** (shipped, in these numbers) was measured in a
  controlled same-build before/after at **+26–39%** decode at tight budgets — same
  bytes, less wall time. See
  [`examples/llamacpp/README_MOE.md`](../examples/llamacpp/README_MOE.md).
- **Even so, native cpu-offload still leads on this VM** (~7 vs ~3 tok/s) at
  slightly lower peak VRAM (7568 vs 7824–8592): llama's plain `memcpy` of the same
  tensor set pipelines better, and VRAMER still pays per-token pager bookkeeping +
  the compulsory routing-id device sync, and re-pages at 4 MiB granularity, so its
  real H2D (36–40 GiB) is ~2.6× the active bytes (14.3 GiB) at these sub–working-set
  budgets. Faster pager staging is the next core lever (ROADMAP 3a).
- All VRAMER rows are **byte-identical** to the resident baseline.
- **Run-to-run variance:** absolute tok/s moves ~10–30% between runs on this
  passthrough VM, so compare rows **within one run** — all rows in this doc are from
  a single fresh run on current `master` — not across runs or against other reports.

---

## Honest interpretation — what is and isn't a win

**Correctness: a win, unconditional.** Every VRAMER row, dense and MoE, at every
budget, produced output **byte-identical** to the fully-resident baseline under
greedy decoding. Paging weights/experts through a budget smaller than the set
does not perturb a single token.

**VRAM footprint: a tie, not a win — and read the fine print.** VRAMER removes the
paged set from *resident* VRAM (dense: the 206 MiB FFN set; MoE: ~7.7 GiB of
experts, which is what lets the MoE model drop from 13.9 GiB to ~8 GiB and would
let it run on a card where it otherwise would not fit resident). But native
`-ot …=CPU` removes the *same* tensors and lands at the *same* peak — and VRAMER
then adds its budget on top, so VRAMER's peak is marginally **higher** than
native's. VRAMER's distinct footprint property is not a lower peak; it is that the
peak is a number **you set** (the budget), decoupled from the model size, whereas
native offload's transient buffer is whatever llama.cpp chooses (7.5 GiB for this
MoE — itself unbounded, and itself a potential OOM on a smaller card).

**Throughput: a loss for single-stream decode, and that is expected.** On this VM,
`resident` ≫ `native partial offload` ≫ `cpu-offload` > `vramer`. Any
configuration that puts weights on the host is bounded by the ~5.7 GiB/s PCIe
wall; VRAMER additionally pays pager bookkeeping and a per-op device sync, so it
lands below llama's own offload. For a latency-bound single token stream there is
no throughput win to be had here, and we do not claim one.

**Selectivity: a genuine, measured MoE win in kind.** VRAMER streams only the
active experts (7.5 % of a full-stack copy). That is the mechanism that makes
budget-bounded MoE paging viable at all. It does not translate into a tok/s win
over native on *this* hardware/model because native MoE offload is also selective
and avoids the pager overhead — but the selectivity is real and is the foundation
VRAMER's value rests on in the regime these two small-enough-to-fit models don't
reach.

**Bottom line.** VRAMER's honest value proposition from this table is *correctness
+ an explicit, tunable VRAM ceiling + demonstrable MoE selectivity*, not decode
throughput. For single-stream dense decode it ties the PCIe wall like native
CPU-offload (and trails it by the pager overhead); for single-stream MoE decode on
a card big enough to hold the model resident, native offload is the better choice.
VRAMER earns its keep when the paged working set cannot be held resident at all
and must live under a hard, named VRAM budget — the case to benchmark next on a
model that overflows the card.

## Reproduce

```bash
# 1. VRAMER lib + the fork (from the repo root)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
bash examples/llamacpp/adapter/build_fork.sh /tmp/llama.cpp /tmp/llama-vramer-bench

# 2. the whole comparison (both models), pinned to GPU 0
GPU=0 NPRED=64 bash benchmarks/run_comparison.sh both     # or: dense | moe
```

Raw per-row logs (llama.cpp stderr incl. the `[VRAMER]` exit summary, the decoded
text, the VRAM samples, and the machine-readable row) and the assembled table are
written under [`../benchmarks/raw/`](../benchmarks/raw/) — the committed copies
there are the exact logs that produced the numbers above.

The driver `benchmarks/run_comparison.sh` only measures: it builds nothing, and
touches neither the adapter nor the VRAMER library.
