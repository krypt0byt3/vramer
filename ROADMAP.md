# Roadmap

Every item is judged by one question: **does it either raise effective PCIe
bandwidth, or target a workload where you don't re-stream everything every token?**
Paging an oversized dense model for single-stream decode is PCIe-bound — see
[`docs/TUNING.md`](docs/TUNING.md). Everything below is scoped against that reality.

## Tier 1 — biggest leverage

1. ✅ **Prefetch/compute overlap (double-buffering).** *Done.* The pager's
   `acquire()` already returns an async lease and `bench` already issues
   `prefetch_async` ahead of the demand cursor, so transfer overlaps compute. Added
   the missing GPU regression test (`prefetch_overlap`) and an honest measurement
   harness (`bench --sync-per-page`). Measured ~**2× vs a blocking consumer**;
   neutral-but-free (hit rate 0 → ~99%) for an already-pipelined, PCIe-bound scan.
   *Remaining:* push the same layer-ahead overlap into the llama.cpp adapter itself
   (currently node-granular with a blocking gather).
2. 🟢 **A real MoE demo (Qwen1.5-MoE-A2.7B).** *Selectivity + correctness done;
   speed improved, floor documented.* The llama.cpp hook pages **only the active
   experts** per token (reads the `MUL_MAT_ID` routing ids): ~90% of the model on
   host through a 0.5 GiB budget, **byte-identical**, ~7.4% streamed/token (13.5×
   less than dense), hit-rate 36→70% with budget. Overlapping the scattered
   expert-slice transfers then recovered **+26–39% decode at tight budgets**
   (512 MiB 2.89→4.01 tok/s). **Remaining floor (honest):** ~510–590 MiB of active
   experts cross PCIe per token → ~20 tok/s hard ceiling for this 14 B model, set by
   the ~5.7 GiB/s PCIe uplink itself (not pager software — see 3a). Detail in
   `examples/llamacpp/README_MOE.md`.

3a. ✅ **Faster pager staging** (surfaced by #2). *Investigated and resolved —
    honest negative.* Hypothesis was that page-granular staging left the wire idle
    (a ~4 GiB/s symptom). Added a concurrent transfer-stream pool
    (`PagerConfig::transfer_streams`, `bench --transfer-streams`) so demand loads and
    prefetch workers issue H2D on distinct streams instead of serializing on one.
    **Measured: no win on this A2/VM** — a *single* async stream already saturates
    the ~5.7 GiB/s uplink once prefetch overlaps the source→pinned copy (steady state
    ~5.5–5.7 GiB/s at 1, 2, or 4 streams); the ~4 GiB/s was a cold first-launch
    outlier, not a steady-state ceiling. Multiple same-direction H2D copies share one
    uplink and can't exceed it. The knob is correct and zero-cost at its default (1)
    and would pay off only where a single stream can't fill the link (multi-rail /
    higher-bandwidth). **Takeaway: the bottleneck is the physical link, not staging
    software → see #10.**
3. 🔴 **Compression-on-wire (nvCOMP). RESOLVED — honest negative** (see
   `docs/COMPRESSION.md`). The premise (weights compress ~2×) does not hold. Measured
   on real tensor bytes: **Q4_K_M is incompressible (~1.02×)** — already near entropy,
   so #3 is dead for the quantized production models — and **BF16 tops out at
   ~1.23–1.28×**, below the 1.3× go/no-go threshold and far from 2×. Worse, on the A2
   the nvCOMP GPU decompressor (GDeflate ~3.4 GB/s) runs *slower* than the ~5.7 GiB/s
   PCIe link, so decompression would just become the new bottleneck: every codec is a
   wash or a net loss (effective `min(link×ratio, decode)` < baseline). Mechanism is
   sound (nvCOMP decodes byte-exact; `MemorySource` already abstracts the tier) but
   the economics aren't. Realistic ceiling ≈ **1.25×, fp16/bf16 only, and only on a
   GPU whose decode rate ≫ PCIe** — strictly worse and narrower than #10's free ~2×.
   Probe: `benchmarks/compression/`. **Not worth pursuing for quantized VRAMER.**

## Tier 2 — credibility and reach

4. **A proper `ggml_backend_buffer_type` + scheduler hook** to replace the env-flag
   adapter — usable without patching llama.cpp.
5. 🟢 **Honest comparative benchmarks.** *Done for llama.cpp.* `docs/BENCHMARKS.md`
   compares VRAMER vs `llama.cpp -ngl` partial offload vs native `-ot=CPU` offload vs
   resident, on dense (Qwen2.5-0.5B) and MoE (Qwen1.5-MoE-A2.7B), same hardware/run.
   Honest finding: VRAMER's wins are **correctness** (byte-identical through a
   sub-set budget), an **explicit tunable VRAM ceiling**, and **MoE selectivity**
   (7.5% streamed) — **not** single-stream decode tok/s, where it ties/loses the PCIe
   wall vs native offload (expected). The **batched-serving** regime is now measured
   too (`docs/BENCHMARKS.md`, "Batched serving"): per-step weight-load amortization is
   real — aggregate tok/s scales with batch size B and bytes/token fall (dense tracks
   1/B near-ideally; MoE sub-linearly as selectivity erodes). Honest verdict:
   batching *partially* rescues the case but never overtakes native CPU-offload on
   this VM, because native amortizes the same selective copy and moves ~1.7× fewer
   bytes. The **too-big-to-fit regime** is now measured too (`docs/BENCHMARKS.md`,
   "Too big to fit"; `examples/llamacpp/README_BIGMODEL.md`): Qwen2.5-32B Q4_K_M
   (18.5 GiB) on a 15 GiB A2 — *full* offload OOMs, but native partial offload still
   runs it, so "impossible without VRAMER" is **false**. The honest win is narrower
   and real: VRAMER is the only way to run this over-card model at a peak VRAM **you
   name** (e.g. 8 GiB), byte-identical, and by caching hot FFN pages (72–83% hit) it
   **beats blind native CPU-offload by ~30%** (0.43 vs 0.33 tok/s) *at a fixed
   sub-card VRAM cap*. It still loses raw throughput to the only-fitting `-ngl 49`
   (~1 tok/s, which pins ~100% of the card). *Remaining:* DeepSpeed ZeRO-Infinity /
   FSDP comparison; a model too big for host **RAM** (→ #6, GPUDirect/NVMe).
6. **GPUDirect Storage (cuFile) / `DevDaxSource`** — stream straight from NVMe
   (models larger than *RAM*) and the CXL/PMEM tier story. `MemorySource` already
   abstracts the tier.

## Tier 3 — research bets

7. **QLoRA fine-tuning demo** — frozen 4-bit base is read-only and layer-ordered.
8. **vLLM KV-connector experiment** (`OffloadingConnector` / `KVConnectorBase_V1`).
9. **Auto-NUMA detection**; **TensorRT-LLM adapter**.
10. **Hardware investigation** *(now the clearest bandwidth lever — both software
    levers are exhausted).* 3a (more transfer streams) and #3 (compression-on-wire)
    both dead-ended at the same ~5.7 GiB/s wall: one stream already saturates it, and
    quantized weights don't compress. So the ceiling is the physical link, not the
    pager. The A2 rig measured ~5.7 vs nominal ~12 GiB/s H2D — likely VM/QEMU
    passthrough; bare metal (or PCIe ACS/relaxed-ordering tuning) may give a near-free
    ~2× to **every** paged workload at once, with no decode cost. Highest leverage left.

See the [project wiki](https://github.com/krypt0byt3/vramer/wiki) for the full
reasoning behind each item.
