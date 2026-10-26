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
   experts cross PCIe per token → ~20 tok/s hard ceiling for this 14 B model; and
   the pager's page-granular staging currently hits only ~4 of ~12 GiB/s. *Next
   lever:* multi-worker async staging + expert-sized pages (item 3a). Detail in
   `examples/llamacpp/README_MOE.md`.

3a. **Faster pager staging** (surfaced by #2). The page-granular path
    (acquire→wait_on→D2D per page, single prefetch worker) achieves only ~4 GiB/s of
    the A2's ~12 GiB/s wall. Multi-worker async staging and a page size matched to
    the workload's natural unit (the ~1.8 MiB expert slice) should reclaim much of
    the gap toward the wire — the highest-leverage *core* follow-up.
3. **Compression-on-wire (nvCOMP).** Being PCIe-bound, compressing weights ~2×
   before H2D and decompressing on-GPU is a near-direct 2× to effective bandwidth.

## Tier 2 — credibility and reach

4. **A proper `ggml_backend_buffer_type` + scheduler hook** to replace the env-flag
   adapter — usable without patching llama.cpp.
5. 🟢 **Honest comparative benchmarks.** *Done for llama.cpp.* `docs/BENCHMARKS.md`
   compares VRAMER vs `llama.cpp -ngl` partial offload vs native `-ot=CPU` offload vs
   resident, on dense (Qwen2.5-0.5B) and MoE (Qwen1.5-MoE-A2.7B), same hardware/run.
   Honest finding: VRAMER's wins are **correctness** (byte-identical through a
   sub-set budget), an **explicit tunable VRAM ceiling**, and **MoE selectivity**
   (7.5% streamed) — **not** single-stream decode tok/s, where it ties/loses the PCIe
   wall vs native offload (expected). *Remaining:* DeepSpeed ZeRO-Infinity / FSDP
   comparison, and the batched-serving regime where the throughput case actually lives.
6. **GPUDirect Storage (cuFile) / `DevDaxSource`** — stream straight from NVMe
   (models larger than *RAM*) and the CXL/PMEM tier story. `MemorySource` already
   abstracts the tier.

## Tier 3 — research bets

7. **QLoRA fine-tuning demo** — frozen 4-bit base is read-only and layer-ordered.
8. **vLLM KV-connector experiment** (`OffloadingConnector` / `KVConnectorBase_V1`).
9. **Auto-NUMA detection**; **TensorRT-LLM adapter**.
10. **Hardware investigation** — the A2 rig measured ~5.7 vs nominal ~12 GiB/s H2D,
    likely VM passthrough; bare metal may give a free ~2×.

See the [project wiki](https://github.com/krypt0byt3/vramer/wiki) for the full
reasoning behind each item.
