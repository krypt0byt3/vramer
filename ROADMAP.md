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
2. 🔄 **A real MoE demo (Qwen-MoE / Mixtral-8x7B).** *In progress.* Only active
   experts stream per token (`moe_active_fraction` ≪ 1), so the locality wall mostly
   doesn't apply — the differentiated story. The planner already models this.
3. **Compression-on-wire (nvCOMP).** Being PCIe-bound, compressing weights ~2×
   before H2D and decompressing on-GPU is a near-direct 2× to effective bandwidth.

## Tier 2 — credibility and reach

4. **A proper `ggml_backend_buffer_type` + scheduler hook** to replace the env-flag
   adapter — usable without patching llama.cpp.
5. **Honest comparative benchmarks** vs `llama.cpp -ngl`, DeepSpeed ZeRO-Infinity,
   and FSDP CPU-offload on identical hardware.
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
