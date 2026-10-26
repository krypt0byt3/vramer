# Roadmap

Every item is judged by one question: **does it either raise effective PCIe
bandwidth, or target a workload where you don't re-stream everything every token?**
Paging an oversized dense model for single-stream decode is PCIe-bound — see
[`docs/TUNING.md`](docs/TUNING.md). Everything below is scoped against that reality.

## Tier 1 — biggest leverage

1. **Prefetch/compute overlap (double-buffering) in the llama.cpp adapter.** The
   gather is currently a blocking sync — H2D and compute are serialized. Overlapping
   next-layer H2D with the current layer's matmul hides transfer latency behind
   compute already being paid for. Directly improves the measured ~9.3 tok/s paged
   decode; changes no physics-level limit. **In progress.**
2. **A real MoE demo (Mixtral-8x7B / Qwen-MoE).** Only active experts stream per
   token (`moe_active_fraction` ≪ 1), so the locality wall mostly doesn't apply —
   the differentiated story. The planner already models this.
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
