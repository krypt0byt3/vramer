# Integrating VRAMER with inference engines

VRAMER is a **paging primitive, not an engine.** It owns one thing: a fixed
VRAM budget that it fills, on demand, from a byte-addressable `MemorySource`
(host RAM, mmap'd file, CXL/PMEM, or memory borrowed from another runtime).
An inference engine keeps owning the model graph, the scheduler, the kernels,
and the KV cache. VRAMER just answers the question *"is this chunk of bytes
resident in VRAM right now, and if not, get it there."*

That means integration is always at a **memory-ownership boundary**, never a
rewrite. There are two such boundaries in a modern LLM engine.

## Boundary A — weight / tensor residency (VRAMER's primary target)

When a model doesn't fit in VRAM, engines today either refuse to load it or do
a coarse CPU offload (vLLM's `--cpu-offload-gb`, which parks whole layers in
host RAM and copies them in per step). VRAMER replaces that with page-granular,
LRU/scan-aware residency over an explicit budget.

The hook is `BorrowedMemorySource`: wrap the engine's host-resident weight
buffer (e.g. a pinned PyTorch CPU tensor, or an mmap of the checkpoint) and let
VRAMER page it:

```cpp
// Engine owns the full weights in host memory (any tier).
auto src = std::make_shared<vramer::BorrowedMemorySource>(
    host_weight_base, host_weight_bytes, "qwen2.5-14b-w4a16");

vramer::PagerConfig cfg;
cfg.vram_budget      = 6ull << 30;   // fixed, independent of model size
cfg.page_size        = 64ull << 20;  // 64 MiB sweet spot on A2 (see README)
cfg.prefetch_workers = 1;            // overlap next layer's load with this one
vramer::CudaPager pager(src, cfg);

// Per layer, on the compute stream:
auto lease = pager.acquire(layer_page);
lease.wait_on(compute_stream);        // ordering, not a device sync
launch_gemm(lease.device_ptr(), ...); // engine kernel reads resident bytes
lease.release_on(compute_stream);     // frees for eviction after this layer
pager.prefetch_async(next_layer_page);// pipeline the next layer
```

Because the eviction order for inference is *known* (layers are scanned in
order every token), this is exactly where generic LRU fails — see the A2
measurements in the README — and where VRAMER's planned scan-aware eviction
plus `prefetch_async` pay off. **llama.cpp is the most tractable first adapter:**
its tensors already carry a `data` pointer and an mmap backing (`MmapFileSource`
maps the gguf directly), so interception is a localized change at the ggml
tensor-load path rather than a scheduler change.

## Boundary B — KV-cache block backing (vLLM's connector API)

vLLM already offloads KV cache well in-tree: the **OffloadingConnector**
(`KVConnectorBase_V1`) stages blocks through pinned host memory over a DMA
engine, routes all KV through host DRAM as the primary tier, and LRU-evicts on
the CPU side. Out-of-tree backends (LMCache, Mooncake, NVIDIA KVBM) plug in
through that same connector interface.

So for KV cache, VRAMER should **be a connector, not compete with one.** The
connector interface is the seam: implement `KVConnectorBase_V1` such that
VRAMER's `CudaPager` owns the device-side block residency and a `MemorySource`
owns the host/secondary tier. VRAMER's differentiation here is *not* "do KV
offload" (vLLM does that) but **what sits behind the host tier**: a
`MemorySource` can be CXL-attached memory, PMEM, or an mmap'd file, which the
built-in connector does not model. That is the honest niche — exotic memory
tiers and out-of-core, exposed through one uniform byte interface.

## The FFI reality

Engines are Python (vLLM, TRT-LLM) or C++ (llama.cpp); VRAMER is C++/CUDA.
A thin **C ABI** (`vramer_pager_create / acquire / release / prefetch / stats`)
is the right boundary — it lets vLLM bind via ctypes/pybind and lets a future
Rust engine (candle / mistral.rs) bind via `cudarc` + FFI without a rewrite.
This is v1.1 target #5. Keep the GPU pointer (`lease.device_ptr()`) as the
currency crossing the boundary; never copy tensors across the FFI line.

## Recommended sequencing

1. Land lock-free source reads + scan-aware eviction (README v1.1 #1–2) so the
   primitive is actually fast and correct for the inference access pattern.
2. Expose the C ABI (#5).
3. **llama.cpp weight-paging adapter** (Boundary A) — simplest ownership model,
   immediate "run a model bigger than VRAM" demo.
4. **vLLM KV-connector experiment** (Boundary B) — only to showcase a
   CXL/PMEM/file-backed host tier, which is the part vLLM doesn't already do.

## What VRAMER should *not* become

Do not grow kernels, attention, batching, or sampling into VRAMER. Those are
solved extremely well by vLLM / TRT-LLM / llama.cpp, and building them would
bury VRAMER's one defensible contribution: an explicit, measurable,
memory-tier-agnostic residency pager that any of those engines can call.
