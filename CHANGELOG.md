# Changelog

All notable changes to VRAMER are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[Semantic Versioning](https://semver.org/). VRAMER is pre-1.0: APIs may change.

## [Unreleased]

### Added
- **`prefetch_overlap` test** — GPU regression coverage for the prefetch/compute
  overlap (double-buffering) path: a cyclic scan through a cache smaller than the
  working set, byte-verifying every acquired page (including prefetched, evicted,
  and reloaded pages) and asserting prefetch-ahead raises the demand hit rate.
- **`bench --sync-per-page`** — models a naive blocking-consumer (the "blocking
  gather sync" pathology), so the overlap win is measurable against a serialized
  baseline (~2× on an 8 GiB/4 GiB RAM scan: 2.6 → 5.7 GiB/s).
- **`ROADMAP.md`** — prioritized enhancement plan (Tier 1–3) and a project wiki.
- **MoE expert-paging demo** (`examples/llamacpp/README_MOE.md`,
  `adapter/run_demo_moe.sh`) — the llama.cpp hook is now expert-aware: for a 3-D
  expert stack it reads the token's `MUL_MAT_ID` routing ids and pages **only the
  active experts' slices** (2-D weights still page whole, so the dense demo is
  unchanged). On Qwen1.5-MoE-A2.7B: ~90% of the model paged from host through a
  0.5 GiB budget, byte-identical output, ~7.4% of the expert set streamed per token
  (13.5× less than dense-equivalent), hit-rate 36→70% with budget. Added a
  `VRAMER_EVICT` knob (LRU suits MoE's popular-expert reuse).
- **MoE expert-transfer overlap** — the hook now queues the (usually scattered)
  active-expert slice transfers back-to-back on the paging stream and blocks once,
  instead of a blocking sync per run. Byte-identical; **+26–39% decode at tight
  budgets** (512 MiB 2.89→4.01 tok/s; 2048 MiB 3.85→4.86), a wash once the hit-rate
  is high. Same `bytes_h2d` — same traffic, less wall time.

### Notes
- Measured honestly: prefetch-ahead converts demand misses to hits (0 → ~99%) but
  for an already-pipelined consumer the RAM/pinned scan is PCIe-bandwidth-bound, so
  it does not beat the wire; the wall-clock payoff appears against a blocking consumer.
- MoE paging proves *selectivity + correctness* and now part of *speed*: overlapping
  the expert-slice transfers recovers +26–39% at tight budgets. A hard floor remains
  — ~510–590 MiB of active experts cross PCIe per token, so this 14 B model is capped
  near ~20 tok/s vs 62.9 resident; and the pager's page-granular staging currently
  achieves only ~4 GiB/s of the ~12 GiB/s wall (multi-worker async staging is the
  next lever). See `examples/llamacpp/README_MOE.md`.

## [0.2.0] — 2026-10-25

First public release. Builds on the original v1.0 pager with a large set of
performance, correctness, integration, and adoption features.

### Added
- **Scan-aware MRU eviction** (`EvictionPolicy::Mru`, `--eviction mru`) — fixes the
  LRU 0%-reuse collapse on cyclic layer scans (hit rate ≈ resident fraction).
- **Lock-free source reads** — `MemorySource::read` + async H2D run outside the
  cache lock, with coalesced same-page loads; gives a 1-deep load/compute pipeline.
- **Pinned direct-DMA fast path** — `PinnedHostSource` / pinned `BorrowedMemorySource`
  are DMA'd straight to the GPU via `MemorySource::pinned_host_ptr`, skipping staging.
- **Async prefetch worker pool** (`prefetch_async`) with a capacity-bounded,
  dedup'd queue that never thrashes the cache.
- **Write-back** for mutable pages (`acquire_mutable` + `flush()` to a
  `WritableMemorySource`; flushed on eviction).
- **Multi-GPU** independent caches (`bench --gpus 0,1`).
- **NUMA-aware source placement** (`--numa-node N|auto`, `mbind`, no libnuma dep).
- **Telemetry**: demand hit rate, prefetch/direct-load counts, source-read &
  stall time, H2D/D2H bytes, sync-fallback warning.
- **`vramer probe`** — raw pageable/pinned H2D/D2H bandwidth per GPU + hardware-profile export.
- **`vramer plan`** streaming-cost estimator (tokens/s ceiling, "fits / PCIe-bound" verdict).
- **C ABI** (`vramer_c.h`) + shared `libvramer.so` for Python/Rust bindings.
- **CMake install + package config** (`find_package(vramer)`), pkg-config.
- Expanded **profiles system**: richer hardware/model schemas, a detector
  (PCIe/NVLink/NUMA), and a planner with quant × batch matrices + PCIe-gen sweeps.
- **Verified ggml-cuda integration** and a **working end-to-end llama.cpp
  weight-paging adapter** (byte-identical generation with weights paged through a
  sub-model budget); QLoRA adapter design + skeleton.
- CLI ergonomics: `--version`, `--help`, friendlier errors.
- Docs: `docs/INTEGRATION.md`, `docs/LLAMACPP_ADAPTER.md`, `docs/QLORA_ADAPTER.md`,
  `docs/TUNING.md`; examples and a `scripts/smoke.sh`.

### Changed
- Version set to 0.2.0 to reflect pre-1.0 maturity (engine adapters are proven
  mechanisms / proofs of concept, not yet production-optimized).

### Measured (2× NVIDIA A2, SM86)
- 64 MiB page-size sweet spot; MRU restores reuse lost by LRU on sequential scans;
  this setup is PCIe-link-bound (~5.5–5.7 GiB/s H2D), so pinning gains little here.

## [1.0.0] — initial
- Fixed-budget LRU device cache, `MemorySource` abstraction (mmap / anonymous RAM /
  borrowed), pinned staging ring, async H2D on a dedicated stream, page leases,
  `detect`/`plan`/`bench` CLI. No Unified Memory dependency.
