# Changelog

All notable changes to VRAMER are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[Semantic Versioning](https://semver.org/). VRAMER is pre-1.0: APIs may change.

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
