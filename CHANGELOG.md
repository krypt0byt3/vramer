# Changelog

All notable changes to VRAMER are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[Semantic Versioning](https://semver.org/). VRAMER is pre-1.0: APIs may change.

## [Unreleased]

### Added
- **Larger-than-RAM disk tier** (`docs/LARGER_THAN_RAM.md`, "Too big to fit — in
  host RAM" in `docs/BENCHMARKS.md`, `benchmarks/larger_than_ram/`, raw logs under
  `benchmarks/raw/ltr_*.log`, new ctest `file_source`) — pages a **120 GiB**
  backing file (larger than this box's 78 GiB RAM, so unloadable natively) through
  a 4 GiB VRAM budget via the existing `MmapFileSource`, byte-verified (checksum
  matches across every page-size/prefetch config; `file_source` does full
  every-byte verification incl. reload-after-eviction). Honest finding: the
  disk → page cache → pinned staging → GPU path is **disk-bound at SATA-SSD speed
  (~0.4 GiB/s cold)**, ~14× slower than the same store served warm from the page
  cache (~5.5 GiB/s), and the pager extracts ~96% of the device's cold read
  bandwidth — so the win is **correctness + a bounded footprint for a >RAM working
  set**, not throughput. Numbers are a floor from a shared, contended SATA SSD;
  GPUDirect Storage (cuFile) is the documented next step (not installed; a QEMU VM
  wouldn't do true P2P DMA). No core change — reuses `MmapFileSource` and
  `bench --source file`.
- **Too-big-to-fit demo** (`examples/llamacpp/run_demo_bigmodel.sh`,
  `examples/llamacpp/README_BIGMODEL.md`, "Too big to fit" in `docs/BENCHMARKS.md`,
  raw logs under `benchmarks/raw/bigmodel/`) — runs **Qwen2.5-32B Q4_K_M (18.5 GiB)
  on a single 15 GiB A2**. Full offload (`-ngl 99`) OOMs; VRAMER pages the 15.7 GiB
  FFN set through a budget you name (4 or 8 GiB), byte-identical to native offload.
  Honest finding: native partial offload *also* runs it (so not "impossible without
  VRAMER"), but VRAMER is the only way to run it at a hard, tunable sub-card VRAM
  ceiling, and by caching hot pages (72–83% hit) it beats blind CPU-offload by ~30%
  (0.43 vs 0.33 tok/s) at a fixed cap.
- **Compression-on-wire evaluation** (`docs/COMPRESSION.md`,
  `benchmarks/compression/`) — measured, not built. Roadmap #3 resolved as an honest
  negative: Q4_K_M weights are incompressible (~1.02×) and BF16 tops out ~1.25×
  (below the 1.3× threshold), and on the A2 the GPU decompressor is slower than the
  ~5.7 GiB/s PCIe link, so every codec is a wash or net loss. No core change.
- **Concurrent transfer-stream pool** (`PagerConfig::transfer_streams`,
  `bench --transfer-streams N`) — the pager round-robins H2D (and D2H writeback)
  across a pool of non-blocking CUDA streams so demand loads and prefetch workers
  overlap on the wire instead of serializing on one stream. Default `1` preserves
  exact prior behavior. Each copy records its slot `load_done` and the staging
  `h2d_done` on the stream that issued it; `synchronize()`/destructor drain all
  streams. `prefetch_overlap` extended to byte-verify cyclic scans under 2 and 4
  concurrent streams. **Measured no win on the A2/VM rig** — one async stream already
  saturates the ~5.7 GiB/s uplink; the knob stands to help only where a single stream
  can't fill the link (multi-rail / higher-bandwidth). See `ROADMAP.md` item 3a.
- **Batched-serving benchmark** (`benchmarks/run_batched.sh`, raw logs under
  `benchmarks/raw/batched/`, new "Batched serving" section in `docs/BENCHMARKS.md`) —
  sweeps batch size B via `llama-batched-bench` and reports aggregate decode tok/s and
  H2D bytes/generated-token for resident / native CPU-offload / VRAMER. Finding:
  per-step weight-load amortization is real (bytes/token fall with B; dense ≈ 1/B,
  MoE sub-linearly as selectivity erodes), so batching *partially* rescues the
  throughput case — but VRAMER never overtakes native offload on this VM, which
  amortizes the same selective copy and moves ~1.7× fewer bytes. Output stays
  byte-identical to resident across the batch.
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
- **`docs/BENCHMARKS.md`** — an honest, reproducible comparison (`benchmarks/run_comparison.sh`
  + raw logs): VRAMER vs `llama.cpp -ngl` vs native `-ot=CPU` offload vs resident, on a
  dense and a MoE model, same hardware/run. Finding: VRAMER wins on correctness
  (byte-identical), a tunable VRAM ceiling, and MoE selectivity (7.5% streamed) — not
  single-stream decode tok/s, which ties/loses the PCIe wall vs native offload.
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
