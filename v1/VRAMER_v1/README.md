# VRAMER v1.0

**VRAMER** is an explicit host-memory ↔ GPU-VRAM pager for model/tensor data.

The v1.0 goal is intentionally narrow: prove the memory hierarchy and give later
TensorRT-LLM / llama.cpp / custom-kernel adapters a safe paging primitive.

## Architectural invariant: VRAMER is not DDR4-specific

VRAMER never identifies the host tier as DDR4. The core depends only on the
`MemorySource` interface:

```cpp
class MemorySource {
public:
    virtual uint64_t size() const noexcept = 0;
    virtual std::string name() const = 0;
    virtual void read(uint64_t offset, void* dst, size_t bytes) = 0;
};
```

Today v1.0 ships three sources:

- `MmapFileSource` — any Linux mmap-able file/device
- `AnonymousRamSource` — ordinary host memory from the OS
- `BorrowedMemorySource` — memory owned by another runtime

That means the algorithm is independent of DDR generation. DDR3, DDR4, DDR5,
LPDDR, NUMA RAM, PMEM, and OS-exposed CXL memory can all sit behind a source.
Only bandwidth/latency/topology change.

A small **pinned staging ring** is allocated with CUDA because asynchronous H2D
copies require page-locked host memory. NVIDIA explicitly recommends pinned
memory be used sparingly as a staging area rather than pinning enormous model
allocations. VRAMER therefore does **not** pin the entire backing model.

VRAMER v1.0 also deliberately does **not** use `cudaMallocManaged()`. Residency
is explicit and measurable.

## Data path

```text
MemorySource
(DDR3/4/5, CXL, mmap file, etc.)
       |
       | CPU memcpy/read
       v
small pinned staging ring
       |
       | cudaMemcpyAsync H2D
       v
fixed-budget GPU cache
       |
       v
inference/compute stream
```

The GPU cache is fixed-size and LRU-evicted. A `PageLease` prevents an in-use
page from being evicted. CUDA events make transfers and compute ordered without
a device-wide synchronization on the normal path.

## What v1.0 already does

- fixed VRAM budget (`cudaMalloc`), independent of model size
- page-granular residency and LRU eviction
- host-side `MemorySource` abstraction
- mmap-backed model/file source
- real anonymous-RAM source for host-memory experiments
- 2+ pinned staging buffers
- asynchronous H2D copies on a dedicated CUDA stream
- compute-stream/event synchronization
- leases to prevent eviction while GPU work is in flight
- optional background **prefetch worker pool** (`prefetch_async`) with a
  capacity-bounded queue and resident-page dedup (never thrashes the cache)
- **telemetry**: demand hit rate, prefetch count, source-read time, stall time,
  H2D bytes/rate, and a one-time warning if `release_on()` is omitted
- `detect`, `plan`, and `bench` CLI commands
- PCI BDF + NUMA-node reporting
- no Unified Memory / `cudaMallocManaged()` dependency

See [`docs/INTEGRATION.md`](docs/INTEGRATION.md) for how VRAMER plugs into
inference engines (vLLM's KV connector API, llama.cpp weight paging).

## What v1.0 is NOT yet

It is not yet a transparent TensorRT-LLM or llama.cpp weight provider. Those
adapters require interception at each framework's tensor/layer ownership
boundary. v1.0 is the pager those adapters will call.

It also does not pretend NVMe swap is GPU RAM. `MmapFileSource` is useful for
out-of-core experiments, but repeated faults from storage during every token
will be much slower than keeping the active model set in host RAM.

## Requirements

- Linux
- CMake 3.24+
- C++17 compiler
- NVIDIA CUDA Toolkit
- NVIDIA GPU

For RTX 50-series / SM120, use **CUDA 12.8 or newer**; CUDA 12.8 is the first
toolkit with SM120 compiler support.

## Build

### Native build (recommended)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CMake defaults to the GPU present on the build machine (`native`).

### Explicit architecture

NVIDIA A2 / SM86:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

RTX 5060 / SM120 (CUDA >= 12.8):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
```

## Use

Detect hardware/topology:

```bash
./build/vramer detect
```

Plan residency:

```bash
./build/vramer plan \
  --model-bytes 12GiB \
  --vram-budget 6GiB \
  --page-size 64MiB
```

Exercise 12 GiB of real host RAM through a 6 GiB GPU cache:

```bash
./build/vramer bench \
  --gpu 0 \
  --source ram \
  --bytes 12GiB \
  --vram-budget 6GiB \
  --page-size 64MiB \
  --staging 2 \
  --passes 2
```

Exercise a model/file through the mmap source:

```bash
./build/vramer bench \
  --gpu 0 \
  --source file \
  --file /models/model.gguf \
  --vram-budget 6GiB \
  --page-size 64MiB
```

On a NUMA server such as the R730xd, bind the process to the CPU/memory node
closest to the GPU when testing host-RAM bandwidth, for example:

```bash
numactl --cpunodebind=0 --membind=0 ./build/vramer bench ...
```

`vramer detect` reports the GPU's Linux NUMA node so this can be automated in a
later release.

## Recommended first experiments

Before the RTX 5060 arrives, run v1.0 on the A2s to validate the pager and find
an effective page size. Start with 16, 32, 64, 128, and 256 MiB pages.

When the RTX 5060 arrives, build for `sm_120` and deliberately sweep the resident
fraction, e.g. 100%, 80%, 60%, 40%, to measure the PCIe spill penalty.

## Measured on NVIDIA A2 (SM86, PCIe Gen4)

A residency/page-size sweep (`scripts/resident_sweep.sh`, raw data in
`benchmarks/a2-residency-sweep.txt`) surfaced three load-bearing facts:

1. **Generic LRU gives 0% reuse on a sequential full scan.** At any resident
   fraction below 100%, multi-pass demand hit rate is 0% — a scan larger than
   the cache evicts page 0 right before the next pass needs it (the classic
   LRU/Belady sequential-scan pathology). For token-by-token layer streaming,
   LRU is the *wrong* policy once the model exceeds the VRAM budget. A
   scan-aware / layer-order eviction policy is therefore a correctness-shaped
   win, not a micro-optimization.
2. **The CPU source→pinned copy dominates and is serialized by the cache
   mutex.** Throughput is flat (~5.6 GiB/s) across resident fractions and the
   `src read` counter is ~85–90% of per-page time. Moving the source read out
   from under the lock (so it overlaps H2D/compute) is the top throughput lever
   — and the reason the current prefetch worker reaches 98% hit rate but no
   speedup on single-pass streaming.
3. **64 MiB is the page-size sweet spot on A2** (16 MiB loses ~10% to
   per-transfer overhead; 128/256 MiB add nothing).

## v1.1 targets (reordered by measured impact)

1. **Lock-free source reads** — reserve slot + private staging under the lock,
   run `MemorySource::read` and issue the H2D *outside* it, coalescing
   concurrent/duplicate loads of the same page via an in-progress map. This is
   what makes the prefetch worker actually overlap. (Correctness hazard to
   respect: a coalescing waiter must block on a `load_done` event that is
   genuinely *not-ready* from reservation until the real `cudaMemcpyAsync`
   records it — do not reuse the pre-completed init event.)
2. **Scan-aware / predictive layer-order eviction** instead of generic LRU.
3. multi-GPU independent caches and transfer streams
4. topology-aware automatic NUMA binding / source allocation
5. a C ABI so Rust/Python engines can bind the pager (see `docs/INTEGRATION.md`)
6. llama.cpp weight-paging adapter, then a vLLM KV-connector experiment
7. TensorRT-LLM adapter
8. optional CXL/PMEM-specific `MemorySource` implementations where the OS/API
   exposes useful placement controls

