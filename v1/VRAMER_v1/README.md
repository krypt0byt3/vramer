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
- **lock-free source reads**: the `MemorySource::read` and async H2D run outside
  the cache lock (slot + staging reserved first), with concurrent same-page loads
  coalesced — so the demand thread pipelines the next page's load under the
  current page's compute, and prefetch workers genuinely overlap
- **scan-aware eviction** (`EvictionPolicy::Mru`): MRU retains a stable resident
  prefix on a cyclic scan where LRU collapses to 0% reuse (see measurements)
- optional background **prefetch worker pool** (`prefetch_async`) with a
  capacity-bounded queue and resident-page dedup (never thrashes the cache)
- **telemetry**: demand hit rate, prefetch count, source-read time, stall time,
  H2D bytes/rate, and a one-time warning if `release_on()` is omitted
- **multi-GPU independent caches** — one cache/stream/staging ring per GPU,
  driven concurrently (`bench --gpus 0,1`); ~2× aggregate H2D on 2× A2
- **write-back** of mutable pages (`acquire_mutable` + `flush()`): dirty device
  pages are flushed D2H to a `WritableMemorySource` on eviction or flush
- **NUMA-aware source placement** (`--numa-node N` / `AnonymousRamSource`): binds
  host pages to the GPU-local node before first-touch (no-op on non-NUMA systems)
- **pinned direct-DMA fast path**: a `PinnedHostSource` (or a `pinned` borrowed
  source) is DMA'd straight to the GPU, skipping the staging copy — the pager
  takes it automatically via `MemorySource::pinned_host_ptr`
- **`vramer probe`**: measures raw pageable-vs-pinned H2D/D2H bandwidth per GPU
  and emits a hardware profile the Python planner consumes
- a **C ABI** (`vramer_c.h`) + a **shared `libvramer.so`** for Python/Rust
  bindings without a C++ toolchain, plus reference code in `examples/` (llama.cpp
  PoC, QLoRA skeleton, and a runnable NumPy+ctypes paging demo)
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

The build produces `libvramer.a`, `libvramer.so` (for ctypes/FFI), and the
`vramer` CLI.

## Install / use as a library

```bash
cmake --install build --prefix /usr/local   # headers, libs, CMake pkg, pkg-config
```

Consume from CMake:

```cmake
find_package(vramer REQUIRED)
target_link_libraries(myapp PRIVATE vramer::vramer)        # or vramer::vramer_shared
```

or from pkg-config / plain compilers:

```bash
pkg-config --cflags --libs vramer
```

Python/Rust bind the C ABI (`vramer_c.h`) against `libvramer.so` — see
`examples/python_paging_demo.py`.

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

1. **Generic LRU gives 0% reuse on a sequential full scan** (now fixed by MRU).
   Under LRU, at any resident fraction below 100%, multi-pass demand hit rate is
   0% — a scan larger than the cache evicts page 0 right before the next pass
   needs it (the classic LRU/Belady pathology). `EvictionPolicy::Mru` retains a
   stable resident prefix instead: measured MRU hit rate tracks ~2/3 × resident
   fraction (the cyclic-scan optimum given a cold first pass) — e.g. at 75%
   resident, 0% → 49% hit rate and 4.38 s → 2.35 s. Raw data:
   `benchmarks/a2-residency-sweep-v1.1.txt`.
2. **The CPU source→pinned copy dominated and was serialized by the cache
   mutex** (now fixed by lock-free reads). The `src read` counter was ~85–90% of
   per-page time. The read + async H2D now run outside the lock, giving the
   demand thread a 1-deep load/compute pipeline; single-pass streaming improved
   (~3.09 s → ~2.90 s on 16 GiB/4 GiB) and the concurrency stress test sped up
   accordingly (threads no longer serialize on the read).
3. **64 MiB is the page-size sweet spot on A2** (16 MiB loses ~10% to
   per-transfer overhead; 128/256 MiB add nothing).
4. **This A2 setup is PCIe-bound, not copy-bound.** `vramer probe` reports raw
   H2D of ~5.5 GiB/s pageable and only ~5.7 GiB/s *pinned* — so the pinned
   direct-DMA path (which removes the CPU source copy entirely, `src read` → 0)
   buys only ~3% here: the link itself is the ceiling. On hardware where the
   source copy dominates, the same path is a much larger win. Always `probe`
   before assuming where the bottleneck is.

Note on prefetch workers: with lock-free reads the single demand thread already
hides the source read behind the current page's H2D/compute, so explicit
prefetch workers add little for *single-stream sequential* access (they still
reach ~99% hit rate). Their expected payoff is predictable *non-sequential*
access and multi-GPU, where one demand thread cannot pipeline far enough ahead.

## v1.1 targets

Done in this release:
- lock-free source reads (coalescing hazard handled — waiters block on
  `load_cv_` until the real H2D records `load_done`, never on a stale event);
- scan-aware MRU eviction;
- multi-GPU independent caches — each GPU gets its own cache, transfer stream,
  and staging ring; `vramer bench --gpus 0,1` drives them concurrently. Measured
  on 2× A2: ~5.5 GiB/s each → **10.9 GiB/s aggregate** (clean ~2× scaling, no
  host-bandwidth contention);
- a **C ABI** (`include/vramer/vramer_c.h`) so Python (ctypes) and Rust (FFI /
  `cudarc`) can drive the pager without C++ — opaque handles, error-code
  convention, no exceptions across the boundary, CUDA streams passed as `void*`;
- **NUMA source binding** (`--numa-node N`, `mbind`-based, no libnuma dependency);
- **write-back** for mutable pages (`acquire_mutable` + `flush()`), validated by
  `tests/test_writeback.cu`;
- **verified ggml-cuda integration**: `examples/llamacpp/` runs real
  `ggml_mul_mat` on the CUDA backend against VRAMER-paged GGUF weights — a
  138 MiB weight through a half-size budget matched a CPU reference to 1.5e-8,
  and a whole transformer block cycled through a sub-layer budget at 77.8% MRU
  reuse. Plus the C-ABI PoC (paged a real Qwen2.5-14B shard at 98.3% hit rate),
  a runnable Python demo, and design docs `docs/LLAMACPP_ADAPTER.md` +
  `docs/QLORA_ADAPTER.md`.

Remaining (the full engine adapters, which need their external trees/hardware):

1. a *full* llama.cpp adapter: a custom `ggml_backend_buffer_type` + per-layer
   scheduler hook inside llama.cpp. The mechanism is already verified in
   `examples/llamacpp/` — a real `ggml_mul_mat` consumes a 138 MiB weight paged
   through a half-size budget (matched a CPU reference to 1.5e-8), and a whole
   transformer block cycles through a sub-layer budget at 77.8% MRU reuse. What
   remains is wiring it into llama.cpp's own token loop; the tensor/page
   contiguity cost (D2D gather, or page_size ≥ largest tensor) is the trade-off.
2. a *working* QLoRA `PagedFrozenLinear` for bitsandbytes/PEFT (design in
   `docs/QLORA_ADAPTER.md`; helps only when the 4-bit base + activations exceed
   VRAM, and is PCIe-bound for the paged fraction)
3. automatic NUMA detection (bind each per-GPU source to that GPU's reported
   node without a manual flag)
4. predictive layer-order prefetch (feed the real layer schedule to the worker)
5. a vLLM KV-connector experiment showcasing a CXL/PMEM/file-backed host tier
6. TensorRT-LLM adapter (analogous to llama.cpp Boundary A via the C ABI +
   a custom plugin/allocator)
7. a `DevDaxSource` for character `/dev/dax` CXL/PMEM. Note: file-backed PMEM
   (fsdax) and CXL-as-a-file already work today through `MmapFileSource`.

