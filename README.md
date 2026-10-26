# VRAMER

![status: experimental](https://img.shields.io/badge/status-experimental-orange)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)

**VRAMER** is an explicit host-memory ↔ GPU-VRAM pager for model/tensor data. It
holds a *fixed* VRAM budget and fills it on demand from a byte-addressable
`MemorySource` (host RAM, pinned host memory, an mmap'd file, or — by design —
CXL/PMEM), so a GPU can work on models **larger than its VRAM**. It is a paging
primitive meant to sit *under* an inference/fine-tuning engine, not to replace one.

> **Status: experimental (0.2.0, pre-1.0).** The core pager, C ABI, Python package,
> tooling and tests are solid and GPU-verified; the engine adapters (llama.cpp,
> QLoRA) are working proofs of mechanism, not yet production-optimized. APIs may
> change. **Read [`docs/TUNING.md`](docs/TUNING.md) first** — paging an oversized
> model is PCIe-bound, and that guide tells you exactly when it helps.

## Quickstart (C++ / CLI)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86  # 120 for RTX 50xx
cmake --build build -j
ctest --test-dir build                       # 5 GPU tests
./build/vramer --version
./build/vramer detect                        # GPUs, VRAM, NUMA
./build/vramer probe                         # real H2D bandwidth (pageable vs pinned)
./build/vramer plan --model-bytes 40GiB --vram-budget 16GiB   # will paging help, how fast?
./scripts/smoke.sh 86                         # build + test + demos in one shot
```

## Quickstart (Python)

```bash
pip install -e python/                        # the `vramer` package; planner works with no GPU
vramer-plan --profile profiles/current-machine.json --model profiles/llama-70b.json --matrix
# with a built libvramer.so (auto-found, or set VRAMER_LIB), drive the pager:
python -c "from vramer import RamSource, Pager
with Pager(RamSource(64<<20), gpu=0, vram_budget=16<<20) as p:
    print('pages:', p.page_count())"
```

## Install as a library

```bash
cmake --install build --prefix /usr/local     # headers, libs, CMake package, pkg-config
# then, from another CMake project:
#   find_package(vramer REQUIRED); target_link_libraries(app PRIVATE vramer::vramer)
```

## Repository layout

| Path | What |
|------|------|
| `CMakeLists.txt`, `include/`, `src/`, `tests/` | the C++17/CUDA pager, C ABI, and GPU test suite |
| `examples/` | C-ABI reference, runnable Python demo, verified **ggml-cuda** demos, and a working **llama.cpp** weight-paging adapter (`examples/llamacpp/`) |
| `python/` | pip-installable **`vramer`** package — `Pager`/`Lease`/sources + planner + CLIs |
| `profiles/` | hardware/model profiles for the planner ([schema](profiles/README.md)) |
| `docs/` | [`TUNING.md`](docs/TUNING.md), [`LIBRARY.md`](docs/LIBRARY.md) (full library reference), [`INTEGRATION.md`](docs/INTEGRATION.md), [`LLAMACPP_ADAPTER.md`](docs/LLAMACPP_ADAPTER.md), [`QLORA_ADAPTER.md`](docs/QLORA_ADAPTER.md) |
| `scripts/`, `benchmarks/` | one-shot smoke/sweep scripts and measured A2 data |
| `experiments/` | early A2 exploration & raw logs (historical, not built) |

Large model/engine/checkpoint data is intentionally **not** tracked (`.gitignore`).

## What works today (measured on 2× NVIDIA A2, SM86)

- Fixed-budget, page-granular residency with **scan-aware MRU eviction** (turns the
  LRU 0%-reuse sequential-scan collapse into hit-rate ≈ resident fraction).
- **Lock-free source reads**, a background **prefetch worker pool**, and a **pinned
  direct-DMA fast path** (skips the staging copy).
- **Write-back** of mutable pages (`acquire_mutable` + `flush`).
- **Multi-GPU** independent caches (~2× aggregate H2D across two A2s).
- A **C ABI** + shared `libvramer.so`, installable via CMake `find_package` / pkg-config,
  and a **Python package** on top.
- A **verified ggml-cuda path** and a **working end-to-end llama.cpp adapter**:
  byte-identical generation with a weight set paged through a VRAM budget smaller
  than it.
- **MoE expert paging** (Qwen1.5-MoE-A2.7B): only the *active* experts stream per
  token (~7.4% of the stack — **13.5× less** traffic than paging it whole), keeping
  **~90% of the model on host** through a budget as small as 0.5 GiB, with
  **byte-identical** output and pager hit-rate rising 36→70% as the budget grows
  (real MoE cross-token locality). This proves the selectivity + correctness thesis;
  throughput optimization (async expert prefetch) is the open follow-up. See
  [`examples/llamacpp/README_MOE.md`](examples/llamacpp/README_MOE.md).

See [`docs/LIBRARY.md`](docs/LIBRARY.md) for the full reference, roadmap, and honest
limits (paging an oversized model is PCIe-bound — `vramer plan` estimates by how much).

## Docs & community

📖 **[Project Wiki](https://github.com/krypt0byt3/vramer/wiki)** — getting started,
architecture, tuning, integrations, API reference, roadmap, FAQ.

[`ROADMAP.md`](ROADMAP.md) · [`CONTRIBUTING.md`](CONTRIBUTING.md) ·
[`CHANGELOG.md`](CHANGELOG.md) · [`SECURITY.md`](SECURITY.md) ·
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

## License

MIT — see [`LICENSE`](LICENSE).
