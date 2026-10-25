# VRAMER

![status: experimental](https://img.shields.io/badge/status-experimental-orange)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)

**VRAMER** is an explicit host-memory ↔ GPU-VRAM pager for model/tensor data: it
holds a *fixed* VRAM budget and fills it on demand from a byte-addressable
`MemorySource` (host RAM, an mmap'd file, or — by design — CXL/PMEM), so a GPU
can work on models larger than its VRAM. It is a **paging primitive**, meant to
sit under an inference/fine-tuning engine, not to replace one.

> **Status: experimental (0.2.0, pre-1.0).** The core pager, C ABI, tooling and
> tests are solid and GPU-verified; the engine adapters (llama.cpp, QLoRA) are
> working proofs of mechanism, not yet production-optimized. APIs may change.
> **Read [`docs/TUNING.md`](v1/VRAMER_v1/docs/TUNING.md) first** — paging an
> oversized model is PCIe-bound, and that guide tells you when it helps.

## Where things are

| Path | What it is |
|------|------------|
| **`v1/VRAMER_v1/`** | **The library** — the maintained C++17/CUDA pager, CLI, tests, C ABI, and docs. Start here. |
| `v1/VRAMER_v1/README.md` | Full library docs: build, install, usage, measured results, roadmap. |
| `v1/VRAMER_v1/docs/` | Integration design (vLLM connector, llama.cpp adapter, QLoRA). |
| `v1/VRAMER_v1/examples/` | C-ABI reference programs, a runnable Python paging demo, and verified ggml-cuda integration demos. |
| `src/*.py`, `a2_*.cu`, `*.txt` | Early A2 experiments, Python tooling, and benchmark logs (historical). |

Large model/engine/checkpoint data is intentionally **not** tracked (see
`.gitignore`).

## Quickstart

```bash
cd v1/VRAMER_v1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
ctest --test-dir build                      # 5 tests
./build/vramer detect                       # GPUs, VRAM, NUMA
./build/vramer plan --model-bytes 40GiB --vram-budget 16GiB   # will paging help?
./scripts/smoke.sh 86                        # build + test + demos in one shot
```

## What works today (measured on 2× NVIDIA A2, SM86)

- Fixed-budget, page-granular residency with **scan-aware MRU eviction** (turns
  the LRU 0%-reuse sequential-scan collapse into hit-rate ≈ resident fraction).
- **Lock-free source reads** (overlapped load/compute pipeline), a background
  **prefetch worker pool**, and rich **telemetry**.
- **Write-back** of mutable pages (`acquire_mutable` + `flush`).
- **Multi-GPU** independent caches (~2× aggregate H2D across two A2s).
- A **C ABI** + shared `libvramer.so`, installable via CMake `find_package` /
  pkg-config.
- A **verified ggml-cuda path**: a real `ggml_mul_mat` consumes a 138 MiB weight
  paged through a half-size budget, matching a CPU reference to 1.5e-8.

See `v1/VRAMER_v1/README.md` for the details, the roadmap, and honest limits
(the full llama.cpp/QLoRA engine adapters are the next step; paging an
oversized model is PCIe-bound — the `plan` command estimates by how much).

## Docs & community

- [`docs/TUNING.md`](v1/VRAMER_v1/docs/TUNING.md) — when VRAMER helps, the levers, the measure-don't-assume workflow
- [`docs/INTEGRATION.md`](v1/VRAMER_v1/docs/INTEGRATION.md), [`LLAMACPP_ADAPTER.md`](v1/VRAMER_v1/docs/LLAMACPP_ADAPTER.md), [`QLORA_ADAPTER.md`](v1/VRAMER_v1/docs/QLORA_ADAPTER.md) — engine integration
- [`profiles/README.md`](profiles/README.md) — the hardware/model profile schemas and planner
- [`CONTRIBUTING.md`](CONTRIBUTING.md) · [`CHANGELOG.md`](CHANGELOG.md) · [`SECURITY.md`](SECURITY.md) · [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

## License

MIT — see [`v1/VRAMER_v1/LICENSE`](v1/VRAMER_v1/LICENSE).
