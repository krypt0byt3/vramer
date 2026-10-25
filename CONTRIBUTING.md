# Contributing to VRAMER

Thanks for your interest! VRAMER is an explicit host-memory ↔ GPU-VRAM pager.
The maintained library lives in **`v1/VRAMER_v1/`**; Python tooling is in `src/`
and profiles in `profiles/`.

## Prerequisites

- Linux, NVIDIA GPU, CUDA Toolkit (12.8+ for SM120 / RTX 50-series)
- CMake ≥ 3.24, a C++17 compiler
- Python 3.9+ (for the planner/profile tooling; no PyTorch needed)

## Build & test

```bash
cd v1/VRAMER_v1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86  # 120 for RTX 50xx
cmake --build build -j
ctest --test-dir build --output-on-failure   # requires an NVIDIA GPU
./scripts/smoke.sh 86                          # build + tests + demos in one shot
```

The tests (`tests/`) and most demos **require a GPU**. CI on CPU-only runners
only compiles the code; GPU tests run on self-hosted runners or locally.

## Project layout

| Path | What |
|---|---|
| `v1/VRAMER_v1/include/vramer/` | public headers (C++ API + `vramer_c.h` C ABI) |
| `v1/VRAMER_v1/src/` | `memory_source.cpp`, `cuda_pager.cu`, `main.cu` (CLI), `vramer_c.cpp` |
| `v1/VRAMER_v1/tests/` | ctest suite |
| `v1/VRAMER_v1/examples/` | C-ABI PoC, Python demo, verified ggml-cuda + llama.cpp adapter |
| `v1/VRAMER_v1/docs/` | integration & tuning docs |
| `src/`, `profiles/` | Python detect/plan/score tooling and hardware/model profiles |

## Conventions

- C++17; keep to the surrounding style. Wrap CUDA calls in the `VRAMER_CUDA(...)` macro.
- **No C++ exception may cross the C ABI** (`vramer_c.cpp`): catch and convert to the error-code + message convention.
- New backing tiers implement `MemorySource` (or `WritableMemorySource`); add a
  `pinned_host_ptr` override if the tier is page-locked. Keep the core free of
  any assumption about the memory *technology* — it's bytes behind a source.
- Add a ctest for new pager behavior; data-correctness tests should verify bytes
  on-GPU against a known pattern (see `tests/test_pager.cu`).

## Pull requests

1. Branch from `master`.
2. Keep changes focused; update docs/CHANGELOG for user-visible changes.
3. Ensure `cmake --build` is warning-clean and `ctest` passes locally (on a GPU).
4. Describe what you measured — this project lives and dies by honest numbers.

By contributing you agree your work is licensed under the repository's LICENSE.
