# VRAMER × llama.cpp — real end-to-end weight paging

This adapter makes **llama.cpp generate correct text while the model's FFN
weights are paged on demand from the GGUF file through a fixed VRAM budget that
is *smaller* than the paged weight set.** It is the full integration promised by
[`../../../docs/LLAMACPP_ADAPTER.md`](../../../docs/LLAMACPP_ADAPTER.md), built on
the mechanism proven in `../vramer_ggml_mulmat.cpp` / `../vramer_ggml_layer.cpp`.

## How it works (the honest mechanism)

llama.cpp already supports weights that live in **host** memory whose matmuls run
on the **GPU**: the scheduler uploads such a weight H2D into a bounded, reused
per-split device buffer immediately before the op (`ggml_backend_sched_copy_input`
→ its `callback_copy` hook, in `ggml/src/ggml-backend.cpp`). We exploit exactly
that path:

1. `-ot "ffn_(down|gate|up)\.weight=CPU"` routes the 72 FFN weight tensors
   (≈206 MiB) to host memory, so **they are not resident in VRAM**.
2. `GGML_OP_OFFLOAD_MIN_BATCH=1` makes the scheduler offload those FFN mul_mats
   to the GPU even during single-token decode (default threshold is 32), so the
   host-weight→GPU upload fires every token.
3. The fork redirects that upload into **VRAMER**: instead of a plain H2D memcpy
   of the host copy, `vramer_llama_copy_hook` pages the tensor's bytes out of the
   GGUF file through a VRAMER pager with a small fixed budget
   (`acquire / wait_on / D2D-gather into the kernel's dst buffer / release_on`,
   one-ahead prefetch, MRU eviction — the sequence from the proven examples).

The bytes delivered are identical to the file, so **generation is byte-identical
to a full-offload baseline**, but the FFN weights only ever occupy the VRAMER
budget in VRAM.

## Files

| file | role |
|---|---|
| `vramer_llama_hook.{h,cpp}` | the adapter: GGUF offset map + VRAMER pager + the copy hook. **No VRAMER repo files are modified.** |
| `llama_cpp_fork.patch` | the entire llama.cpp change: ~7 lines calling the hook from `llama_context::sched_copy_experts`, plus `src/CMakeLists.txt` wiring. |
| `build_fork.sh` | reproduce the fork from a clean upstream checkout + build. |
| `run_demo.sh` | one command: baseline vs VRAMER-paged generation + paging stats. |

## Reproduce

```bash
# 1. build VRAMER (worktree) and the fork
cmake -S v1/VRAMER_v1 -B v1/VRAMER_v1/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build v1/VRAMER_v1/build -j
bash v1/VRAMER_v1/examples/llamacpp/adapter/build_fork.sh /tmp/llama.cpp /tmp/llama-vramer

# 2. run the demo (uses the free GPU — set GPU=<idx>)
GPU=1 bash v1/VRAMER_v1/examples/llamacpp/adapter/run_demo.sh
```

### Env knobs (read by the hook)
- `VRAMER_GGUF`  — GGUF path; **setting it enables the hook** (unset ⇒ stock llama.cpp).
- `VRAMER_BUDGET` — VRAM budget in bytes (default 32 MiB).
- `VRAMER_PAGE`  — page size in bytes (default 4 MiB).
- `VRAMER_GPU`   — CUDA device index for the pager (default 0).
- plus `GGML_OP_OFFLOAD_MIN_BATCH=1` and `-ot "ffn_(down|gate|up)\.weight=CPU"`
  on the llama.cpp command line.

## Verified result (Qwen2.5-0.5B-Instruct Q4_K_M, NVIDIA A2 / SM86)

- **Output: byte-identical** to the full-offload baseline (greedy, `--temp 0`).
- **VRAM weights: 373.73 MiB → 167.60 MiB** (CUDA0 model buffer) — the 206 MiB
  FFN set is gone from VRAM and served through a **32 MiB** VRAMER budget.
- **Real paging:** ~72 FFN tensors paged per token, hit-rate ≈50%, and
  multi-GiB `bytes_h2d` of genuine host→device traffic over a 32-token run.

## Scope / limitations
- The FFN weights are also held once in host RAM (llama's own loaded copy);
  VRAMER additionally reads them from the file (served from the OS page cache).
  The *VRAM* budget is what is reduced and verified here. A host-free variant
  would use `vramer_source_borrowed` over llama's host buffer.
- The copy hook forces node-granular uploads; this is a correctness/mechanism
  demonstration, not a throughput-optimized path (every token re-pages the set).
