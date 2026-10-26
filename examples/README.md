# VRAMER examples & adapter references

Reference code showing how to drive the pager through the C ABI
(`include/vramer/vramer_c.h`). These are starting points for the engine adapters
described in [`../docs/INTEGRATION.md`](../docs/INTEGRATION.md),
[`../docs/LLAMACPP_ADAPTER.md`](../docs/LLAMACPP_ADAPTER.md), and
[`../docs/QLORA_ADAPTER.md`](../docs/QLORA_ADAPTER.md).

## `llamacpp_poc.c`

Pure-C program (does **not** require llama.cpp) that demonstrates the exact
C-ABI call pattern a weight-paging adapter's per-layer loop uses: for each page,
`prefetch_async(next)` → `acquire` → `wait_on` → use the device pointer →
`release_on`, then prints cache stats. Point it at any large file.

Build (after a normal `cmake --build build`) and run:

```bash
cd examples
gcc -I../include llamacpp_poc.c ../build/libvramer.a -lcudart -lstdc++ -lpthread -o /tmp/llamacpp_poc
/tmp/llamacpp_poc /path/to/model.safetensors 512MiB
```

Validated against a real 3.7 GiB Qwen2.5-14B shard paged through a 512 MiB
budget: **98.3% hit rate** with one-ahead prefetch (only the cold first page
misses).

## `llamacpp/` — real ggml-cuda integration (verified)

Two standalone programs that link **both** llama.cpp's ggml/ggml-cuda and VRAMER,
proving that real CUDA kernels consume VRAMER-paged weights. They need a built
llama.cpp checkout (see `docs/LLAMACPP_ADAPTER.md`).

```bash
# build llama.cpp with CUDA first (one time):
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build /tmp/llama.cpp/build -j

cd examples/llamacpp && make LLAMA=/tmp/llama.cpp
CUDA_VISIBLE_DEVICES=1 ./vramer_ggml_mulmat /path/to/model.gguf
CUDA_VISIBLE_DEVICES=1 ./vramer_ggml_layer  /path/to/model.gguf
```

- `vramer_ggml_mulmat.cpp` — pages one large GGUF weight (e.g. 138 MiB
  `output.weight`) through a budget **half its size**, D2D-gathers it into
  contiguous device scratch, runs a real `ggml_mul_mat` on the CUDA backend, and
  verifies the result against a CPU-from-file reference (**max_abs_diff 1.5e-8**).
- `vramer_ggml_layer.cpp` — cycles all 7 weight tensors of a real transformer
  block through an 8 MiB / 2-slot budget (< the 10.37 MiB layer) over two
  simulated tokens; MRU gives 77.8% reuse, all 7 numerically verified.

The remaining step to a *full* adapter is a custom `ggml_backend_buffer_type` +
scheduler hook inside llama.cpp itself; the load-bearing mechanism is proven here.

## `qlora_paged_linear.py`

Python (`ctypes`) skeleton for a QLoRA-style paged frozen-base layer: a
`PagedWeightProvider` over the C ABI plus a torch-guarded `PagedFrozenLinear`.
Imports and runs without PyTorch (prints the computed page layout for a toy
config). Full execution needs a **shared** `libvramer.so` (ctypes can't load the
static `.a` — see the doc for the `-DBUILD_SHARED_LIBS=ON` build), PyTorch, and a
real quantized model.

```bash
python3 examples/qlora_paged_linear.py          # binding self-check, no torch needed
```

## `python_paging_demo.py`

Minimal **runnable** Python demo (needs only `numpy` + a built `libvramer.so`):
wraps a NumPy array as a `vramer_source_borrowed` host tier, pages it through a
small VRAM budget, and verifies via the pager's own `bytes_h2d` that every byte
reached the GPU. Proves the Python → ctypes → C ABI → pager → GPU data path end
to end without PyTorch.

```bash
cmake --build build          # produces build/libvramer.so (VRAMER_BUILD_SHARED=ON)
python3 examples/python_paging_demo.py
# -> source=64 MiB budget=16 MiB page=4 MiB pages=16
#    hits=0 misses=16 evictions=12 bytes_h2d=67108864 (64 MiB)
#    python paging demo OK: NumPy -> borrowed source -> pager -> GPU verified
```
