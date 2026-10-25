# vramer (Python)

A clean, pip-installable Python front end for **VRAMER**, the fixed-budget VRAM
pager. It gives Python / ML users two things without ever touching C++:

1. **A decode performance planner** (`vramer.plan` / `vramer.score` /
   `vramer.detect`) — a pure analytical model of PCIe-bound LLM decode. **No GPU
   and no compiled library required.**
2. **A Pythonic pager** (`vramer.Pager`) — a thin, safe `ctypes` wrapper over the
   `libvramer.so` **C ABI** (`include/vramer/vramer_c.h`). This part needs the
   separately-built shared library and a CUDA device.

> This package *wraps* the VRAMER C ABI. It does **not** build or ship the CUDA
> library. Build `libvramer.so` from the C++ project (see below) and point the
> package at it via `$VRAMER_LIB`.

## Install

From the repository root:

```bash
pip install -e python/
# or, if build isolation is unavailable:
python3 -m pip install -e python/ --no-build-isolation
```

Optional extras:

```bash
pip install -e "python/[dev]"     # pytest, for the test suite
pip install -e "python/[numpy]"   # numpy, for the BorrowedSource examples
```

Verify:

```bash
python3 -c "import vramer; print(vramer.__version__)"   # -> 0.2.0
```

## Quickstart 1 — the planner (no GPU, no libvramer.so)

Programmatic API:

```python
from vramer import plan, score

# Profiles may be a dict, a path to a JSON file, or a bundled profile name.
est = plan.estimate("current-machine", "qwen14b", quant="q4", batch=1)
print(est["verdict"], est["decode_tok_s"])

# Force a small resident budget so the model spills and must stream over PCIe:
est = plan.estimate("current-machine", "qwen14b", quant="q4",
                    vram_budget="1GiB", batch=8)
print(est["decode_tok_s"], "tok/s aggregate;",
      est["latency_s"] * 1000, "ms/token per sequence")

rec = score.score("current-machine", "qwen14b", vram_budget="1GiB")
print(rec["recommendation"], rec["paging_worthwhile"], rec["reason"])
```

Bundled profiles live in `vramer/profiles/` (`current-machine`, `qwen14b`,
`llama-70b`, `mixtral-8x7b`); list them with `plan.bundled_profiles()`.

CLI (installed as console scripts, or via `python -m vramer`):

```bash
vramer-plan   --profile current-machine --model qwen14b --vram-budget 1GiB --matrix
vramer-score  --profile current-machine --model llama-70b --vram-budget 8GiB
vramer-detect --json                     # emit a vramer-hw-profile/1 for this host

python -m vramer plan   --profile current-machine --model qwen14b
python -m vramer detect
python -m vramer score  --profile current-machine --model mixtral-8x7b
```

The planner CLIs accept either a bundled profile name (as above) or a path to any
`vramer-hw-profile/1` / `vramer-model-profile/1` JSON file, e.g. one produced by
`vramer-detect --output my-host.json` or the profiles under `/vramer/profiles/`.

## Quickstart 2 — the pager (needs libvramer.so + CUDA)

Build the shared library from the C++ project:

```bash
cd v1/VRAMER_v1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j                   # -> v1/VRAMER_v1/build/libvramer.so
export VRAMER_LIB=$PWD/build/libvramer.so
```

Then drive it from Python:

```python
from vramer import RamSource, Pager, EvictionPolicy

# Page a 32 MiB host RAM tier through an 8 MiB VRAM budget on GPU 0.
with Pager(RamSource(32 << 20), gpu=0, vram_budget=8 << 20,
           page_size=4 << 20, eviction=EvictionPolicy.MRU) as pager:
    for pid in range(pager.page_count()):
        with pager.acquire(pid) as lease:      # pins the page, resident in VRAM
            lease.wait_on(0)                   # order compute stream 0 after H2D
            # ... launch a kernel reading lease.device_ptr (int) ...
        # leaving the `with` releases the lease on stream 0
    pager.synchronize()
    print(pager.stats())   # {'hits':..,'misses':..,'bytes_h2d':.., ...}
```

Zero-copy over memory you already own (e.g. a NumPy array) uses `BorrowedSource`
— VRAMER does not copy it, so keep it alive via `keepalive`:

```python
import ctypes, numpy as np
from vramer import BorrowedSource, Pager

buf = np.zeros(64 << 20, dtype=np.uint8)
ptr = buf.ctypes.data_as(ctypes.c_void_p).value
src = BorrowedSource(ptr, buf.nbytes, keepalive=buf, label="numpy-demo")
with Pager(src, gpu=0, vram_budget=16 << 20, page_size=4 << 20) as pager:
    ...
```

A `FileSource(path)` pages an mmap-backed file instead.

Every non-zero C return raises `vramer.VramerError` carrying the C error message.

## How `VRAMER_LIB` works

When a `Pager` is first constructed, the package locates `libvramer.so` in this
order (first hit wins):

1. **`$VRAMER_LIB`** — an explicit path to the `.so` (recommended).
2. **`$VRAMER_LIB_DIR`** — a directory to look in.
3. Common build locations relative to the installed package and the current
   directory: `build/libvramer.so`, `v1/VRAMER_v1/build/libvramer.so`, …
4. The system dynamic loader (bare `libvramer.so`, honouring `LD_LIBRARY_PATH`
   and `/usr/local/lib`, `/usr/lib`, `/usr/lib64`).

If it cannot be found, a clear `RuntimeError` explains how to build it and set
`VRAMER_LIB`. Importing `vramer` and using the **planner** never triggers this —
the library is only needed for the pager.

## Testing

```bash
cd python && python3 -m pytest -q
```

Planner tests run anywhere. The pager test auto-skips unless `libvramer.so` is
loadable and a CUDA device is present. Set `VRAMER_TEST_GPU` to pick a GPU.

## What maps to what

| Python                                   | C ABI (`vramer_c.h`)                |
| ---------------------------------------- | ----------------------------------- |
| `RamSource(nbytes)`                      | `vramer_source_ram`                 |
| `FileSource(path)`                       | `vramer_source_file`                |
| `BorrowedSource(ptr, nbytes)`            | `vramer_source_borrowed`            |
| `Pager(...)` / `.close()`                | `vramer_pager_create` / `_destroy`  |
| `Pager.page_count()`                     | `vramer_pager_page_count`           |
| `Pager.acquire(id) -> Lease`             | `vramer_acquire`                    |
| `Lease.wait_on(stream)`                  | `vramer_lease_wait_on`              |
| `Lease.release_on(stream)`               | `vramer_lease_release_on`           |
| `Pager.prefetch_async(id)`               | `vramer_pager_prefetch_async`       |
| `Pager.drain_prefetch_queue()`           | `vramer_pager_drain_prefetch_queue` |
| `Pager.synchronize()`                    | `vramer_pager_synchronize`          |
| `Pager.stats()`                          | `vramer_pager_stats`                |
