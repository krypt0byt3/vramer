# VRAMER ↔ llama.cpp weight-paging adapter

This document extends **Boundary A** (weight/tensor residency) from
[`INTEGRATION.md`](INTEGRATION.md) to llama.cpp/ggml specifically. It is an
engineering design, not a pitch: where the integration is clean it says so, and
where it fights the framework it says that too.

> **Status (verified on NVIDIA A2, SM86).** The load-bearing mechanism is
> proven by `examples/llamacpp/` (see `examples/README.md`):
> - a real `ggml_mul_mat` on the CUDA backend consumes a 138 MiB GGUF weight
>   paged through a budget **half its size** (D2D-gather into contiguous
>   scratch), result matching a CPU-from-file reference to **1.5e-8**;
> - all 7 weight tensors of a real transformer block cycled through an 8 MiB /
>   2-slot budget (< the 10.37 MiB layer) with MRU reuse (77.8% hit rate), all
>   numerically verified.
>
> What remains for a *full* adapter is integration, not research: a custom
> `ggml_backend_buffer_type` + a per-layer scheduler hook inside llama.cpp (§2/§4
> below). The contiguity problem (§3/§5) is the main hot-path cost.

The one-sentence thesis: **VRAMER lets llama.cpp run a model whose weights
exceed VRAM by paging the GGUF through a fixed VRAM budget during the token
loop, as an alternative to `--n-gpu-layers` partial offload.** VRAMER does not
replace ggml's kernels, scheduler, or KV cache — it only answers "are these
weight bytes resident on the device right now, and if not, get them there."

A standalone, compile-verified reference for the exact C ABI call sequence this
design relies on lives at [`../examples/llamacpp_poc.c`](../examples/llamacpp_poc.c).

---

## 1. How llama.cpp/ggml loads weights today

The relevant path, end to end:

1. **GGUF file → `llama_model_loader`.** The loader (`src/llama-model-loader.cpp`)
   parses the GGUF header (tensor names, shapes, dtypes, and byte offsets into
   the file's data section) and, by default, `mmap`s the whole file
   (`llama_mmap`, in `src/llama-mmap.cpp`). Each tensor's data is a region of
   that mapping.
2. **Per-tensor `ggml_tensor`.** For every tensor the loader creates a
   `ggml_tensor` whose `->data` ends up pointing at the bytes for that tensor.
   Where those bytes physically live depends on the backend buffer the tensor is
   allocated in:
   - **CPU backend:** `->data` points directly into the host mmap (or a host
     malloc when `--no-mmap`). No copy.
   - **CUDA backend:** the loader allocates a `ggml_backend_buffer` of type
     `ggml_backend_cuda_buffer_type(device)` (see `ggml/src/ggml-cuda/ggml-cuda.cu`),
     and `llama_model_loader::load_data_for` / `ggml_backend_tensor_set` copies
     the tensor's bytes from the file mapping into that device buffer with
     `cudaMemcpy`. After load, `tensor->data` is a **device pointer** into that
     CUDA buffer and stays fixed for the model's lifetime.
3. **`--n-gpu-layers` (the current out-of-core story).** llama.cpp decides
   per-tensor which backend buffer to use. Layers `[0, n_gpu_layers)` get CUDA
   buffers; the rest stay in host (CPU-backend) buffers. This is the baseline we
   must beat: it is *static* — a layer is either fully on GPU for the whole run
   or never.
4. **Graph compute.** `ggml_backend_sched` (`ggml/src/ggml-backend.cpp`) splits
   the op graph across backends and runs each split on its backend. For CUDA that
   is `ggml_backend_cuda_graph_compute`, which dispatches each op's kernel; every
   source tensor the kernel reads is dereferenced through its (fixed) `->data`
   device pointer.

### Real interception points

There are exactly two places VRAMER can plug in, matching the two strategies
below:

- **The CUDA backend buffer type** — `ggml_backend_cuda_buffer_type` plus the
  tensor `->data` / `->extra` fields. A buffer type is an interface
  (`ggml_backend_buffer_type_i`: `alloc_buffer`, `get_alloc_size`, …) and a
  buffer is `ggml_backend_buffer_i` (`get_base`, `set_tensor`, `get_tensor`,
  `free_buffer`, …). A **custom buffer type** is a first-class, documented
  extension point.
- **The scheduler / compute path** — `ggml_backend_sched_graph_compute` and, one
  level down, the per-split op loop in `ggml_backend_cuda_graph_compute`. This is
  where you can know "layer L is about to run" and drive prefetch/residency.

---

## 2. The core tension (state it plainly)

**ggml assumes a stable device pointer per tensor for the entire compute.** A
`ggml_tensor->data` is set once at load and read by every kernel that touches
the tensor, across every op and every token. The whole backend design — graph
capture, CUDA graphs, `ggml_backend_sched` split planning — leans on that
pointer not moving.

**VRAMER's whole point is that pointers move.** A page is resident only while
leased; outside its lease it can be evicted and its slot reused, so the device
address backing a given file offset is valid only between `acquire` and
`release_on`. VRAMER is honest about this: `vramer_acquire` returns the
`out_devptr` *for this lease*, and `vramer_lease_release_on` makes the slot
reclaimable.

Reconciling "stable pointer forever" with "valid only within a lease" is the
entire adapter problem. Two viable strategies:

### Strategy (a): custom ggml CUDA buffer type backed by a VRAMER MemorySource

Implement a new `ggml_backend_buffer_type` whose buffers do **not** hold the
tensor bytes resident. Instead:

- `get_alloc_size` reports the tensor's size as usual, but the "allocation" is
  just a record of **which file offset / VRAMER page(s)** this tensor occupies.
  The GGUF file is the `MemorySource` (`vramer_source_file`).
- The tensor's `->data` is a **stable sentinel** (e.g. the tensor's file offset
  encoded as a fake pointer, or an index stored in `->extra`), never a real
  device address at load time.
- You **hook op execution** so that immediately before a kernel reads a tensor,
  the adapter calls `vramer_acquire(page_of(tensor))`, patches the kernel's
  source pointer to the lease's `out_devptr`, runs the op with
  `vramer_lease_wait_on(stream)` ordering, and after the op calls
  `vramer_lease_release_on(stream)`. **The lease spans exactly one op.**

```c
/* Per op, for each weight source tensor the op reads: */
vramer_lease* lease = NULL; void* devptr = NULL; size_t n = 0;
if (vramer_acquire(pager, page_of(tensor), &lease, &devptr, &n, err, sizeof err))
    return fail(err);
vramer_lease_wait_on(lease, (void*)stream, err, sizeof err); /* order, not sync */
/* point the kernel at devptr for the duration of this op */
launch_op_reading(devptr, /* ... */ stream);
vramer_lease_release_on(lease, (void*)stream, err, sizeof err); /* consumes lease */
```

Tradeoffs:
- **+** Correct by construction: the pointer is only ever used inside its lease.
- **+** Finest residency granularity; the budget can be far smaller than a layer.
- **−** Requires intercepting op dispatch to patch source pointers and bracket
  each op with acquire/release. That is invasive — it fights the "pointer set
  once" assumption head-on — and it **breaks CUDA graph capture** (the pointer
  isn't stable across replays), so you lose that optimization.
- **−** Per-op acquire/release overhead on the hot path (see §5).

### Strategy (b): layer-granular prefetch + residency at the scheduler level

Keep tensors in a real (but *budget-bounded*) residency pool and drive it from
the scheduler's knowledge of layer order. Before computing layer L:

- `vramer_pager_prefetch_async` the pages of **L's** tensors (and, with
  look-ahead, L+1…L+N), then
- `vramer_acquire` + `vramer_lease_wait_on` each of L's tensor pages, hold the
  leases for the **duration of the layer's ops**, and `vramer_lease_release_on`
  once the layer is done.

This matches VRAMER's design directly: `prefetch_async` + MRU + the *known*
cyclic layer scan order. Rather than patch every op, you hold a layer's pages
resident across that layer's whole subgraph, so the device pointers are stable
*within the layer* — long enough for the split's ops and even a per-layer CUDA
graph.

Tradeoffs:
- **+** Far fewer acquire/release calls (per layer, not per op).
- **+** Aligns with VRAMER's measured strengths: MRU on a cyclic scan, prefetch
  look-ahead, lock-free pipelined loads.
- **+** Pointers stable within a layer → compatible with per-layer graph capture.
- **−** The budget must hold a whole layer's resident set at once (plus
  look-ahead), so it is coarser than (a).
- **−** Still needs a hook to patch each layer's tensor `->data` to the current
  lease pointers before the layer runs and to release after — less invasive than
  (a) but not zero.

**Recommendation:** ship a **hybrid** — a custom buffer type (a) for the
*mechanism* (tensors that live in the file, not VRAM), driven at
**layer granularity** (b) for the *policy* (acquire/hold/release per layer,
prefetch ahead). The milestone in §4 is the smallest honest slice of this.

---

## 3. Mapping VRAMER features onto llama.cpp

| VRAMER feature | llama.cpp mapping |
|---|---|
| Page size (64 MiB sweet spot on A2) | Weights are paged in 64 MiB units of the GGUF data section, independent of tensor boundaries. |
| `MemorySource` = `vramer_source_file(gguf)` | The mmap'd GGUF *is* the host tier. No extra copy of the model in host RAM beyond the OS page cache. |
| MRU eviction | The per-token layer scan is cyclic over the same pages; MRU keeps a stable resident prefix where LRU gives 0% reuse (README fact #1). |
| `prefetch_async` + `prefetch_ahead` | Look ahead N layers using the *known* scan order and prefetch their pages while the current layer computes. |
| Multi-GPU independent caches | llama.cpp tensor-split assigns tensors to devices; give **each device its own pager** over the same `MemorySource`, each with its own budget/stream/staging ring. |
| `PageLease` | Prevents eviction of a tensor's page while a kernel reading it is in flight. |

### Page vs tensor granularity (the packing problem)

A 64 MiB page rarely lines up with tensor boundaries. Two cases:

- **Tensor larger than a page** (a big `attn`/`ffn` weight matrix): the tensor
  spans several consecutive pages `[p0 … pk]`. The adapter must acquire *all* of
  them and ensure they are **contiguous in the device budget** for the kernel to
  read the tensor as one array — which VRAMER's fixed-slot cache does **not**
  guarantee (slots are reused in arbitrary order). Options: (i) make the page
  size ≥ the largest tensor so every tensor fits in one page (wastes budget and
  loses the 64 MiB sweet spot), or (ii) have the adapter copy the tensor's pages
  into a contiguous scratch region after acquiring (an extra D2D copy). This is
  the single thorniest implementation detail — see §5 risk.
- **Tensor smaller than a page** (norms, biases, small embeddings, quantized
  blocks): many tensors share one page. This is *good* — one `acquire` makes
  several tensors resident at once (page-packing amortizes the lease). The
  adapter should group tensors by page and acquire per page, not per tensor,
  which also cuts the per-op overhead of Strategy (a).

For a first milestone, **choose a page size ≥ the largest weight tensor in the
target GGUF** so no tensor straddles a page boundary. That sidesteps the
contiguity problem entirely at the cost of a larger (but still fixed) page.

---

## 4. First milestone (actually achievable)

**Goal:** run a model whose weights exceed VRAM by paging the GGUF through a
fixed budget for the token loop, accepting per-op lease overhead, and measure
tokens/s against `--n-gpu-layers` partial offload.

Concretely:

1. **Prove the call pattern** with no llama.cpp at all — done: see
   `examples/llamacpp_poc.c`. It sweeps every 64 MiB page of a real file
   through a fixed budget using exactly the acquire / wait_on / (read) /
   release_on + prefetch sequence an adapter's layer loop uses, and prints hit
   rate and H2D bytes. (Measured: a 3.7 GiB shard through a 512 MiB budget →
   98.3% hit rate from one-ahead prefetch, 1 cold miss.)
2. **Custom CUDA buffer type** that allocates "page-backed" tensors: `->data` is
   a sentinel, real bytes live in the GGUF `MemorySource`. Pick the page size ≥
   the largest tensor (sidestep §3 contiguity).
3. **Scheduler hook** (Strategy b): before each layer, prefetch the next layer's
   pages and acquire the current layer's; after each layer, release. Patch the
   layer's tensor pointers to the lease devptrs.
4. **Measure** tokens/s and prefill time at several budgets (e.g. 40/60/80% of
   model size) vs llama.cpp's own `--n-gpu-layers` set to the same effective VRAM
   footprint. The bar to clear: at a budget where partial offload can place K of
   N layers on GPU, VRAMER paging *all* N layers through the same VRAM should be
   competitive or better because it never runs a layer on the CPU backend.

Keep the first run single-GPU, single-stream, greedy decode, batch 1. That is
the regime where VRAMER's lock-free pipelined load already hides the source read
behind compute (README).

---

## 5. Honest limitations

- **Per-op (or per-layer) acquire/release overhead.** Every lease is event
  bookkeeping on the hot path. Strategy (a) pays it per op; (b) amortizes it per
  layer. If the budget is large enough that pages stay resident, the hits are
  cheap, but the acquire/release calls still execute.
- **Dequantization is unchanged.** VRAMER pages the *quantized* bytes exactly as
  they sit in the GGUF; ggml's existing dequant kernels still run. VRAMER does
  not reduce compute, only where the weights live.
- **The stable-pointer assumption.** This is the real fight (§2). A custom buffer
  type with per-op pointer patching is incompatible with CUDA graph capture;
  layer-granular residency recovers per-layer graphs but not whole-graph capture.
- **Tensor/page contiguity** (§3): tensors spanning multiple non-contiguous
  slots need either page-size ≥ largest-tensor or a D2D gather. This is the
  biggest single technical risk.
- **It does not help if the model already fits.** When the whole model is
  resident, paging is pure overhead; `llamacpp_poc.c` prints a NOTE in that case.
  VRAMER's niche is strictly weights-exceed-VRAM.
- **NVMe is not VRAM.** If the GGUF `MemorySource` faults from storage (not the
  OS page cache) every token, it will be far slower than keeping the active set
  in host RAM (README caveat). The win case is model-in-host-RAM, not
  model-on-disk-per-token.

---

## 6. Code sketches (against the real `vramer_c.h`)

### 6.1 Create one pager per device over the GGUF

```c
#include "vramer/vramer_c.h"

char err[256];
vramer_source* src = vramer_source_file("model.gguf", err, sizeof err);
if (!src) { fprintf(stderr, "source: %s\n", err); return 1; }

vramer_config cfg;
memset(&cfg, 0, sizeof cfg);
cfg.gpu              = device_id;        /* one pager per tensor-split device */
cfg.vram_budget      = 6ull << 30;       /* fixed, independent of model size  */
cfg.page_size        = 64ull << 20;      /* or >= largest tensor (see §3)     */
cfg.staging_buffers  = 2;
cfg.prefetch_workers = 1;
cfg.eviction         = VRAMER_EVICT_MRU; /* cyclic layer scan (README #1)     */

vramer_pager* pager = vramer_pager_create(src, &cfg, err, sizeof err);
if (!pager) { fprintf(stderr, "pager: %s\n", err); return 1; }
vramer_source_free(src);                 /* pager holds its own ref           */
```

### 6.2 Per-layer loop (Strategy b), inside the scheduler hook

```c
/* page_of(tensor): file_offset(tensor) / cfg.page_size, precomputed at load. */

void run_layer(vramer_pager* pager, const layer_t* L, const layer_t* Lnext,
               cudaStream_t stream) {
    char err[256];

    /* 1. Prefetch the NEXT layer's pages while this layer computes. */
    for (size_t i = 0; i < Lnext->n_pages; ++i)
        vramer_pager_prefetch_async(pager, Lnext->pages[i], err, sizeof err);

    /* 2. Acquire this layer's pages; hold leases across the layer's ops. */
    vramer_lease* leases[MAX_PAGES_PER_LAYER];
    void*         dptr[MAX_PAGES_PER_LAYER];
    for (size_t i = 0; i < L->n_pages; ++i) {
        size_t n = 0;
        if (vramer_acquire(pager, L->pages[i], &leases[i], &dptr[i], &n,
                           err, sizeof err)) { /* handle + release prior leases */ }
        vramer_lease_wait_on(leases[i], (void*)stream, err, sizeof err);
    }

    /* 3. Patch each tensor's ->data to its page's lease devptr, then run the
     *    layer's ggml ops on `stream` (kernels read the resident bytes). */
    bind_tensor_ptrs(L, dptr);
    ggml_backend_cuda_graph_compute_layer(L, stream);

    /* 4. Release after the layer; slots become reclaimable past this stream pt. */
    for (size_t i = 0; i < L->n_pages; ++i)
        vramer_lease_release_on(leases[i], (void*)stream, err, sizeof err);
}
```

### 6.3 Reading telemetry to compare against the baseline

```c
vramer_stats st; memset(&st, 0, sizeof st);
vramer_pager_stats(pager, &st);
double hit_rate = (st.hits + st.misses)
    ? 100.0 * (double)st.hits / (double)(st.hits + st.misses) : 0.0;
fprintf(stderr, "hit_rate=%.1f%% h2d=%.1f MiB evictions=%llu\n",
        hit_rate, (double)st.bytes_h2d / (1u << 20),
        (unsigned long long)st.evictions);
```

The full, compile-and-run-verified version of this pattern (with CUDA stream
setup, D2H residency check, and error handling) is
[`../examples/llamacpp_poc.c`](../examples/llamacpp_poc.c).
