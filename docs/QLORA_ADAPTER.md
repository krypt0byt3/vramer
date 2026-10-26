# A read-only QLoRA weight-paging adapter on VRAMER

VRAMER is a **read-only** pager: it moves bytes host → VRAM on demand over a
fixed budget, and it has **no write-back / dirty-page path yet**. That one fact
decides what training it can and cannot accelerate.

QLoRA fits VRAMER's model almost exactly, and for a non-obvious reason: in
QLoRA the large tensor is the **frozen, 4-bit quantized base**. It is never
written during training — gradients flow only into the small LoRA adapters — and
it is read in a predictable layer-by-layer scan. A read-only, scan-aware pager
is precisely the right primitive for a large, immutable, sequentially-scanned
tensor.

Full fine-tuning is the opposite shape: weights, gradients, and optimizer state
are all **mutable** and must be written back to the host tier. That is a
PCIe-bound, write-heavy, out-of-core problem already served well by
**DeepSpeed ZeRO-Infinity** and **FSDP (CPU/NVMe offload)**. VRAMER should not
try to be those; it has no write-back path, and bolting one on would make it a
worse version of tools that already exist. This adapter is deliberately scoped
to the read-only case.

---

## 1. Memory math: where paging actually helps

Let `P` = parameter count. Per-parameter resident cost, roughly:

| Regime | Weights | Grads | Optimizer (Adam m+v) | ≈ bytes/param |
|---|---|---|---|---|
| Full FT (fp16 weights, fp32 Adam) | 2 | 2 | 8 (+4 fp32 master) | **~16** |
| LoRA on fp16 base | 2 (frozen) | ~0 (base) | ~0 (base) | ~2 for base |
| **QLoRA (4-bit frozen base)** | **~0.5** (NF4) | 0 (base) | 0 (base) | **~0.5 for base** |

The QLoRA trainable state (LoRA `A`,`B` + their fp32 Adam moments) is governed
by the rank `r`, not by `P`. For a rank-16 adapter on attention + MLP projections
it is on the order of **0.1–0.5% of `P`**, i.e. a few hundred MB even for a 70B
model. That part always stays resident.

So the dominant resident cost in QLoRA is:

```
base_resident ≈ 0.5 bytes/param  +  activations (batch × seq × hidden × layers)
```

Concrete 4-bit base sizes (NF4, ~0.5 B/param plus small quant metadata):

| Model | 4-bit base | Fits on 24 GB? | Fits on 48 GB? |
|---|---|---|---|
| 7B | ~3.5 GB | yes | yes |
| 13B | ~6.5 GB | yes | yes |
| 70B | **~35 GB** | **no** | yes (tight, + activations) |
| 100B+ | **50 GB+** | no | often no |

**Bottom line — when VRAMER helps:**

- VRAMER helps **only** when the 4-bit frozen base **plus activations** exceed
  VRAM. That is the 70B-on-a-24GB-card, or 100B+ regime: the base can't be fully
  resident, so you page it through a fixed budget and keep the adapters resident.
- If the 4-bit base **already fits** (7B/13B on a 24 GB card, 70B on a 48–80 GB
  card with headroom), **you do not need VRAMER** — plain QLoRA
  (bitsandbytes + PEFT) keeps the whole base resident and will be strictly
  faster. Paging only adds PCIe traffic you didn't need.
- VRAMER does **not** reduce the trainable/activation footprint; those are not
  paged here (they're mutable). Use gradient checkpointing / smaller batch for
  those. VRAMER addresses exactly one axis: the frozen base not fitting.

The honest value proposition is **"train a model whose base won't fit, at the
cost of PCIe,"** not "train faster."

---

## 2. Integration point: `PagedFrozenLinear`

The adapter is a drop-in replacement for the **frozen base's** linear layers —
i.e. `bitsandbytes.nn.Linear4bit` as used by HF `transformers` when a model is
loaded with `load_in_4bit=True`. The LoRA path is unchanged from ordinary PEFT.

`forward(x)` does four things (see `examples/qlora_paged_linear.py` for the
skeleton):

1. **Acquire** the page(s) holding this layer's quantized weight via the C ABI
   (`vramer_acquire`), getting a device pointer (`out_devptr`) and byte count.
   `vramer_lease_wait_on(lease, stream)` orders the compute stream *after* the
   H2D copy without a device-wide sync.
2. **Dequant + matmul** reading directly from that device pointer. A custom (or
   bitsandbytes) kernel dequantizes NF4 → compute dtype and computes
   `base_out = x @ dequant(W)ᵀ`. The weight is never copied into a torch tensor
   and **never written**.
3. **Release** each lease (`vramer_lease_release_on(lease, stream)`), allowing
   those pages to be evicted after the stream passes that point.
4. **LoRA path** `lora_out = (x @ Aᵀ) @ Bᵀ · (α/r)` using fully resident,
   trainable `A`,`B` parameters; `out = base_out + lora_out (+ bias)`.

Why no write-back is needed: `requires_grad=False` on the base, `True` only on
`A`,`B`. Autograd never produces a gradient w.r.t. the paged weight, so VRAMER's
page is read, used, and dropped — the clean read-only contract.

The currency across the FFI boundary is the **GPU pointer**, never a copied
tensor (consistent with `docs/INTEGRATION.md`).

---

## 3. Laying the frozen base out as a pageable source

VRAMER pages a single byte-addressable `MemorySource`. You give it the frozen
quantized base as **one contiguous byte buffer**, the concatenation of every
frozen layer's quantized weight (plus its packed quant metadata if you choose to
page it too), in **layer order** so the scan is sequential:

```
[ layer0 qweight | layer1 qweight | layer2 qweight | ... | layerN qweight ]
  off0=0           off1             off2                    offN
```

Two source choices:

- **`vramer_source_borrowed(base, bytes, label, ...)`** — the layout already
  lives in host RAM (e.g. a pinned CPU tensor holding the packed 4-bit base).
  Zero-copy; VRAMER does **not** take ownership, so the Python side must keep the
  buffer alive for the pager's lifetime (the skeleton holds a `_keepalive` ref).
- **`vramer_source_file(path, ...)`** — the layout is an mmap-able file on disk
  (a pre-packed base checkpoint). Good for bases too large even for host RAM, but
  every miss then faults from storage (see limitations).

**Page-id computation.** VRAMER slices the buffer into `page_size` pages:

```
page_id(byte_offset) = byte_offset // page_size
```

For a layer at `offset` of length `nbytes`, the pages covering it are
`page_id(offset) .. page_id(offset + nbytes - 1)` inclusive
(`PagedWeightProvider.pages_for_range`).

- **Sub-page tensors**: a small layer may share one page with its neighbours.
  One `acquire` serves it; neighbours sharing the page are effectively prefetched.
- **Multi-page tensors**: a large layer spans several pages. `forward` must
  acquire **all** of them, `wait_on` each, run the kernel over the contiguous
  device region, then release all. Note VRAMER pages are independent device
  slots and are **not guaranteed contiguous** in device memory, so the kernel
  must consume a list of (devptr, nbytes) segments, or you must size `page_size`
  so each layer is ≤ 1 page (simplest, recommended for a first cut).

Choose `page_size` to balance per-transfer overhead (README: 64 MiB is the A2
sweet spot; 16 MiB loses ~10%) against alignment to layer boundaries.

---

## 4. Access pattern: a predictable bidirectional scan

- **Forward**: layers `0 → N`. A clean forward scan.
- **Backward**: layers `N → 0`, re-reading the **same frozen weights** to form
  the adapter gradients.

Both directions are predictable sequential scans, which is exactly the regime
where:

- **MRU eviction (`VRAMER_EVICT_MRU`)** beats LRU. On a cyclic scan larger than
  the cache, LRU gives 0% reuse (it evicts the page you'll need next); MRU
  retains a stable resident prefix (README A2 data: hit rate ≈ ⅔ × resident
  fraction). Use MRU here.
- **`vramer_pager_prefetch_async(next_layer_page)`** overlaps the next layer's
  H2D with the current layer's compute.

**Backward re-read cost.** Backward reads the frozen base a second time per
step. When the resident fraction is high enough that a layer is still cached from
forward, the pager serves it as a hit; otherwise it is extra H2D traffic. This
doubles the *potential* read traffic for the paged fraction and is the main
reason step time scales with how little of the base fits. (Gradient
checkpointing adds a partial *third* scan for recomputed activations, but that
touches activations, not the paged base.)

---

## 5. Honest limitations

- **Per-layer acquire/release overhead inside the training loop.** Every layer,
  every direction, every step pays FFI + lease + event-ordering cost. For a
  70B/80-layer model that's thousands of acquire/release pairs per step.
- **Dequant on every access.** VRAMER caches the **quantized** bytes, not a
  dequantized tensor. There is no persistent dequantized cache, so NF4 → compute
  dtype dequant runs on every access (forward and backward). A would-be
  dequantized cache would be larger than the 4-bit source and defeat the point.
- **PCIe-bound for the paged fraction.** Throughput is capped by H2D bandwidth
  (A2: ~5.5 GiB/s/GPU measured). The smaller the resident fraction, the more the
  step is dominated by paging. This **trades speed for the ability to train at
  all** — if the base already fits, you lose only.
- **Interception plumbing is non-trivial.** You must find and replace the frozen
  base's `Linear4bit` modules (bitsandbytes/transformers internals), pack the
  base into a contiguous layer-ordered buffer with correct per-layer offsets and
  quant metadata, and supply a dequant+matmul kernel that reads a device pointer
  (and possibly multiple page segments). None of this is a one-liner.
- **No contiguity guarantee across pages** (see §3) — multi-page layers need a
  segment-aware kernel or `page_size ≥ max layer size`.
- **Read-only by design.** Anything that would write the base (e.g. actually
  fine-tuning base weights, or merging adapters in place) is out of scope until
  VRAMER grows a write-back path.

---

## 6. Build note: a shared library is required for ctypes

The project builds a **static** `build/libvramer.a`. `ctypes.CDLL` needs a
**shared object**. Produce `libvramer.so` without editing `CMakeLists.txt` by
building shared, e.g.:

```bash
# Option A: ask CMake for shared libraries (separate build dir, no edits).
cmake -S . -B build-shared -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build-shared -j
# -> build-shared/libvramer.so

# Option B: relink the compiled objects / static lib into a .so by hand.
# (CUDA runtime must be linked; adjust -L/-l for your toolkit.)
g++ -shared -o build/libvramer.so \
    -Wl,--whole-archive build/libvramer.a -Wl,--no-whole-archive \
    -L"${CUDA_HOME:-/usr/local/cuda}/lib64" -lcudart -lpthread
```

Then point the Python side at it:

```bash
export VRAMER_LIB=/vramer/build/libvramer.so
python3 examples/qlora_paged_linear.py
```

The skeleton defaults `VRAMER_LIB` to `build/libvramer.so` and does **not** fail
at import if the library is missing — it raises only when a pager is actually
constructed, so the binding code stays importable for inspection and testing.

---

## 7. A realistic first milestone

1. **Single GPU, base-doesn't-fit demo.** Take a model whose 4-bit base exceeds
   the card (e.g. 70B NF4 ≈ 35 GB on a 24 GB GPU), pack the base into a contiguous
   host buffer, page it through a fixed budget (MRU + prefetch), and run a real
   QLoRA step. **Success metric:** measured step time vs. a baseline that simply
   **OOMs** — i.e. demonstrate training that was otherwise impossible, and report
   the PCIe-bound step-time penalty honestly (`vramer_pager_stats`: hit rate,
   `bytes_h2d`, `stall_ns`).
2. **Resident-fraction sweep.** Vary the budget (100% → 40% of base resident) and
   plot step time vs. resident fraction to quantify the spill curve.
3. **Multi-GPU.** One independent pager per device (VRAMER already supports
   per-GPU caches/streams/staging), each paging its model shard's frozen base.

Only after (1) shows a working, measured train-the-untrainable result is the
plumbing worth generalizing across architectures.
