#!/usr/bin/env python3
"""QLoRA-style weight-paging adapter built on VRAMER's C ABI (reference skeleton).

This file is a *reference skeleton*, not a finished training integration. Its job
is to show, in syntactically valid and importable Python, how a QLoRA fine-tune
would drive VRAMER's C ABI (``include/vramer/vramer_c.h``) to page a frozen,
4-bit quantized base model through a fixed VRAM budget while the small LoRA
adapters, their fp32 optimizer state, and activations stay resident.

Design rationale lives in ``docs/QLORA_ADAPTER.md``. The short version:

* VRAMER is a **read-only** pager (no write-back path yet). QLoRA fits because
  the quantized base is **frozen** (never written) and is read in layer order
  (a predictable scan). Only the LoRA ``A``/``B`` matrices + their optimizer
  state + activations are trainable/resident, and those are tiny.
* Full fine-tuning (mutable weights + grads + Adam offloaded to host) is a
  different, PCIe-bound, write-heavy problem; use DeepSpeed ZeRO-Infinity / FSDP
  for that, not this.

Importability contract:

* ``import qlora_paged_linear`` must succeed **without** PyTorch installed and
  **without** a built ``libvramer.so`` present. The ctypes load is wrapped so it
  raises only on first *use*, and the torch-dependent class is defined only when
  torch imports. Running this module directly (``python3 qlora_paged_linear.py``)
  exercises the ctypes structures and prints a toy page layout.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional, Tuple, Union

# ---------------------------------------------------------------------------
# ctypes binding layer for vramer_c.h
# ---------------------------------------------------------------------------

#: Default shared-library location. NOTE: the project builds a *static*
#: ``libvramer.a`` by default; ctypes.CDLL needs a *shared* object. See
#: docs/QLORA_ADAPTER.md for the cmake/g++ line that produces libvramer.so.
_DEFAULT_LIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "build", "libvramer.so"
)

#: Override with the VRAMER_LIB environment variable.
VRAMER_LIB_PATH = os.environ.get("VRAMER_LIB", _DEFAULT_LIB)

#: Size of the error buffer passed to fallible C ABI calls.
_ERRLEN = 512


# vramer_eviction enum
VRAMER_EVICT_LRU = 0
VRAMER_EVICT_MRU = 1


class VramerConfig(ctypes.Structure):
    """Mirror of ``vramer_config`` in vramer_c.h.

    struct {
        int      gpu;
        uint64_t vram_budget;
        uint64_t page_size;
        unsigned staging_buffers;
        unsigned prefetch_workers;
        int      eviction;   // vramer_eviction
    }
    """

    _fields_ = [
        ("gpu", ctypes.c_int),
        ("vram_budget", ctypes.c_uint64),
        ("page_size", ctypes.c_uint64),
        ("staging_buffers", ctypes.c_uint),
        ("prefetch_workers", ctypes.c_uint),
        ("eviction", ctypes.c_int),
    ]


class VramerStats(ctypes.Structure):
    """Mirror of ``vramer_stats`` in vramer_c.h (all uint64).

    struct {
        uint64_t hits, misses, evictions, bytes_h2d;
        uint64_t prefetches, prefetch_queue_skips;
        uint64_t source_read_ns, stall_ns, lease_sync_fallbacks;
    }
    """

    _fields_ = [
        ("hits", ctypes.c_uint64),
        ("misses", ctypes.c_uint64),
        ("evictions", ctypes.c_uint64),
        ("bytes_h2d", ctypes.c_uint64),
        ("prefetches", ctypes.c_uint64),
        ("prefetch_queue_skips", ctypes.c_uint64),
        ("source_read_ns", ctypes.c_uint64),
        ("stall_ns", ctypes.c_uint64),
        ("lease_sync_fallbacks", ctypes.c_uint64),
    ]


# Opaque handle types. We treat them as void* (c_void_p) at the ctypes layer.
_SourcePtr = ctypes.c_void_p
_PagerPtr = ctypes.c_void_p
_LeasePtr = ctypes.c_void_p


_lib_cache: Optional[ctypes.CDLL] = None
_lib_load_error: Optional[str] = None


def _load_lib() -> ctypes.CDLL:
    """Load libvramer and wire up argtypes/restypes. Cached after first success.

    Raises RuntimeError (not at import time) if the shared library cannot be
    found or loaded. This keeps ``import`` working for inspection/testing even
    when VRAMER has only been built as a static ``.a``.
    """
    global _lib_cache, _lib_load_error
    if _lib_cache is not None:
        return _lib_cache
    try:
        lib = ctypes.CDLL(VRAMER_LIB_PATH)
    except OSError as exc:  # pragma: no cover - depends on build artifacts
        _lib_load_error = str(exc)
        raise RuntimeError(
            "Could not load libvramer shared library from "
            f"{VRAMER_LIB_PATH!r}: {exc}\n"
            "VRAMER builds a static libvramer.a by default; build a shared "
            "object and/or set the VRAMER_LIB environment variable. See "
            "docs/QLORA_ADAPTER.md for the build line."
        ) from exc

    _bind_signatures(lib)
    _lib_cache = lib
    return lib


def _bind_signatures(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restypes for the subset of the C ABI we use."""
    errbuf_t = ctypes.c_char_p  # we pass a mutable buffer via create_string_buffer
    size_t = ctypes.c_size_t

    # --- sources ---
    lib.vramer_source_borrowed.restype = _SourcePtr
    lib.vramer_source_borrowed.argtypes = [
        ctypes.c_void_p,  # const void* base
        ctypes.c_uint64,  # uint64_t bytes
        ctypes.c_char_p,  # const char* label
        ctypes.c_char_p,  # char* errbuf
        size_t,           # size_t errlen
    ]

    lib.vramer_source_file.restype = _SourcePtr
    lib.vramer_source_file.argtypes = [
        ctypes.c_char_p,  # const char* path
        ctypes.c_char_p,  # char* errbuf
        size_t,           # size_t errlen
    ]

    lib.vramer_source_free.restype = None
    lib.vramer_source_free.argtypes = [_SourcePtr]

    # --- pager ---
    lib.vramer_pager_create.restype = _PagerPtr
    lib.vramer_pager_create.argtypes = [
        _SourcePtr,
        ctypes.POINTER(VramerConfig),
        ctypes.c_char_p,
        size_t,
    ]

    lib.vramer_pager_destroy.restype = None
    lib.vramer_pager_destroy.argtypes = [_PagerPtr]

    lib.vramer_pager_page_count.restype = ctypes.c_uint64
    lib.vramer_pager_page_count.argtypes = [_PagerPtr]

    lib.vramer_pager_resident_capacity_pages.restype = ctypes.c_uint64
    lib.vramer_pager_resident_capacity_pages.argtypes = [_PagerPtr]

    lib.vramer_pager_stats.restype = None
    lib.vramer_pager_stats.argtypes = [_PagerPtr, ctypes.POINTER(VramerStats)]

    lib.vramer_pager_prefetch_async.restype = ctypes.c_int
    lib.vramer_pager_prefetch_async.argtypes = [
        _PagerPtr,
        ctypes.c_uint64,
        ctypes.c_char_p,
        size_t,
    ]

    lib.vramer_pager_synchronize.restype = ctypes.c_int
    lib.vramer_pager_synchronize.argtypes = [_PagerPtr, ctypes.c_char_p, size_t]

    # --- lease lifecycle ---
    lib.vramer_acquire.restype = ctypes.c_int
    lib.vramer_acquire.argtypes = [
        _PagerPtr,
        ctypes.c_uint64,                    # page_id
        ctypes.POINTER(_LeasePtr),          # vramer_lease** out_lease
        ctypes.POINTER(ctypes.c_void_p),    # void** out_devptr
        ctypes.POINTER(ctypes.c_size_t),    # size_t* out_bytes
        ctypes.c_char_p,
        size_t,
    ]

    lib.vramer_lease_wait_on.restype = ctypes.c_int
    lib.vramer_lease_wait_on.argtypes = [
        _LeasePtr,
        ctypes.c_void_p,  # void* stream
        ctypes.c_char_p,
        size_t,
    ]

    lib.vramer_lease_release_on.restype = ctypes.c_int
    lib.vramer_lease_release_on.argtypes = [
        _LeasePtr,
        ctypes.c_void_p,  # void* stream
        ctypes.c_char_p,
        size_t,
    ]

    lib.vramer_lease_destroy.restype = None
    lib.vramer_lease_destroy.argtypes = [_LeasePtr]

    # silence unused-name lint; errbuf_t documents intent above
    del errbuf_t


def _errbuf() -> ctypes.Array:
    return ctypes.create_string_buffer(_ERRLEN)


def _errmsg(buf: ctypes.Array) -> str:
    try:
        return buf.value.decode("utf-8", "replace")
    except Exception:  # pragma: no cover
        return "<unknown error>"


# ---------------------------------------------------------------------------
# PagedWeightProvider: thin, torch-free wrapper around a pager handle
# ---------------------------------------------------------------------------


class PagedWeightProvider:
    """Owns a VRAMER source + pager over a frozen, quantized base weight buffer.

    The buffer is laid out as a single contiguous byte range (the concatenation
    of every frozen layer's quantized weights, in layer order). VRAMER slices it
    into fixed-size pages; callers acquire the page(s) covering a given byte range
    and read the resident device pointer.

    This class is deliberately torch-free so it can be unit-tested and reasoned
    about on its own. See PagedFrozenLinear for the torch-side integration.
    """

    def __init__(
        self,
        source_path_or_bytes: Union[str, bytes, bytearray, memoryview, int],
        *,
        vram_budget: int,
        page_size: int,
        gpu: int = 0,
        staging_buffers: int = 2,
        prefetch_workers: int = 1,
        eviction: int = VRAMER_EVICT_MRU,
        total_bytes: Optional[int] = None,
        label: str = "qlora-frozen-base",
    ) -> None:
        self.page_size = int(page_size)
        self.vram_budget = int(vram_budget)
        self._lib = _load_lib()  # raises here if the .so is missing
        self._source = None
        self._pager = None
        # Keep a reference to any borrowed buffer so it is not garbage-collected
        # while VRAMER still points at it. VRAMER does NOT copy borrowed memory.
        self._keepalive = None

        err = _errbuf()
        if isinstance(source_path_or_bytes, str):
            # On-disk layout -> mmap-backed file source.
            self._source = self._lib.vramer_source_file(
                source_path_or_bytes.encode("utf-8"), err, _ERRLEN
            )
        else:
            # In-RAM layout -> borrowed source (zero-copy; we must keep it alive).
            base_ptr, nbytes = _as_c_buffer(source_path_or_bytes, total_bytes)
            self._keepalive = source_path_or_bytes
            self._source = self._lib.vramer_source_borrowed(
                base_ptr, ctypes.c_uint64(nbytes), label.encode("utf-8"),
                err, _ERRLEN,
            )
        if not self._source:
            raise RuntimeError(f"vramer_source_* failed: {_errmsg(err)}")

        cfg = VramerConfig(
            gpu=int(gpu),
            vram_budget=ctypes.c_uint64(self.vram_budget).value,
            page_size=ctypes.c_uint64(self.page_size).value,
            staging_buffers=int(staging_buffers),
            prefetch_workers=int(prefetch_workers),
            eviction=int(eviction),
        )
        err = _errbuf()
        self._pager = self._lib.vramer_pager_create(
            self._source, ctypes.byref(cfg), err, _ERRLEN
        )
        if not self._pager:
            raise RuntimeError(f"vramer_pager_create failed: {_errmsg(err)}")

    # -- page math --------------------------------------------------------

    def page_id_for_offset(self, byte_offset: int) -> int:
        """page_id = layer_byte_offset // page_size (see docs/QLORA_ADAPTER.md)."""
        return int(byte_offset) // self.page_size

    def pages_for_range(self, byte_offset: int, nbytes: int) -> range:
        """Every page_id covering [byte_offset, byte_offset+nbytes).

        A layer's quantized weight may be sub-page (shares a page with its
        neighbours) or multi-page (spans several). Both are handled here; the
        caller must acquire *all* returned pages before launching its kernel.
        """
        first = self.page_id_for_offset(byte_offset)
        last = self.page_id_for_offset(byte_offset + max(int(nbytes), 1) - 1)
        return range(first, last + 1)

    # -- lifecycle --------------------------------------------------------

    def acquire(self, page_id: int) -> Tuple[int, int, int]:
        """Acquire a page. Returns (lease_handle, devptr, nbytes).

        ``lease_handle`` and ``devptr`` are integers (opaque pointer values).
        The lease pins the page against eviction until released.
        """
        lease = _LeasePtr()
        devptr = ctypes.c_void_p()
        nbytes = ctypes.c_size_t()
        err = _errbuf()
        rc = self._lib.vramer_acquire(
            self._pager, ctypes.c_uint64(int(page_id)),
            ctypes.byref(lease), ctypes.byref(devptr), ctypes.byref(nbytes),
            err, _ERRLEN,
        )
        if rc != 0:
            raise RuntimeError(f"vramer_acquire(page={page_id}) failed: {_errmsg(err)}")
        return (lease.value or 0, devptr.value or 0, int(nbytes.value))

    def wait_on(self, lease_handle: int, stream: int = 0) -> None:
        """Order the compute stream after this page's H2D transfer (not a sync)."""
        err = _errbuf()
        rc = self._lib.vramer_lease_wait_on(
            _LeasePtr(lease_handle), ctypes.c_void_p(stream), err, _ERRLEN
        )
        if rc != 0:
            raise RuntimeError(f"vramer_lease_wait_on failed: {_errmsg(err)}")

    def release(self, lease_handle: int, stream: int = 0) -> None:
        """Release (consumes the lease) so the page may be evicted after `stream`."""
        err = _errbuf()
        rc = self._lib.vramer_lease_release_on(
            _LeasePtr(lease_handle), ctypes.c_void_p(stream), err, _ERRLEN
        )
        if rc != 0:
            raise RuntimeError(f"vramer_lease_release_on failed: {_errmsg(err)}")

    def prefetch(self, page_id: int) -> None:
        """Hint VRAMER to start loading `page_id` in the background."""
        err = _errbuf()
        rc = self._lib.vramer_pager_prefetch_async(
            self._pager, ctypes.c_uint64(int(page_id)), err, _ERRLEN
        )
        if rc != 0:
            raise RuntimeError(f"vramer_pager_prefetch_async failed: {_errmsg(err)}")

    def page_count(self) -> int:
        return int(self._lib.vramer_pager_page_count(self._pager))

    def resident_capacity_pages(self) -> int:
        return int(self._lib.vramer_pager_resident_capacity_pages(self._pager))

    def stats(self) -> VramerStats:
        out = VramerStats()
        self._lib.vramer_pager_stats(self._pager, ctypes.byref(out))
        return out

    def close(self) -> None:
        if self._pager:
            self._lib.vramer_pager_destroy(self._pager)
            self._pager = None
        if self._source:
            self._lib.vramer_source_free(self._source)
            self._source = None
        self._keepalive = None

    def __enter__(self) -> "PagedWeightProvider":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:  # pragma: no cover
            pass


def _as_c_buffer(
    buf: Union[bytes, bytearray, memoryview, int], total_bytes: Optional[int]
) -> Tuple[ctypes.c_void_p, int]:
    """Return (base_pointer, nbytes) for a Python buffer or raw address int."""
    if isinstance(buf, int):
        if total_bytes is None:
            raise ValueError("total_bytes is required when passing a raw address")
        return ctypes.c_void_p(buf), int(total_bytes)
    if isinstance(buf, (bytes, bytearray, memoryview)):
        mv = memoryview(buf)
        nbytes = total_bytes if total_bytes is not None else mv.nbytes
        base = (ctypes.c_char * mv.nbytes).from_buffer(
            bytearray(mv) if isinstance(buf, (bytes, memoryview)) else buf
        )
        return ctypes.cast(base, ctypes.c_void_p), int(nbytes)
    raise TypeError(f"Unsupported buffer type: {type(buf)!r}")


# ---------------------------------------------------------------------------
# PagedFrozenLinear: torch-side drop-in for the frozen 4-bit base linear
# ---------------------------------------------------------------------------
#
# Guarded so the module imports without torch. The real integration subclasses
# nn.Module and intercepts the frozen base's Linear layers (bitsandbytes
# Linear4bit / HF transformers linears).

try:  # pragma: no cover - exercised only when torch is installed
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # ImportError or a broken torch build
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


if _HAS_TORCH:

    class PagedFrozenLinear(nn.Module):  # type: ignore[misc]
        """Drop-in for a frozen 4-bit base Linear, with a resident LoRA adapter.

        The frozen quantized weight for this layer lives at a known byte offset
        inside the provider's backing buffer; its bytes are paged in on demand.
        Only the LoRA ``A``/``B`` matrices are trainable and resident, so no
        gradient ever touches the paged base -> VRAMER's read-only model is
        sufficient (no write-back needed).
        """

        def __init__(
            self,
            provider: PagedWeightProvider,
            weight_byte_offset: int,
            weight_nbytes: int,
            in_features: int,
            out_features: int,
            *,
            r: int = 16,
            lora_alpha: float = 32.0,
            bias: bool = False,
            quant_meta: Optional[dict] = None,
        ) -> None:
            super().__init__()
            self.provider = provider
            self.weight_byte_offset = int(weight_byte_offset)
            self.weight_nbytes = int(weight_nbytes)
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.r = int(r)
            self.scaling = float(lora_alpha) / float(r) if r else 1.0
            # Quantization metadata (blocksize, absmax/scale tensors, dtype)
            # needed by the real dequant kernel. Kept resident; it is tiny.
            self.quant_meta = quant_meta or {}

            # --- LoRA adapters: resident + trainable ---
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            # lora_B stays zero at init so the adapter is a no-op on step 0.

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter("bias", None)

        def forward(self, x):  # noqa: D401
            # ---- 1. Page in the frozen quantized base weight for this layer ----
            pages = self.provider.pages_for_range(
                self.weight_byte_offset, self.weight_nbytes
            )
            # Compute stream as an integer handle. In a real integration this is
            # torch.cuda.current_stream().cuda_stream (a cudaStream_t value).
            stream = 0
            # stream = torch.cuda.current_stream(x.device).cuda_stream

            leases = []
            try:
                # Prefetch the NEXT layer's first page while we work (pipeline).
                # next_first_page = self.provider.page_id_for_offset(
                #     self.weight_byte_offset + self.weight_nbytes)
                # self.provider.prefetch(next_first_page)

                dev_segments = []
                for pid in pages:
                    lease, devptr, nbytes = self.provider.acquire(pid)
                    # Order compute after the H2D copy for this page.
                    self.provider.wait_on(lease, stream)
                    leases.append(lease)
                    dev_segments.append((devptr, nbytes))

                # ---- 2. Dequant + matmul reading from the device pointer(s) ----
                #
                # REAL KERNEL GOES HERE. Given `dev_segments` (contiguous device
                # bytes of this layer's 4-bit weight) plus `self.quant_meta`, a
                # custom/bitsandbytes kernel dequantizes to the compute dtype and
                # does `base_out = x @ dequant(W)^T`. The kernel reads directly
                # from VRAMER's device pointer -- the weight is NEVER copied back
                # to a torch tensor and NEVER written.
                #
                # Example shape of the real call (pseudocode):
                #   base_out = paged_dequant_matmul(
                #       x, dev_segments, self.quant_meta,
                #       self.in_features, self.out_features)
                #
                # Skeleton placeholder (keeps the graph valid without a kernel):
                base_out = x.new_zeros(*x.shape[:-1], self.out_features)

            finally:
                # ---- 3. Release leases so these pages can be evicted ----
                for lease in leases:
                    self.provider.release(lease, stream)

            # ---- 4. Resident, trainable LoRA path: y = base + (x A^T) B^T * s ---
            lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
            out = base_out + lora_out * self.scaling
            if self.bias is not None:
                out = out + self.bias
            return out

else:

    class PagedFrozenLinear:  # type: ignore[no-redef]
        """Placeholder when torch is unavailable; instantiation requires torch."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PagedFrozenLinear requires PyTorch. Install torch to use the "
                "training integration; the ctypes binding layer "
                "(PagedWeightProvider) works without it."
            )


# ---------------------------------------------------------------------------
# __main__: exercise the binding code without torch or a built .so
# ---------------------------------------------------------------------------


def _demo_page_layout() -> None:
    """Construct the ctypes structs and print a toy frozen-base page layout."""
    print("VRAMER QLoRA paged-linear skeleton -- binding self-check")
    print("=" * 60)
    print(f"VRAMER_LIB (env VRAMER_LIB or default): {VRAMER_LIB_PATH}")
    print(f"PyTorch available: {_HAS_TORCH}")
    print()

    # Toy config: a 70B-ish model, 4-bit base, paged through an 8 GiB budget.
    page_size = 64 * 1024 * 1024          # 64 MiB (A2 sweet spot)
    vram_budget = 8 * 1024 * 1024 * 1024  # 8 GiB fixed budget

    cfg = VramerConfig(
        gpu=0,
        vram_budget=vram_budget,
        page_size=page_size,
        staging_buffers=2,
        prefetch_workers=1,
        eviction=VRAMER_EVICT_MRU,
    )
    print("VramerConfig (sizeof=%d bytes):" % ctypes.sizeof(cfg))
    print(f"  gpu={cfg.gpu} vram_budget={cfg.vram_budget} "
          f"page_size={cfg.page_size}")
    print(f"  staging_buffers={cfg.staging_buffers} "
          f"prefetch_workers={cfg.prefetch_workers} eviction={cfg.eviction}")
    print()

    stats = VramerStats()
    print("VramerStats (sizeof=%d bytes) zero-initialized: hits=%d misses=%d"
          % (ctypes.sizeof(stats), stats.hits, stats.misses))
    print()

    # Toy frozen-base layout: 8 layers, each a quantized weight block.
    # (Numbers illustrative; a real layout comes from the checkpoint.)
    per_layer_bytes = 90 * 1024 * 1024  # ~90 MiB of 4-bit weight per layer
    num_layers = 8
    total_bytes = per_layer_bytes * num_layers

    resident_pages = vram_budget // page_size
    total_pages = (total_bytes + page_size - 1) // page_size
    print(f"Frozen base: {num_layers} layers x {per_layer_bytes // (1024*1024)} "
          f"MiB = {total_bytes // (1024*1024)} MiB total")
    print(f"page_size = {page_size // (1024*1024)} MiB -> {total_pages} pages; "
          f"budget {vram_budget // (1024*1024*1024)} GiB "
          f"-> {resident_pages} pages resident")
    print()
    print("Per-layer page mapping (page_id = byte_offset // page_size):")
    print("  layer   byte_offset      first_page  last_page  #pages")
    for layer in range(num_layers):
        off = layer * per_layer_bytes
        first = off // page_size
        last = (off + per_layer_bytes - 1) // page_size
        print(f"  {layer:5d}   {off:13d}   {first:9d}  {last:9d}  "
              f"{last - first + 1:6d}")
    print()
    print("Forward scans layers 0..N, backward N..0 -> predictable; use MRU "
          "eviction + prefetch of the next layer. Backward re-reads the same "
          "frozen weights (served from cache when the resident fraction allows).")
    print()
    print("NOTE: full execution needs (1) a built shared libvramer.so, "
          "(2) PyTorch + CUDA, and (3) a real quantized model + dequant kernel. "
          "This self-check only exercises the ctypes binding code.")


if __name__ == "__main__":
    _demo_page_layout()
