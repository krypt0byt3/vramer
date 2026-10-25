"""Pythonic wrapper around the VRAMER C ABI pager.

This is the high-level, object-oriented surface over ``vramer_c.h``. It wraps the
acquire / wait_on / release_on lease lifecycle in context managers and turns
non-zero C return codes into Python exceptions carrying the C error message.

Importing this module does not require the compiled library; the library is
loaded lazily the first time a :class:`Pager` is constructed. Example::

    import numpy as np, ctypes
    from vramer import BorrowedSource, Pager, EvictionPolicy

    buf = np.zeros(64 << 20, dtype=np.uint8)
    ptr = buf.ctypes.data_as(ctypes.c_void_p).value
    src = BorrowedSource(ptr, buf.nbytes, keepalive=buf, label="demo")

    with Pager(src, gpu=0, vram_budget=16 << 20, page_size=4 << 20) as pager:
        for pid in range(pager.page_count()):
            with pager.acquire(pid) as lease:
                lease.wait_on(0)          # order compute stream after H2D
                # ... launch a kernel reading lease.device_ptr ...
        print(pager.stats())
"""

from __future__ import annotations

import ctypes
import enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from . import _lib
from ._lib import VramerConfig, VramerStats

__all__ = [
    "EvictionPolicy",
    "VramerError",
    "RamSource",
    "FileSource",
    "BorrowedSource",
    "Pager",
    "Lease",
]


class VramerError(RuntimeError):
    """Raised when a VRAMER C ABI call fails, carrying the C errbuf message."""


class EvictionPolicy(enum.IntEnum):
    """Page eviction policy (mirror of ``vramer_eviction``)."""

    LRU = _lib.VRAMER_EVICT_LRU
    MRU = _lib.VRAMER_EVICT_MRU


# ---------------------------------------------------------------------------
# Sources -- thin handles over vramer_source_{ram,file,borrowed}
# ---------------------------------------------------------------------------

class _Source:
    """Base class for a memory source. Holds construction params; the C handle
    is created lazily by :class:`Pager` and freed once the pager owns it."""

    def _create(self, lib: ctypes.CDLL) -> ctypes.c_void_p:  # pragma: no cover
        raise NotImplementedError

    # Keepalive (buffers that must outlive the pager); default none.
    keepalive: Any = None


class RamSource(_Source):
    """An anonymous host RAM tier of ``nbytes`` bytes (``vramer_source_ram``).

    VRAMER allocates and owns the host buffer. ``numa_node`` is accepted for API
    forward-compat but the current C ABI has no NUMA argument, so it is a no-op.
    """

    def __init__(self, nbytes: int, numa_node: int = -1):
        self.nbytes = int(nbytes)
        self.numa_node = int(numa_node)

    def _create(self, lib: ctypes.CDLL) -> ctypes.c_void_p:
        err = _lib.errbuf()
        src = lib.vramer_source_ram(ctypes.c_uint64(self.nbytes), err, _lib.ERRLEN)
        if not src:
            raise VramerError(f"vramer_source_ram failed: {_lib.errmsg(err)}")
        return ctypes.c_void_p(src)


class PinnedSource(_Source):
    """A page-locked (pinned) host tier of ``nbytes`` (``vramer_source_pinned``).

    DMA'd straight to the GPU (direct-DMA fast path, no staging copy). Pinned
    memory is scarce -- use it for a hot subset, not an entire large model; for
    the general case use :class:`RamSource`.
    """

    def __init__(self, nbytes: int):
        self.nbytes = int(nbytes)

    def _create(self, lib: ctypes.CDLL) -> ctypes.c_void_p:
        err = _lib.errbuf()
        src = lib.vramer_source_pinned(ctypes.c_uint64(self.nbytes), err, _lib.ERRLEN)
        if not src:
            raise VramerError(f"vramer_source_pinned failed: {_lib.errmsg(err)}")
        return ctypes.c_void_p(src)


class FileSource(_Source):
    """An mmap-backed file tier (``vramer_source_file``)."""

    def __init__(self, path: Union[str, Path]):
        self.path = str(path)

    def _create(self, lib: ctypes.CDLL) -> ctypes.c_void_p:
        err = _lib.errbuf()
        src = lib.vramer_source_file(self.path.encode("utf-8"), err, _lib.ERRLEN)
        if not src:
            raise VramerError(
                f"vramer_source_file({self.path!r}) failed: {_lib.errmsg(err)}"
            )
        return ctypes.c_void_p(src)


class BorrowedSource(_Source):
    """A zero-copy view over host memory you own (``vramer_source_borrowed``).

    ``ptr`` is the integer base address, ``nbytes`` the length. VRAMER does NOT
    copy the buffer, so it must stay alive and at a fixed address for the whole
    lifetime of the pager -- pass the owning Python object as ``keepalive`` and
    this class holds a reference to it.

    Set ``pinned=True`` when the borrowed region is page-locked (e.g. a pinned
    PyTorch tensor) to enable the direct-DMA fast path (no staging copy).
    """

    def __init__(
        self,
        ptr: int,
        nbytes: int,
        pinned: bool = False,
        *,
        label: str = "borrowed",
        keepalive: Any = None,
    ):
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)
        self.pinned = bool(pinned)
        self.label = label if not pinned else f"{label}(pinned)"
        self.keepalive = keepalive

    def _create(self, lib: ctypes.CDLL) -> ctypes.c_void_p:
        err = _lib.errbuf()
        src = lib.vramer_source_borrowed(
            ctypes.c_void_p(self.ptr), ctypes.c_uint64(self.nbytes),
            self.label.encode("utf-8"), ctypes.c_int(1 if self.pinned else 0),
            err, _lib.ERRLEN,
        )
        if not src:
            raise VramerError(f"vramer_source_borrowed failed: {_lib.errmsg(err)}")
        return ctypes.c_void_p(src)


# ---------------------------------------------------------------------------
# Lease
# ---------------------------------------------------------------------------

class Lease:
    """A pinned, resident page. Keeps the page from being evicted until released.

    Use as a context manager; on exit the lease is released on stream 0 if it has
    not already been released. ``device_ptr`` is the integer CUDA device address
    of the resident page bytes.
    """

    def __init__(self, pager: "Pager", handle: int, device_ptr: int, nbytes: int):
        self._pager = pager
        self._handle = handle
        self.device_ptr = int(device_ptr)
        self.nbytes = int(nbytes)
        self._released = False

    def wait_on(self, stream: int = 0) -> None:
        """Order ``stream`` after this page's H2D transfer (not a host sync)."""
        if self._released:
            raise VramerError("Lease already released")
        lib = self._pager._lib
        err = _lib.errbuf()
        rc = lib.vramer_lease_wait_on(
            ctypes.c_void_p(self._handle), ctypes.c_void_p(stream), err, _lib.ERRLEN
        )
        if rc != 0:
            raise VramerError(f"vramer_lease_wait_on failed: {_lib.errmsg(err)}")

    def release_on(self, stream: int = 0) -> None:
        """Release the page (consumes the lease) so it may be evicted after
        ``stream`` completes. Idempotent: a second call is a no-op."""
        if self._released:
            return
        lib = self._pager._lib
        err = _lib.errbuf()
        rc = lib.vramer_lease_release_on(
            ctypes.c_void_p(self._handle), ctypes.c_void_p(stream), err, _lib.ERRLEN
        )
        # The lease handle is consumed regardless of rc.
        self._released = True
        if rc != 0:
            raise VramerError(f"vramer_lease_release_on failed: {_lib.errmsg(err)}")

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc) -> None:
        self.release_on(0)


# ---------------------------------------------------------------------------
# Pager
# ---------------------------------------------------------------------------

class Pager:
    """A CUDA pager over a :class:`_Source`, paging it through a fixed VRAM budget.

    Constructor keyword arguments mirror ``vramer_config``. Use as a context
    manager so the underlying pager is destroyed deterministically.
    """

    def __init__(
        self,
        source: _Source,
        *,
        gpu: int = 0,
        vram_budget: int,
        page_size: int = 64 << 20,
        staging_buffers: int = 2,
        prefetch_workers: int = 0,
        eviction: Union[EvictionPolicy, int] = EvictionPolicy.MRU,
    ):
        if not isinstance(source, _Source):
            raise TypeError(
                "source must be a RamSource, FileSource, or BorrowedSource"
            )
        self._lib = _lib.load_lib()
        self._pager: Optional[ctypes.c_void_p] = None
        # Hold the source's keepalive buffer (if any) for the pager's lifetime.
        self._keepalive = getattr(source, "keepalive", None)

        self.gpu = int(gpu)
        self.vram_budget = int(vram_budget)
        self.page_size = int(page_size)

        # Create the C source handle, build the pager, then free the source
        # handle (the pager holds its own reference).
        src_handle = source._create(self._lib)
        try:
            cfg = VramerConfig(
                gpu=int(gpu),
                vram_budget=ctypes.c_uint64(int(vram_budget)).value,
                page_size=ctypes.c_uint64(int(page_size)).value,
                staging_buffers=int(staging_buffers),
                prefetch_workers=int(prefetch_workers),
                eviction=int(eviction),
            )
            err = _lib.errbuf()
            pager = self._lib.vramer_pager_create(
                src_handle, ctypes.byref(cfg), err, _lib.ERRLEN
            )
            if not pager:
                raise VramerError(
                    f"vramer_pager_create failed: {_lib.errmsg(err)}"
                )
            self._pager = ctypes.c_void_p(pager)
        finally:
            self._lib.vramer_source_free(src_handle)

    # -- introspection ----------------------------------------------------

    def _check_open(self) -> None:
        if self._pager is None:
            raise VramerError("Pager is closed")

    def page_count(self) -> int:
        self._check_open()
        return int(self._lib.vramer_pager_page_count(self._pager))

    def resident_capacity_pages(self) -> int:
        self._check_open()
        return int(self._lib.vramer_pager_resident_capacity_pages(self._pager))

    def stats(self) -> Dict[str, int]:
        self._check_open()
        out = VramerStats()
        self._lib.vramer_pager_stats(self._pager, ctypes.byref(out))
        return out.as_dict()

    # -- lease lifecycle --------------------------------------------------

    def acquire(self, page_id: int) -> Lease:
        """Acquire (pin + make resident) a page; returns a :class:`Lease`."""
        self._check_open()
        lease = ctypes.c_void_p()
        devptr = ctypes.c_void_p()
        nbytes = ctypes.c_size_t()
        err = _lib.errbuf()
        rc = self._lib.vramer_acquire(
            self._pager, ctypes.c_uint64(int(page_id)),
            ctypes.byref(lease), ctypes.byref(devptr), ctypes.byref(nbytes),
            err, _lib.ERRLEN,
        )
        if rc != 0:
            raise VramerError(
                f"vramer_acquire(page={page_id}) failed: {_lib.errmsg(err)}"
            )
        return Lease(self, lease.value or 0, devptr.value or 0, int(nbytes.value))

    def prefetch_async(self, page_id: int) -> None:
        """Hint VRAMER to start loading ``page_id`` in the background."""
        self._check_open()
        err = _lib.errbuf()
        rc = self._lib.vramer_pager_prefetch_async(
            self._pager, ctypes.c_uint64(int(page_id)), err, _lib.ERRLEN
        )
        if rc != 0:
            raise VramerError(
                f"vramer_pager_prefetch_async(page={page_id}) failed: "
                f"{_lib.errmsg(err)}"
            )

    def drain_prefetch_queue(self) -> None:
        self._check_open()
        err = _lib.errbuf()
        rc = self._lib.vramer_pager_drain_prefetch_queue(self._pager, err, _lib.ERRLEN)
        if rc != 0:
            raise VramerError(
                f"vramer_pager_drain_prefetch_queue failed: {_lib.errmsg(err)}"
            )

    def synchronize(self) -> None:
        """Block until all outstanding pager work has completed."""
        self._check_open()
        err = _lib.errbuf()
        rc = self._lib.vramer_pager_synchronize(self._pager, err, _lib.ERRLEN)
        if rc != 0:
            raise VramerError(
                f"vramer_pager_synchronize failed: {_lib.errmsg(err)}"
            )

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self._pager is not None:
            self._lib.vramer_pager_destroy(self._pager)
            self._pager = None
        self._keepalive = None

    def __enter__(self) -> "Pager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
