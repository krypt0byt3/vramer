"""Locate, load, and bind the VRAMER C ABI shared library (``libvramer.so``).

This module defines the ctypes ``Structure`` mirrors of ``vramer_c.h`` and a
lazy loader. Importing this module (and the whole ``vramer`` package) NEVER
requires the shared library to be present -- the planner works lib-free. The
loader only runs, and only raises, when the pager is actually used.

Library search order (first hit wins):
  1. ``$VRAMER_LIB``                          -- explicit override (file path)
  2. ``$VRAMER_LIB_DIR``                      -- a directory to search
  3. common build locations relative to the installed package and CWD
  4. the system loader (bare ``libvramer.so``, honouring LD_LIBRARY_PATH)
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Optional

# --- vramer_eviction enum (mirror of vramer_c.h) ---------------------------
VRAMER_EVICT_LRU = 0
VRAMER_EVICT_MRU = 1

#: Error buffer size passed to fallible C ABI calls.
ERRLEN = 512

_LIB_NAME = "libvramer.so"


class VramerConfig(ctypes.Structure):
    """Mirror of ``vramer_config`` in vramer_c.h.

    ::

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

    ::

        struct {
            uint64_t hits, misses, evictions, bytes_h2d;
            uint64_t prefetches, prefetch_queue_skips;
            uint64_t source_read_ns, stall_ns, lease_sync_fallbacks;
            uint64_t bytes_d2h, direct_loads;
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
        ("bytes_d2h", ctypes.c_uint64),
        ("direct_loads", ctypes.c_uint64),
    ]

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name, _ in self._fields_}


# Opaque handle types -- void* at the ctypes layer.
_SourcePtr = ctypes.c_void_p
_PagerPtr = ctypes.c_void_p
_LeasePtr = ctypes.c_void_p


def _candidate_paths() -> List[Path]:
    """Build the ordered list of candidate shared-library paths."""
    cands: List[Path] = []

    env_lib = os.environ.get("VRAMER_LIB")
    if env_lib:
        cands.append(Path(env_lib))

    env_dir = os.environ.get("VRAMER_LIB_DIR")
    if env_dir:
        cands.append(Path(env_dir) / _LIB_NAME)

    # Relative to the installed package: walk up looking for a build/ dir.
    here = Path(__file__).resolve().parent
    roots = [here, *here.parents]
    rel = [
        Path("libvramer.so"),
        Path("build") / _LIB_NAME,
        Path("v1") / "VRAMER_v1" / "build" / _LIB_NAME,
        Path("VRAMER_v1") / "build" / _LIB_NAME,
    ]
    for root in roots:
        for r in rel:
            cands.append(root / r)

    # Relative to the current working directory.
    cwd = Path.cwd()
    for r in rel:
        cands.append(cwd / r)

    # Common system locations.
    for sysdir in ("/usr/local/lib", "/usr/lib", "/usr/lib64"):
        cands.append(Path(sysdir) / _LIB_NAME)

    return cands


def _find_lib() -> Optional[str]:
    """Return the first existing candidate path, else None."""
    seen = set()
    for c in _candidate_paths():
        try:
            rc = c.resolve()
        except Exception:
            rc = c
        if rc in seen:
            continue
        seen.add(rc)
        if c.is_file():
            return str(c)
    return None


_lib_cache: Optional[ctypes.CDLL] = None


def load_lib() -> ctypes.CDLL:
    """Load ``libvramer.so`` and bind argtypes/restypes. Cached after success.

    Raises a clear, actionable :class:`RuntimeError` when the library cannot be
    found or loaded. This is only ever called from pager code paths, so simply
    importing ``vramer`` never requires the compiled library.
    """
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache

    path = _find_lib()
    tried = "\n  ".join(dict.fromkeys(str(p) for p in _candidate_paths()))

    if path is None:
        # Last resort: let the system loader try a bare name (LD_LIBRARY_PATH).
        try:
            lib = ctypes.CDLL(_LIB_NAME)
        except OSError:
            raise RuntimeError(
                "Could not locate the VRAMER shared library "
                f"({_LIB_NAME}).\n\n"
                "The compiled CUDA library is NOT shipped in the wheel; build "
                "it from the C++ project and point VRAMER at it:\n\n"
                "  cd v1/VRAMER_v1\n"
                "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "
                "-DCMAKE_CUDA_ARCHITECTURES=86\n"
                "  cmake --build build -j\n\n"
                "Then set VRAMER_LIB to the built .so, e.g.:\n"
                "  export VRAMER_LIB=/abs/path/to/v1/VRAMER_v1/build/"
                f"{_LIB_NAME}\n\n"
                f"Searched these paths:\n  {tried}"
            )
    else:
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:
            raise RuntimeError(
                f"Found {path!r} but failed to load it: {exc}\n"
                "This usually means the CUDA runtime it links against is not "
                "available (no GPU / driver, or a CUDA version mismatch)."
            ) from exc

    _bind_signatures(lib)
    _lib_cache = lib
    return lib


def _bind_signatures(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restypes for the full C ABI surface we use."""
    c_char_p = ctypes.c_char_p
    size_t = ctypes.c_size_t
    E = [c_char_p, size_t]  # (errbuf, errlen)

    # --- sources ---
    lib.vramer_source_ram.restype = _SourcePtr
    lib.vramer_source_ram.argtypes = [ctypes.c_uint64, *E]

    lib.vramer_source_pinned.restype = _SourcePtr
    lib.vramer_source_pinned.argtypes = [ctypes.c_uint64, *E]

    lib.vramer_source_file.restype = _SourcePtr
    lib.vramer_source_file.argtypes = [c_char_p, *E]

    lib.vramer_source_borrowed.restype = _SourcePtr
    lib.vramer_source_borrowed.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, c_char_p, ctypes.c_int, *E
    ]

    lib.vramer_source_free.restype = None
    lib.vramer_source_free.argtypes = [_SourcePtr]

    # --- pager ---
    lib.vramer_pager_create.restype = _PagerPtr
    lib.vramer_pager_create.argtypes = [
        _SourcePtr, ctypes.POINTER(VramerConfig), *E
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
    lib.vramer_pager_prefetch_async.argtypes = [_PagerPtr, ctypes.c_uint64, *E]

    lib.vramer_pager_drain_prefetch_queue.restype = ctypes.c_int
    lib.vramer_pager_drain_prefetch_queue.argtypes = [_PagerPtr, *E]

    lib.vramer_pager_synchronize.restype = ctypes.c_int
    lib.vramer_pager_synchronize.argtypes = [_PagerPtr, *E]

    # --- lease lifecycle ---
    lib.vramer_acquire.restype = ctypes.c_int
    lib.vramer_acquire.argtypes = [
        _PagerPtr,
        ctypes.c_uint64,                    # page_id
        ctypes.POINTER(_LeasePtr),          # vramer_lease** out_lease
        ctypes.POINTER(ctypes.c_void_p),    # void** out_devptr
        ctypes.POINTER(ctypes.c_size_t),    # size_t* out_bytes
        *E,
    ]

    lib.vramer_lease_wait_on.restype = ctypes.c_int
    lib.vramer_lease_wait_on.argtypes = [_LeasePtr, ctypes.c_void_p, *E]

    lib.vramer_lease_release_on.restype = ctypes.c_int
    lib.vramer_lease_release_on.argtypes = [_LeasePtr, ctypes.c_void_p, *E]

    lib.vramer_lease_destroy.restype = None
    lib.vramer_lease_destroy.argtypes = [_LeasePtr]


def errbuf() -> ctypes.Array:
    return ctypes.create_string_buffer(ERRLEN)


def errmsg(buf: ctypes.Array) -> str:
    try:
        return buf.value.decode("utf-8", "replace")
    except Exception:
        return "<unknown error>"
