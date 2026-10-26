#!/usr/bin/env python3
"""Minimal, RUNNABLE Python demo of the VRAMER C ABI (no PyTorch needed).

Proves the full Python -> ctypes -> C ABI -> pager -> GPU path with real data:
it wraps a NumPy array as a `vramer_source_borrowed` host tier, pages it through
a small VRAM budget, and verifies via the pager's own stats that every byte was
transferred H2D. (Reading the device pointer back into Python would need cupy or
torch; here we confirm the transfer through `bytes_h2d`, which is sufficient to
prove the data path works end to end.)

Run:  python3 examples/python_paging_demo.py
Needs: numpy, a built libvramer.so (cmake --build build => build/libvramer.so).
"""
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.environ.get("VRAMER_LIB", os.path.join(HERE, "..", "build", "libvramer.so"))


class VramerConfig(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_int),
        ("vram_budget", ctypes.c_uint64),
        ("page_size", ctypes.c_uint64),
        ("staging_buffers", ctypes.c_uint),
        ("prefetch_workers", ctypes.c_uint),
        ("eviction", ctypes.c_int),
    ]


class VramerStats(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "hits", "misses", "evictions", "bytes_h2d", "prefetches",
        "prefetch_queue_skips", "source_read_ns", "stall_ns",
        "lease_sync_fallbacks", "bytes_d2h")]


def load_lib(path):
    lib = ctypes.CDLL(path)
    E = [ctypes.c_char_p, ctypes.c_size_t]  # (errbuf, errlen)
    lib.vramer_source_borrowed.restype = ctypes.c_void_p
    lib.vramer_source_borrowed.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_int, *E]
    lib.vramer_source_free.argtypes = [ctypes.c_void_p]
    lib.vramer_pager_create.restype = ctypes.c_void_p
    lib.vramer_pager_create.argtypes = [ctypes.c_void_p, ctypes.POINTER(VramerConfig), *E]
    lib.vramer_pager_destroy.argtypes = [ctypes.c_void_p]
    lib.vramer_pager_page_count.restype = ctypes.c_uint64
    lib.vramer_pager_page_count.argtypes = [ctypes.c_void_p]
    lib.vramer_pager_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(VramerStats)]
    lib.vramer_pager_synchronize.argtypes = [ctypes.c_void_p, *E]
    lib.vramer_acquire.restype = ctypes.c_int
    lib.vramer_acquire.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), *E]
    lib.vramer_lease_wait_on.restype = ctypes.c_int
    lib.vramer_lease_wait_on.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *E]
    lib.vramer_lease_release_on.restype = ctypes.c_int
    lib.vramer_lease_release_on.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *E]
    return lib


def main():
    try:
        lib = load_lib(LIB)
    except OSError as e:
        print(f"could not load {LIB}: {e}\nBuild it: cmake --build build (produces build/libvramer.so)")
        return 1

    err = ctypes.create_string_buffer(256)

    # A 64 MiB host buffer with a known per-page pattern, exposed as a borrowed source.
    PAGE = 4 << 20            # 4 MiB pages
    NBYTES = 64 << 20         # 64 MiB source
    BUDGET = 16 << 20         # 16 MiB VRAM budget -> 4 slots, source is 4x bigger
    buf = np.zeros(NBYTES, dtype=np.uint8)
    for p in range(NBYTES // PAGE):
        buf[p * PAGE] = (p * 37 + 101) & 0xFF   # marker byte per page

    base = buf.ctypes.data_as(ctypes.c_void_p)
    src = lib.vramer_source_borrowed(base, NBYTES, b"numpy-demo", 0, err, 256)
    if not src:
        print("source_borrowed failed:", err.value.decode()); return 1

    cfg = VramerConfig(gpu=0, vram_budget=BUDGET, page_size=PAGE,
                       staging_buffers=2, prefetch_workers=1, eviction=1)  # 1 = MRU
    pager = lib.vramer_pager_create(src, ctypes.byref(cfg), err, 256)
    if not pager:
        print("pager_create failed:", err.value.decode())
        lib.vramer_source_free(src); return 1
    lib.vramer_source_free(src)  # pager holds its own reference

    pages = lib.vramer_pager_page_count(pager)
    print(f"source={NBYTES>>20} MiB  budget={BUDGET>>20} MiB  page={PAGE>>20} MiB  pages={pages}")

    # Page the whole source through the small budget, like a one-token layer scan.
    for p in range(pages):
        lease = ctypes.c_void_p()
        devptr = ctypes.c_void_p()
        nbytes = ctypes.c_size_t()
        rc = lib.vramer_acquire(pager, p, ctypes.byref(lease), ctypes.byref(devptr),
                                ctypes.byref(nbytes), err, 256)
        if rc != 0:
            print(f"acquire({p}) failed:", err.value.decode()); return 1
        lib.vramer_lease_wait_on(lease, None, err, 256)       # stream 0 waits for H2D
        # (a real consumer would launch a kernel reading devptr.value here)
        lib.vramer_lease_release_on(lease, None, err, 256)    # consumes the lease

    lib.vramer_pager_synchronize(pager, err, 256)
    st = VramerStats()
    lib.vramer_pager_stats(pager, ctypes.byref(st))
    print(f"hits={st.hits} misses={st.misses} evictions={st.evictions} "
          f"bytes_h2d={st.bytes_h2d} ({st.bytes_h2d>>20} MiB)")

    ok = st.bytes_h2d == NBYTES and st.misses == pages
    lib.vramer_pager_destroy(pager)
    if ok:
        print("python paging demo OK: NumPy -> borrowed source -> pager -> GPU verified")
        return 0
    print("FAIL: unexpected stats"); return 1


if __name__ == "__main__":
    sys.exit(main())
