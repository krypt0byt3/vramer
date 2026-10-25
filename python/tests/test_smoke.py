"""Smoke tests for the vramer package.

* The planner tests run anywhere -- no GPU and no compiled library.
* The pager test runs only if ``libvramer.so`` can be loaded AND a CUDA device
  is present; otherwise it is skipped.
"""

import ctypes
import os

import pytest

import vramer
from vramer import plan, score


# ---------------------------------------------------------------------------
# Planner / analytical model (must pass anywhere)
# ---------------------------------------------------------------------------

def test_version():
    assert vramer.__version__ == "0.2.0"


def test_bundled_profiles_present():
    names = plan.bundled_profiles()
    assert "qwen14b" in names
    assert "current-machine" in names


def test_estimate_on_bundled_profile():
    # 14.77B params at q4 (0.5 B/param) ~= 6.88 GiB of weights; with no budget
    # the full two-A2 VRAM is resident, so a 14B model FITS.
    est = plan.estimate("current-machine", "qwen14b", quant="q4", batch=1)
    assert est["model"] == "Qwen2.5-14B-Instruct"
    assert est["weight_bytes"] == pytest.approx(14.77e9 * 0.5, rel=1e-3)
    assert est["fits"] is True
    assert est["verdict"] == "FITS"


def test_estimate_spill_is_pcie_bound():
    # Force a tiny resident budget so the model spills and must stream.
    est = plan.estimate(
        "current-machine", "qwen14b",
        quant="q4", batch=1, vram_budget="1GiB",
    )
    assert est["fits"] is False
    assert est["spill_bytes"] > 0
    assert est["decode_tok_s"] is not None and est["decode_tok_s"] > 0
    assert "PCIe-bound" in est["verdict"]
    # Aggregate throughput scales with batch; per-seq latency does not.
    est8 = plan.estimate(
        "current-machine", "qwen14b",
        quant="q4", batch=8, vram_budget="1GiB",
    )
    assert est8["decode_tok_s"] == pytest.approx(est["decode_tok_s"] * 8, rel=1e-6)
    assert est8["latency_s"] == pytest.approx(est["latency_s"], rel=1e-9)


def test_score_recommendation():
    rec = score.score("current-machine", "qwen14b", vram_budget="1GiB")
    assert rec["model"] == "Qwen2.5-14B-Instruct"
    assert rec["recommendation"] in ("fit", "page", "none")
    assert "reason" in rec


# ---------------------------------------------------------------------------
# Pager (requires libvramer.so + a CUDA device)
# ---------------------------------------------------------------------------

def _lib_available():
    try:
        from vramer import _lib
        _lib.load_lib()
        return True
    except Exception:
        return False


def _cuda_available():
    # Heuristic: a CUDA runtime we can dlopen and at least one device.
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13",
                 "libcudart.so.11"):
        try:
            rt = ctypes.CDLL(name)
        except OSError:
            continue
        count = ctypes.c_int(0)
        try:
            if rt.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0:
                return True
        except Exception:
            continue
    # Fallback: presence of an nvidia device node.
    return os.path.exists("/dev/nvidia0")


@pytest.mark.skipif(
    not (_lib_available() and _cuda_available()),
    reason="libvramer.so not loadable or no CUDA device present",
)
def test_pager_ram_source_round_trip():
    from vramer import Pager, RamSource

    PAGE = 4 << 20       # 4 MiB pages
    NBYTES = 32 << 20    # 32 MiB source
    BUDGET = 8 << 20     # 8 MiB budget -> source is 4x bigger -> real paging

    gpu = int(os.environ.get("VRAMER_TEST_GPU", "0"))

    with Pager(
        RamSource(NBYTES), gpu=gpu, vram_budget=BUDGET, page_size=PAGE,
        staging_buffers=2, prefetch_workers=0,
    ) as pager:
        pages = pager.page_count()
        assert pages == NBYTES // PAGE
        for pid in range(pages):
            with pager.acquire(pid) as lease:
                assert lease.device_ptr != 0
                assert lease.nbytes == PAGE
                lease.wait_on(0)
        pager.synchronize()
        st = pager.stats()

    assert st["bytes_h2d"] == NBYTES
    assert st["misses"] == pages
    assert st["evictions"] > 0  # budget smaller than source -> evictions happen
