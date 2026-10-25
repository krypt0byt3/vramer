"""VRAMER -- a pip-installable Python front end for the VRAMER VRAM pager.

Two independent pieces live here:

* **The planner** (:mod:`vramer.planner`, also exposed as ``vramer.plan``) plus
  the scorer (:mod:`vramer.score`) and detector (:mod:`vramer.detect`): a pure
  analytical decode performance model. Needs NO GPU and NO compiled library::

      from vramer import plan
      est = plan.estimate("current-machine", "qwen14b", quant="q4", batch=1)
      print(est["decode_tok_s"], est["verdict"])

* **The pager** (:mod:`vramer.pager`): a Pythonic ctypes wrapper over the
  ``libvramer.so`` C ABI. Requires the separately-built shared library (found via
  ``$VRAMER_LIB`` or common build paths) and a CUDA device::

      from vramer import RamSource, Pager
      with Pager(RamSource(64 << 20), gpu=0, vram_budget=16 << 20) as pager:
          with pager.acquire(0) as lease:
              lease.wait_on(0)
          print(pager.stats())

Importing this package NEVER requires the compiled library; the pager loads it
lazily on first use, so the planner works everywhere.
"""

from __future__ import annotations

__version__ = "0.2.0"

# Analytical model (lib-free).
from . import detect, planner, score  # noqa: E402
from . import planner as plan  # noqa: E402  (alias: `from vramer import plan`)

# Pager wrapper (ctypes; lib loaded lazily on first Pager construction).
from .pager import (  # noqa: E402
    BorrowedSource,
    EvictionPolicy,
    FileSource,
    Lease,
    Pager,
    RamSource,
    PinnedSource,
    VramerError,
)

__all__ = [
    "__version__",
    # analytical model
    "plan",
    "planner",
    "score",
    "detect",
    # pager API
    "Pager",
    "Lease",
    "RamSource",
    "PinnedSource",
    "FileSource",
    "BorrowedSource",
    "EvictionPolicy",
    "VramerError",
]
