#!/usr/bin/env python3
"""VRAMER decode performance model + planner CLI.

This module is the ONE source of truth for the VRAMER decode performance model.
It is a clean port of ``src/vramer_plan.py`` exposed as an importable API so
that ML/Python users can estimate decode throughput programmatically::

    from vramer import plan
    hw = plan.load_profile("current-machine")     # bundled or path
    model = plan.load_profile("qwen14b")
    est = plan.estimate(hw, model, quant="q4", batch=1)
    print(est["decode_tok_s"], est["verdict"])

The scorer (``vramer.score``) imports the same functions so the numbers agree.

Needs NO GPU and NO compiled ``libvramer.so`` -- it is a pure analytical model.

------------------------------------------------------------------------------
PERFORMANCE MODEL (the physics)
------------------------------------------------------------------------------
LLM decode (batch=1, one token) reads ~100% of the DENSE model's weights per
token. When weights don't fit in VRAM, the spill is re-streamed over PCIe every
token, so decode is PCIe-bound::

    decode_tok_s(aggregate) = effective_H2D_bandwidth / bytes_streamed_per_token
    bytes_streamed_per_token = spill_bytes * moe_active_fraction / batch
    spill_bytes              = max(0, weight_bytes_at_quant - resident_bytes)
    weight_bytes_at_quant    = params * bytes_per_param(quant)

Batching reads each weight ONCE for the whole batch, so aggregate throughput
scales with batch while PER-SEQUENCE latency is unchanged::

    per_sequence_latency_s/token = spill_bytes * moe_active_fraction
                                   / effective_H2D_bandwidth
                                 = batch / decode_tok_s(aggregate)

If spill <= 0 the model fits: decode is VRAM-bandwidth-bound (not PCIe) and
paging adds nothing -- VRAMER does not help, and we say so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

VERSION = "0.2.0"
GIB = 1024 ** 3
MIB = 1024 ** 2

ProfileLike = Union[str, Path, Dict[str, Any]]

# ------------------------------------------------------------
# Constants: quant and PCIe tables
# ------------------------------------------------------------

#: Default bytes-per-parameter per quant (model profiles may override).
DEFAULT_BYTES_PER_PARAM = {
    "fp16": 2.0,
    "q8": 1.0,
    "q4": 0.5,
    "q4_k": 0.5625,
}

QUANTS = ["fp16", "q8", "q4", "q4_k"]

# PCIe one-way effective (pinned) GiB/s per lane, by generation.
_PCIE_PINNED_GIBS_PER_LANE = {
    1: 0.19,
    2: 0.38,
    3: 0.75,
    4: 1.625,
    5: 3.125,
    6: 6.25,
}

# Named bus configs used by the PCIe sweep.
PCIE_SWEEP = [
    ("gen3x16", 3, 16),
    ("gen4x8", 4, 8),
    ("gen4x16", 4, 16),
    ("gen5x16", 5, 16),
]

# Pageable staging is roughly half of the pinned ceiling.
PAGEABLE_FRACTION = 0.5


def pcie_pinned_gibs(gen: int, width: int) -> float:
    """Effective one-way pinned H2D bandwidth (GiB/s) for a gen/width link."""
    per_lane = _PCIE_PINNED_GIBS_PER_LANE.get(int(gen), 0.0)
    return per_lane * int(width)


# ------------------------------------------------------------
# Profile loading helpers
# ------------------------------------------------------------

#: Directory of profiles bundled with the package.
PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


def bundled_profiles() -> List[str]:
    """Return the names (without .json) of profiles bundled with the package."""
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON document from a filesystem path."""
    with open(path, "r") as f:
        return json.load(f)


def load_profile(profile: ProfileLike) -> Dict[str, Any]:
    """Resolve a hardware/model profile to a dict.

    Accepts an already-loaded dict (returned as-is), a path to a JSON file, or
    the bare name of a profile bundled with the package (e.g. ``"qwen14b"`` or
    ``"current-machine"``).
    """
    if isinstance(profile, dict):
        return profile
    text = str(profile)
    p = Path(text)
    if p.is_file():
        return load_json(p)
    # Try as a bundled profile name (with or without .json).
    name = text[:-5] if text.endswith(".json") else text
    candidate = PROFILES_DIR / f"{name}.json"
    if candidate.is_file():
        return load_json(candidate)
    raise FileNotFoundError(
        f"Profile {profile!r} not found as a path or bundled profile. "
        f"Bundled profiles: {bundled_profiles()}"
    )


def bytes_per_param(model: Dict[str, Any], quant: str) -> float:
    table = dict(DEFAULT_BYTES_PER_PARAM)
    table.update(model.get("bytes_per_param", {}) or {})
    if quant not in table:
        raise ValueError(f"Unknown quant {quant!r} (have {sorted(table)})")
    return float(table[quant])


# ------------------------------------------------------------
# Size parsing (e.g. "16GiB", "40GB", "512MiB")
# ------------------------------------------------------------

_UNIT = {
    "": 1,
    "b": 1,
    "k": 1000, "kb": 1000, "kib": 1024,
    "m": 1000 ** 2, "mb": 1000 ** 2, "mib": MIB,
    "g": 1000 ** 3, "gb": 1000 ** 3, "gib": GIB,
    "t": 1000 ** 4, "tb": 1000 ** 4, "tib": 1024 ** 4,
}


def parse_bytes(text: Union[str, int]) -> int:
    if isinstance(text, int):
        return text
    s = str(text).strip().lower()
    num = ""
    unit = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            unit += ch
    unit = unit.strip()
    if unit not in _UNIT:
        raise ValueError(f"Bad size {text!r}")
    return int(float(num) * _UNIT[unit])


# ------------------------------------------------------------
# MoE active fraction
# ------------------------------------------------------------

def moe_active_fraction(model: Dict[str, Any]) -> float:
    """Blended fraction of weight bytes that stream per token.

    Dense layers stream fully (1.0); expert tensors stream
    active_experts/n_experts. Approximated with expert_param_fraction.
    """
    moe = model.get("moe")
    if not moe:
        return 1.0
    n = float(moe.get("n_experts", 1) or 1)
    active = float(moe.get("active_experts", n))
    expert_frac = float(moe.get("expert_param_fraction", 0.9))
    expert_frac = max(0.0, min(1.0, expert_frac))
    sparse = active / n if n > 0 else 1.0
    return (1.0 - expert_frac) * 1.0 + expert_frac * sparse


# ------------------------------------------------------------
# Effective bandwidth from hardware profile
# ------------------------------------------------------------

def effective_bandwidth(
    profile: Dict[str, Any],
    gpus: int,
    pinned: bool,
    pcie_gen_override: Optional[int] = None,
    pcie_width_override: Optional[int] = None,
) -> Tuple[float, float, str]:
    """Return (total_gibs, per_gpu_gibs, source_label).

    Uses measured fields when present in the profile, otherwise the PCIe
    gen/width table. Multi-GPU sums bandwidth across the selected GPUs.
    """
    gpu_list = profile.get("gpus", [])
    if not gpu_list:
        gen = pcie_gen_override or 4
        width = pcie_width_override or 8
        pinned_bw = pcie_pinned_gibs(gen, width)
        bw = pinned_bw if pinned else pinned_bw * PAGEABLE_FRACTION
        return bw * gpus, bw, f"table(gen{gen}x{width}, no GPUs in profile)"

    n = min(gpus, len(gpu_list)) if gpus else len(gpu_list)
    n = max(1, n)
    chosen = gpu_list[:n]

    per_gpu: List[float] = []
    sources = set()

    for g in chosen:
        gen = pcie_gen_override or int(g.get("pcie_gen", 0) or 0)
        width = pcie_width_override or int(g.get("pcie_width", 0) or 0)

        measured_pinned = float(g.get("h2d_pinned_gibs", 0.0) or 0.0)
        measured_pageable = float(g.get("h2d_pageable_gibs", 0.0) or 0.0)

        overridden = bool(pcie_gen_override or pcie_width_override)

        if pinned and measured_pinned > 0 and not overridden:
            per_gpu.append(measured_pinned)
            sources.add("measured-pinned")
        elif (not pinned) and measured_pageable > 0 and not overridden:
            per_gpu.append(measured_pageable)
            sources.add("measured-pageable")
        else:
            pinned_bw = pcie_pinned_gibs(gen, width) if gen and width else 0.0
            if pinned_bw <= 0:
                pinned_bw = pcie_pinned_gibs(4, 8)
                sources.add("table-default(gen4x8)")
            else:
                sources.add(f"table(gen{gen}x{width})")
            bw = pinned_bw if pinned else pinned_bw * PAGEABLE_FRACTION
            per_gpu.append(bw)

    if gpus and gpus > len(chosen) and per_gpu:
        template = per_gpu[-1]
        while len(per_gpu) < gpus:
            per_gpu.append(template)

    total = sum(per_gpu)
    label = "+".join(sorted(sources))
    label += " (pinned)" if pinned else " (pageable)"
    return total, (per_gpu[0] if per_gpu else 0.0), label


# ------------------------------------------------------------
# Core estimate (low level: already-resolved bandwidth + residency)
# ------------------------------------------------------------

def estimate_core(
    model: Dict[str, Any],
    quant: str,
    resident_bytes: float,
    batch: int,
    total_bw_gibs: float,
    moe_frac: float,
) -> Dict[str, Any]:
    """Return a decode estimate for one fully-resolved configuration.

    ``total_bw_gibs`` is the aggregate effective H2D bandwidth (GiB/s). This is
    the low-level primitive; most callers want :func:`estimate`.
    """
    params = float(model["params"])
    bpp = bytes_per_param(model, quant)
    weight_bytes = params * bpp

    resident_bytes = max(0.0, min(resident_bytes, weight_bytes))
    spill_bytes = max(0.0, weight_bytes - resident_bytes)

    bw_bytes_s = total_bw_gibs * GIB
    fits = spill_bytes <= 0

    effective_spill = spill_bytes * moe_frac
    bytes_per_token = effective_spill / max(1, batch)

    if fits:
        decode_tok_s = None
        latency_s = None
        verdict = "FITS"
    elif bw_bytes_s <= 0:
        decode_tok_s = 0.0
        latency_s = float("inf")
        verdict = "NO-BANDWIDTH"
    else:
        decode_tok_s = bw_bytes_s / bytes_per_token
        latency_s = effective_spill / bw_bytes_s
        verdict = "PCIe-bound (slow)" if latency_s > 0.2 else "PCIe-bound (usable)"

    return {
        "quant": quant,
        "bytes_per_param": bpp,
        "batch": batch,
        "weight_bytes": weight_bytes,
        "resident_bytes": resident_bytes,
        "spill_bytes": spill_bytes,
        "moe_fraction": moe_frac,
        "effective_spill_bytes": effective_spill,
        "bytes_per_token": bytes_per_token,
        "total_bw_gibs": total_bw_gibs,
        "fits": fits,
        "decode_tok_s": decode_tok_s,
        "latency_s": latency_s,
        "verdict": verdict,
    }


def resolve_resident_bytes(
    model: Dict[str, Any],
    quant: str,
    profile: Dict[str, Any],
    gpus: int,
    *,
    vram_budget: Optional[Union[str, int]] = None,
    resident_fraction: Optional[float] = None,
) -> Tuple[float, str]:
    """Figure out resident_bytes from vram_budget or resident_fraction.

    Default: total VRAM of the selected GPUs minus a small runtime reserve.
    """
    params = float(model["params"])
    weight_bytes = params * bytes_per_param(model, quant)

    if resident_fraction is not None:
        frac = max(0.0, min(1.0, resident_fraction))
        return weight_bytes * frac, f"resident-fraction={frac:.2f}"

    if vram_budget is not None:
        budget = parse_bytes(vram_budget)
        return float(budget), f"vram-budget={vram_budget}"

    gpu_list = profile.get("gpus", [])
    n = min(gpus, len(gpu_list)) if gpus else len(gpu_list)
    n = max(1, n) if gpu_list else gpus
    if gpu_list:
        per = float(gpu_list[0].get("vram_bytes", 0))
        budget = per * n * 0.90
        return budget, f"detected-vram x{n} @90%"
    return 0.0, "no-vram-info"


# ------------------------------------------------------------
# High-level programmatic API
# ------------------------------------------------------------

def estimate(
    hw_profile: ProfileLike,
    model_profile: ProfileLike,
    *,
    quant: str = "q4",
    batch: int = 1,
    gpus: int = 0,
    pinned: bool = True,
    vram_budget: Optional[Union[str, int]] = None,
    resident_fraction: Optional[float] = None,
    pcie_gen: Optional[int] = None,
    pcie_width: Optional[int] = None,
    moe: bool = True,
) -> Dict[str, Any]:
    """Estimate decode throughput for a hardware + model profile.

    ``hw_profile`` and ``model_profile`` may each be a dict, a path to a JSON
    file, or the name of a bundled profile. Returns the estimate dict from
    :func:`estimate_core`, enriched with ``model``, ``gpus``, ``pinned`` and
    ``bandwidth_label``.
    """
    profile = load_profile(hw_profile)
    model = load_profile(model_profile)

    n_gpus = gpus or len(profile.get("gpus", [])) or 1
    bw, _per_gpu, bw_label = effective_bandwidth(
        profile, n_gpus, pinned, pcie_gen, pcie_width
    )
    moe_frac = moe_active_fraction(model) if moe else 1.0
    resident_bytes, rlabel = resolve_resident_bytes(
        model, quant, profile, n_gpus,
        vram_budget=vram_budget, resident_fraction=resident_fraction,
    )
    est = estimate_core(model, quant, resident_bytes, batch, bw, moe_frac)
    est["model"] = model.get("name", "?")
    est["gpus"] = n_gpus
    est["pinned"] = pinned
    est["bandwidth_label"] = bw_label
    est["resident_label"] = rlabel
    return est


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

def human_bytes(n: float) -> str:
    n = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def fmt_toks(v: Optional[float]) -> str:
    if v is None:
        return "  fits "
    if v >= 1000:
        return f"{v:7.0f}"
    if v >= 10:
        return f"{v:7.1f}"
    return f"{v:7.2f}"


# ------------------------------------------------------------
# Single-config detailed report
# ------------------------------------------------------------

def print_single(model, profile, args, gpus, pinned, bw, bw_label, est,
                 resident_label):
    print()
    print("=" * 72)
    print(f" VRAMER {VERSION} -- DECODE PLAN")
    print("=" * 72)

    moe = model.get("moe")
    print()
    print("MODEL")
    print(f"  Name            : {model['name']}")
    print(f"  Params          : {model['params'] / 1e9:.1f} B")
    print(f"  Architecture    : {model.get('architecture', 'dense')}")
    if moe:
        print(
            f"  MoE             : {moe['active_experts']}/{moe['n_experts']} "
            f"experts active  (stream fraction {est['moe_fraction']:.3f})"
        )
    print(f"  Quant           : {est['quant']} "
          f"({est['bytes_per_param']} bytes/param)")
    print(f"  Weights @ quant : {human_bytes(est['weight_bytes'])}")

    print()
    print("HARDWARE")
    print(f"  GPUs used       : {gpus}")
    print(f"  Staging         : {'pinned' if pinned else 'pageable'}")
    print(f"  Effective H2D   : {bw:.2f} GiB/s  [{bw_label}]")
    print(f"  VRAM budget     : {human_bytes(est['resident_bytes'])} "
          f"[{resident_label}]")

    kv = model.get("kv_bytes_per_token", 0)
    if kv:
        print(f"  KV cache/token  : {human_bytes(kv)} "
              f"(competes for VRAM budget; not in weight stream)")

    print()
    print("DECODE ESTIMATE")
    print(f"  Resident        : {human_bytes(est['resident_bytes'])}")
    print(f"  Spill           : {human_bytes(est['spill_bytes'])}")
    if moe and est["spill_bytes"] > 0:
        print(f"  Streamed/token  : {human_bytes(est['effective_spill_bytes'])}"
              f"  (after MoE sparsity)")
    print(f"  Batch           : {est['batch']}")
    print(f"  Bytes/token     : {human_bytes(est['bytes_per_token'])}"
          f"  (amortized across batch)")

    print()
    if est["fits"]:
        print("  VERDICT: FITS -- the whole model is resident. Decode is bound")
        print("  by VRAM bandwidth, NOT PCIe. Nothing streams per token, so")
        print("  VRAMER paging adds no traffic and does not help here.")
    elif est["decode_tok_s"] == 0.0:
        print("  VERDICT: NO BANDWIDTH -- profile has no usable H2D figure.")
    else:
        print(f"  Decode tok/s    : {est['decode_tok_s']:.2f}  (aggregate, "
              f"batch={est['batch']})")
        print(f"  Per-seq latency : {est['latency_s'] * 1000:.1f} ms/token  "
              f"(unchanged by batch)")
        print(f"  VERDICT: {est['verdict']}")
        print("  (PCIe-bound upper bound; real compute overlaps but adds its")
        print("   own time. Raise the budget or quantize to shrink the spill.)")
    print()


# ------------------------------------------------------------
# Matrix mode
# ------------------------------------------------------------

def run_matrix(model, profile, args, gpus, pinned, bw, bw_label):
    quants = args.matrix_quants or ["fp16", "q8", "q4"]
    batches = args.matrix_batches or [1, 8, 32]
    moe_frac = moe_active_fraction(model) if args.moe else 1.0

    print()
    print("=" * 72)
    print(f" VRAMER {VERSION} -- DECODE MATRIX  (aggregate tok/s)")
    print("=" * 72)
    print(f" Model: {model['name']}  ({model['params']/1e9:.0f}B, "
          f"{model.get('architecture','dense')})")
    print(f" GPUs: {gpus}   Staging: {'pinned' if pinned else 'pageable'}   "
          f"Bandwidth: {bw:.1f} GiB/s [{bw_label}]")

    print()
    header = "  quant \\ batch |" + "".join(f"{b:>9}" for b in batches)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for q in quants:
        resident_bytes, rlabel = resolve_resident_bytes(
            model, q, profile, gpus,
            vram_budget=args.vram_budget, resident_fraction=args.resident_fraction,
        )
        cells = []
        fits_row = False
        for b in batches:
            est = estimate_core(model, q, resident_bytes, b, bw, moe_frac)
            if est["fits"]:
                fits_row = True
                cells.append("   fits ")
            else:
                cells.append(fmt_toks(est["decode_tok_s"]) + " ")
        tag = "  (fits)" if fits_row else ""
        print(f"  {q:>6}        |" + "".join(f"{c:>9}" for c in cells) + tag)

    print()
    budget, blabel = resolve_resident_bytes(
        model, quants[0], profile, gpus,
        vram_budget=args.vram_budget, resident_fraction=args.resident_fraction,
    )
    if args.resident_fraction is not None:
        print(f"  Budget: {blabel} (scales with quant)")
    else:
        print(f"  VRAM budget: {human_bytes(budget)} [{blabel}]")
    print("  'fits' = model resident at that quant (VRAM-bw bound, not PCIe).")
    print("  tok/s is AGGREGATE decode throughput; per-sequence latency is")
    print("  tok/s / batch and does not improve with batching.")
    print()


# ------------------------------------------------------------
# PCIe sweep mode
# ------------------------------------------------------------

def run_pcie_sweep(model, profile, args, gpus, pinned):
    quant = args.quant
    batch = args.batch
    moe_frac = moe_active_fraction(model) if args.moe else 1.0
    resident_bytes, rlabel = resolve_resident_bytes(
        model, quant, profile, gpus,
        vram_budget=args.vram_budget, resident_fraction=args.resident_fraction,
    )

    print()
    print("=" * 72)
    print(f" VRAMER {VERSION} -- PCIe SWEEP  (what-if newer bus)")
    print("=" * 72)
    print(f" Model: {model['name']}  quant={quant}  batch={batch}  GPUs={gpus}")
    print(f" Staging: {'pinned' if pinned else 'pageable'}   "
          f"VRAM budget: {human_bytes(resident_bytes)} [{rlabel}]")
    print()
    print(f"  {'bus':<10} {'per-GPU':>10} {'aggregate':>11} "
          f"{'tok/s':>10} {'ms/token':>10}  verdict")
    print("  " + "-" * 68)

    for name, gen, width in PCIE_SWEEP:
        per_gpu_pinned = pcie_pinned_gibs(gen, width)
        per_gpu = per_gpu_pinned if pinned else per_gpu_pinned * PAGEABLE_FRACTION
        total = per_gpu * gpus
        est = estimate_core(model, quant, resident_bytes, batch, total, moe_frac)

        if est["fits"]:
            toks = "fits"
            ms = "-"
            verdict = "FITS (VRAM-bound)"
        else:
            toks = f"{est['decode_tok_s']:.1f}"
            ms = f"{est['latency_s'] * 1000:.0f}"
            verdict = est["verdict"]

        print(f"  {name:<10} {per_gpu:>8.1f}GiB {total:>8.1f}GiB "
              f"{toks:>10} {ms:>10}  {verdict}")

    print()
    print("  Aggregate = per-GPU x GPUs (bandwidth sums across GPUs).")
    print("  The SLOWER of GPU-supported link and slot caps the real bus.")
    print()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vramer-plan",
        description="VRAMER decode planner (angle investigation)",
    )
    ap.add_argument("--profile", required=True,
                    help="hardware profile (path or bundled name)")
    ap.add_argument("--model", required=True,
                    help="model profile (path or bundled name)")

    ap.add_argument("--vram-budget", help="resident weight budget, e.g. 16GiB")
    ap.add_argument("--resident-fraction", type=float,
                    help="fraction of weights kept resident (0..1)")

    ap.add_argument("--quant", default="q4", choices=QUANTS,
                    help="quantization (default q4)")
    ap.add_argument("--batch", type=int, default=1,
                    help="serving batch size (aggregate throughput)")

    ap.add_argument("--gpus", type=int, default=0,
                    help="number of GPUs to sum bandwidth over "
                         "(default: all in profile)")

    staging = ap.add_mutually_exclusive_group()
    staging.add_argument("--pinned", dest="pinned", action="store_true",
                         help="pinned host staging (default; ~2x pageable)")
    staging.add_argument("--pageable", dest="pinned", action="store_false",
                         help="pageable host staging (~half of pinned)")
    ap.set_defaults(pinned=True)

    ap.add_argument("--pcie-gen", type=int, help="override PCIe generation")
    ap.add_argument("--pcie-width", type=int, help="override PCIe link width")

    ap.add_argument("--moe", dest="moe", action="store_true", default=True,
                    help="apply MoE sparsity from the model profile (default)")
    ap.add_argument("--no-moe", dest="moe", action="store_false",
                    help="treat the model as fully dense (ignore MoE sparsity)")

    ap.add_argument("--matrix", action="store_true",
                    help="print a quant x batch grid of tok/s")
    ap.add_argument("--matrix-quants", nargs="+",
                    help="quants for matrix rows (default fp16 q8 q4)")
    ap.add_argument("--matrix-batches", nargs="+", type=int,
                    help="batches for matrix cols (default 1 8 32)")

    ap.add_argument("--pcie-sweep", action="store_true",
                    help="sweep tok/s across gen3x16/gen4x8/gen4x16/gen5x16")

    ap.add_argument("--json", action="store_true",
                    help="emit the single-config estimate as JSON")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    profile = load_profile(args.profile)
    model = load_profile(args.model)

    gpus = args.gpus or len(profile.get("gpus", [])) or 1

    bw, _per_gpu_bw, bw_label = effective_bandwidth(
        profile, gpus, args.pinned, args.pcie_gen, args.pcie_width
    )

    if args.matrix:
        run_matrix(model, profile, args, gpus, args.pinned, bw, bw_label)
        if args.pcie_sweep:
            run_pcie_sweep(model, profile, args, gpus, args.pinned)
        return

    if args.pcie_sweep:
        run_pcie_sweep(model, profile, args, gpus, args.pinned)
        return

    moe_frac = moe_active_fraction(model) if args.moe else 1.0
    resident_bytes, rlabel = resolve_resident_bytes(
        model, args.quant, profile, gpus,
        vram_budget=args.vram_budget, resident_fraction=args.resident_fraction,
    )
    est = estimate_core(model, args.quant, resident_bytes, args.batch, bw, moe_frac)

    if args.json:
        out = dict(est)
        out["model"] = model["name"]
        out["gpus"] = gpus
        out["pinned"] = args.pinned
        out["bandwidth_label"] = bw_label
        print(json.dumps(out, indent=2,
                         default=lambda x: None if x == float("inf") else x))
    else:
        print_single(model, profile, args, gpus, args.pinned, bw, bw_label,
                     est, rlabel)


if __name__ == "__main__":
    main()
