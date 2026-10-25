#!/usr/bin/env python3

"""
VRAMER decode planner — the angle-investigation tool.

Loads a hardware profile (vramer-hw-profile/1) + a model profile
(vramer-model-profile/1) and estimates decode throughput (tokens/sec) and
per-sequence latency under a transparent, PCIe-faithful performance model.

It lets you sweep the angles that actually move the number:
  quantization, batch size, resident fraction / VRAM budget, PCIe generation,
  number of GPUs, pinned vs pageable staging, and MoE sparsity.

------------------------------------------------------------------------------
PERFORMANCE MODEL (the physics — see profiles/README.md for the full write-up)
------------------------------------------------------------------------------
LLM decode (batch=1, one token) reads ~100% of the DENSE model's weights per
token. When weights don't fit in VRAM, the spill is re-streamed over PCIe every
token, so decode is PCIe-bound:

    decode_tok_s(aggregate) = effective_H2D_bandwidth / bytes_streamed_per_token

    bytes_streamed_per_token = spill_bytes * moe_active_fraction / batch
    spill_bytes              = max(0, weight_bytes_at_quant - resident_bytes)
    weight_bytes_at_quant    = params * bytes_per_param(quant)

Batching reads each weight ONCE for the whole batch, so aggregate throughput
scales with batch while PER-SEQUENCE latency is unchanged:

    per_sequence_latency_s/token = spill_bytes * moe_active_fraction
                                   / effective_H2D_bandwidth
                                 = batch / decode_tok_s(aggregate)

MoE: only the routed (active) experts stream, so for expert tensors the
fraction is active_experts/total_experts; dense layers still stream fully. We
blend these with the model's expert_param_fraction (approximate).

If spill <= 0 the model fits: decode is VRAM-bandwidth-bound (not PCIe) and
paging adds nothing — VRAMER does not help, and we say so.

Prefill is compute-bound and amortizes weight reads over all prompt tokens, so
it is NOT PCIe-bound; this tool focuses on decode.
"""

import argparse
import json
import sys
from pathlib import Path


VERSION = "0.2.0"
GIB = 1024 ** 3
MIB = 1024 ** 2


# ------------------------------------------------------------
# Constants: quant and PCIe tables
# ------------------------------------------------------------

# Default bytes-per-parameter per quant (model profiles may override).
DEFAULT_BYTES_PER_PARAM = {
    "fp16": 2.0,
    "q8": 1.0,
    "q4": 0.5,
    "q4_k": 0.5625,
}

QUANTS = ["fp16", "q8", "q4", "q4_k"]

# PCIe one-way effective (pinned) GiB/s per lane, by generation.
# Calibrated to reproduce the measured/accepted table:
#   gen3x16 ~= 12, gen4x8 ~= 13, gen4x16 ~= 25/26, gen5x8 ~= 25, gen5x16 ~= 50.
_PCIE_PINNED_GIBS_PER_LANE = {
    1: 0.19,
    2: 0.38,
    3: 0.75,
    4: 1.625,
    5: 3.125,
    6: 6.25,
}

# Named bus configs used by --pcie-sweep.
PCIE_SWEEP = [
    ("gen3x16", 3, 16),
    ("gen4x8", 4, 8),
    ("gen4x16", 4, 16),
    ("gen5x16", 5, 16),
]

# Pageable staging is roughly half of the pinned ceiling (measured on A2:
# ~5.5 GiB/s pageable vs ~12 GiB/s pinned ceiling on Gen3/4 x8-x16).
PAGEABLE_FRACTION = 0.5


def pcie_pinned_gibs(gen, width):
    """Effective one-way pinned H2D bandwidth (GiB/s) for a gen/width link."""
    per_lane = _PCIE_PINNED_GIBS_PER_LANE.get(int(gen), 0.0)
    return per_lane * int(width)


# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def bytes_per_param(model, quant):
    table = dict(DEFAULT_BYTES_PER_PARAM)
    table.update(model.get("bytes_per_param", {}) or {})
    if quant not in table:
        raise SystemExit(f"Unknown quant '{quant}' (have {sorted(table)})")
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


def parse_bytes(text):
    s = str(text).strip().lower()
    num = ""
    unit = ""
    for ch in s:
        if ch.isdigit() or ch == "." :
            num += ch
        else:
            unit += ch
    unit = unit.strip()
    if unit not in _UNIT:
        raise SystemExit(f"Bad size '{text}'")
    return int(float(num) * _UNIT[unit])


# ------------------------------------------------------------
# MoE active fraction
# ------------------------------------------------------------

def moe_active_fraction(model):
    """
    Blended fraction of weight bytes that stream per token.
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
    profile,
    gpus,
    pinned,
    pcie_gen_override,
    pcie_width_override,
):
    """
    Return (total_gibs, per_gpu_gibs, source_label).

    Uses measured fields when present in the profile, otherwise the PCIe
    gen/width table. Multi-GPU sums bandwidth across the selected GPUs.
    """
    gpu_list = profile.get("gpus", [])
    if not gpu_list:
        # No detected GPUs: fall back to a single synthetic link from overrides.
        gen = pcie_gen_override or 4
        width = pcie_width_override or 8
        pinned_bw = pcie_pinned_gibs(gen, width)
        bw = pinned_bw if pinned else pinned_bw * PAGEABLE_FRACTION
        return bw * gpus, bw, f"table(gen{gen}x{width}, no GPUs in profile)"

    # Use up to `gpus` detected GPUs as the template (they're usually uniform).
    n = min(gpus, len(gpu_list)) if gpus else len(gpu_list)
    n = max(1, n)
    chosen = gpu_list[:n]

    per_gpu = []
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
                # Last-resort default: Gen4 x8 pinned ceiling.
                pinned_bw = pcie_pinned_gibs(4, 8)
                sources.add("table-default(gen4x8)")
            else:
                sources.add(f"table(gen{gen}x{width})")
            bw = pinned_bw if pinned else pinned_bw * PAGEABLE_FRACTION
            per_gpu.append(bw)

    # If the user asked for more GPUs than detected, scale by replication.
    if gpus and gpus > len(chosen) and per_gpu:
        template = per_gpu[-1]
        while len(per_gpu) < gpus:
            per_gpu.append(template)

    total = sum(per_gpu)
    label = "+".join(sorted(sources))
    if not pinned:
        label += " (pageable)"
    else:
        label += " (pinned)"
    return total, (per_gpu[0] if per_gpu else 0.0), label


# ------------------------------------------------------------
# Core estimate
# ------------------------------------------------------------

def estimate(
    model,
    quant,
    resident_bytes,
    batch,
    total_bw_gibs,
    moe_frac,
):
    """
    Return a dict with the decode estimate for one configuration.
    total_bw_gibs is the aggregate effective H2D bandwidth (GiB/s).
    """
    params = float(model["params"])
    bpp = bytes_per_param(model, quant)
    weight_bytes = params * bpp

    resident_bytes = max(0.0, min(resident_bytes, weight_bytes))
    spill_bytes = max(0.0, weight_bytes - resident_bytes)

    bw_bytes_s = total_bw_gibs * GIB

    fits = spill_bytes <= 0

    # Bytes streamed per output token (amortized across the batch).
    effective_spill = spill_bytes * moe_frac
    bytes_per_token = effective_spill / max(1, batch)

    if fits:
        decode_tok_s = None            # not PCIe-bound; VRAM-bandwidth-bound
        latency_s = None
        verdict = "FITS"
    elif bw_bytes_s <= 0:
        decode_tok_s = 0.0
        latency_s = float("inf")
        verdict = "NO-BANDWIDTH"
    else:
        decode_tok_s = bw_bytes_s / bytes_per_token
        # Per-sequence latency is independent of batch.
        latency_s = effective_spill / bw_bytes_s
        if latency_s > 0.2:
            verdict = "PCIe-bound (slow)"
        else:
            verdict = "PCIe-bound (usable)"

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


def resolve_resident_bytes(model, quant, args, profile, gpus):
    """
    Figure out resident_bytes from --vram-budget or --resident-fraction.
    Default: total VRAM of the selected GPUs minus a small runtime reserve.
    """
    params = float(model["params"])
    weight_bytes = params * bytes_per_param(model, quant)

    if args.resident_fraction is not None:
        frac = max(0.0, min(1.0, args.resident_fraction))
        return weight_bytes * frac, f"resident-fraction={frac:.2f}"

    if args.vram_budget is not None:
        budget = parse_bytes(args.vram_budget)
        return float(budget), f"vram-budget={args.vram_budget}"

    # Default: detected VRAM across the selected GPUs, 90% usable for weights.
    gpu_list = profile.get("gpus", [])
    n = min(gpus, len(gpu_list)) if gpus else len(gpu_list)
    n = max(1, n) if gpu_list else gpus
    if gpu_list:
        per = float(gpu_list[0].get("vram_bytes", 0))
        budget = per * n * 0.90
        return budget, f"detected-vram x{n} @90%"
    return 0.0, "no-vram-info"


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

def human_bytes(n):
    n = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024


def fmt_toks(v):
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
    print(f" VRAMER {VERSION} — DECODE PLAN")
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
        print("  VERDICT: FITS — the whole model is resident. Decode is bound")
        print("  by VRAM bandwidth, NOT PCIe. Nothing streams per token, so")
        print("  VRAMER paging adds no traffic and does not help here.")
    elif est["decode_tok_s"] == 0.0:
        print("  VERDICT: NO BANDWIDTH — profile has no usable H2D figure.")
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
    moe_frac = moe_active_fraction(model)

    print()
    print("=" * 72)
    print(f" VRAMER {VERSION} — DECODE MATRIX  (aggregate tok/s)")
    print("=" * 72)
    print(f" Model: {model['name']}  ({model['params']/1e9:.0f}B, "
          f"{model.get('architecture','dense')})")
    print(f" GPUs: {gpus}   Staging: {'pinned' if pinned else 'pageable'}   "
          f"Bandwidth: {bw:.1f} GiB/s [{bw_label}]")

    # Resident budget is fixed across the grid (user-specified or detected).
    # Resolve per-quant since fraction mode scales with weight size.
    print()
    header = "  quant \\ batch |" + "".join(f"{b:>9}" for b in batches)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for q in quants:
        resident_bytes, rlabel = resolve_resident_bytes(
            model, q, args, profile, gpus
        )
        cells = []
        fits_row = False
        for b in batches:
            est = estimate(model, q, resident_bytes, b, bw, moe_frac)
            if est["fits"]:
                fits_row = True
                cells.append("   fits ")
            else:
                cells.append(fmt_toks(est["decode_tok_s"]) + " ")
        tag = "  (fits)" if fits_row else ""
        print(f"  {q:>6}        |" + "".join(f"{c:>9}" for c in cells) + tag)

    # Show the VRAM budget assumption (same for all rows unless fraction mode).
    print()
    budget, blabel = resolve_resident_bytes(model, quants[0], args,
                                            profile, gpus)
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
    moe_frac = moe_active_fraction(model)
    resident_bytes, rlabel = resolve_resident_bytes(
        model, quant, args, profile, gpus
    )

    print()
    print("=" * 72)
    print(f" VRAMER {VERSION} — PCIe SWEEP  (what-if newer bus)")
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
        est = estimate(model, quant, resident_bytes, batch, total, moe_frac)

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
# Main
# ------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="VRAMER decode planner (angle investigation)"
    )
    ap.add_argument("--profile", required=True,
                    help="hardware profile (vramer-hw-profile/1)")
    ap.add_argument("--model", required=True,
                    help="model profile (vramer-model-profile/1)")

    # Budget / residency (mutually complementary).
    ap.add_argument("--vram-budget",
                    help="resident weight budget, e.g. 16GiB")
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

    ap.add_argument("--pcie-gen", type=int,
                    help="override PCIe generation (what-if)")
    ap.add_argument("--pcie-width", type=int,
                    help="override PCIe link width (what-if)")

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

    profile = load_json(args.profile)
    model = load_json(args.model)

    gpus = args.gpus or len(profile.get("gpus", [])) or 1

    bw, per_gpu_bw, bw_label = effective_bandwidth(
        profile, gpus, args.pinned, args.pcie_gen, args.pcie_width
    )

    moe_frac = moe_active_fraction(model) if args.moe else 1.0

    # Mode: matrix
    if args.matrix:
        run_matrix(model, profile, args, gpus, args.pinned, bw, bw_label)
        if args.pcie_sweep:
            run_pcie_sweep(model, profile, args, gpus, args.pinned)
        return

    # Mode: pcie sweep
    if args.pcie_sweep:
        run_pcie_sweep(model, profile, args, gpus, args.pinned)
        return

    # Mode: single config
    resident_bytes, rlabel = resolve_resident_bytes(
        model, args.quant, args, profile, gpus
    )
    est = estimate(model, args.quant, resident_bytes, args.batch, bw, moe_frac)

    if args.json:
        out = dict(est)
        out["model"] = model["name"]
        out["gpus"] = gpus
        out["pinned"] = args.pinned
        out["bandwidth_label"] = bw_label
        print(json.dumps(out, indent=2, default=lambda x: None
                         if x == float("inf") else x))
    else:
        print_single(model, profile, args, gpus, args.pinned, bw, bw_label,
                     est, rlabel)


if __name__ == "__main__":
    main()
