#!/usr/bin/env python3

"""
VRAMER scorer — a recommendation.

Given a hardware profile (vramer-hw-profile/1) and a model profile
(vramer-model-profile/1), search the config space (quant x batch, at the given
VRAM budget / residency) and report the best decode throughput found, plus an
honest verdict on whether host->GPU paging is worthwhile at all.

Reuses the exact performance model from vramer_plan.py so the numbers agree.
"""

import argparse
import json
import sys
from pathlib import Path

# Import the shared performance model from the planner (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vramer_plan as vp  # noqa: E402


VERSION = "0.2.0"
GIB = 1024 ** 3


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def search(model, profile, args):
    gpus = args.gpus or len(profile.get("gpus", [])) or 1

    bw, per_gpu_bw, bw_label = vp.effective_bandwidth(
        profile, gpus, args.pinned, args.pcie_gen, args.pcie_width
    )

    moe_frac = vp.moe_active_fraction(model) if args.moe else 1.0

    quants = args.quants or ["fp16", "q8", "q4", "q4_k"]
    batches = args.batches or [1, 8, 32]

    configs = []
    best = None
    best_fit = None

    for q in quants:
        resident_bytes, rlabel = vp.resolve_resident_bytes(
            model, q, args, profile, gpus
        )
        for b in batches:
            est = vp.estimate(model, q, resident_bytes, b, bw, moe_frac)
            est["resident_label"] = rlabel
            configs.append(est)

            if est["fits"]:
                # Prefer the highest-precision config that fits (least batch).
                key = (vp.QUANTS.index(q), b)
                if best_fit is None or key < best_fit[0]:
                    best_fit = (key, est)
                continue

            tok = est["decode_tok_s"] or 0.0
            if best is None or tok > (best["decode_tok_s"] or 0.0):
                best = est

    return {
        "gpus": gpus,
        "bandwidth_gibs": bw,
        "bandwidth_label": bw_label,
        "pinned": args.pinned,
        "moe_fraction": moe_frac,
        "configs": configs,
        "best_streaming": best,
        "best_fit": best_fit[1] if best_fit else None,
    }


def recommend(model, result):
    best_fit = result["best_fit"]
    best = result["best_streaming"]

    rec = {
        "vramer_score_version": 1,
        "vramer_version": VERSION,
        "model": model["name"],
        "gpus": result["gpus"],
        "bandwidth_gibs": round(result["bandwidth_gibs"], 2),
        "bandwidth_label": result["bandwidth_label"],
    }

    if best_fit is not None:
        rec["recommendation"] = "fit"
        rec["paging_worthwhile"] = False
        rec["best_config"] = {
            "quant": best_fit["quant"],
            "batch": best_fit["batch"],
            "resident_gib": round(best_fit["resident_bytes"] / GIB, 2),
            "verdict": best_fit["verdict"],
        }
        rec["reason"] = (
            f"The model fits resident at {best_fit['quant']} within the "
            f"budget. Decode is VRAM-bandwidth-bound, not PCIe-bound; "
            f"host paging (VRAMER) adds no benefit at this quant."
        )
    elif best is not None and best.get("decode_tok_s"):
        latency_ms = best["latency_s"] * 1000
        worthwhile = best["decode_tok_s"] >= 1.0
        rec["recommendation"] = "page"
        rec["paging_worthwhile"] = bool(worthwhile)
        rec["best_config"] = {
            "quant": best["quant"],
            "batch": best["batch"],
            "resident_gib": round(best["resident_bytes"] / GIB, 2),
            "spill_gib": round(best["spill_bytes"] / GIB, 2),
            "decode_tok_s": round(best["decode_tok_s"], 2),
            "latency_ms_per_token": round(latency_ms, 1),
            "verdict": best["verdict"],
        }
        if worthwhile:
            rec["reason"] = (
                f"No config fits resident; best streaming config is "
                f"{best['quant']} @ batch {best['batch']} -> "
                f"{best['decode_tok_s']:.1f} tok/s aggregate "
                f"({latency_ms:.0f} ms/token per sequence). PCIe-bound but "
                f"usable — VRAMER paging is worthwhile here."
            )
        else:
            rec["reason"] = (
                f"Best streaming config is {best['quant']} @ batch "
                f"{best['batch']} but only {best['decode_tok_s']:.2f} tok/s "
                f"aggregate ({latency_ms:.0f} ms/token). Firmly PCIe-bound; "
                f"paging works but is slow — quantize harder, add GPUs, or "
                f"move to a faster bus."
            )
    else:
        rec["recommendation"] = "none"
        rec["paging_worthwhile"] = False
        rec["reason"] = "No usable bandwidth in the profile."

    return rec


def print_score(model, result, rec):
    print()
    print("=" * 72)
    print(f" VRAMER {VERSION} — RECOMMENDATION")
    print("=" * 72)

    print()
    print(f"  Model        : {rec['model']}  ({model['params']/1e9:.0f}B, "
          f"{model.get('architecture','dense')})")
    print(f"  GPUs         : {rec['gpus']}")
    print(f"  Bandwidth    : {rec['bandwidth_gibs']:.1f} GiB/s "
          f"[{rec['bandwidth_label']}]")
    if result["moe_fraction"] < 1.0:
        print(f"  MoE fraction : {result['moe_fraction']:.3f} "
              f"(active expert streaming)")

    print()
    print("  CONFIG SPACE (aggregate tok/s)")
    print(f"    {'quant':>6} {'batch':>6} {'resident':>10} {'spill':>10} "
          f"{'tok/s':>9} {'ms/tok':>8}  verdict")
    print("    " + "-" * 66)
    for est in result["configs"]:
        resident = vp.human_bytes(est["resident_bytes"])
        spill = vp.human_bytes(est["spill_bytes"])
        if est["fits"]:
            toks = "fits"
            ms = "-"
        else:
            toks = f"{est['decode_tok_s']:.1f}" if est["decode_tok_s"] else "0"
            ms = (f"{est['latency_s']*1000:.0f}"
                  if est["latency_s"] not in (None, float("inf")) else "-")
        print(f"    {est['quant']:>6} {est['batch']:>6} {resident:>10} "
              f"{spill:>10} {toks:>9} {ms:>8}  {est['verdict']}")

    print()
    print("  BEST CONFIG")
    bc = rec.get("best_config", {})
    for k, v in bc.items():
        print(f"    {k:<22}: {v}")

    print()
    print(f"  PAGING WORTHWHILE : {rec['paging_worthwhile']}")
    print()
    print("  VERDICT")
    for line in _wrap(rec["reason"], 68):
        print(f"    {line}")
    print()
    print("  NOTE: PCIe-bound upper bounds. Real decode also pays compute,")
    print("  attention over the KV cache, and imperfect transfer/compute")
    print("  overlap. Aggregate tok/s scales with batch; per-sequence latency")
    print("  does not.")
    print()


def _wrap(text, width):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def build_parser():
    ap = argparse.ArgumentParser(
        description="VRAMER recommendation / scorer"
    )
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model", required=True)

    ap.add_argument("--vram-budget")
    ap.add_argument("--resident-fraction", type=float)

    ap.add_argument("--gpus", type=int, default=0)

    staging = ap.add_mutually_exclusive_group()
    staging.add_argument("--pinned", dest="pinned", action="store_true")
    staging.add_argument("--pageable", dest="pinned", action="store_false")
    ap.set_defaults(pinned=True)

    ap.add_argument("--pcie-gen", type=int)
    ap.add_argument("--pcie-width", type=int)

    ap.add_argument("--moe", dest="moe", action="store_true", default=True)
    ap.add_argument("--no-moe", dest="moe", action="store_false")

    ap.add_argument("--quants", nargs="+",
                    help="quants to search (default fp16 q8 q4 q4_k)")
    ap.add_argument("--batches", nargs="+", type=int,
                    help="batches to search (default 1 8 32)")

    ap.add_argument("--json", action="store_true")
    ap.add_argument("--output")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    profile = load_json(args.profile)
    model = load_json(args.model)

    result = search(model, profile, args)
    rec = recommend(model, result)

    if args.json:
        print(json.dumps(rec, indent=2))
    else:
        print_score(model, result, rec)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"Score saved: {args.output}")


if __name__ == "__main__":
    main()
