#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path


VERSION = "0.1.0"
GIB = 1024 ** 3


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def gib(n):
    return n / GIB


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def file_size(path):
    try:
        return path.stat().st_size
    except Exception:
        return 0


# ------------------------------------------------------------
# Model inspection
# ------------------------------------------------------------

def inspect_model(path_string):
    path = Path(path_string)

    info = {
        "path": str(path),
        "exists": path.exists(),
        "type": "unknown",
        "format": "unknown",
        "backend_hint": "unknown",
        "ternary": False,
        "size_bytes": 0,
        "size_gib": 0.0,
        "files": [],
    }

    if not path.exists():
        return info

    name = path.name.lower()

    # --------------------------------------------------------
    # Single model file
    # --------------------------------------------------------

    if path.is_file():
        size = file_size(path)

        info["size_bytes"] = size
        info["size_gib"] = round(gib(size), 3)
        info["files"] = [str(path)]

        suffix = path.suffix.lower()

        if suffix == ".gguf":
            info["type"] = "gguf"
            info["format"] = "gguf"

            ternary_terms = (
                "bonsai",
                "bitnet",
                "ternary",
                "1.58",
                "tq1",
                "tq2",
                "ptq1",
                "pq2",
            )

            is_ternary = any(x in name for x in ternary_terms)
            info["ternary"] = is_ternary

            if "bonsai" in name:
                info["backend_hint"] = "prism_llama"

            elif "bitnet" in name or "1.58" in name:
                info["backend_hint"] = "bitnet"

            elif is_ternary:
                info["backend_hint"] = "ternary_auto"

            else:
                info["backend_hint"] = "llama_cpp"

        return info

    # --------------------------------------------------------
    # Directory model
    # --------------------------------------------------------

    safetensors = list(path.glob("*.safetensors"))

    if safetensors:
        total = sum(file_size(p) for p in safetensors)

        info["type"] = "huggingface"
        info["format"] = "safetensors"
        info["backend_hint"] = "tensorrt_llm"
        info["size_bytes"] = total
        info["size_gib"] = round(gib(total), 3)
        info["files"] = [str(p) for p in safetensors]

        # Look at config architecture/name for future ternary HF models.
        config = path / "config.json"

        if config.exists():
            try:
                cfg = load_json(config)
                text = json.dumps(cfg).lower()

                if any(x in text for x in (
                    "bitnet",
                    "ternary",
                    "1.58",
                    "bitlinear",
                )):
                    info["ternary"] = True
                    info["backend_hint"] = "bitnet"
            except Exception:
                pass

    return info


# ------------------------------------------------------------
# GPU selection
# ------------------------------------------------------------

def select_gpus(profile, requested):
    gpus = profile.get("gpus", [])

    if not gpus:
        return []

    if requested == "auto":
        return list(gpus)

    # --gpus 4 means choose best 4
    if requested.isdigit():
        count = int(requested)

        ranked = sorted(
            gpus,
            key=lambda x: (
                x.get("vram_free_mib", 0),
                x.get("vram_total_mib", 0),
            ),
            reverse=True,
        )

        return ranked[:count]

    # --gpus 0,2,3 means exact IDs
    ids = []

    for x in requested.split(","):
        x = x.strip()
        if x:
            ids.append(int(x))

    by_index = {
        int(g["index"]): g
        for g in gpus
    }

    return [
        by_index[i]
        for i in ids
        if i in by_index
    ]


# ------------------------------------------------------------
# P2P / topology parsing
# ------------------------------------------------------------

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def clean_ansi(text):
    return ANSI_RE.sub("", text or "")


def parse_p2p(raw, selected_ids):
    """
    Return True only if every selected GPU pair reports OK.
    """

    if len(selected_ids) <= 1:
        return True

    text = clean_ansi(raw)

    matrix = {}

    for line in text.splitlines():
        parts = line.split()

        if not parts:
            continue

        if not parts[0].startswith("GPU"):
            continue

        try:
            src = int(parts[0][3:])
        except Exception:
            continue

        values = parts[1:]

        for dst, value in enumerate(values):
            matrix[(src, dst)] = value

    for a in selected_ids:
        for b in selected_ids:
            if a == b:
                continue

            if matrix.get((a, b)) != "OK":
                return False

    return True


def has_fast_gpu_link(profile, selected_ids):
    """
    Detect obvious NVLink-style connections from topo output.
    """

    if len(selected_ids) <= 1:
        return False

    topo = clean_ansi(
        profile.get("topology", {}).get(
            "nvidia_smi_topo", ""
        )
    )

    return bool(re.search(r"\bNV\d+\b", topo))


# ------------------------------------------------------------
# Memory policy
# ------------------------------------------------------------

def host_ram_policy(profile):
    total = float(
        profile.get("host", {}).get("ram_gib", 0)
    )

    # Keep RAM for OS, runtime, filesystem cache, etc.
    reserve = max(
        6.0,
        min(16.0, total * 0.20)
    )

    usable = max(0.0, total - reserve)

    return {
        "total_gib": round(total, 2),
        "reserve_gib": round(reserve, 2),
        "usable_gib": round(usable, 2),
    }


def gpu_memory_policy(gpu):
    total = gpu.get("vram_total_mib", 0) / 1024.0
    free = gpu.get("vram_free_mib", 0) / 1024.0

    # Conservative generic runtime reserve.
    #
    # Backend-specific planners can refine this later.
    reserve = max(1.5, total * 0.10)

    usable = max(
        0.0,
        min(free, total - reserve)
    )

    return {
        "index": gpu["index"],
        "total_gib": round(total, 2),
        "free_gib": round(free, 2),
        "reserve_gib": round(reserve, 2),
        "usable_weight_gib": round(usable, 2),
    }


# ------------------------------------------------------------
# Backend selection
# ------------------------------------------------------------

def choose_backend(model, requested):
    if requested != "auto":
        return requested

    return model.get(
        "backend_hint",
        "unknown"
    )


# ------------------------------------------------------------
# Parallel strategy
# ------------------------------------------------------------

def choose_parallel(
    gpu_count,
    p2p_complete,
    fast_link,
    backend,
):
    if gpu_count <= 1:
        return {
            "tp": 1,
            "pp": 1,
            "strategy": "single_gpu",
            "reason":
                "Only one GPU selected.",
        }

    # For strong GPU interconnects, TP becomes viable.
    if p2p_complete and fast_link:
        return {
            "tp": gpu_count,
            "pp": 1,
            "strategy": "tensor_parallel",
            "reason":
                "Selected GPUs have complete peer access "
                "and a fast GPU interconnect.",
        }

    # Generic PCIe/non-P2P case:
    # favor pipeline partitioning.
    return {
        "tp": 1,
        "pp": gpu_count,
        "strategy": "pipeline_parallel",
        "reason":
            "Direct GPU peer access is incomplete or "
            "no fast GPU interconnect was detected.",
    }


# ------------------------------------------------------------
# Residency planner
# ------------------------------------------------------------

def choose_residency(
    model,
    backend,
    gpu_policies,
):
    weight_gib = model["size_gib"]

    if not gpu_policies:
        return {
            "fraction": 0.0,
            "percent": 0,
            "resident_gib": 0.0,
            "overflow_gib": weight_gib,
        }

    capacity = sum(
        x["usable_weight_gib"]
        for x in gpu_policies
    )

    # --------------------------------------------------------
    # Ternary / GGUF
    # --------------------------------------------------------

    if model["ternary"]:
        # If it fits, don't invent an overflow tier.
        if weight_gib <= capacity:
            fraction = 1.0
        else:
            fraction = min(
                0.95,
                capacity / weight_gib
            )

    # --------------------------------------------------------
    # TensorRT native weight streaming
    # --------------------------------------------------------

    elif backend == "tensorrt_llm":
        if weight_gib <= capacity * 0.85:
            fraction = 1.0
        else:
            # Our first validated conservative streaming target.
            fraction = min(
                0.80,
                capacity / weight_gib
            )

    # --------------------------------------------------------
    # Generic GGUF / other engines
    # --------------------------------------------------------

    else:
        if weight_gib <= capacity:
            fraction = 1.0
        else:
            fraction = min(
                0.90,
                capacity / weight_gib
            )

    fraction = max(0.0, min(1.0, fraction))

    resident = weight_gib * fraction
    overflow = max(0.0, weight_gib - resident)

    return {
        "fraction": round(fraction, 4),
        "percent": round(fraction * 100, 1),
        "resident_gib": round(resident, 2),
        "overflow_gib": round(overflow, 2),
        "gpu_capacity_gib": round(capacity, 2),
    }


# ------------------------------------------------------------
# Planner
# ------------------------------------------------------------

def make_plan(
    profile,
    model,
    gpu_request,
    backend_request,
):
    selected = select_gpus(
        profile,
        gpu_request,
    )

    selected_ids = [
        int(g["index"])
        for g in selected
    ]

    topology = profile.get(
        "topology", {}
    )

    p2p_complete = parse_p2p(
        topology.get("p2p_read", ""),
        selected_ids,
    ) and parse_p2p(
        topology.get("p2p_write", ""),
        selected_ids,
    )

    fast_link = has_fast_gpu_link(
        profile,
        selected_ids,
    )

    backend = choose_backend(
        model,
        backend_request,
    )

    parallel = choose_parallel(
        len(selected),
        p2p_complete,
        fast_link,
        backend,
    )

    gpu_policies = [
        gpu_memory_policy(g)
        for g in selected
    ]

    host = host_ram_policy(profile)

    residency = choose_residency(
        model,
        backend,
        gpu_policies,
    )

    overflow_fits = (
        residency["overflow_gib"]
        <= host["usable_gib"]
    )

    return {
        "vramer_plan_version": 1,
        "vramer_version": VERSION,

        "model": model,

        "backend": backend,

        "hardware": {
            "selected_gpu_ids": selected_ids,
            "gpu_count": len(selected),
            "p2p_complete": p2p_complete,
            "fast_gpu_link": fast_link,
            "gpus": gpu_policies,
            "host_memory": host,
        },

        "parallel": parallel,

        "memory": {
            "residency": residency,
            "host_overflow_fits":
                overflow_fits,
        },

        "status":
            "ready"
            if overflow_fits
            else "insufficient_memory",
    }


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

def print_plan(plan):
    model = plan["model"]
    hw = plan["hardware"]
    par = plan["parallel"]
    mem = plan["memory"]["residency"]

    print()
    print("=" * 68)
    print(" VRAMER 0.1.0 — EXECUTION PLAN")
    print("=" * 68)

    print()
    print("MODEL")
    print(f"  Path          : {model['path']}")
    print(f"  Format        : {model['format']}")
    print(f"  Storage       : {model['size_gib']:.2f} GiB")
    print(f"  Ternary       : {model['ternary']}")

    print()
    print("BACKEND")
    print(f"  Selected      : {plan['backend']}")

    print()
    print("GPU SELECTION")
    print(
        "  GPUs          : "
        + ",".join(
            str(x)
            for x in hw["selected_gpu_ids"]
        )
    )
    print(f"  Count         : {hw['gpu_count']}")
    print(f"  Full P2P      : {hw['p2p_complete']}")
    print(f"  Fast link     : {hw['fast_gpu_link']}")

    for gpu in hw["gpus"]:
        print()
        print(f"  GPU {gpu['index']}")
        print(
            f"    VRAM        : "
            f"{gpu['total_gib']:.2f} GiB"
        )
        print(
            f"    Free        : "
            f"{gpu['free_gib']:.2f} GiB"
        )
        print(
            f"    Reserve     : "
            f"{gpu['reserve_gib']:.2f} GiB"
        )
        print(
            f"    Weight cap  : "
            f"{gpu['usable_weight_gib']:.2f} GiB"
        )

    print()
    print("PARALLEL PLAN")
    print(f"  Strategy      : {par['strategy']}")
    print(f"  TP            : {par['tp']}")
    print(f"  PP            : {par['pp']}")
    print(f"  Reason        : {par['reason']}")

    print()
    print("MEMORY PLAN")
    print(
        f"  Residency     : "
        f"{mem['percent']:.1f}%"
    )
    print(
        f"  GPU resident  : "
        f"{mem['resident_gib']:.2f} GiB"
    )
    print(
        f"  RAM overflow  : "
        f"{mem['overflow_gib']:.2f} GiB"
    )

    host = hw["host_memory"]

    print(
        f"  System RAM    : "
        f"{host['total_gib']:.2f} GiB"
    )
    print(
        f"  RAM reserve   : "
        f"{host['reserve_gib']:.2f} GiB"
    )
    print(
        f"  RAM usable    : "
        f"{host['usable_gib']:.2f} GiB"
    )

    print()
    print("STATUS")
    print(f"  {plan['status'].upper()}")
    print()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="VRAMER execution planner"
    )

    ap.add_argument(
        "--profile",
        required=True,
        help="hardware profile from vramer detect",
    )

    ap.add_argument(
        "--model",
        required=True,
        help="model file or directory",
    )

    ap.add_argument(
        "--gpus",
        default="auto",
        help=(
            "auto, GPU count (e.g. 4), "
            "or explicit list (e.g. 0,2,3)"
        ),
    )

    ap.add_argument(
        "--backend",
        default="auto",
        choices=[
            "auto",
            "tensorrt_llm",
            "llama_cpp",
            "prism_llama",
            "bitnet",
        ],
    )

    ap.add_argument(
        "--json",
        action="store_true",
    )

    ap.add_argument(
        "--output",
        help="write plan JSON",
    )

    args = ap.parse_args()

    profile = load_json(args.profile)
    model = inspect_model(args.model)

    if not model["exists"]:
        raise SystemExit(
            f"Model not found: {args.model}"
        )

    plan = make_plan(
        profile,
        model,
        args.gpus,
        args.backend,
    )

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print_plan(plan)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(plan, f, indent=2)

        print(
            f"Plan saved: {args.output}"
        )


if __name__ == "__main__":
    main()
