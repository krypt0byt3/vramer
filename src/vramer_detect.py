#!/usr/bin/env python3

"""
VRAMER hardware detector.

Emits a `vramer-hw-profile/1` document that describes the host and every GPU:
VRAM, PCIe gen/width, NUMA placement, NVLink peers, and (left for the C++
`vramer probe` tool, or the Python `vramer_bandwidth.py` helper) measured
host<->device copy bandwidth.

The schema is intentionally the SAME shape the C++ `vramer probe` tool emits,
so `vramer_plan.py` / `vramer_score.py` consume either producer transparently.

Everything is best-effort: missing tools (`nvidia-smi`, `lscpu`) and missing
sysfs nodes degrade gracefully to zeros / defaults instead of raising.
"""

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


VERSION = "0.2.0"
SCHEMA = "vramer-hw-profile/1"

GIB = 1024 ** 3
MIB = 1024 ** 2

# Map PCIe link speed (GT/s) -> PCIe generation, for the sysfs fallback path.
_GTS_TO_GEN = {
    "2.5": 1,
    "5.0": 2,
    "5": 2,
    "8.0": 3,
    "8": 3,
    "16.0": 4,
    "16": 4,
    "32.0": 5,
    "32": 5,
    "64.0": 6,
    "64": 6,
}


# ------------------------------------------------------------
# Shell helpers
# ------------------------------------------------------------

def run(cmd):
    """Run a command, returning stripped stdout on success else None."""
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return None


def read_text(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None


def read_int(path):
    raw = read_text(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


# ------------------------------------------------------------
# Host facts
# ------------------------------------------------------------

def host_ram_bytes():
    # /proc/meminfo is the canonical Linux source.
    raw = read_text("/proc/meminfo")
    if raw:
        m = re.search(r"MemTotal:\s+(\d+)\s*kB", raw)
        if m:
            return int(m.group(1)) * 1024
    # Portable fallback.
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except Exception:
        return 0


def numa_node_count():
    # Preferred: count node* dirs under /sys.
    try:
        nodes = [
            n for n in os.listdir("/sys/devices/system/node")
            if re.fullmatch(r"node\d+", n)
        ]
        if nodes:
            return len(nodes)
    except Exception:
        pass

    # Fallback: lscpu.
    out = run(["lscpu"])
    if out:
        m = re.search(r"NUMA node\(s\):\s+(\d+)", out)
        if m:
            return int(m.group(1))

    return 1


# ------------------------------------------------------------
# PCIe / BDF helpers
# ------------------------------------------------------------

def normalize_bdf(bus_id):
    """
    nvidia-smi reports BDF as '00000000:06:10.0' (8-digit domain).
    sysfs uses '0000:06:10.0' (4-digit domain). Normalize to sysfs form.
    """
    if not bus_id:
        return None
    b = bus_id.strip().lower()
    parts = b.split(":")
    if len(parts) == 3:
        domain, bus, devfn = parts
        try:
            domain = f"{int(domain, 16):04x}"
        except Exception:
            domain = domain[-4:].rjust(4, "0")
        return f"{domain}:{bus}:{devfn}"
    return b


def sysfs_pcie(bdf):
    """
    Best-effort PCIe gen/width/numa straight from sysfs for one BDF.
    Returns (gen, width, numa_node) with None where unavailable.
    """
    gen = None
    width = None
    numa = None

    if not bdf:
        return gen, width, numa

    base = Path("/sys/bus/pci/devices") / bdf

    # max_link_speed looks like "8.0 GT/s PCIe"
    speed = read_text(base / "max_link_speed")
    if speed:
        tok = speed.split()[0]
        gen = _GTS_TO_GEN.get(tok)

    width = read_int(base / "max_link_width")
    if width is not None and width <= 0:
        width = None

    numa = read_int(base / "numa_node")

    return gen, width, numa


# ------------------------------------------------------------
# NVIDIA GPU discovery
# ------------------------------------------------------------

def _sm_from_compute_cap(raw):
    # nvidia-smi compute_cap is e.g. "8.6"; pass through, else "".
    if raw and re.fullmatch(r"\d+\.\d+", raw):
        return raw
    return ""


def query_gpus():
    """
    Build the `gpus` array using nvidia-smi for name/vram/pcie/compute_cap,
    sysfs for the authoritative PCIe gen/width/numa, and nvidia-smi nvlink
    for NVLink peers.
    """
    fields = [
        "index",
        "name",
        "uuid",
        "pci.bus_id",
        "memory.total",
        "compute_cap",
        "pcie.link.gen.max",
        "pcie.link.width.max",
    ]
    query = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]

    out = run(query)
    if not out:
        return []

    gpus = []

    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < len(fields):
            continue

        try:
            index = int(cols[0])
        except Exception:
            continue

        name = cols[1]
        uuid = cols[2]
        bus_id = cols[3]

        try:
            vram_mib = int(cols[4])
        except Exception:
            vram_mib = 0

        sm = _sm_from_compute_cap(cols[5])

        # nvidia-smi gen/width (integer gen, integer width).
        smi_gen = _safe_int(cols[6])
        smi_width = _safe_int(cols[7])

        bdf = normalize_bdf(bus_id)
        sys_gen, sys_width, numa = sysfs_pcie(bdf)

        # Prefer nvidia-smi gen/width (reflects the negotiated max link),
        # fall back to sysfs where nvidia-smi gave nothing.
        pcie_gen = smi_gen if smi_gen else (sys_gen or 0)
        pcie_width = smi_width if smi_width else (sys_width or 0)

        gpu = {
            "index": index,
            "name": name,
            "uuid": uuid,
            "sm": sm,
            "bdf": bdf or "",
            "vram_bytes": vram_mib * MIB,
            "pcie_gen": pcie_gen,
            "pcie_width": pcie_width,
            "copy_engines": copy_engines_for(name),
            "numa_node": numa if numa is not None else -1,
            "nvlink_peers": [],
            "nvlink_gibs": 0.0,
            # Measured bandwidth fields. Left 0.0 here; the C++ probe or
            # vramer_bandwidth.py fills them (and --merge carries them over).
            "h2d_pageable_gibs": 0.0,
            "h2d_pinned_gibs": 0.0,
            "d2h_pinned_gibs": 0.0,
        }

        gpus.append(gpu)

    annotate_nvlink(gpus)

    return gpus


def _safe_int(raw):
    try:
        return int(float(raw))
    except Exception:
        return 0


def copy_engines_for(name):
    """
    cudaDeviceProp.asyncEngineCount isn't reachable from pure Python.
    Best-effort: 2 is correct for A2 and most modern datacenter parts.
    """
    return 2


def annotate_nvlink(gpus):
    """
    Fill nvlink_peers / nvlink_gibs from `nvidia-smi nvlink -s`.
    Silently leaves the defaults (no NVLink) when unsupported.
    """
    out = run(["nvidia-smi", "nvlink", "-s"])
    if not out:
        return

    # Output is grouped per-GPU: "GPU 0: NVIDIA A2 (...)" then per-link lines.
    current = None
    link_speeds = {}

    for line in out.splitlines():
        m = re.match(r"GPU (\d+):", line)
        if m:
            current = int(m.group(1))
            link_speeds.setdefault(current, [])
            continue
        if current is None:
            continue
        # "Link 0: 26.562 GB/s" style lines indicate active links.
        lm = re.search(r"Link \d+:\s+([\d.]+)\s*GB/s", line)
        if lm:
            link_speeds[current].append(float(lm.group(1)))

    by_index = {g["index"]: g for g in gpus}
    for idx, speeds in link_speeds.items():
        if idx in by_index and speeds:
            g = by_index[idx]
            # Aggregate per-direction NVLink bandwidth (GB/s -> GiB/s approx).
            total_gbs = sum(speeds)
            g["nvlink_gibs"] = round(total_gbs * (1e9 / GIB), 3)
            # Without per-peer parsing, mark all other GPUs as candidate peers.
            g["nvlink_peers"] = [
                o["index"] for o in gpus if o["index"] != idx
            ]


# ------------------------------------------------------------
# Profile assembly
# ------------------------------------------------------------

def detect():
    ram = host_ram_bytes()

    profile = {
        "schema": SCHEMA,
        "vramer_version": VERSION,
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "host": {
            "hostname": platform.node(),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count() or 0,
            "ram_bytes": ram,
            "numa_nodes": numa_node_count(),
        },
        "gpus": query_gpus(),
    }

    return profile


def merge_measured(profile, existing):
    """
    Carry measured bandwidth (and NVLink) fields from an existing hw profile
    into a freshly detected one, matching GPUs by uuid then by index.
    The C++ probe / bandwidth tool own those fields; detect must not clobber
    them on a re-run.
    """
    old_gpus = existing.get("gpus", [])
    by_uuid = {g.get("uuid"): g for g in old_gpus if g.get("uuid")}
    by_index = {g.get("index"): g for g in old_gpus}

    measured_keys = (
        "h2d_pageable_gibs",
        "h2d_pinned_gibs",
        "d2h_pinned_gibs",
        "nvlink_gibs",
    )

    for gpu in profile.get("gpus", []):
        old = by_uuid.get(gpu.get("uuid")) or by_index.get(gpu.get("index"))
        if not old:
            continue
        for key in measured_keys:
            val = old.get(key, 0.0)
            if val:
                gpu[key] = val
        # Preserve discovered NVLink peers if the new run found none.
        if not gpu.get("nvlink_peers") and old.get("nvlink_peers"):
            gpu["nvlink_peers"] = old["nvlink_peers"]

    return profile


# ------------------------------------------------------------
# Human-readable output
# ------------------------------------------------------------

def print_human(profile):
    host = profile["host"]

    print()
    print("=" * 70)
    print(f" VRAMER {VERSION} — HARDWARE PROFILE  ({profile['schema']})")
    print("=" * 70)

    print()
    print("HOST")
    print(f"  OS           : {host['os']} {host['os_release']}")
    print(f"  Arch         : {host['machine']}")
    print(f"  Logical CPUs : {host['logical_cpus']}")
    print(f"  System RAM   : {host['ram_bytes'] / GIB:.2f} GiB")
    print(f"  NUMA nodes   : {host['numa_nodes']}")

    print()
    print("GPUS")
    if not profile["gpus"]:
        print("  (none detected — nvidia-smi unavailable or no NVIDIA GPU)")
    for g in profile["gpus"]:
        print()
        print(f"  GPU {g['index']}: {g['name']}  (SM {g['sm'] or '?'})")
        print(f"    VRAM         : {g['vram_bytes'] / GIB:.2f} GiB")
        print(f"    PCIe         : Gen{g['pcie_gen']} x{g['pcie_width']}")
        print(f"    Copy engines : {g['copy_engines']}")
        print(f"    NUMA node    : {g['numa_node']}")
        if g["nvlink_peers"]:
            print(
                f"    NVLink       : peers {g['nvlink_peers']} "
                f"@ {g['nvlink_gibs']:.1f} GiB/s"
            )
        else:
            print("    NVLink       : none")
        measured = (
            g["h2d_pinned_gibs"]
            or g["h2d_pageable_gibs"]
            or g["d2h_pinned_gibs"]
        )
        if measured:
            print(
                "    Measured H2D : "
                f"pageable {g['h2d_pageable_gibs']:.2f} / "
                f"pinned {g['h2d_pinned_gibs']:.2f} GiB/s"
            )
        else:
            print(
                "    Measured BW  : none "
                "(run vramer probe / vramer_bandwidth.py)"
            )

    print()
    print("SUMMARY")
    print(f"  GPU count    : {len(profile['gpus'])}")
    total_vram = sum(g["vram_bytes"] for g in profile["gpus"])
    print(f"  Total VRAM   : {total_vram / GIB:.2f} GiB")
    print(f"  System RAM   : {host['ram_bytes'] / GIB:.2f} GiB")
    print()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="VRAMER hardware detector (emits vramer-hw-profile/1)"
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="print the JSON profile to stdout",
    )
    ap.add_argument(
        "--output",
        help="write the hardware profile to this file",
    )
    ap.add_argument(
        "--merge",
        help=(
            "an existing hw profile whose measured bandwidth / NVLink fields "
            "should be carried into this detection"
        ),
    )

    args = ap.parse_args()

    profile = detect()

    if args.merge:
        try:
            with open(args.merge, "r") as f:
                existing = json.load(f)
            profile = merge_measured(profile, existing)
        except Exception as exc:
            print(f"(merge skipped: {exc})")

    if args.json:
        print(json.dumps(profile, indent=2))
    else:
        print_human(profile)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"Profile saved: {path}")


if __name__ == "__main__":
    main()
