#!/usr/bin/env python3

import argparse
import ctypes
import datetime
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


VERSION = "0.1.0"


def run(cmd):
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


def system_ram_bytes():
    # Linux / Unix
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except Exception:
        pass

    # Windows
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.ullTotalPhys)
    except Exception:
        return 0


def parse_nvidia_gpus():
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.free,pci.bus_id,"
        "pcie.link.gen.current,pcie.link.gen.max,"
        "pcie.link.width.current,pcie.link.width.max",
        "--format=csv,noheader,nounits",
    ]

    out = run(query)

    if not out:
        return []

    gpus = []

    for line in out.splitlines():
        fields = [x.strip() for x in line.split(",")]

        if len(fields) < 10:
            continue

        try:
            gpu = {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "vram_total_mib": int(fields[3]),
                "vram_free_mib": int(fields[4]),
                "pci_bus_id": fields[5],
                "pcie_gen_current": fields[6],
                "pcie_gen_max": fields[7],
                "pcie_width_current": fields[8],
                "pcie_width_max": fields[9],
            }
        except Exception:
            continue

        gpu["vram_total_gib"] = round(
            gpu["vram_total_mib"] / 1024.0, 2
        )
        gpu["vram_free_gib"] = round(
            gpu["vram_free_mib"] / 1024.0, 2
        )

        gpus.append(gpu)

    return gpus


def get_topology():
    topo = run(["nvidia-smi", "topo", "-m"])
    return topo or ""


def get_p2p_matrix(kind):
    out = run(["nvidia-smi", "topo", "-p2p", kind])
    return out or ""


def detect():
    total_ram = system_ram_bytes()

    profile = {
        "vramer_profile_version": 1,
        "vramer_version": VERSION,
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "host": {
            "hostname": platform.node(),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "ram_bytes": total_ram,
            "ram_gib": round(total_ram / (1024**3), 2)
                if total_ram else 0,
        },
        "cuda": {
            "nvidia_smi_available":
                shutil.which("nvidia-smi") is not None,
        },
        "gpus": parse_nvidia_gpus(),
        "topology": {
            "nvidia_smi_topo": get_topology(),
            "p2p_read": get_p2p_matrix("r"),
            "p2p_write": get_p2p_matrix("w"),
        },
    }

    profile["summary"] = {
        "gpu_count": len(profile["gpus"]),
        "total_vram_gib": round(
            sum(g["vram_total_gib"] for g in profile["gpus"]),
            2,
        ),
        "system_ram_gib": profile["host"]["ram_gib"],
    }

    return profile


def print_human(profile):
    print()
    print("=" * 66)
    print(f" VRAMER {VERSION} — HARDWARE DETECTION")
    print("=" * 66)

    host = profile["host"]

    print()
    print("HOST")
    print(f"  OS           : {host['os']} {host['os_release']}")
    print(f"  Architecture : {host['machine']}")
    print(f"  CPUs         : {host['logical_cpus']}")
    print(f"  System RAM   : {host['ram_gib']:.2f} GiB")

    print()
    print("GPUS")

    if not profile["gpus"]:
        print("  No NVIDIA GPUs detected.")
    else:
        for gpu in profile["gpus"]:
            print()
            print(f"  GPU {gpu['index']}: {gpu['name']}")
            print(
                f"    VRAM       : "
                f"{gpu['vram_total_gib']:.2f} GiB"
            )
            print(
                f"    Free       : "
                f"{gpu['vram_free_gib']:.2f} GiB"
            )
            print(f"    PCI Bus    : {gpu['pci_bus_id']}")
            print(
                f"    PCIe       : Gen{gpu['pcie_gen_current']} "
                f"x{gpu['pcie_width_current']} current"
            )
            print(
                f"    PCIe max   : Gen{gpu['pcie_gen_max']} "
                f"x{gpu['pcie_width_max']}"
            )

    print()
    print("SUMMARY")
    print(
        f"  GPU count    : "
        f"{profile['summary']['gpu_count']}"
    )
    print(
        f"  Total VRAM   : "
        f"{profile['summary']['total_vram_gib']:.2f} GiB"
    )
    print(
        f"  System RAM   : "
        f"{profile['summary']['system_ram_gib']:.2f} GiB"
    )

    if profile["topology"]["nvidia_smi_topo"]:
        print()
        print("GPU TOPOLOGY")
        print(profile["topology"]["nvidia_smi_topo"])

    print()


def main():
    parser = argparse.ArgumentParser(
        description="VRAMER hardware detector"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="print JSON profile",
    )

    parser.add_argument(
        "--output",
        help="write hardware profile to this file",
    )

    args = parser.parse_args()

    profile = detect()

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
