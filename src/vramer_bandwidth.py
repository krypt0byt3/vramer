#!/usr/bin/env python3

import argparse
import datetime
import json
import threading
import time
from pathlib import Path

try:
    import torch
except ImportError:
    raise SystemExit(
        "PyTorch is not installed. "
        "VRAMER bandwidth probing currently requires torch."
    )


GIB = 1024 ** 3
MIB = 1024 ** 2


def load_profile(path):
    with open(path, "r") as f:
        return json.load(f)


def save_profile(path, profile):
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


def gpu_ids_from_profile(profile):
    return [
        int(g["index"])
        for g in profile.get("gpus", [])
    ]


def allocate_pair(device, size_bytes):
    torch.cuda.set_device(device)

    host = torch.empty(
        size_bytes,
        dtype=torch.uint8,
        pin_memory=True,
    )

    dev = torch.empty(
        size_bytes,
        dtype=torch.uint8,
        device=f"cuda:{device}",
    )

    # Touch the host allocation.
    host.fill_(0x5A)

    return host, dev


def measure_direction(
    device,
    host,
    dev,
    iterations,
    direction,
):
    torch.cuda.set_device(device)

    stream = torch.cuda.Stream(
        device=device
    )

    # Warmup
    with torch.cuda.stream(stream):
        for _ in range(2):
            if direction == "h2d":
                dev.copy_(
                    host,
                    non_blocking=True
                )
            else:
                host.copy_(
                    dev,
                    non_blocking=True
                )

    stream.synchronize()

    start = torch.cuda.Event(
        enable_timing=True
    )
    end = torch.cuda.Event(
        enable_timing=True
    )

    with torch.cuda.stream(stream):
        start.record(stream)

        for _ in range(iterations):
            if direction == "h2d":
                dev.copy_(
                    host,
                    non_blocking=True
                )
            else:
                host.copy_(
                    dev,
                    non_blocking=True
                )

        end.record(stream)

    end.synchronize()

    elapsed_ms = start.elapsed_time(end)

    total_bytes = (
        host.numel()
        * host.element_size()
        * iterations
    )

    seconds = elapsed_ms / 1000.0

    gib_s = (
        total_bytes / GIB / seconds
        if seconds > 0 else 0
    )

    return gib_s


def measure_single_gpu(
    device,
    size_bytes,
    iterations,
):
    host, dev = allocate_pair(
        device,
        size_bytes
    )

    h2d = measure_direction(
        device,
        host,
        dev,
        iterations,
        "h2d",
    )

    d2h = measure_direction(
        device,
        host,
        dev,
        iterations,
        "d2h",
    )

    del dev
    del host

    torch.cuda.empty_cache()

    return {
        "h2d_gib_s": round(h2d, 3),
        "d2h_gib_s": round(d2h, 3),
    }


def measure_concurrent_h2d(
    devices,
    size_bytes,
    iterations,
):
    if not devices:
        return 0.0, {}

    buffers = {}

    for device in devices:
        buffers[device] = allocate_pair(
            device,
            size_bytes
        )

    barrier = threading.Barrier(
        len(devices)
    )

    elapsed = {}
    errors = {}

    def worker(device):
        try:
            torch.cuda.set_device(device)

            host, dev = buffers[device]

            stream = torch.cuda.Stream(
                device=device
            )

            # Warm up this GPU.
            with torch.cuda.stream(stream):
                dev.copy_(
                    host,
                    non_blocking=True
                )

            stream.synchronize()

            barrier.wait()

            t0 = time.perf_counter()

            with torch.cuda.stream(stream):
                for _ in range(iterations):
                    dev.copy_(
                        host,
                        non_blocking=True
                    )

            stream.synchronize()

            elapsed[device] = (
                time.perf_counter() - t0
            )

        except Exception as exc:
            errors[device] = str(exc)

    threads = [
        threading.Thread(
            target=worker,
            args=(device,),
            daemon=True,
        )
        for device in devices
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    if errors:
        print(
            "Concurrent benchmark errors:",
            errors,
        )

    valid = [
        elapsed[d]
        for d in devices
        if d in elapsed
    ]

    if not valid:
        aggregate = 0.0
    else:
        # Threads begin together. Completion time of the
        # slowest transfer approximates the total interval.
        wall = max(valid)

        total_bytes = (
            size_bytes
            * iterations
            * len(valid)
        )

        aggregate = (
            total_bytes / GIB / wall
            if wall > 0 else 0.0
        )

    per_gpu_effective = {}

    for device, seconds in elapsed.items():
        total_bytes = (
            size_bytes
            * iterations
        )

        speed = (
            total_bytes / GIB / seconds
            if seconds > 0 else 0.0
        )

        per_gpu_effective[str(device)] = round(
            speed,
            3
        )

    for device in devices:
        host, dev = buffers[device]
        del dev
        del host

    torch.cuda.empty_cache()

    return (
        round(aggregate, 3),
        per_gpu_effective,
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "VRAMER host/GPU bandwidth probe"
        )
    )

    ap.add_argument(
        "--profile",
        required=True,
        help="VRAMER hardware JSON profile",
    )

    ap.add_argument(
        "--chunk-mib",
        type=int,
        default=256,
        help="transfer chunk size (default: 256 MiB)",
    )

    ap.add_argument(
        "--iterations",
        type=int,
        default=16,
        help="copies per test (default: 16)",
    )

    ap.add_argument(
        "--update",
        action="store_true",
        help="write benchmark results into profile",
    )

    args = ap.parse_args()

    profile = load_profile(
        args.profile
    )

    devices = gpu_ids_from_profile(
        profile
    )

    if not devices:
        raise SystemExit(
            "No GPUs in hardware profile."
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable to PyTorch."
        )

    visible = torch.cuda.device_count()

    print()
    print("=" * 66)
    print(
        " VRAMER 0.1.0 — BANDWIDTH PROBE"
    )
    print("=" * 66)

    print()
    print(
        f"Chunk size : {args.chunk_mib} MiB"
    )
    print(
        f"Iterations : {args.iterations}"
    )
    print(
        f"CUDA GPUs  : {visible}"
    )
    print()

    size_bytes = (
        args.chunk_mib * MIB
    )

    per_gpu = {}

    for device in devices:
        print(
            f"Testing GPU {device}..."
        )

        result = measure_single_gpu(
            device,
            size_bytes,
            args.iterations,
        )

        per_gpu[str(device)] = result

        print(
            f"  H2D: {result['h2d_gib_s']:.3f} GiB/s"
        )
        print(
            f"  D2H: {result['d2h_gib_s']:.3f} GiB/s"
        )

    print()
    print(
        "Testing concurrent H2D..."
    )

    aggregate, concurrent = (
        measure_concurrent_h2d(
            devices,
            size_bytes,
            args.iterations,
        )
    )

    print()
    print("CONCURRENT H2D")

    for gpu, speed in concurrent.items():
        print(
            f"  GPU {gpu}: "
            f"{speed:.3f} GiB/s"
        )

    print(
        f"  Aggregate: "
        f"{aggregate:.3f} GiB/s"
    )

    result = {
        "timestamp":
            datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),

        "provider": "pytorch_cuda",

        "chunk_mib":
            args.chunk_mib,

        "iterations":
            args.iterations,

        "per_gpu":
            per_gpu,

        "concurrent_h2d": {
            "per_gpu_gib_s":
                concurrent,

            "aggregate_gib_s":
                aggregate,
        },
    }

    if args.update:
        profile.setdefault(
            "benchmarks", {}
        )

        profile["benchmarks"][
            "host_gpu_bandwidth"
        ] = result

        save_profile(
            args.profile,
            profile,
        )

        print()
        print(
            "Updated profile:"
        )
        print(
            f"  {args.profile}"
        )

    print()


if __name__ == "__main__":
    main()
