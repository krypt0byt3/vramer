#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

VERSION = "0.1.0"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_bandwidth(profile):
    try:
        return profile["benchmarks"]["host_gpu_bandwidth"]
    except KeyError:
        return None


def score(profile, plan):
    bw = get_bandwidth(profile)

    result = {
        "vramer_score_version": 1,
        "vramer_version": VERSION,
        "bandwidth_available": bw is not None,
    }

    if not bw:
        result["status"] = "no_bandwidth_benchmark"
        return result

    selected = [
        str(x)
        for x in plan["hardware"]["selected_gpu_ids"]
    ]

    per_gpu = bw.get("per_gpu", {})

    h2d = {}

    for gpu in selected:
        if gpu in per_gpu:
            h2d[gpu] = float(
                per_gpu[gpu].get(
                    "h2d_gib_s",
                    0.0,
                )
            )

    aggregate = float(
        bw.get(
            "concurrent_h2d", {}
        ).get(
            "aggregate_gib_s",
            0.0,
        )
    )

    sum_individual = sum(h2d.values())

    scaling = (
        aggregate / sum_individual
        if sum_individual > 0
        else 0.0
    )

    memory = plan["memory"]["residency"]

    overflow = float(
        memory.get(
            "overflow_gib",
            0.0,
        )
    )

    gpu_count = len(selected)

    pp = int(
        plan["parallel"].get(
            "pp",
            1,
        )
    )

    tp = int(
        plan["parallel"].get(
            "tp",
            1,
        )
    )

    # --------------------------------------------------------
    # Ideal concurrent host-transfer floor
    # --------------------------------------------------------

    if aggregate > 0 and overflow > 0:
        ideal_concurrent_s = (
            overflow / aggregate
        )
    else:
        ideal_concurrent_s = 0.0

    # --------------------------------------------------------
    # Pipeline stage transfer floor
    #
    # Assume approximately equal weight distribution between
    # PP stages. Each stage has to consume its share of the
    # RAM-backed weights.
    #
    # This is transfer-only. It does NOT include:
    #   GPU compute
    #   NCCL
    #   kernel launch overhead
    #   TensorRT scheduling
    #   KV cache
    #   imperfect overlap
    # --------------------------------------------------------

    pp_sequential_s = 0.0

    if (
        overflow > 0
        and pp > 1
        and gpu_count > 0
    ):
        overflow_per_stage = (
            overflow / pp
        )

        stage_devices = selected[:pp]

        for gpu in stage_devices:
            speed = h2d.get(
                gpu,
                0.0,
            )

            if speed > 0:
                pp_sequential_s += (
                    overflow_per_stage
                    / speed
                )

    elif (
        overflow > 0
        and gpu_count == 1
    ):
        gpu = selected[0]
        speed = h2d.get(
            gpu,
            0.0,
        )

        if speed > 0:
            pp_sequential_s = (
                overflow / speed
            )

    # --------------------------------------------------------
    # Transfer-only throughput ceilings
    # --------------------------------------------------------

    ideal_tok_s = (
        1.0 / ideal_concurrent_s
        if ideal_concurrent_s > 0
        else None
    )

    pp_tok_s = (
        1.0 / pp_sequential_s
        if pp_sequential_s > 0
        else None
    )

    # --------------------------------------------------------
    # Host feed classification
    # --------------------------------------------------------

    if scaling >= 0.90:
        feed_class = "excellent"
    elif scaling >= 0.75:
        feed_class = "good"
    elif scaling >= 0.50:
        feed_class = "shared_bottleneck"
    else:
        feed_class = "severe_shared_bottleneck"

    # --------------------------------------------------------
    # Strategy observations
    # --------------------------------------------------------

    observations = []

    if scaling >= 0.90 and gpu_count > 1:
        observations.append(
            "Concurrent host-to-GPU bandwidth scales "
            "almost linearly across the selected GPUs."
        )

    if (
        pp > 1
        and not plan["hardware"].get(
            "p2p_complete",
            False,
        )
        and scaling >= 0.75
    ):
        observations.append(
            "Pipeline parallelism is favorable because "
            "direct P2P is unavailable while host feeding "
            "scales well across GPUs."
        )

    if overflow <= 0:
        observations.append(
            "The planned model is fully GPU resident; "
            "host weight bandwidth should not be a primary "
            "decode bottleneck."
        )

    if overflow > 0:
        observations.append(
            "Transfer-floor estimates represent only "
            "RAM-to-GPU weight movement and are not "
            "predicted end-to-end token latency."
        )

    result.update({
        "status": "ready",

        "selected_gpus": selected,

        "host_feed": {
            "per_gpu_h2d_gib_s": h2d,
            "sum_individual_h2d_gib_s":
                round(sum_individual, 3),

            "aggregate_h2d_gib_s":
                round(aggregate, 3),

            "concurrent_scaling_efficiency":
                round(scaling, 4),

            "classification":
                feed_class,
        },

        "model_memory": {
            "ram_backed_weights_gib":
                round(overflow, 3),
        },

        "transfer_floor": {
            "ideal_concurrent_seconds_per_token":
                round(ideal_concurrent_s, 4),

            "pipeline_sequential_seconds_per_token":
                round(pp_sequential_s, 4),

            "ideal_concurrent_transfer_tok_s":
                (
                    round(ideal_tok_s, 3)
                    if ideal_tok_s
                    else None
                ),

            "pipeline_transfer_tok_s":
                (
                    round(pp_tok_s, 3)
                    if pp_tok_s
                    else None
                ),
        },

        "parallel": {
            "tp": tp,
            "pp": pp,
        },

        "observations": observations,
    })

    return result


def print_score(result):
    print()
    print("=" * 68)
    print(" VRAMER 0.1.0 — PERFORMANCE SCORE")
    print("=" * 68)

    if result["status"] != "ready":
        print()
        print("No bandwidth benchmark available.")
        print()
        return

    feed = result["host_feed"]
    floor = result["transfer_floor"]
    mem = result["model_memory"]

    print()
    print("HOST → GPU")

    for gpu, speed in (
        feed["per_gpu_h2d_gib_s"].items()
    ):
        print(
            f"  GPU {gpu:<3}       "
            f"{speed:.3f} GiB/s"
        )

    print()
    print(
        f"  Sum links     "
        f"{feed['sum_individual_h2d_gib_s']:.3f} GiB/s"
    )

    print(
        f"  Concurrent    "
        f"{feed['aggregate_h2d_gib_s']:.3f} GiB/s"
    )

    print(
        f"  Scaling       "
        f"{feed['concurrent_scaling_efficiency'] * 100:.1f}%"
    )

    print(
        f"  Classification: "
        f"{feed['classification']}"
    )

    print()
    print("MODEL STREAMING")

    print(
        f"  RAM-backed weights : "
        f"{mem['ram_backed_weights_gib']:.2f} GiB"
    )

    print()
    print("TRANSFER-ONLY FLOOR")

    print(
        f"  Ideal concurrent   : "
        f"{floor['ideal_concurrent_seconds_per_token']:.3f} s/token"
    )

    if floor[
        "ideal_concurrent_transfer_tok_s"
    ] is not None:
        print(
            f"                       "
            f"{floor['ideal_concurrent_transfer_tok_s']:.2f} tok/s"
        )

    print(
        f"  PP sequential      : "
        f"{floor['pipeline_sequential_seconds_per_token']:.3f} s/token"
    )

    if floor[
        "pipeline_transfer_tok_s"
    ] is not None:
        print(
            f"                       "
            f"{floor['pipeline_transfer_tok_s']:.2f} tok/s"
        )

    print()
    print("NOTE")
    print(
        "  These are PCIe transfer-only limits."
    )
    print(
        "  Actual inference also includes compute,"
    )
    print(
        "  synchronization, NCCL, runtime overhead,"
    )
    print(
        "  and imperfect transfer/compute overlap."
    )

    print()
    print("OBSERVATIONS")

    for item in result["observations"]:
        print(f"  - {item}")

    print()


def main():
    ap = argparse.ArgumentParser(
        description="VRAMER performance scorer"
    )

    ap.add_argument(
        "--profile",
        required=True,
    )

    ap.add_argument(
        "--plan",
        required=True,
    )

    ap.add_argument(
        "--json",
        action="store_true",
    )

    ap.add_argument(
        "--output",
    )

    args = ap.parse_args()

    profile = load_json(args.profile)
    plan = load_json(args.plan)

    result = score(
        profile,
        plan,
    )

    if args.json:
        print(
            json.dumps(
                result,
                indent=2,
            )
        )
    else:
        print_score(result)

    if args.output:
        save_json(
            args.output,
            result,
        )

        print(
            f"Score saved: {args.output}"
        )


if __name__ == "__main__":
    main()
