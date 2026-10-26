# VRAMER profiles

This directory holds two kinds of JSON profile plus the tooling's example
inputs/outputs. The Python tools in `../src/` consume them:

```
vramer-detect   ->  hardware profile  (vramer-hw-profile/1)
                       (the C++ `vramer probe` tool emits the SAME shape)
vramer-plan     <-  hardware profile + model profile  -> decode tok/s
vramer-score    <-  hardware profile + model profile  -> recommendation
```

- Hardware profiles describe **a machine** (host RAM, NUMA, GPUs, PCIe, NVLink,
  and — when probed — measured H2D/D2H bandwidth).
- Model profiles describe **a model** (params, architecture, per-quant byte
  costs, layers, MoE routing, KV footprint).

---

## 1. Hardware profile — `vramer-hw-profile/1`

Emitted by `vramer-detect` (and, by design, the C++ `vramer probe`). The
planner consumes either producer. Measured bandwidth fields are left `0.0` by
the Python detector and filled by the C++ probe or `vramer_bandwidth.py`;
`vramer-detect --merge <old>` carries existing measured values forward.

```json
{
  "schema": "vramer-hw-profile/1",
  "vramer_version": "0.2.0",
  "timestamp": "2026-10-25T22:35:18Z",
  "host": { "ram_bytes": 0, "numa_nodes": 1,
            "hostname": "", "os": "Linux", "logical_cpus": 0 },
  "gpus": [
    { "index": 0, "name": "NVIDIA A2", "sm": "8.6", "uuid": "", "bdf": "0000:06:10.0",
      "vram_bytes": 0,
      "pcie_gen": 4, "pcie_width": 8, "copy_engines": 2, "numa_node": -1,
      "nvlink_peers": [], "nvlink_gibs": 0.0,
      "h2d_pageable_gibs": 0.0, "h2d_pinned_gibs": 0.0, "d2h_pinned_gibs": 0.0 }
  ]
}
```

| Field | Meaning | Source |
|-------|---------|--------|
| `host.ram_bytes` | total system RAM | `/proc/meminfo` |
| `host.numa_nodes` | NUMA node count | `/sys/devices/system/node`, `lscpu` |
| `gpus[].vram_bytes` | total VRAM | `nvidia-smi memory.total` |
| `gpus[].sm` | compute capability | `nvidia-smi compute_cap` |
| `gpus[].pcie_gen` / `pcie_width` | **max** negotiated link | `nvidia-smi`, else `/sys/.../max_link_speed\|max_link_width` |
| `gpus[].numa_node` | GPU's NUMA node (-1 = none) | `/sys/bus/pci/devices/<bdf>/numa_node` |
| `gpus[].copy_engines` | async copy engines (best-effort, default 2) | heuristic (not Python-reachable) |
| `gpus[].nvlink_peers` / `nvlink_gibs` | NVLink topology / bandwidth | `nvidia-smi nvlink -s` |
| `gpus[].h2d_pinned_gibs` | measured pinned H2D ceiling | C++ probe / `vramer_bandwidth.py` |
| `gpus[].h2d_pageable_gibs` | measured pageable H2D | ditto |
| `gpus[].d2h_pinned_gibs` | measured pinned D2H | ditto |

`current-machine.json` is this repo's detected profile (2× NVIDIA A2, PCIe Gen3
x16). `r730.json` is an older-schema (`vramer_profile_version: 1`) capture kept
for reference; regenerate it with `vramer-detect` to get the new schema.

---

## 2. Model profile — `vramer-model-profile/1`

```json
{
  "schema": "vramer-model-profile/1",
  "name": "Llama-3-70B",
  "params": 70600000000,
  "architecture": "dense",          // "dense" | "moe"
  "n_layers": 80,
  "hidden_size": 8192,
  "bytes_per_param": { "fp16": 2.0, "q8": 1.0, "q4": 0.5, "q4_k": 0.5625 },
  "kv_bytes_per_token": 655360,
  "moe": null                        // or the MoE block below
}
```

MoE models add:

```json
"architecture": "moe",
"moe": { "n_experts": 8, "active_experts": 2, "expert_param_fraction": 0.95 }
```

| Field | Meaning |
|-------|---------|
| `params` | total parameter count (for MoE: **all** experts, i.e. the on-disk size) |
| `architecture` | `dense` or `moe` |
| `bytes_per_param` | per-quant byte cost; overrides the built-in defaults below |
| `n_layers`, `hidden_size` | informational / for KV math |
| `kv_bytes_per_token` | approx fp16 K+V bytes/token across all layers; competes for the VRAM budget but is **not** part of the per-token weight stream |
| `moe.n_experts` / `active_experts` | total vs routed (top-k) experts |
| `moe.expert_param_fraction` | fraction of params living in expert FFNs (~0.95 for Mixtral); blends dense vs sparse streaming |

Built-in `bytes_per_param` defaults: `fp16=2.0, q8=1.0, q4=0.5, q4_k=0.5625`.

Examples here: `qwen14b.json` (14B dense), `llama-70b.json` (70B dense),
`mixtral-8x7b.json` (8x7B MoE). The `qwen14b-plan.json` / `qwen14b-score.json`
files are historical **outputs** of the old tools, not model profiles.

---

## 3. The performance model (the physics)

LLM **decode** (batch=1, one token) reads ~100% of the *dense* model's weights
per token. When weights don't fit in VRAM the spill is re-streamed over PCIe
every token, so decode is PCIe-bound:

```
decode_tok_s(aggregate) = effective_H2D_bandwidth / bytes_streamed_per_token

bytes_streamed_per_token = spill_bytes * moe_active_fraction / batch
spill_bytes              = max(0, weight_bytes_at_quant - resident_bytes)
weight_bytes_at_quant    = params * bytes_per_param(quant)
```

- **Batch**: batched serving reads each weight **once** for the whole batch, so
  aggregate throughput multiplies by `batch` while **per-sequence latency is
  unchanged**:
  `per_sequence_latency = spill_bytes * moe_active_fraction / effective_H2D_bandwidth = batch / decode_tok_s`.
- **MoE**: only routed experts stream. `moe_active_fraction` blends dense layers
  (1.0) with expert tensors (`active_experts / n_experts`) using
  `expert_param_fraction`. Dense models use `1.0`.
- **Fits (`spill <= 0`)**: paging is unneeded. Decode is **VRAM-bandwidth-bound**
  (not PCIe); VRAMER adds no traffic and the tools say so instead of pretending
  it helps.
- **Prefill** is compute-bound and amortizes weight reads over all prompt
  tokens, so it is **not** PCIe-bound; the tools focus on decode.

### Effective H2D bandwidth

Measured fields (`h2d_pinned_gibs` / `h2d_pageable_gibs`) are used when present.
Otherwise a PCIe gen/width table (one-way effective GiB/s) is used:

| bus | GiB/s (pinned) |
|-----|----------------|
| gen3x16 | 12 |
| gen4x8  | 13 |
| gen4x16 | 25 |
| gen5x8  | 25 |
| gen5x16 | 50 |

Pinned is ~2x pageable (pageable ≈ half the pinned ceiling; measured ~5.5 GiB/s
pageable vs ~12 GiB/s pinned ceiling on the A2). Multi-GPU (tensor/pipeline
parallel) **sums** bandwidth across GPUs. The slower of GPU-supported link and
slot caps the real bus.

---

## 4. Example commands

Detect this machine and save a hardware profile:

```bash
vramer-detect --output current-machine.json
# carry forward measured bandwidth from a previous probe:
vramer-detect --merge current-machine.json --output current-machine.json
```

Single decode estimate (what-ifs via flags):

```bash
vramer-plan --profile current-machine.json --model llama-70b.json \
    --quant q4 --batch 8 --vram-budget 24GiB --pinned
```

Matrix mode (quant x batch grid) + PCIe what-if sweep:

```bash
vramer-plan --profile current-machine.json --model llama-70b.json --matrix
vramer-plan --profile current-machine.json --model llama-70b.json \
    --quant q4 --batch 1 --pcie-sweep
```

Recommendation / scorer:

```bash
vramer-score --profile current-machine.json --model llama-70b.json
vramer-score --profile current-machine.json --model mixtral-8x7b.json --json
```

Useful flags (plan & score): `--resident-fraction F`, `--gpus N`,
`--pcie-gen G --pcie-width W` (what-if newer bus), `--pageable`, `--no-moe`.

> All figures are PCIe-bound **upper bounds**. Real decode also pays GPU
> compute, attention over the KV cache, kernel/NCCL overhead, and imperfect
> transfer/compute overlap.
