# VRAMER tuning & FAQ — will it help me, and how fast will it be?

VRAMER lets a GPU work on models **larger than its VRAM** by paging weights from
host memory (or a file) over PCIe. Whether that's *fast enough* depends entirely
on your access pattern. This guide is the honest version.

## The one equation

```
decode tokens/sec  ≈  H2D_bandwidth  /  bytes_streamed_per_token
```

- **`H2D_bandwidth`** — your PCIe host→device speed. Measure it: `vramer probe`.
- **`bytes_streamed_per_token`** — how much weight has to cross PCIe per token:
  ```
  bytes/token = spill_bytes * moe_active_fraction / batch_size
  spill_bytes = max(0, model_bytes_at_quant - resident_bytes)
  ```

Everything below is a way to shrink the numerator's denominator or raise the bandwidth.

## Why decode is the hard case (the swap analogy, and where it breaks)

VRAMER *is* "swap for VRAM." But swap is fast because of **locality** — you rarely
touch what's on disk. **Autoregressive decode has no locality**: generating each
token reads ~100% of a dense model's weights once. So the spilled fraction is
re-fetched **every token** at PCIe speed. The resident fraction is served from
cache (use MRU — see below); the spill is a per-token tax. This is physics, not a
VRAMER limitation — every offloading approach (DeepSpeed ZeRO-Infinity, FSDP,
`llama.cpp --n-gpu-layers`) hits the same wall.

## When VRAMER is a good idea

| Regime | Verdict |
|---|---|
| Model **fits** in VRAM | Don't use paging — it only adds overhead. `vramer plan` will tell you. |
| Model **slightly exceeds** VRAM (small spill) | 👍 Great fit — stretch VRAM 10–30% for a modest cost. |
| **Batched serving** (many sequences) | 👍 Weights read once per batch → aggregate throughput scales with batch size. |
| **MoE** models | 👍 Only active experts stream — much smaller spill. |
| **Prefill / fine-tuning (QLoRA)** | 👍 Compute-heavy, so the transfer hides behind compute. |
| **Interactive single-stream chat, model 2–3× VRAM** | 👎 PCIe-bound, sub-1 tok/s. Quantize, add VRAM, or batch. |

## The levers, most impactful first

1. **Quantize.** 4-bit weights = 4× less PCIe traffic than fp16. The single biggest lever.
2. **Batch** (for serving). Aggregate throughput ≈ `batch × bandwidth / spill`.
3. **Shrink the spill**: raise the budget, or page only the big tensors (keep embeddings/norms/attention resident, page the FFN — the FFN is ~⅔ of the weights).
4. **MRU eviction** (`--eviction mru`). For the cyclic layer scan, LRU gives **0% reuse**; MRU keeps a stable resident prefix (hit rate ≈ resident fraction). Use MRU whenever you page.
5. **Pinned source** (`--source pinned`, or a pinned borrowed source). Removes the CPU source→staging copy. Big win **if** you're copy-bound — `vramer probe` tells you (if `h2d_pinned` ≫ `h2d_pageable`, pin; if they're equal you're bus-bound and pinning won't help).
6. **Prefetch** (`--prefetch-workers 1 --prefetch-ahead N`). Hides transfer latency behind compute. Helps compute-bound / predictable access; little for trivial single-stream streaming.
7. **Multi-GPU** (`--gpus 0,1`). Pool VRAM and sum bandwidth; with NVLink, stream peer-to-peer instead of over the host bus.
8. **Page size.** 64 MiB is the measured sweet spot on Ampere; 16 MiB loses ~10% to per-transfer overhead.

## Measure, don't assume (the workflow)

```bash
# 1. What's my real bandwidth, and am I bus-bound or copy-bound?
vramer probe --emit-profile hw.json
python3 ../../src/vramer_detect.py --merge hw.json --output profiles/me.json

# 2. Will paging help for my model, and at what speed?
python3 ../../src/vramer_plan.py --profile profiles/me.json \
        --model profiles/llama-70b.json --gpus 2 --pinned --matrix

# 3. What would a newer bus buy me?
python3 ../../src/vramer_plan.py --profile profiles/me.json \
        --model profiles/llama-70b.json --quant q4 --pcie-sweep

# 4. Best config for my hardware?
python3 ../../src/vramer_score.py --profile profiles/me.json --model profiles/llama-70b.json
```

## Gotchas we measured

- **Pinned ≈ pageable → you're PCIe-link-bound.** On a virtualized/passed-through
  A2 we measured ~5.5 GiB/s pageable and only ~5.7 pinned: the link, not the copy,
  was the ceiling, so pinning bought ~3%. `probe` catches this.
- **Low bandwidth for your PCIe gen?** A Gen3 ×16 link should do ~12 GiB/s; if you
  measure ~half that, the link may be downtrained or VM-limited — a possible free
  ~2× on bare metal before buying anything. Check `nvidia-smi -q | grep -A3 "Link"`.
- **A prefetch queue with no bound thrashes** — VRAMER caps it automatically; don't
  prefetch further ahead than the cache can hold.
- **`acquire_mutable` requires a writable source** (`AnonymousRamSource` /
  `PinnedHostSource`), not an mmap'd read-only file.
