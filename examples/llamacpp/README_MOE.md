# VRAMER × llama.cpp — MoE expert paging

This demo runs a **Mixture-of-Experts** model with its per-layer expert stacks
living on **host RAM** and only the experts that are **active for each token**
streamed into a small VRAM budget through the VRAMER pager. It is the MoE
counterpart of [`adapter/README.md`](adapter/README.md) (dense weight paging)
and shares the same ~7-line llama.cpp fork.

## Why MoE is the interesting case

A dense model offloaded to host hits a PCIe wall: *every* weight is needed for
*every* token, so the whole offloaded set must stream per token (the prior dense
demo measured **208 → 9.3 tok/s**, ~4.5% of resident, when ~206 MiB of FFN was
paged every token).

A MoE layer routes each token to only a few of its experts. **Qwen1.5-MoE-A2.7B**
has **60 experts/layer, 4 active per token** across **24 layers**. So the model
can keep its whole ~7.7 GiB expert set on host and stream only the ~active
fraction per token — the "no cross-token locality" wall mostly does not apply to
the *size* of what moves, only to the (unavoidable) fact that a fresh expert set
is a compulsory miss each token.

## What the adapter does for MoE

`vramer_llama_copy_hook` is called from `llama_context::sched_copy_experts` (the
MoE expert-copy path). When the tensor being uploaded is a 3-D expert stack
(`ne[2] = n_expert`, e.g. `blk.N.ffn_down_exps.weight`), the hook:

1. reads the token's routing ids from the split graph's `MUL_MAT_ID` node
   (`src[2]`) — exactly how llama.cpp itself picks the active experts;
2. pages **only the active experts' slices** out of the GGUF through the VRAMER
   budget (D2D-gather into the per-split device buffer at each expert's offset);
3. leaves the inactive experts (the large majority) untouched on host.

2-D weights (dense FFN, attention) still page whole-tensor, so the dense demo is
unaffected.

## Model

Qwen1.5-MoE-A2.7B-Chat, Q4_K_M GGUF (~8.84 GiB; ~7.73 GiB of it is expert
stacks). Re-download:

```bash
mkdir -p /tmp/models
curl -L -o /tmp/models/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf \
  https://huggingface.co/RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf/resolve/main/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf
```

## Reproduce

```bash
# from the repo root: build VRAMER + the fork (same as the dense demo)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
bash examples/llamacpp/adapter/build_fork.sh /tmp/llama.cpp /tmp/llama-vramer

# run the MoE demo (free GPU via GPU=<idx>)
GPU=0 bash examples/llamacpp/adapter/run_demo_moe.sh
```

### Env knobs (read by the hook)
- `VRAMER_GGUF`   — GGUF path; **setting it enables the hook**.
- `VRAMER_BUDGET` — VRAM budget in bytes (demo uses 512 MiB).
- `VRAMER_PAGE`   — page size in bytes (demo uses 4 MiB).
- `VRAMER_EVICT`  — `0`=LRU, `1`=MRU (default MRU). LRU helps MoE keep popular experts.
- `VRAMER_GPU`    — CUDA device for the pager.
- plus `GGML_OP_OFFLOAD_MIN_BATCH=1` and
  `-ot "ffn_(down|gate|up)_exps\.weight=CPU"` on the command line.

## Measured result (NVIDIA A2 / SM86, Qwen1.5-MoE-A2.7B-Chat Q4_K_M)

Prompt: *"Explain what a mixture-of-experts model is in two sentences."*,
greedy (`--temp 0`), 64 decode tokens. Expert set on host = **7920 MiB across
72 tensors** (24 layers × {down,gate,up}_exps), i.e. **~90% of the 8.84 GiB
model lives on host**.

| run | VRAM budget | page / evict | peak VRAM (nvidia-smi) | decode tok/s | % of resident | pager hit-rate | bytes_h2d |
|---|---|---|---|---|---|---|---|
| **resident** (full offload) | — | — | 13946 MiB | **62.91** | 100% | — | — |
| expert-paged | 512 MiB | 4 MiB / MRU | 8080 MiB | 2.01 | 3.2% | 36.2% | 102.3 GiB |
| expert-paged | 512 MiB | 2 MiB / LRU | 8080 MiB | 3.01 | 4.8% | 50.0% | 74.9 GiB |
| expert-paged | 2048 MiB | 2 MiB / LRU | 9616 MiB | 3.83 | 6.1% | 60.2% | 54.2 GiB |
| expert-paged | 4096 MiB | 2 MiB / LRU | 11664 MiB | 5.26 | 8.4% | 70.5% | 36.5 GiB |

(idle VRAM = 14 MiB; every budget is **smaller** than the 7920 MiB expert set.)

**Output quality:** every paged run is **byte-identical** to the resident
baseline. The per-token VRAMER-gathered expert buffer reproduces exactly the
same bytes as a resident expert stack.

**Selectivity (the MoE property):** per token only **7.4%** of the expert set is
streamed (20805 active-expert slices over the run; a full-stack copy would have
moved 514800 MiB — so **13.5× less traffic** than paging the whole expert set
every token, which is what a dense model of the same footprint forces).

**Cross-token locality is real for MoE:** the pager hit-rate rises 36% → 70% as
the budget grows (a few experts are "popular" and stay resident). A dense paged
model has essentially no such reuse.

### Honest read

The adapter delivers the **VRAM reduction** (7.9 GiB of experts off VRAM, served
through a budget as small as 0.5 GiB) and the **selectivity + correctness**
(byte-identical output, only 7.4% of the expert set moving per token). It does
**not**, on this A2 / PCIe setup, deliver *near-resident throughput*: decode runs
at **3–8% of the 62.9 tok/s resident baseline** — a ratio similar to the dense
wall (the prior 0.5B dense demo was 208 → 9.3 tok/s, 4.5%).

Why: even at 7.4% selectivity, the active experts are still **~510 MiB/token** of
real H2D traffic for this 14 B model, and this adapter pages **synchronously,
node-by-node, with no compute/transfer overlap** (as its parent README states,
it is a correctness/mechanism path). That makes decode PCIe-and-stall-bound. The
trend (bigger budget → higher hit-rate → less `bytes_h2d` → faster) shows the
headroom: a throughput-optimized path (async double-buffered expert prefetch
overlapped with the attention/shared-expert compute, page sizing matched to the
~1.6–3 MiB expert slice) is what would turn the confirmed *byte* advantage into a
*speed* advantage. That optimization is future work; the numbers here are the
honest current state.
