# VRAMER × llama.cpp — running a model that is *too big to fit*

This is the demo where VRAMER is **not** competing with full GPU offload — it is
**enabling something full offload cannot do at all**: running a model whose
weights are **larger than the GPU's entire VRAM**.

- **Model:** Qwen2.5-32B-Instruct, Q4_K_M GGUF — **19,851,336,576 bytes (18.49 GiB)**
  of weights, from
  [`bartowski/Qwen2.5-32B-Instruct-GGUF`](https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF).
- **GPU:** 1× NVIDIA A2, **15356 MiB (~15 GiB)** VRAM, `CUDA_VISIBLE_DEVICES=0`.
- The weights are **~3.5 GiB larger than the whole card.** Full offload is
  physically impossible; the interesting question is *how* you run it anyway.

```bash
# download the model (~18.5 GiB; needs ~20 GiB free disk)
mkdir -p /tmp/models
wget -O /tmp/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf

# build VRAMER + the fork (once) — see adapter/README.md
cmake -S /vramer -B /vramer/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build /vramer/build -j
bash /vramer/examples/llamacpp/adapter/build_fork.sh /tmp/llama.cpp /tmp/llama-vramer-batched

# run the four-way demo on GPU 0
GPU=0 bash /vramer/examples/llamacpp/run_demo_bigmodel.sh
```

## The four ways to run it (same model, same GPU, greedy, `-c 2048`, 48 decode tokens)

| # | config | what it does | fits? |
|---|---|---|---|
| (a) | `-ngl 99` **full offload** | every weight resident in VRAM | **NO — OOM** |
| (b) | `-ngl 49` **native partial** | 49/64 layers on GPU, the rest run on the CPU | yes |
| (b') | `-ngl 99 -ot FFN=CPU` **native cpu-offload** | FFN weights kept on host, llama.cpp's own per-token H2D copy | yes |
| (c) | `-ngl 99 -ot FFN=CPU` **VRAMER** | the *same* FFN set, every H2D upload served by the VRAMER pager through a **fixed VRAM budget you choose** | yes |

`-ngl 49` is the **largest** native partial offload that fits: `-ngl 50` already
OOMs (the KV cache + compute buffers no longer fit beside 50 layers of weights).

### (a) The OOM proof — full offload is impossible

```
$ llama-completion -m Qwen2.5-32B-Instruct-Q4_K_M.gguf -ngl 99 -c 2048 ...
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 18508.35 MiB on device 0:
    cudaMalloc failed: out of memory
ggml_backend_buft_alloc_buffer_n_default: failed to allocate CUDA0 buffer of size 19407413248
llama_model_load: error loading model: unable to allocate CUDA0 buffer
   -> exit code 1
```

The model buffer alone is **18508 MiB** against a **15356 MiB** card. There is no
`-ngl` value that offloads everything; this is the floor VRAMER steps under.

### What VRAMER pages

With `-ot 'ffn_(down|gate|up)\.weight=CPU'`, the **192 FFN weight tensors
(64 layers × 3) — 15693.75 MiB — live on host and are *not* resident in VRAM.**
That FFN set **by itself is larger than the entire 15 GiB card.** VRAMER streams
it through a fixed budget (4 GiB / 8 GiB below), so peak VRAM = the resident
remainder (attention + embeddings + output + norms, ~3 GiB) + KV + the budget.

## Results

NVIDIA A2 (SM86, 15356 MiB), GPU 0, greedy `--temp 0`, `-c 2048`, 48 decode
tokens, prompt `"The capital of France is"`. Each timed row run 2×; tok/s shown
as the range. Peak VRAM is `nvidia-smi memory.used` sampled at 10 Hz.

| config | fits? | decode tok/s | peak VRAM | FFN H2D over 48 tok | pager hit-rate | output |
|---|---|---|---|---|---|---|
| **(a) full offload** `-ngl 99` | **NO — OOM** | — | fails allocating 18508 MiB | — | — | — |
| **(b) native partial** `-ngl 49` | yes | **0.98–1.00** | **14774–14935 MiB** (≈ whole card) | — (49 layers resident) | — | reference |
| **(b') native cpu-offload** `-ot FFN=CPU` | yes | 0.33 | **3778–4003 MiB** | (llama internal copy) | — | reference |
| **(c) VRAMER / 4 GiB budget** | yes | 0.29–0.32 | **7874 MiB** | ~587 GiB | 71.7–77.8 % | **byte-identical** |
| **(c) VRAMER / 8 GiB budget** | yes | 0.41–0.43 | **11970 MiB** | ~386 GiB | 81.3–83.3 % | **byte-identical** |

The paged FFN set is **15693.75 MiB** (192 tensors); the per-token demand over
the run is **768994 MiB (751 GiB)**. VRAMER serves that through a 4 GiB or 8 GiB
window — the H2D column is the traffic that actually crossed PCIe after cache
hits, so a bigger budget (higher hit-rate) moves less: 751 GiB demand → 587 GiB
(4 GiB budget) / 386 GiB (8 GiB budget) of real H2D.

All VRAMER rows produce output **byte-identical** to the native cpu-offload run
(greedy, `--temp 0`): the bytes VRAMER gathers into the matmul's destination
buffer are exactly the file's bytes, so the math is unchanged.

## Honest interpretation

**The honest truth first: VRAMER is *not* the only way to run this model.**
llama.cpp's own native offload — partial (`-ngl 49`) and cpu-offload
(`-ot …=CPU`) — both run the 32B fine. So the framing "impossible without
VRAMER" is wrong: what is impossible is only **full** offload (`-ngl 99`), and
nobody needs full offload to run a too-big model. We say this plainly.

**What the four rows actually show is a VRAM-vs-speed frontier, and VRAMER's
value is that it is the only option that lets you *name the VRAM point*:**

- **Native partial (`-ngl 49`) is the fastest (~1.0 tok/s) but eats the whole
  card (~14.9 GiB).** Its VRAM is not a knob — it is quantised to whole layers,
  and here the *only* K that fits (`-ngl 50` OOMs on the KV cache) pins you at
  ~100 % VRAM with no headroom to share the GPU.
- **Native cpu-offload is the most VRAM-frugal (~3.9 GiB) but gives you no
  control and no caching:** it re-copies every FFN tensor every token, lands at
  0.33 tok/s, and its footprint is whatever llama.cpp picks.
- **VRAMER lets you set the ceiling: run the same 32B at a peak VRAM *you*
  choose — 7.9 GiB (4 GiB budget) or 12.0 GiB (8 GiB budget) — with
  byte-identical output.** Because it *caches* hot FFN pages (72–83 % hit-rate)
  instead of re-copying blindly, VRAMER@8 GiB reaches **0.43 tok/s — ~30 % faster
  than native cpu-offload** — by spending VRAM it is allowed to spend. At a 4 GiB
  budget it matches cpu-offload's speed (~0.30 tok/s) while still leaving you the
  budget as an explicit dial.

**Verdict — when does the too-big-to-fit demo show a clear VRAMER win?** Not on
raw throughput (native partial wins that, at the cost of the whole card) and not
on minimum VRAM (native cpu-offload wins that, with no knob). VRAMER wins
precisely when **you must run a model that overflows the card under a *specific,
hard VRAM ceiling you choose* — below what the only-fitting `-ngl K` needs, but
with more speed than blind cpu-offload buys at that ceiling.** Concretely: to
leave VRAM free to share the GPU with another process, to fit a bigger KV cache
beside the weights, or to hold a model on a smaller card than any partial-offload
K would fit — VRAMER is the only one of the three that turns "peak VRAM" into a
parameter, and it does so with provably identical output. If you simply want the
fastest decode and can spend all 15 GiB, use `-ngl 49`; if you want the smallest
footprint and don't care about the knob, use native cpu-offload. VRAMER owns the
middle: *"run this 32B at exactly N GiB — your number — correctly."*

All runs were pinned to GPU 0; raw per-row logs (llama.cpp stderr incl. the
`[VRAMER]` summary, decoded text, VRAM samples) are under
[`../../benchmarks/raw/bigmodel/`](../../benchmarks/raw/bigmodel/).
