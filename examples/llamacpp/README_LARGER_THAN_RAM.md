# VRAMER × llama.cpp — a model *larger than host RAM*

This is the llama.cpp counterpart to [`../../docs/LARGER_THAN_RAM.md`](../../docs/LARGER_THAN_RAM.md)
(roadmap #6). That doc paged a 120 GiB *synthetic* working set, larger than the
78 GiB host RAM, through a bounded VRAM budget and byte-verified it. Here we do
the same with a **real LLM**: run Qwen2.5-72B-Instruct **F16 — 135.9 GiB of
weights, larger than the machine's 78 GiB of host RAM** — generating correct
text while *both* the host-RAM footprint *and* the VRAM footprint are held to
explicit ceilings you name, on a single 15 GiB card. Output is **byte-identical**
to the native reference.

| | |
|---|---|
| Model | **Qwen2.5-72B-Instruct, F16 GGUF** (ungated, [`Qwen/Qwen2.5-72B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF), fp16 42-shard split merged to one file) |
| On-disk size | **145,925,989,312 B = 135.9 GiB** (963 tensors) |
| Host RAM | **78.5 GiB** (`MemTotal` 84,328,517,632 B) → **model is 1.73× RAM** |
| GPU | 1× NVIDIA A2, 15356 MiB, `CUDA_VISIBLE_DEVICES=0` |
| Disk | `/dev/sda4` ext4, `QEMU HARDDISK` (SATA-class SSD, cold ceiling ~0.4–0.5 GB/s) |

> ### Read this first — honesty caveats
> - **Native llama.cpp can ALSO run a >RAM model.** With its default mmap, the OS
>   page cache streams weights from disk on demand and evicts under pressure, so a
>   model larger than RAM runs — it just *thrashes the disk*. We measure that
>   native baseline below (both pure-CPU `-ngl 0` and GPU cpu-offload) and report
>   it plainly. The claim here is **NOT** "only VRAMER can run it."
> - **VRAMER's honest win is a *doubly-bounded, tunable footprint*** — a VRAM
>   budget you name **and** (via a cgroup memory cap) a host-RAM ceiling far below
>   the model size — with **byte-identical** output to the native GPU reference.
>   The native reference's host RAM ballooned to **75.78 GiB** (it fills all RAM
>   with page cache); the VRAMER run was confined to a **24 GiB** cgroup cap and
>   still produced identical text.
> - **This tier is DISK-BOUND on a SATA SSD and SLOW by design.** Decode runs at a
>   *fraction of a token per second* (~0.002 tok/s). The story is **correctness +
>   bounded footprint**, not throughput. Every number is a **floor** on a shared,
>   contended disk (other processes do I/O on this box).
> - **Not GPUDirect Storage.** The path is disk → OS page cache → pinned staging →
>   GPU. GDS remains the documented next step (roadmap), not a claim made here.

## Why a single-file merge, and why attention is paged too

Two integration facts shaped this demo:

1. **The adapter hook opens one GGUF.** `adapter/vramer_llama_hook.cpp` builds a
   single tensor→offset map from `VRAMER_GGUF` and pages from one file, so a
   **split** GGUF must first be merged:
   `llama-gguf-split --merge <shard-00001-of-00042> <out.gguf>`.
2. **On a 15 GiB card, FFN-only routing is not enough for a 72B.** The too-big-to-
   fit (VRAM) demo ([`README_BIGMODEL.md`](README_BIGMODEL.md)) routed only the FFN
   weights to host. For Qwen2.5-72B F16 the *attention* projections alone are
   ~24 GiB resident — they overflow the card by themselves. So here we route the
   FFN **and** the attention/output projections to host and page them all:

   ```
   -ot '(ffn_(down|gate|up)|attn_(q|k|v|output))\.weight=CPU'
   ```

   Only `token_embd` (~2.5 GiB F16) and the norms stay resident. The resulting
   paged set is **131.25 GiB across 560 tensors** — the overwhelming majority of
   the 135.9 GiB model is paged, never resident in VRAM.

## The three configurations (same prompt / seed / greedy `--temp 0`, `-c 512`)

Prompt `"The capital of France is"`, 4 decode tokens (1 for the pure-CPU control).

| # | config | weights | matmuls | VRAMER | host-RAM cap |
|---|---|---|---|---|---|
| (1) | `native-cpu` `-ngl 0` | host (mmap) | **CPU** | off | none (natural) |
| (2) | `native-cpuoff` `-ngl 99 -ot …=CPU` | host (mmap) | **GPU** | off | none (natural) |
| (3) | `vramer/4GiB` `-ngl 99 -ot …=CPU` | host, **paged from GGUF** | **GPU** | **budget 4 GiB** | **cgroup cap 24 GiB** |

- **(1)** is the honesty control: can stock llama.cpp run a >RAM model at all?
- **(2)** is the **byte reference**: identical GPU math to (3), but weights delivered
  by llama.cpp's own per-split H2D copy instead of the VRAMER pager. (3) must match
  (2) byte-for-byte — *not* (1), whose CPU matmuls use a different reduction order.
- **(3)** is the headline: the same GPU execution, every weight upload served by
  the VRAMER pager through a **fixed VRAM budget**, the whole process confined to a
  **cgroup host-RAM cap** (`systemd-run --scope -p MemoryMax=24G -p MemorySwapMax=0`).

## Results (real logs under [`../../benchmarks/raw/llama_ltr/`](../../benchmarks/raw/llama_ltr/))

| config | runs? | decode tok/s | prefill tok/s | load | host RAM (peak) | peak VRAM | output |
|---|---|---|---|---|---|---|---|
| **(1) native-cpu `-ngl 0`** | **yes** | n/a (1 tok, see note) | 0.0068 (146.1 s/tok) | 1527.5 s | **75.59 GiB** RSS (uncapped) | 2814 MiB (CUDA ctx only) | `Paris` |
| **(2) native-cpuoff** (reference) | yes | **0.0020** (502.8 s/tok) | 0.016 (60.8 s/tok) | 659.8 s | **75.78 GiB** RSS (uncapped) | 3294 MiB | `Paris. What is` |
| **(3) vramer/4GiB, 24GiB cap** | yes | **0.0015** (645.6 s/tok) | 0.010 (101.0 s/tok) | 1310.2 s | **24 GiB** (cgroup cap, anon 189 MB) | 7390 MiB | `Paris. What is` — **byte-identical** |

**VRAMER paging summary (phase 3, from the `[VRAMER]` atexit report):**

```
paged weight set (host, NOT resident in VRAM): 134400.00 MiB across 560 tensors
VRAM budget: 4096.00 MiB (page 4.00 MiB, 1024 slots) -- SMALLER than paged set
copies served from VRAMER: 2800            (560 tensors x 5 forward passes)
bytes streamed: 672000.00 MiB (656.25 GiB)
pager: hits=149634 misses=21166 hit_rate=87.6% evictions=164246
bytes_h2d=661080.00 MiB (645.6 GiB)   <- real host->device paging traffic
```

> **Note on config (1).** Pure-CPU `-ngl 0` **also runs the >RAM model** (rc=0,
> output `Paris`, peak RSS 75.59 GiB) — the honesty control: stock llama.cpp with
> default mmap streams the weights from disk and generates correctly, just
> impractically slowly (prefill 0.0068 tok/s). Its single decode token's per-token
> time is `n/a` because llama.cpp's perf counter does not report a decode rate at
> `n_predict = 1`; the prefill rate and the fact that it completed are the usable
> signal. The ~2.8 GiB VRAM is just the CUDA context (no weights are offloaded).

### The headline, read carefully

- **A 135.9 GiB model (1.73× host RAM) generated correct text** on a 15 GiB card,
  with **131.25 GiB of weights paged** through a **4 GiB VRAM budget** (budget is
  1/33rd of the paged set) and **byte-identical** output to the native reference.
- **Host RAM was held to an enforced 24 GiB** (cgroup `memory.peak` = `memory.max`
  = 24 GiB; swap disabled), versus the native reference's **75.78 GiB** (which
  fills all of RAM with page cache). The capped run's *anonymous* memory was only
  **189 MB** — essentially all of its footprint is reclaimable page cache that the
  cgroup holds down; the kernel performed **~505 GiB of direct reclaim**
  (`pgsteal_direct` ≈ 132 M pages) during the run, continuously evicting and
  re-faulting from disk. That is the bounded footprint, proven.
- **Caveat on the RSS number.** `/usr/bin/time -v` reported ~37 GiB "Maximum
  resident set size" for the capped run — *higher* than the 24 GiB cap — because
  llama.cpp mmaps the GGUF **and** VRAMER's `MmapFileSource` independently mmaps the
  same file in the same process, so shared physical pages are counted twice in
  `VmRSS` but once by the cgroup. The **cgroup accounting (24 GiB) is authoritative**
  for the physical host-RAM bound; the raw `*.cgroup` capture is committed alongside
  the logs.
- **The cap, not VRAMER, is what bounds host RAM.** Both configs use mmap, so a
  cgroup cap could be applied to native too (it would also run, thrashing harder).
  VRAMER's distinct contribution is the **named VRAM budget** (native cpu-offload's
  device buffer is sized by llama.cpp, not a knob; native partial `-ngl K` is
  quantized to whole layers and here no K ≤ card leaves room) **plus byte-identical
  GPU execution** under that bounded RAM.
- **Speed is a floor and it is slow.** VRAMER@24 GiB-cap decodes at ~0.0015 tok/s
  (~646 s/token), ~1.3× slower than the *uncapped* native reference (~0.0020 tok/s,
  ~503 s/token): the tighter RAM cap leaves less page cache, so more of every token's
  131 GiB weight sweep comes from the SATA disk. This is the honest cost of the
  bounded footprint on this tier. On NVMe + GPUDirect Storage the disk would stop
  being the wall — the documented next step.

## Reproduce

```bash
# 0. build VRAMER + the adapter fork (see adapter/README.md)
cmake -S /vramer -B /vramer/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build /vramer/build -j
bash /vramer/examples/llamacpp/adapter/build_fork.sh /tmp/llama.cpp /tmp/llama-vramer-ltr
cmake --build /tmp/llama-vramer-ltr/build -j --target llama-gguf-split

# 1. download an UNGATED GGUF larger than host RAM, then MERGE the shards to one file
#    (Qwen2.5-72B-Instruct fp16 = 42 shards, 135.9 GiB total). Model dir OUTSIDE the repo.
hf download Qwen/Qwen2.5-72B-Instruct-GGUF --include 'qwen2.5-72b-instruct-fp16-*' \
    --local-dir /var/tmp/ltr-models        # or curl each resolve/main/<shard>
/tmp/llama-vramer-ltr/build/bin/llama-gguf-split --merge \
    /var/tmp/ltr-models/qwen2.5-72b-instruct-fp16-00001-of-00042.gguf \
    /var/tmp/ltr-models/qwen2.5-72b-instruct-f16-merged.gguf
#    confirm the merged file exceeds host RAM before claiming "> RAM":
stat -c%s /var/tmp/ltr-models/qwen2.5-72b-instruct-f16-merged.gguf   # > MemTotal

# 2. run all three configs (native-cpu / native-cpuoff / vramer-capped) + byte diff
CUDA_VISIBLE_DEVICES=0 \
MODEL=/var/tmp/ltr-models/qwen2.5-72b-instruct-f16-merged.gguf \
LLAMA=/tmp/llama-vramer-ltr/build/bin/llama-completion \
NPRED=4 NPRED_CPU=1 CTX=512 RAMCAP=24G BUDGETS=$((4*1024*1024*1024)) \
bash /vramer/examples/llamacpp/run_larger_than_ram.sh
```

The script (`run_larger_than_ram.sh`) is parameterized — it takes the model path,
never hardcodes the giant download — drops caches before each row, wraps the
VRAMER row in the cgroup cap, samples VRAM/iostat/`free`, and diffs (3) against (2).
