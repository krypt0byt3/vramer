# Too big to fit — *in host RAM*

[`BENCHMARKS.md`](BENCHMARKS.md) ends on the "too big to fit (VRAM)" case: a
model whose weights overflow the **card** (18.49 GiB on a 15 GiB A2), paged
through a named VRAM budget. This doc takes the premise one tier down the memory
hierarchy and asks the forward question (roadmap #6): can VRAMER page a working
set that overflows **host RAM**, streaming it from disk through a bounded VRAM
budget, *correctly*, with a footprint you name?

**Answer: yes — demonstrated with a 120 GiB backing file on a box with 78 GiB
RAM and 15 GiB cards, paged through a 4 GiB VRAM budget, byte-verified.** The
honest catch is in the second half: this tier is **disk-bound at SATA-SSD
speed (~0.4 GiB/s)**, roughly **14× slower** than when the same backing store
fits in RAM and is served from the OS page cache. The contribution here is
*correctness and a bounded footprint for a >RAM working set*, not throughput.

> ### Read this first — honesty caveats
> - **Shared, contended SATA SSD.** The backing store is a single `QEMU
>   HARDDISK` (`/dev/sda`, ext4). The true cold ceiling measured ~0.4–0.46 GB/s,
>   i.e. **SATA-class, not NVMe**. The box was **not** idle: other processes do
>   I/O on the same device during these runs. `iostat -x sda` showed read
>   throughput dipping from ~0.59 GB/s to ~0.05 GB/s mid-run and wide
>   run-to-run variance (e.g. one 256 MiB-page rep took 561 s, another 290 s).
>   **Treat every number here as a floor, not a clean benchmark.** Each cell was
>   run 2–3× and reported as a range.
> - **This is NOT GPUDirect Storage.** cuFile/GDS is not installed, and this
>   QEMU VM would not do true disk→GPU peer-to-peer DMA anyway. The measured
>   path is **disk → OS page cache → pinned staging → GPU** (`cudaMemcpyAsync`).
>   GDS's win is mostly on NVMe; on a SATA SSD at ~0.4 GB/s it would not move
>   this result. GDS remains the documented next step (roadmap), not a claim
>   made here.

## The scenario

| | |
|---|---|
| Host RAM | **78 GiB** (`sysinfo` totalram 84,328,517,632 B) |
| GPUs | 2× NVIDIA A2, 15 GiB each; all runs `CUDA_VISIBLE_DEVICES=1` (GPU 0 reserved) |
| Disk | `/dev/sda4` ext4, `QEMU HARDDISK`, 428 GB free at start |
| Backing file | **120 GiB** (`128,849,018,880` B) at `/var/tmp/vramer-ltr-data/ltr120.bin` |
| File vs RAM | **120 GiB > 78 GiB** — cannot be held in RAM, let alone the 15 GiB card |
| VRAM budget | 4 GiB per run (also valid at 2–8 GiB) |

A model this size is impossible to even *load* natively on this machine: it
exceeds host RAM, so there is nowhere to put it before offload. VRAMER never
holds more than the budget resident; it pages 64 MiB (or 16/256 MiB) windows on
demand and evicts under a hard cap.

### Deterministic, position-verifiable pattern

The backing file is filled so every byte is a function of its position: 4 KiB
block *b* (absolute from the file start) is filled with the byte `b & 0xff`.
This mirrors the `(i/page)&0xff` convention `AnonymousRamSource` already uses and
lets the benchmark's built-in checksum be predicted in closed form.

`vramer bench`'s `sample_kernel` sums **one byte per 4096-byte block** over the
whole scan (`--compute-reps 1`). With this pattern the sampled byte of block *b*
is `b & 0xff`, so for a file of *M* MiB the expected checksum is exactly

```
checksum = M * 32640          (32640 = 0+1+…+255, one full MiB of blocks)
```

This is **independent of the pager's `--page-size`** (any page size that is a
multiple of 4096 and tiles the file yields the identical checksum) — a strong
cross-configuration consistency check. The generator
([`../benchmarks/larger_than_ram/gen_pattern.c`](../benchmarks/larger_than_ram/gen_pattern.c))
prints the expected value; for 120 GiB it is **4,010,803,200**.

> The bench kernel samples 1 byte / 4 KiB, so on the 120 GiB run it byte-checks
> ~31.5 M positions across the whole file — enough to catch a wrong offset, a
> short read, or truncation, but it does not read every byte. Full every-byte
> verification through the real pager (cold load **and** reload-after-eviction)
> is done on a small file by the CI test
> [`../tests/test_file_source.cu`](../tests/test_file_source.cu) (ctest
> `file_source`), which CI can afford where a >RAM scan cannot.

## Step 1 — the raw disk ceiling (what bounds everything)

```bash
dd if=/dev/zero of=ceiling.bin bs=1M count=8192 conv=fsync    # 8 GiB
sync; echo 3 > /proc/sys/vm/drop_caches
dd if=ceiling.bin of=/dev/null bs=4M iflag=direct             # O_DIRECT
sync; echo 3 > /proc/sys/vm/drop_caches
dd if=ceiling.bin of=/dev/null bs=4M                          # buffered
```

| read, 8 GiB file | GB/s (×3) | note |
|---|---|---|
| `dd iflag=direct` (O_DIRECT) | 3.5 / 4.5 / 4.4 | **misleading** — see below |
| buffered, cache-cold | 1.9 / 1.6 / 1.8 | still host-cache-aided |

**These small-file numbers are NOT the cold ceiling.** On this QEMU VM, guest
`O_DIRECT` does not bypass the *host's* page cache, so an 8 GiB file is partly
served from host RAM — which is exactly why it reads *faster* than a SATA SSD's
~0.55 GB/s cap. The honest cold ceiling is the one taken with a file far larger
than any cache slice: the **120 GiB cold scan below, where `iostat -x sda`
measured the device delivering ~0.46 GB/s (min 0.14, max 0.54) at ~74 % util**
(and the 40 GiB cold scan, 0.39 GiB/s). That is SATA-class, and it — not the PCIe
wall (~5.7 GiB/s on this VM) and not the pager — is the bound on this tier.
Sustained *write* of the 120 GiB file likewise ran at ~51 MB/s average including
the final 120 GiB `fsync`.

## Step 2 — correctness proof (the headline)

```bash
sync; echo 3 > /proc/sys/vm/drop_caches     # force cold: nothing in page cache
CUDA_VISIBLE_DEVICES=1 vramer bench --source file --file ltr120.bin \
  --vram-budget 4GiB --page-size 64MiB --gpu 0
```

```
Source size:  120 GiB            VRAM budget:  4.00 GiB per GPU
elapsed:     290.633 s           H2D bytes:   120 GiB
H2D rate*:   0.41 GiB/s          hit rate:    0.0%
misses:      1920                evictions:   1856
src read:    290186.456 ms       stall:       11.181 ms
checksum:    4010803200          <-- expected 4010803200  ✓ byte-verified
```

A **120 GiB working set, larger than the 78 GiB of host RAM, paged correctly
through a 4 GiB VRAM window on a 15 GiB card** — a configuration that cannot be
loaded natively at all. The checksum matches the closed-form expectation, and
matched it in **every** page-size and prefetch configuration below.

`src read` is **290,186 ms of 290,633 ms wall = 99.85 %**: the run is almost
entirely the `MemorySource::read` (mmap → pinned staging) waiting on the disk.
`stall` (staging/eviction waits) is 11 ms; the H2D copy and GPU work hide
entirely behind the disk. The pager delivered 0.41 GiB/s end-to-end while the
device delivered ~0.43 GiB/s (iostat), so **VRAMER extracts ~96 % of the disk's
cold read bandwidth** — essentially all of it. The overhead this tier was
feared to add (mmap+memcpy through the page cache, the pinned staging copy) is
invisible next to a 0.4 GiB/s disk.

## Step 3 — isolating the RAM boundary (cold/disk vs warm/cache)

The same binary, same pattern, same 4 GiB budget, two regimes:

| regime | file | source | effective GiB/s | elapsed | checksum |
|---|---|---|---|---|---|
| **cold / disk-bound** | 120 GiB (> RAM) | real disk I/O | **0.32–0.43** | 279–376 s | ✓ 4,010,803,200 |
| **cold / disk-bound** | 40 GiB (< RAM) | real disk I/O (cache dropped) | **0.39** | 102 s | ✓ 1,336,934,400 |
| **warm / cache-bound** | 40 GiB (< RAM) | OS page cache | **5.45–5.68** | 7.0–7.3 s | ✓ 1,336,934,400 |

The 40 GiB file fits in the 78 GiB RAM, so after one cold scan the OS page cache
holds all of it (`buff/cache` = 41 GiB) and re-scans are served from RAM at
**~5.5 GiB/s — about 14× the disk-bound rate, right at the ~5.7 GiB/s PCIe wall**.
On the warm runs `src read` collapses from 102 s to ~5 s (a RAM memcpy) and the
remaining time is `stall` on the single H2D transfer stream — i.e. warm, VRAMER
is pager/PCIe-bound, exactly like the RAM-source benchmarks in `BENCHMARKS.md`.

**This is the RAM boundary made visible.** The only difference between the two
40 GiB rows is whether the page cache can hold the file. The 120 GiB file's whole
point is that it *cannot* — a linear scan continually faults pages the cache has
already evicted, forcing the disk on essentially every page (0 % hit, 1856
evictions), which is why it sits at the disk floor and the warm case cannot be
reached no matter the budget.

## Step 4 — page size and prefetch on the disk-bound path

120 GiB, 4 GiB budget, cache dropped before each rep, 2 reps each (plus the
headline as a third 64 MiB/demand point). All checksums ✓ 4,010,803,200.

| config | page | prefetch | effective GiB/s (range) | hit rate | notes |
|---|---|---|---|---|---|
| demand | 16 MiB | none | 0.41 – 0.44 | 0 % | 7680 misses |
| demand | 64 MiB | none | 0.32 – 0.43 | 0 % | 1920 misses (headline 0.41) |
| demand | 256 MiB | none | 0.21 – 0.41 | 0 % | widest variance (contention) |
| prefetch | 64 MiB | 1 worker, ahead 2 | 0.20 – 0.20 | 99.9 % | **slower** |
| prefetch | 64 MiB | 2 workers, ahead 4 | 0.27 – 0.41 | 99.9 % | ~20 s staging stall |

Two honest findings:

1. **Page size barely matters on this tier.** 16/64/256 MiB all land at
   ~0.4 GiB/s on their good reps — the disk, not the paging granularity, is the
   wall. Larger pages (256 MiB = 16 big sequential reads) showed the *widest*
   variance under external contention.
2. **Prefetch workers do not help here, and can hurt.** They raise the *hit
   rate* to 99.9 % (future pages are resident before demand asks), but
   wall-clock **drops** to ~0.20 GiB/s with 1 worker. The demand path already
   reads the mmap strictly sequentially with `MADV_SEQUENTIAL`, so the kernel's
   readahead is already extracting near-peak from a *single* sequential stream;
   a prefetch thread that jumps `ahead` pages turns that into two interleaved
   read streams, disrupting readahead on a device that is latency-bound (4 KiB
   fault-sized reads, `r_await` ~20–40 ms). Two workers add staging-buffer
   contention (~20 s aggregate `stall`). Prefetch's payoff is overlapping
   *compute* with transfer (see `test_prefetch_overlap` / `BENCHMARKS.md`); when
   the bottleneck is the disk itself and there is no heavy compute, it is pure
   overhead. **On a disk tier, demand paging with kernel readahead is the right
   default.**

## Where the overhead goes (cold, 120 GiB)

```
wall 290.6 s = src_read 290.2 s (99.85%, disk) + stall 0.011 s + [H2D + GPU, fully hidden]
pager end-to-end 0.41 GiB/s  vs  device-delivered ~0.43 GiB/s (iostat)  ->  ~96% extracted
```

There is almost nowhere for overhead to hide: the pinned staging copy and the
H2D DMA overlap the next page's disk read and never surface. The ~4 % gap to the
device rate is within run-to-run contention noise. If this were an NVMe/GDS
machine the story would invert — the disk would stop being the wall and the
staging copy would matter — which is precisely why GDS is the next step.

## How this extends the "too big to fit (VRAM)" story

`BENCHMARKS.md` showed VRAMER running a model that overflows the **card** under a
named VRAM budget, byte-identical. This extends the same explicit-ceiling
property one tier down: a working set that overflows **host RAM** also runs,
byte-verified, under a named budget — the difference is only *which* tier
supplies the misses (RAM at ~5.5 GiB/s vs disk at ~0.4 GiB/s) and therefore how
fast. VRAMER's contract is unchanged across both: **you name the resident
ceiling; the working set may be arbitrarily larger than it, up the whole
hierarchy (VRAM < RAM < disk), and the bytes delivered are correct.** What this
tier adds is the honest cost curve: past the RAM boundary you pay disk latency,
and the real throughput unlock there is NVMe + GPUDirect Storage, not a bigger
budget.

## Reproduce

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j

# backing file LARGER than RAM (120 GiB). prints the expected checksum.
cc -O2 -o /tmp/gen_pattern benchmarks/larger_than_ram/gen_pattern.c
/tmp/gen_pattern /var/tmp/vramer-ltr-data/ltr120.bin 122880

# cold page-size/prefetch sweep (drops caches per rep; needs root)
CUDA_VISIBLE_DEVICES=1 FILE=/var/tmp/vramer-ltr-data/ltr120.bin \
  BUDGET=4GiB DROP=1 REPS=2 bash benchmarks/larger_than_ram/run_ltr.sh

# small-file full byte-verification (CI)
CUDA_VISIBLE_DEVICES=1 ctest --test-dir build -R file_source --output-on-failure
```

Raw per-cell logs (bench stdout incl. checksum + PagerStats) and the `iostat`
summaries are under [`../benchmarks/raw/`](../benchmarks/raw/) as `ltr_*.log` —
the committed copies are the exact logs that produced the numbers above.
