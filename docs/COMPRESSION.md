# Roadmap #3 — Compression-on-wire: RESOLVED (honest negative)

**Verdict: do not build the GPU decompression pipeline for VRAMER as it stands.**
The premise of item #3 — "weights compress ~2×, so compress in host RAM, DMA the
smaller payload over PCIe, decompress on the GPU → ~2× effective bandwidth" — does
not survive measurement. It fails twice over, independently:

1. **The economics fail.** The production weights are not compressible anywhere near
   2×. Quantized (Q4_K_M) weights are already near entropy — they do not compress at
   all (~1.02×). Unquantized BF16 tops out at ~1.23–1.28×, well short of the 2× the
   item assumed and below the ~1.3× go/no-go threshold set for this probe.
2. **The hardware fails (on this rig).** Even if the ratio were free, the A2's GPU
   decompressor runs *slower* than the PCIe link it is meant to relieve. GDeflate
   decompresses at ~3.4 GB/s (≈3.2 GiB/s) vs the ~5.7 GiB/s H2D baseline, so
   decompression simply *replaces* the link as the bottleneck. Every codec that is
   fast enough to keep up (LZ4, ANS) has a ~1.0–1.19× ratio.

A clear negative is the deliverable here: it saves the project from building a
nontrivial GPU pipeline that cannot win on this hardware/model mix.

---

## Step 1 — compressibility of real weight bytes

Method: parse each GGUF/safetensors header, **skip the metadata region**, sample
real tensor-data byte ranges, and compress them. CPU `lz4`/`zstd` are used as a
ratio proxy (large window, levels 1 and 19); nvCOMP provides the GPU codecs that
would actually run in the pipeline (GDeflate, LZ4, Deflate, Zstd, ANS) plus real
on-GPU decompress throughput. Scripts: `benchmarks/compression/measure_cpu.py`,
`benchmarks/compression/measure_gpu_nvcomp.py`. All GPU runs on GPU 1 (A2).

### CPU codecs — compression ratio (×), 384 MiB of real tensor bytes per model

| model | lz4-fast | lz4-hc9 | zstd-1 | zstd-19 |
|---|---|---|---|---|
| MoE-A2.7B **Q4_K_M** | 0.997 | 0.998 | 1.021 | 1.023 |
| Qwen2.5-0.5B **Q4_K_M** | 1.002 | 1.004 | 1.021 | 1.023 |
| Qwen2.5-32B **Q4_K_M** | 0.996 | 0.996 | 1.023 | 1.023 |
| Qwen2.5-14B **BF16** | 0.996 | 1.007 | 1.269 | **1.284** |
| Qwen2.5-14B **int8 (w8a16)** | 0.997 | 1.012 | 1.190 | 1.197 |

### GPU codecs (nvCOMP 6.0.0, A2) — ratio + on-GPU decompress throughput, byte-exact

| model | codec | ratio (×) | GPU decomp | exact |
|---|---|---|---|---|
| MoE Q4_K_M | GDeflate (t=1) | 1.030 | 3.4 GB/s | ✓ |
| MoE Q4_K_M | GDeflate (t=4) | 1.030 | 3.1 GB/s | ✓ |
| MoE Q4_K_M | LZ4 | 0.996 | 5.0 GB/s | ✓ |
| MoE Q4_K_M | Deflate | 1.033 | 0.8 GB/s | ✓ |
| MoE Q4_K_M | Zstd | 1.033 | 2.6 GB/s | ✓ |
| MoE Q4_K_M | ANS | 1.004 | 6.8 GB/s | ✓ |
| 14B BF16 | GDeflate (t=1) | 1.229 | 3.4 GB/s | ✓ |
| 14B BF16 | GDeflate (t=4) | 1.228 | 2.9 GB/s | ✓ |
| 14B BF16 | LZ4 | 0.998 | 4.9 GB/s | ✓ |
| 14B BF16 | Deflate | 1.233 | 0.8 GB/s | ✓ |
| 14B BF16 | Zstd | 1.233 | 2.1 GB/s | ✓ |
| 14B BF16 | ANS | 1.191 | 5.4 GB/s | ✓ |

Notes:
- **Window size is irrelevant** for BF16: raising the nvCOMP chunk from 64 KiB to
  4 MiB left the ratio at ~1.23×. The redundancy is per-element entropy in the fp16
  sign/exponent bits, not long-range byte matches — so dictionary/match codecs
  (LZ4) do nothing (~1.0×) and only entropy coders (Deflate/Zstd/GDeflate/ANS) gain.
  LZ4 bitshuffle (MSB/LSB) only lifted LZ4 from 1.00× to 1.01×.
- Q4_K_M byte streams are already block-quantized + packed; there is essentially no
  residual redundancy (~1.02× is at the noise floor, and LZ4 *expands* them).

**Decisive ratio conclusion:** quantized weights are effectively incompressible
(< 1.05×, far under the 1.15× floor). Item #3 is **dead for quantized models**. The
only format with exploitable redundancy is unquantized fp16/bf16, and even there the
ceiling is ~1.23–1.28× — below the 1.3× threshold and nowhere near the assumed 2×.

---

## Step 2 — decision: measurement-only, no pipeline built

Because the ratio is below threshold *and* the on-GPU decompressor is slower than the
link on this rig, there is no configuration in which a compressed path beats a raw
H2D on the A2. Effective delivered bandwidth under ideal H2D/decompress overlap is
`min(link × ratio, decompress_rate)`, against the ~5.7 GiB/s raw baseline:

| scenario | link×ratio | decomp cap | effective | vs 5.7 GiB/s |
|---|---|---|---|---|
| Q4_K_M, GDeflate | 5.7×1.03 = 5.9 | 3.2 GiB/s | **3.2** | 0.56× (loss) |
| Q4_K_M, ANS | 5.7×1.00 = 5.7 | 6.3 GiB/s | **5.7** | ~1.0× (no gain) |
| BF16, GDeflate | 5.7×1.23 = 7.0 | 3.2 GiB/s | **3.2** | 0.56× (loss) |
| BF16, ANS | 5.7×1.19 = 6.8 | 5.0 GiB/s | **5.0** | 0.88× (loss) |

Every realistic row is at best a wash and usually a net loss. Building a
`CompressedMemorySource` + GPU decode kernel would add complexity and a new
bottleneck for zero or negative gain here.

**The mechanism is sound** — nvCOMP decompresses all codecs byte-exactly on GPU 1,
and `MemorySource` (see `include/vramer/memory_source.hpp`) already abstracts the
tier cleanly, so a `CompressedMemorySource` storing per-page compressed blobs and
decoding after the H2D of the compressed bytes is straightforward to add *if* the
economics ever justify it. They do not here, so we stop at the measurement.

---

## Realistic ceiling / recommendation

- **Q4_K_M and other k-quants: do not pursue. Ever.** ~1.02× is the entropy floor;
  no codec or window helps. This is the production format, so #3 has no payoff for
  the main use case.
- **Unquantized fp16/bf16: marginal, and only on a GPU whose decompressor outruns the
  link.** Absolute ceiling ≈ **1.25× effective bandwidth** (the measured ratio),
  reachable only where GPU decode ≫ PCIe (e.g. a datacenter GPU + GDeflate at tens of
  GB/s). On this A2 even that is a net loss because decode < link.
- **The better single-stream lever remains item #10** (bare-metal / passthrough
  tuning), which 3a identified as a likely ~2× for *free*, applies to *all* formats,
  and adds no decode bottleneck. Compression's best case (1.25×, fp16 only, big GPU)
  is strictly worse and narrower.

Bottom line: **#3 is not worth pursuing for VRAMER's quantized production models.**
Keep it shelved; revisit only for an fp16/bf16 workload on hardware where measured
GPU decompress throughput comfortably exceeds the PCIe H2D rate.
