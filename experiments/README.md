# Experiments (historical)

Early exploration on 2× NVIDIA A2 that predates and motivated the VRAMER library.
Kept for provenance; **not part of the build** and not maintained.

- `a2_vram_expand.cu`, `a2_dma.cu`, `a2_prefetch.cu`, `a2_stream.cu` — standalone
  CUDA probes of host↔device transfer, DMA engines, prefetch and streaming on the A2.
- `p2p_test.cu` — a peer-to-peer (GPU↔GPU) access check.
- `*.txt` — raw benchmark logs (residency sweeps, pp2 validation, w4a16/w8a16 runs).
- `pp2_*.sh` — the shell drivers for those runs.

The maintained, measured benchmarks live in [`../benchmarks/`](../benchmarks); the
library itself is at the repository root. Build an experiment ad hoc, e.g.:

```bash
nvcc -O2 -arch=sm_86 experiments/a2_vram_expand.cu -o experiments/a2_vram_expand
```
