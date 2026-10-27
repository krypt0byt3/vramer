# Compression-on-wire probe (roadmap #3)

Measures whether VRAMER's weights are compressible enough for a
compress-in-RAM / DMA / decompress-on-GPU path to raise effective PCIe bandwidth.

**Result: no (see `../../docs/COMPRESSION.md`).** Q4_K_M ~1.02×, BF16 ~1.23–1.28×,
and on the A2 the GPU decompressor is slower than the ~5.7 GiB/s link anyway.

## Scripts

- `measure_cpu.py` — parses GGUF/safetensors headers, skips metadata, samples real
  tensor bytes, reports LZ4 / Zstd(1,19) ratios. CPU ratio proxy.
- `measure_gpu_nvcomp.py` — nvCOMP GDeflate/LZ4/Deflate/Zstd/ANS ratio **and** real
  on-GPU decompress throughput + byte-exactness check. Run on GPU 1.

## Running

nvCOMP is not installed system-wide. Use an isolated venv:

```
python3 -m venv venv
./venv/bin/pip install lz4 zstandard numpy nvidia-nvcomp-cu12
./venv/bin/python measure_cpu.py
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python measure_gpu_nvcomp.py
```

Model paths are hard-coded to the probe rig (`/tmp/models/*.gguf`,
`/vramer/models/Qwen2.5-14B-Instruct/*.safetensors`); edit the `targets` list for
other files.
