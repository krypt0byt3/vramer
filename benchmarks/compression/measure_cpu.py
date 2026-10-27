#!/usr/bin/env python3
"""Compressibility measurement of real weight bytes (CPU codecs as ratio proxy).
Samples real tensor-data byte ranges (header/metadata skipped) and measures
LZ4 and Zstd (levels 1 and 19) ratios + rough throughput."""
import struct, json, time, sys
import numpy as np
import lz4.block as lz4b
import zstandard as zstd

CHUNK = 64 * 1024 * 1024   # 64 MiB per sample
NCHUNKS = 6                 # samples spread across the tensor-data region

def gguf_data_start(path):
    """Return byte offset where aligned tensor data begins in a GGUF file."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', magic
        version, = struct.unpack('<I', f.read(4))
        n_tensors, = struct.unpack('<Q', f.read(8))
        n_kv, = struct.unpack('<Q', f.read(8))
        def rstr():
            ln, = struct.unpack('<Q', f.read(8))
            return f.read(ln)
        GT = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        def read_val(t):
            if t == 8:  # string
                rstr(); return
            if t == 9:  # array
                et, = struct.unpack('<I', f.read(4))
                cnt, = struct.unpack('<Q', f.read(8))
                if et == 8:
                    for _ in range(cnt): rstr()
                else:
                    f.seek(sizes[et]*cnt, 1)
                return
            f.seek(sizes[t], 1)
        align = 32
        for _ in range(n_kv):
            rstr()  # key
            t, = struct.unpack('<I', f.read(4))
            read_val(t)
        # tensor infos
        for _ in range(n_tensors):
            rstr()  # name
            nd, = struct.unpack('<I', f.read(4))
            f.seek(8*nd, 1)  # dims
            f.seek(4, 1)     # type
            f.seek(8, 1)     # offset
        pos = f.tell()
        data_start = (pos + align - 1) // align * align
        return data_start

def safetensors_data_start(path):
    with open(path, 'rb') as f:
        n, = struct.unpack('<Q', f.read(8))
    return 8 + n

def sample_chunks(path, data_start):
    import os
    fsize = os.path.getsize(path)
    region = fsize - data_start
    usable = region - CHUNK
    chunks = []
    with open(path, 'rb') as f:
        for i in range(NCHUNKS):
            if usable <= 0:
                off = data_start; sz = min(CHUNK, region)
            else:
                off = data_start + (usable * i) // max(1, NCHUNKS-1)
                sz = CHUNK
            f.seek(off)
            chunks.append(f.read(sz))
    return chunks

def measure(name, chunks):
    total = sum(len(c) for c in chunks)
    results = {}
    # LZ4 (fast, high-compression mode)
    t=time.time(); comp=[lz4b.compress(c, mode='high_compression', compression=9, store_size=False) for c in chunks]; ct=time.time()-t
    csz=sum(len(x) for x in comp)
    results['lz4-hc9'] = (total/csz, total/ct/1e9)
    t=time.time(); comp=[lz4b.compress(c, store_size=False) for c in chunks]; ct=time.time()-t
    csz=sum(len(x) for x in comp)
    results['lz4-fast'] = (total/csz, total/ct/1e9)
    # Zstd level 1 and 19
    for lvl in (1,19):
        cctx=zstd.ZstdCompressor(level=lvl)
        t=time.time(); comp=[cctx.compress(c) for c in chunks]; ct=time.time()-t
        csz=sum(len(x) for x in comp)
        results[f'zstd-{lvl}'] = (total/csz, total/ct/1e9)
    print(f"\n=== {name}  (sampled {total/1024/1024:.0f} MiB of real tensor bytes) ===")
    for codec,(ratio,gbps) in results.items():
        print(f"  {codec:10s} ratio={ratio:.4f}x   compress={gbps:.2f} GB/s")
    return results

if __name__ == '__main__':
    targets = []
    # Q4_K_M
    for p,label in [
        ('/tmp/models/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf','MoE-A2.7B Q4_K_M'),
        ('/tmp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf','Qwen2.5-0.5B Q4_K_M'),
        ('/tmp/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf','Qwen2.5-32B Q4_K_M'),
    ]:
        ds = gguf_data_start(p); targets.append((label, p, ds))
    # BF16
    targets.append(('Qwen2.5-14B BF16', '/vramer/models/Qwen2.5-14B-Instruct/model-00002-of-00008.safetensors',
                    safetensors_data_start('/vramer/models/Qwen2.5-14B-Instruct/model-00002-of-00008.safetensors')))
    # int8 w8a16 bonus
    targets.append(('Qwen2.5-14B int8(w8a16)', '/vramer/ckpt/qwen2.5-14b-w8a16-pp2/rank0.safetensors',
                    safetensors_data_start('/vramer/ckpt/qwen2.5-14b-w8a16-pp2/rank0.safetensors')))
    allres={}
    for label,path,ds in targets:
        print(f"[{label}] data starts at byte {ds}", file=sys.stderr)
        chunks = sample_chunks(path, ds)
        allres[label]=measure(label, chunks)
    # summary table
    print("\n\n################ SUMMARY (ratio x) ################")
    codecs=['lz4-fast','lz4-hc9','zstd-1','zstd-19']
    print(f"{'model':28s} " + " ".join(f"{c:>9s}" for c in codecs))
    for label in allres:
        print(f"{label:28s} " + " ".join(f"{allres[label][c][0]:9.4f}" for c in codecs))
