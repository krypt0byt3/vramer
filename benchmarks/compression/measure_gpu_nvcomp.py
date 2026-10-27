#!/usr/bin/env python3
"""GPU decompression ratio + throughput on real weight bytes using nvCOMP.
Run with CUDA_VISIBLE_DEVICES=1. Measures GDeflate / LZ4 / Deflate:
 - compression ratio on real tensor bytes
 - GPU decompress throughput (uncompressed GB/s) -> the number that matters
 - byte-exactness check."""
import struct, os, time, sys
import numpy as np
import nvidia.nvcomp as nvcomp

CHUNK = 256 * 1024 * 1024  # one 256 MiB real sample per model

def gguf_data_start(path):
    with open(path,'rb') as f:
        assert f.read(4)==b'GGUF'; struct.unpack('<I',f.read(4))
        nt,=struct.unpack('<Q',f.read(8)); nkv,=struct.unpack('<Q',f.read(8))
        def rstr():
            ln,=struct.unpack('<Q',f.read(8)); return f.read(ln)
        sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        def rv(t):
            if t==8: rstr(); return
            if t==9:
                et,=struct.unpack('<I',f.read(4)); cnt,=struct.unpack('<Q',f.read(8))
                if et==8:
                    for _ in range(cnt): rstr()
                else: f.seek(sizes[et]*cnt,1)
                return
            f.seek(sizes[t],1)
        for _ in range(nkv):
            rstr(); t,=struct.unpack('<I',f.read(4)); rv(t)
        for _ in range(nt):
            rstr(); nd,=struct.unpack('<I',f.read(4)); f.seek(8*nd+4+8,1)
        pos=f.tell(); return (pos+31)//32*32

def st_data_start(path):
    with open(path,'rb') as f:
        n,=struct.unpack('<Q',f.read(8))
    return 8+n

def load_sample(path, ds):
    fsize=os.path.getsize(path); region=fsize-ds
    off=ds + region//3  # a third of the way into the tensor region
    sz=min(CHUNK, region- (region//3))
    with open(path,'rb') as f:
        f.seek(off); buf=f.read(sz)
    return np.frombuffer(buf, dtype=np.uint8)

def measure_gpu(label, host_arr, algos):
    src = nvcomp.as_array(host_arr).cuda()   # H2D once
    stream = src  # noop
    print(f"\n=== {label}  ({host_arr.nbytes/1024/1024:.0f} MiB real tensor bytes) ===")
    for algo, kw in algos:
        try:
            codec = nvcomp.Codec(algorithm=algo, uncomp_chunk_size=65536, **kw)
            comp = codec.encode(src)
            csz = comp.buffer_size
            ratio = host_arr.nbytes / csz
            # warm up + time decode
            dec = codec.decode(comp, '|u1')
            # sync
            _ = dec.cpu()
            N=10
            t=time.time()
            for _ in range(N):
                d = codec.decode(comp, '|u1')
            _ = d.cpu()  # force sync on last
            dt=(time.time()-t)/N
            gbps = host_arr.nbytes/dt/1e9
            # byte-exact check
            back = np.array(dec.cpu())
            ok = (back.nbytes==host_arr.nbytes) and bool((back==host_arr).all())
            print(f"  {algo:10s}{str(kw):28s} ratio={ratio:.4f}x  GPU-decomp={gbps:7.1f} GB/s  exact={ok}")
        except Exception as e:
            print(f"  {algo:10s} FAILED: {repr(e)[:120]}")

if __name__=='__main__':
    algos=[('GDeflate',{'algorithm_type':1}),  # high throughput default
           ('GDeflate',{'algorithm_type':4}),  # higher ratio (~zlib-6)
           ('LZ4',{}),
           ('Deflate',{}),
           ('Zstd',{}),
           ('ANS',{})]
    tgts=[('MoE-A2.7B Q4_K_M','/tmp/models/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf',gguf_data_start),
          ('Qwen2.5-14B BF16','/vramer/models/Qwen2.5-14B-Instruct/model-00002-of-00008.safetensors',st_data_start)]
    for label,path,dsfn in tgts:
        ds=dsfn(path)
        arr=load_sample(path,ds)
        measure_gpu(label, arr, algos)
