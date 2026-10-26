/*
 * llamacpp_poc.c — standalone reference for the VRAMER per-layer call pattern.
 *
 * This is NOT llama.cpp. It is a pure-C program that exercises exactly the
 * VRAMER C ABI sequence a llama.cpp weight-paging adapter would run for each
 * layer of a forward pass (see docs/LLAMACPP_ADAPTER.md). It treats argv[1] as
 * the model file (any file works: a .gguf, a .safetensors shard, or a random
 * blob), maps it with vramer_source_file, and sweeps every page of it once in
 * ascending order — the access pattern of a single token's layer scan — using:
 *
 *     prefetch_async(next page)   // pipeline the next layer's load
 *     acquire(page) -> devptr     // demand the current layer's page
 *     lease_wait_on(stream)       // order compute after the H2D (no device sync)
 *     <1-byte D2H off devptr>     // stand-in for "a kernel reads these bytes"
 *     lease_release_on(stream)    // allow eviction once this op is done
 *
 * and prints the pager telemetry (hits / misses / hit-rate / bytes_h2d) at the
 * end. On any VRAMER error it prints the errbuf and exits non-zero. If there is
 * no CUDA device it skips cleanly and exits 0.
 *
 * ---------------------------------------------------------------------------
 * EXACT COMPILE LINE (run from this examples/ directory):
 *
 *   gcc -I../include llamacpp_poc.c ../build/libvramer.a -lcudart -lstdc++ -lpthread -o /tmp/llamacpp_poc
 *
 * The VRAMER static lib is at ../build/libvramer.a after a normal
 * `cmake -S . -B build && cmake --build build -j`. libcudart lives in the CUDA
 * toolkit; on this box it is on the default linker path
 * (/usr/lib/x86_64-linux-gnu). If your cudart is elsewhere, find it with
 *   find / -name 'libcudart.so*' 2>/dev/null | head
 * and add -L<that dir> (and that dir on LD_LIBRARY_PATH at run time), e.g.:
 *   gcc -I../include llamacpp_poc.c ../build/libvramer.a \
 *       -L/usr/local/cuda/lib64 -lcudart -lstdc++ -lpthread -o /tmp/llamacpp_poc
 * -lstdc++ is required because libvramer.a is C++/CUDA; we link it into a C TU.
 *
 * RUN:
 *   /tmp/llamacpp_poc <model-file> [vram_budget_bytes]
 *   # e.g. page a 3.8 GiB shard through a 512 MiB budget:
 *   /tmp/llamacpp_poc /vramer/models/Qwen2.5-14B-Instruct/model-00001-of-00008.safetensors
 *   # or a throwaway blob:
 *   head -c 300M /dev/urandom > /tmp/blob && /tmp/llamacpp_poc /tmp/blob 268435456
 * ---------------------------------------------------------------------------
 */

#include "vramer/vramer_c.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* VRAMER error check: print errbuf and bail non-zero. */
#define V_FAIL(what)                                                        \
    do {                                                                    \
        fprintf(stderr, "%s: %s\n", (what), err);                          \
        goto fail;                                                          \
    } while (0)

#define CUDA_OK(expr, what)                                                 \
    do {                                                                    \
        cudaError_t _e = (expr);                                           \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "%s: %s\n", (what), cudaGetErrorString(_e));   \
            goto fail;                                                      \
        }                                                                   \
    } while (0)

int main(int argc, char** argv) {
    char err[256];
    vramer_source* src   = NULL;
    vramer_pager*  pager = NULL;
    cudaStream_t   stream = NULL;
    int            have_stream = 0;

    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model-file> [vram_budget_bytes]\n"
                "  simulates a llama.cpp layer scan paging <model-file> through\n"
                "  a fixed VRAM budget (default 512 MiB), page size 64 MiB.\n",
                argv[0]);
        return 2;
    }

    /* No CUDA device => skip cleanly (matches the test_c_api convention). */
    int ndev = 0;
    cudaError_t derr = cudaGetDeviceCount(&ndev);
    if (derr != cudaSuccess || ndev < 1) {
        printf("no CUDA device available; skipping llamacpp_poc\n");
        return 0;
    }
    CUDA_OK(cudaSetDevice(0), "cudaSetDevice");

    const char* path = argv[1];
    uint64_t vram_budget = 512ull << 20; /* 512 MiB default */
    if (argc >= 3) {
        unsigned long long v = strtoull(argv[2], NULL, 10);
        if (v != 0ull) vram_budget = (uint64_t)v;
    }
    const uint64_t page_size = 64ull << 20; /* 64 MiB sweet spot on A2 (README) */

    /* 1. The GGUF/weight file becomes a byte-addressable MemorySource. In a
     *    real adapter this is the mmap'd checkpoint llama.cpp already opens. */
    src = vramer_source_file(path, err, sizeof(err));
    if (!src) V_FAIL("vramer_source_file");

    /* 2. Fixed-budget pager over that source. MRU because the layer scan is
     *    cyclic across tokens — LRU collapses to 0% reuse (README fact #1). */
    vramer_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.gpu              = 0;
    cfg.vram_budget      = vram_budget;
    cfg.page_size        = page_size;
    cfg.staging_buffers  = 2;
    cfg.prefetch_workers = 1;            /* overlap next layer's load */
    cfg.eviction         = VRAMER_EVICT_MRU;

    pager = vramer_pager_create(src, &cfg, err, sizeof(err));
    if (!pager) V_FAIL("vramer_pager_create");

    /* Pager keeps its own shared_ptr ref; releasing our source handle is fine. */
    vramer_source_free(src);
    src = NULL;

    uint64_t pages = vramer_pager_page_count(pager);
    uint64_t cap   = vramer_pager_resident_capacity_pages(pager);
    printf("file=%s\n", path);
    printf("pages=%llu page_size=%lluMiB vram_budget=%lluMiB resident_capacity=%llu pages\n",
           (unsigned long long)pages,
           (unsigned long long)(page_size >> 20),
           (unsigned long long)(vram_budget >> 20),
           (unsigned long long)cap);
    if (pages == 0) {
        fprintf(stderr, "file has zero pages (empty file?)\n");
        goto fail;
    }
    if (cap < pages) {
        printf("budget < file: VRAMER will page %llu pages through %llu slots.\n",
               (unsigned long long)pages, (unsigned long long)cap);
    } else {
        printf("NOTE: budget >= file; whole model is resident, so paging is a "
               "no-op after the first pass (VRAMER does not help here).\n");
    }

    CUDA_OK(cudaStreamCreate(&stream), "cudaStreamCreate");
    have_stream = 1;

    /* 3. One sequential sweep == one token's layer scan. For layer L we:
     *      - prefetch L+1's page (pipeline), then
     *      - acquire L's page, order compute after its H2D, "read" it, release.
     *    This is the exact lease lifetime an adapter would wrap one op in. */
    for (uint64_t pg = 0; pg < pages; ++pg) {
        /* Prefetch the NEXT layer's page while we work on this one. Best-effort:
         * a full queue is a skip, not an error, so we don't treat it as fatal. */
        if (pg + 1 < pages) {
            if (vramer_pager_prefetch_async(pager, pg + 1, err, sizeof(err)) != 0)
                V_FAIL("vramer_pager_prefetch_async");
        }

        vramer_lease* lease = NULL;
        void*  devptr = NULL;
        size_t nbytes = 0;
        if (vramer_acquire(pager, pg, &lease, &devptr, &nbytes, err, sizeof(err)) != 0)
            V_FAIL("vramer_acquire");

        /* Order any work on `stream` after this page's H2D completes. This is a
         * stream-event wait, NOT a device synchronize. */
        if (vramer_lease_wait_on(lease, (void*)stream, err, sizeof(err)) != 0) {
            vramer_lease_destroy(lease);
            V_FAIL("vramer_lease_wait_on");
        }

        /* Stand-in for "a ggml kernel reads this tensor": pull 1 byte back to
         * prove the page is actually resident on the device at devptr. */
        unsigned char got = 0xff;
        cudaError_t ce = cudaMemcpyAsync(&got, devptr, 1, cudaMemcpyDeviceToHost, stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync: %s\n", cudaGetErrorString(ce));
            vramer_lease_destroy(lease);
            goto fail;
        }
        ce = cudaStreamSynchronize(stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "cudaStreamSynchronize: %s\n", cudaGetErrorString(ce));
            vramer_lease_destroy(lease);
            goto fail;
        }
        (void)got; /* value is data-dependent; residency is what we assert */

        /* Done reading this page's bytes: allow it to be evicted after the
         * stream passes this point. CONSUMES the lease handle. */
        if (vramer_lease_release_on(lease, (void*)stream, err, sizeof(err)) != 0)
            V_FAIL("vramer_lease_release_on");
    }

    /* Drain any in-flight prefetch and quiesce before reading stats. */
    if (vramer_pager_drain_prefetch_queue(pager, err, sizeof(err)) != 0)
        V_FAIL("vramer_pager_drain_prefetch_queue");
    if (vramer_pager_synchronize(pager, err, sizeof(err)) != 0)
        V_FAIL("vramer_pager_synchronize");

    /* 4. Telemetry — the numbers an adapter would compare against llama.cpp's
     *    own --n-gpu-layers partial-offload baseline. */
    vramer_stats st;
    memset(&st, 0, sizeof(st));
    vramer_pager_stats(pager, &st);

    {
        uint64_t demand = st.hits + st.misses;
        double hit_rate = demand ? (100.0 * (double)st.hits / (double)demand) : 0.0;
        double h2d_mib  = (double)st.bytes_h2d / (double)(1u << 20);
        printf("---- vramer stats ----\n");
        printf("hits=%llu misses=%llu hit_rate=%.1f%%\n",
               (unsigned long long)st.hits,
               (unsigned long long)st.misses,
               hit_rate);
        printf("evictions=%llu prefetches=%llu prefetch_queue_skips=%llu\n",
               (unsigned long long)st.evictions,
               (unsigned long long)st.prefetches,
               (unsigned long long)st.prefetch_queue_skips);
        printf("bytes_h2d=%llu (%.1f MiB)\n",
               (unsigned long long)st.bytes_h2d, h2d_mib);
    }

    cudaStreamDestroy(stream);
    vramer_pager_destroy(pager);
    printf("llamacpp_poc ok\n");
    return 0;

fail:
    if (have_stream) cudaStreamDestroy(stream);
    if (pager) vramer_pager_destroy(pager);
    if (src)   vramer_source_free(src);
    return 1;
}
