/*
 * Pure-C smoke test for the VRAMER C ABI.
 *
 * Compiled as C (.c). Uses ONLY vramer_c.h for the pager, plus
 * <cuda_runtime.h> solely to create/destroy a stream and run a tiny D2H copy.
 * If no CUDA device is present, it prints a skip message and returns 0.
 */

#include "vramer/vramer_c.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "CHECK failed: %s (%s:%d)\n", (msg),           \
                    __FILE__, __LINE__);                                    \
            abort();                                                        \
        }                                                                   \
    } while (0)

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t _e = (expr);                                           \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                    \
                    cudaGetErrorString(_e), __FILE__, __LINE__);           \
            abort();                                                        \
        }                                                                   \
    } while (0)

int main(void) {
    char err[256];

    int ndev = 0;
    cudaError_t derr = cudaGetDeviceCount(&ndev);
    if (derr != cudaSuccess || ndev < 1) {
        printf("no CUDA device available; skipping c_api test\n");
        return 0;
    }
    CUDA_CHECK(cudaSetDevice(0));

    const uint64_t source_bytes = 8ull << 20; /* 8 MiB */
    const uint64_t page_size    = 1ull << 20; /* 1 MiB */
    const uint64_t vram_budget  = 4ull << 20; /* 4 MiB => 4 slots */

    vramer_source* src = vramer_source_ram(source_bytes, err, sizeof(err));
    CHECK(src != NULL, "vramer_source_ram");
    if (!src) { fprintf(stderr, "%s\n", err); return 1; }

    vramer_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.gpu = 0;
    cfg.vram_budget = vram_budget;
    cfg.page_size = page_size;
    cfg.staging_buffers = 2;
    cfg.prefetch_workers = 1;
    cfg.eviction = VRAMER_EVICT_MRU;

    vramer_pager* pager = vramer_pager_create(src, &cfg, err, sizeof(err));
    if (!pager) { fprintf(stderr, "vramer_pager_create: %s\n", err); abort(); }

    /* The pager keeps its own shared_ptr ref; freeing our handle is safe. */
    vramer_source_free(src);

    uint64_t pages = vramer_pager_page_count(pager);
    uint64_t cap   = vramer_pager_resident_capacity_pages(pager);
    printf("pages=%llu resident_capacity=%llu\n",
           (unsigned long long)pages, (unsigned long long)cap);
    CHECK(pages == source_bytes / page_size, "page_count == 8");
    CHECK(cap == vram_budget / page_size, "resident_capacity == 4");

    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaStreamCreate(&stream));

    long host_page = sysconf(_SC_PAGESIZE);
    CHECK(host_page > 0, "sysconf(_SC_PAGESIZE) > 0");

    for (uint64_t pg = 0; pg < pages; ++pg) {
        vramer_lease* lease = NULL;
        void*  devptr = NULL;
        size_t nbytes = 0;

        int rc = vramer_acquire(pager, pg, &lease, &devptr, &nbytes, err, sizeof(err));
        if (rc != 0) { fprintf(stderr, "vramer_acquire(%llu): %s\n",
                               (unsigned long long)pg, err); abort(); }
        CHECK(lease != NULL, "acquire out_lease");
        CHECK(devptr != NULL, "acquire out_devptr");
        CHECK(nbytes == page_size, "acquire out_bytes");

        rc = vramer_lease_wait_on(lease, (void*)stream, err, sizeof(err));
        if (rc != 0) { fprintf(stderr, "vramer_lease_wait_on: %s\n", err); abort(); }

        /* Pull the first byte of this page off the device. */
        unsigned char got = 0xff;
        CUDA_CHECK(cudaMemcpyAsync(&got, devptr, 1, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        uint64_t src_off = pg * page_size;
        unsigned char want = (unsigned char)((src_off / (uint64_t)host_page) & 0xffu);
        if (got != want) {
            fprintf(stderr, "page %llu: got 0x%02x want 0x%02x\n",
                    (unsigned long long)pg, got, want);
        }
        CHECK(got == want, "device byte matches source pattern");

        rc = vramer_lease_release_on(lease, (void*)stream, err, sizeof(err));
        if (rc != 0) { fprintf(stderr, "vramer_lease_release_on: %s\n", err); abort(); }
    }

    /* Exercise the prefetch path once. */
    int rc = vramer_pager_prefetch_async(pager, 0, err, sizeof(err));
    if (rc != 0) { fprintf(stderr, "vramer_pager_prefetch_async: %s\n", err); abort(); }
    rc = vramer_pager_drain_prefetch_queue(pager, err, sizeof(err));
    if (rc != 0) { fprintf(stderr, "vramer_pager_drain_prefetch_queue: %s\n", err); abort(); }

    rc = vramer_pager_synchronize(pager, err, sizeof(err));
    if (rc != 0) { fprintf(stderr, "vramer_pager_synchronize: %s\n", err); abort(); }

    vramer_stats st;
    memset(&st, 0, sizeof(st));
    vramer_pager_stats(pager, &st);
    printf("hits=%llu misses=%llu bytes_h2d=%llu\n",
           (unsigned long long)st.hits,
           (unsigned long long)st.misses,
           (unsigned long long)st.bytes_h2d);
    CHECK(st.misses > 0, "misses > 0");

    CUDA_CHECK(cudaStreamDestroy(stream));
    vramer_pager_destroy(pager);

    printf("c_api test passed\n");
    return 0;
}
