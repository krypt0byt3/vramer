#ifndef VRAMER_C_H
#define VRAMER_C_H

/*
 * VRAMER C ABI.
 *
 * A pure-C surface over the C++/CUDA VRAMER pager so that non-C++ callers
 * (Python via ctypes, Rust via FFI/cudarc, etc.) can drive the pager without
 * a C++ toolchain.
 *
 * This header is includable from pure C and from C++ (via the extern "C"
 * guard). It intentionally pulls in NO CUDA or C++ headers: only <stdint.h>
 * and <stddef.h>.
 *
 * CUDA streams cross the boundary as void*: a caller holding a cudaStream_t
 * casts it to void* when passing it here, and the implementation casts it
 * back. All VRAMER objects are opaque struct pointers.
 *
 * Error convention: a function that can fail returns int (0 == ok, non-zero
 * == error) and takes trailing (char* errbuf, size_t errlen) that receive a
 * NUL-terminated message on failure. Constructors instead return a handle and
 * return NULL on failure (still writing errbuf). Simple accessors return their
 * value directly and never throw. No C++ exception ever crosses this boundary.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. */
typedef struct vramer_source vramer_source; /* owns a shared_ptr<MemorySource> */
typedef struct vramer_pager  vramer_pager;  /* owns a CudaPager                */
typedef struct vramer_lease  vramer_lease;  /* owns a heap PageLease           */

typedef enum {
    VRAMER_EVICT_LRU = 0,
    VRAMER_EVICT_MRU = 1
} vramer_eviction;

typedef struct {
    int      gpu;
    uint64_t vram_budget;
    uint64_t page_size;
    unsigned staging_buffers;
    unsigned prefetch_workers;
    int      eviction; /* vramer_eviction */
} vramer_config;

typedef struct {
    uint64_t hits, misses, evictions, bytes_h2d;
    uint64_t prefetches, prefetch_queue_skips;
    uint64_t source_read_ns, stall_ns, lease_sync_fallbacks;
} vramer_stats;

/* --- sources (return NULL on failure, writing errbuf) ------------------- */
vramer_source* vramer_source_ram(uint64_t bytes, char* errbuf, size_t errlen);
vramer_source* vramer_source_file(const char* path, char* errbuf, size_t errlen);
vramer_source* vramer_source_borrowed(const void* base, uint64_t bytes,
                                      const char* label, char* errbuf, size_t errlen);
/* Releases this handle's shared_ptr reference. Safe on NULL. */
void vramer_source_free(vramer_source*);

/* --- pager -------------------------------------------------------------- */
vramer_pager* vramer_pager_create(vramer_source* src, const vramer_config* cfg,
                                  char* errbuf, size_t errlen); /* NULL on failure */
void     vramer_pager_destroy(vramer_pager*); /* safe on NULL */
uint64_t vramer_pager_page_count(vramer_pager*);
uint64_t vramer_pager_resident_capacity_pages(vramer_pager*);
void     vramer_pager_stats(vramer_pager*, vramer_stats* out);
int      vramer_pager_prefetch_async(vramer_pager*, uint64_t page_id,
                                     char* errbuf, size_t errlen);
int      vramer_pager_drain_prefetch_queue(vramer_pager*, char* errbuf, size_t errlen);
int      vramer_pager_synchronize(vramer_pager*, char* errbuf, size_t errlen);

/* --- lease lifecycle ---------------------------------------------------- */
/* On success returns 0 and writes *out_lease, *out_devptr, *out_bytes. */
int  vramer_acquire(vramer_pager*, uint64_t page_id, vramer_lease** out_lease,
                    void** out_devptr, size_t* out_bytes, char* errbuf, size_t errlen);
int  vramer_lease_wait_on(vramer_lease*, void* stream, char* errbuf, size_t errlen);
/* CONSUMES the lease: calls PageLease::release_on(stream), then frees the heap
 * wrapper. Do not use the handle afterward. */
int  vramer_lease_release_on(vramer_lease*, void* stream, char* errbuf, size_t errlen);
/* For error paths where you acquired but will not release_on: deletes the
 * wrapper (triggering the C++ conservative sync fallback). Safe on NULL. */
void vramer_lease_destroy(vramer_lease*);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VRAMER_C_H */
