// VRAMER C ABI implementation.
//
// This translation unit is compiled as C++: it includes the real C++/CUDA
// VRAMER headers and bridges them to the pure-C surface declared in
// vramer_c.h. CUDA streams arrive as void* and are cast back to cudaStream_t.
//
// CRITICAL: no C++ exception may cross the C boundary. Every entry point wraps
// its body in try/catch(...) and converts any failure into the C error
// convention (NULL handle or non-zero return, plus a message copied into the
// caller's errbuf).

#include "vramer/vramer_c.h"

#include "vramer/cuda_pager.hpp"
#include "vramer/memory_source.hpp"

#include <cuda_runtime_api.h>

#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

// Copy an exception message into (buf,len) safely: always NUL-terminate,
// truncate as needed, and tolerate buf == NULL / len == 0.
void set_error(char* buf, size_t len, const char* msg) {
    if (!buf || len == 0) return;
    if (!msg) msg = "unknown error";
    size_t n = std::strlen(msg);
    if (n >= len) n = len - 1;
    std::memcpy(buf, msg, n);
    buf[n] = '\0';
}

vramer::EvictionPolicy to_policy(int e) {
    return e == VRAMER_EVICT_MRU ? vramer::EvictionPolicy::Mru
                                 : vramer::EvictionPolicy::Lru;
}

cudaStream_t to_stream(void* s) {
    return static_cast<cudaStream_t>(s);
}

} // namespace

// The opaque structs. Defined only in this .cpp; callers see them as opaque.
struct vramer_source {
    vramer::MemorySourcePtr ptr;
};

struct vramer_pager {
    std::unique_ptr<vramer::CudaPager> pager;
};

struct vramer_lease {
    vramer::PageLease lease;
};

// --- sources --------------------------------------------------------------

extern "C" vramer_source* vramer_source_ram(uint64_t bytes, char* errbuf, size_t errlen) {
    try {
        auto* h = new vramer_source();
        h->ptr = std::make_shared<vramer::AnonymousRamSource>(bytes, /*touch=*/true);
        return h;
    } catch (const std::exception& e) {
        set_error(errbuf, errlen, e.what());
        return nullptr;
    } catch (...) {
        set_error(errbuf, errlen, "unknown error in vramer_source_ram");
        return nullptr;
    }
}

extern "C" vramer_source* vramer_source_pinned(uint64_t bytes, char* errbuf, size_t errlen) {
    try {
        auto* h = new vramer_source();
        h->ptr = std::make_shared<vramer::PinnedHostSource>(bytes, /*touch=*/true);
        return h;
    } catch (const std::exception& e) {
        set_error(errbuf, errlen, e.what());
        return nullptr;
    } catch (...) {
        set_error(errbuf, errlen, "unknown error in vramer_source_pinned");
        return nullptr;
    }
}

extern "C" vramer_source* vramer_source_file(const char* path, char* errbuf, size_t errlen) {
    try {
        if (!path) throw std::invalid_argument("vramer_source_file: path is NULL");
        auto* h = new vramer_source();
        h->ptr = std::make_shared<vramer::MmapFileSource>(std::string(path));
        return h;
    } catch (const std::exception& e) {
        set_error(errbuf, errlen, e.what());
        return nullptr;
    } catch (...) {
        set_error(errbuf, errlen, "unknown error in vramer_source_file");
        return nullptr;
    }
}

extern "C" vramer_source* vramer_source_borrowed(const void* base, uint64_t bytes,
                                                 const char* label, int pinned,
                                                 char* errbuf, size_t errlen) {
    try {
        auto* h = new vramer_source();
        h->ptr = std::make_shared<vramer::BorrowedMemorySource>(
            base, bytes, std::string(label ? label : "borrowed"), pinned != 0);
        return h;
    } catch (const std::exception& e) {
        set_error(errbuf, errlen, e.what());
        return nullptr;
    } catch (...) {
        set_error(errbuf, errlen, "unknown error in vramer_source_borrowed");
        return nullptr;
    }
}

extern "C" void vramer_source_free(vramer_source* s) {
    delete s; // releases the shared_ptr ref; no-op on nullptr
}

// --- pager ----------------------------------------------------------------

extern "C" vramer_pager* vramer_pager_create(vramer_source* src, const vramer_config* cfg,
                                             char* errbuf, size_t errlen) {
    try {
        if (!src) throw std::invalid_argument("vramer_pager_create: source is NULL");
        if (!cfg) throw std::invalid_argument("vramer_pager_create: config is NULL");

        vramer::PagerConfig pc;
        pc.gpu = cfg->gpu;
        pc.vram_budget = cfg->vram_budget;
        pc.page_size = cfg->page_size;
        pc.staging_buffers = cfg->staging_buffers;
        pc.prefetch_workers = cfg->prefetch_workers;
        pc.eviction = to_policy(cfg->eviction);

        auto* h = new vramer_pager();
        // The CudaPager ctor takes MemorySourcePtr by value, so this copy keeps
        // the source alive even after the caller frees its own handle.
        h->pager = std::make_unique<vramer::CudaPager>(src->ptr, pc);
        return h;
    } catch (const std::exception& e) {
        set_error(errbuf, errlen, e.what());
        return nullptr;
    } catch (...) {
        set_error(errbuf, errlen, "unknown error in vramer_pager_create");
        return nullptr;
    }
}

extern "C" void vramer_pager_destroy(vramer_pager* p) {
    delete p; // no-op on nullptr
}

extern "C" uint64_t vramer_pager_page_count(vramer_pager* p) {
    return p ? p->pager->page_count() : 0;
}

extern "C" uint64_t vramer_pager_resident_capacity_pages(vramer_pager* p) {
    return p ? p->pager->resident_capacity_pages() : 0;
}

extern "C" void vramer_pager_stats(vramer_pager* p, vramer_stats* out) {
    if (!p || !out) return;
    const vramer::PagerStats s = p->pager->stats();
    out->hits = s.hits;
    out->misses = s.misses;
    out->evictions = s.evictions;
    out->bytes_h2d = s.bytes_h2d;
    out->prefetches = s.prefetches;
    out->prefetch_queue_skips = s.prefetch_queue_skips;
    out->source_read_ns = s.source_read_ns;
    out->stall_ns = s.stall_ns;
    out->lease_sync_fallbacks = s.lease_sync_fallbacks;
    out->bytes_d2h = s.bytes_d2h;
    out->direct_loads = s.direct_loads;
}

extern "C" int vramer_pager_prefetch_async(vramer_pager* p, uint64_t page_id,
                                           char* errbuf, size_t errlen) {
    try {
        if (!p) throw std::invalid_argument("vramer_pager_prefetch_async: pager is NULL");
        p->pager->prefetch_async(page_id);
        return 0;
    } catch (const std::exception& e) {
        set_error(errbuf, errlen, e.what());
        return 1;
    } catch (...) {
        set_error(errbuf, errlen, "unknown error in vramer_pager_prefetch_async");
        return 1;
    }
}

extern "C" int vramer_pager_drain_prefetch_queue(vramer_pager* p, char* errbuf, size_t errlen) {
    try {
        if (!p) throw std::invalid_argument("vramer_pager_drain_prefetch_queue: pager is NULL");
        p->pager->drain_prefetch_queue();
        return 0;
    } catch (const std::exception& e) {
        set_error(errbuf, errlen, e.what());
        return 1;
    } catch (...) {
        set_error(errbuf, errlen, "unknown error in vramer_pager_drain_prefetch_queue");
        return 1;
    }
}

extern "C" int vramer_pager_synchronize(vramer_pager* p, char* errbuf, size_t errlen) {
    try {
        if (!p) throw std::invalid_argument("vramer_pager_synchronize: pager is NULL");
        p->pager->synchronize();
        return 0;
    } catch (const std::exception& e) {
        set_error(errbuf, errlen, e.what());
        return 1;
    } catch (...) {
        set_error(errbuf, errlen, "unknown error in vramer_pager_synchronize");
        return 1;
    }
}

// --- lease lifecycle ------------------------------------------------------

extern "C" int vramer_acquire(vramer_pager* p, uint64_t page_id, vramer_lease** out_lease,
                              void** out_devptr, size_t* out_bytes, char* errbuf, size_t errlen) {
    try {
        if (!p) throw std::invalid_argument("vramer_acquire: pager is NULL");
        if (!out_lease) throw std::invalid_argument("vramer_acquire: out_lease is NULL");

        vramer::PageLease l = p->pager->acquire(page_id);
        void* dptr = l.device_ptr();
        size_t nbytes = l.bytes();

        auto* h = new vramer_lease{std::move(l)};
        *out_lease = h;
        if (out_devptr) *out_devptr = dptr;
        if (out_bytes) *out_bytes = nbytes;
        return 0;
    } catch (const std::exception& e) {
        set_error(errbuf, errlen, e.what());
        return 1;
    } catch (...) {
        set_error(errbuf, errlen, "unknown error in vramer_acquire");
        return 1;
    }
}

extern "C" int vramer_lease_wait_on(vramer_lease* l, void* stream, char* errbuf, size_t errlen) {
    try {
        if (!l) throw std::invalid_argument("vramer_lease_wait_on: lease is NULL");
        l->lease.wait_on(to_stream(stream));
        return 0;
    } catch (const std::exception& e) {
        set_error(errbuf, errlen, e.what());
        return 1;
    } catch (...) {
        set_error(errbuf, errlen, "unknown error in vramer_lease_wait_on");
        return 1;
    }
}

extern "C" int vramer_lease_release_on(vramer_lease* l, void* stream, char* errbuf, size_t errlen) {
    try {
        if (!l) throw std::invalid_argument("vramer_lease_release_on: lease is NULL");
        l->lease.release_on(to_stream(stream));
        delete l; // consume: free the heap wrapper
        return 0;
    } catch (const std::exception& e) {
        // Still consume the wrapper so the caller never double-frees or leaks.
        delete l;
        set_error(errbuf, errlen, e.what());
        return 1;
    } catch (...) {
        delete l;
        set_error(errbuf, errlen, "unknown error in vramer_lease_release_on");
        return 1;
    }
}

extern "C" void vramer_lease_destroy(vramer_lease* l) {
    // Deleting the wrapper runs ~PageLease, which performs the conservative
    // sync fallback if release_on was never called. Safe on nullptr.
    delete l;
}
