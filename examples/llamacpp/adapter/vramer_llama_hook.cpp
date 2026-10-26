// VRAMER <-> llama.cpp host-weight paging hook. See vramer_llama_hook.h.
//
// On first use (lazy), this opens the GGUF named by env VRAMER_GGUF, builds a
// tensor-name -> (file offset, size) map, and creates a VRAMER pager over the
// file with a fixed VRAM budget (env VRAMER_BUDGET, default 32 MiB) and page
// size (env VRAMER_PAGE, default 4 MiB). For every host WEIGHT tensor the
// scheduler asks to upload, it pages that tensor's bytes out of the file through
// the budget (acquire / wait_on / D2D-gather into the kernel's dst buffer /
// release_on, with one-ahead prefetch -- exactly the sequence proven in
// examples/llamacpp/vramer_ggml_*.cpp) instead of a plain H2D memcpy.

#include "vramer_llama_hook.h"

#include "vramer/vramer_c.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

struct TInfo { uint64_t off; uint64_t size; };

struct VramerHook {
    bool enabled = false;
    bool inited  = false;
    std::mutex mu;

    vramer_pager * pager = nullptr;
    uint64_t page_size = 4ull << 20;
    uint64_t budget    = 32ull << 20;
    int      gpu       = 0;
    cudaStream_t stream = nullptr;

    std::unordered_map<std::string, TInfo> tensors; // name -> file offset/size
    uint64_t paged_set_bytes = 0; // sum of sizes of DISTINCT tensors paged
    uint64_t paged_count     = 0; // number of DISTINCT tensors paged

    // telemetry
    std::atomic<uint64_t> served{0};     // #copies served from VRAMER
    std::atomic<uint64_t> served_bytes{0};
};

VramerHook g;
char errbuf[512];

uint64_t env_u64(const char * k, uint64_t dflt) {
    const char * v = getenv(k);
    if (!v || !*v) return dflt;
    return strtoull(v, nullptr, 10);
}

void report_atexit() {
    if (!g.pager) return;
    vramer_stats st; memset(&st, 0, sizeof st);
    vramer_pager_stats(g.pager, &st);
    uint64_t dem = st.hits + st.misses;
    fprintf(stderr,
        "\n[VRAMER] ===== weight paging summary =====\n"
        "[VRAMER] paged weight set (host, NOT in VRAM): %.2f MiB across %llu tensors\n"
        "[VRAMER] VRAM budget: %.2f MiB (page %.2f MiB, %llu slots) -- %s than paged set\n"
        "[VRAMER] copies served from VRAMER: %llu  (%.2f MiB of weight reads)\n"
        "[VRAMER] pager: hits=%llu misses=%llu hit_rate=%.1f%% evictions=%llu\n"
        "[VRAMER] bytes_h2d=%.2f MiB  <- real host->device paging traffic\n"
        "[VRAMER] ==================================\n",
        g.paged_set_bytes / 1048576.0, (unsigned long long) g.paged_count,
        g.budget / 1048576.0, g.page_size / 1048576.0,
        (unsigned long long) vramer_pager_resident_capacity_pages(g.pager),
        (g.budget < g.paged_set_bytes ? "SMALLER" : "NOT smaller"),
        (unsigned long long) g.served.load(), g.served_bytes.load() / 1048576.0,
        (unsigned long long) st.hits, (unsigned long long) st.misses,
        dem ? 100.0 * st.hits / dem : 0.0, (unsigned long long) st.evictions,
        st.bytes_h2d / 1048576.0);
}

bool init_locked() {
    if (g.inited) return g.enabled;
    g.inited = true;

    const char * path = getenv("VRAMER_GGUF");
    if (!path || !*path) { g.enabled = false; return false; }

    g.budget    = env_u64("VRAMER_BUDGET", 32ull << 20);
    g.page_size = env_u64("VRAMER_PAGE",   4ull  << 20);
    g.gpu       = (int) env_u64("VRAMER_GPU", 0);

    // 1. GGUF metadata -> name -> (abs file offset, size)
    gguf_init_params gp; gp.no_alloc = true; gp.ctx = nullptr;
    gguf_context * gctx = gguf_init_from_file(path, gp);
    if (!gctx) {
        fprintf(stderr, "[VRAMER] FATAL: cannot open GGUF '%s'\n", path);
        g.enabled = false; return false;
    }
    const uint64_t data_off = (uint64_t) gguf_get_data_offset(gctx);
    const int64_t  nt = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < nt; ++i) {
        const char * nm = gguf_get_tensor_name(gctx, i);
        TInfo t;
        t.off  = data_off + (uint64_t) gguf_get_tensor_offset(gctx, i);
        t.size = (uint64_t) gguf_get_tensor_size(gctx, i);
        g.tensors[std::string(nm)] = t;
    }
    gguf_free(gctx);

    // 2. VRAMER pager over the GGUF file, fixed budget, MRU (cyclic layer scan).
    cudaSetDevice(g.gpu);
    vramer_source * srcf = vramer_source_file(path, errbuf, sizeof errbuf);
    if (!srcf) {
        fprintf(stderr, "[VRAMER] FATAL source_file: %s\n", errbuf);
        g.enabled = false; return false;
    }
    vramer_config cfg; memset(&cfg, 0, sizeof cfg);
    cfg.gpu = g.gpu;
    cfg.vram_budget = g.budget;
    cfg.page_size = g.page_size;
    cfg.staging_buffers = 4;
    cfg.prefetch_workers = 1;
    cfg.eviction = VRAMER_EVICT_MRU;
    g.pager = vramer_pager_create(srcf, &cfg, errbuf, sizeof errbuf);
    vramer_source_free(srcf);
    if (!g.pager) {
        fprintf(stderr, "[VRAMER] FATAL pager_create: %s\n", errbuf);
        g.enabled = false; return false;
    }
    cudaStreamCreate(&g.stream);

    g.enabled = true;
    atexit(report_atexit);
    fprintf(stderr,
        "[VRAMER] ENABLED: paging weights from '%s' through a %.0f MiB budget "
        "(page %.0f MiB, %llu slots)\n",
        path, g.budget / 1048576.0, g.page_size / 1048576.0,
        (unsigned long long) vramer_pager_resident_capacity_pages(g.pager));
    return true;
}

// Page the tensor [off,off+size) from VRAMER into contiguous device dst.
bool gather_into(void * dst, uint64_t off, uint64_t size) {
    const uint64_t fp = off / g.page_size;
    const uint64_t lp = (off + size - 1) / g.page_size;
    for (uint64_t pg = fp; pg <= lp; ++pg) {
        if (pg + 1 <= lp) {
            vramer_pager_prefetch_async(g.pager, pg + 1, errbuf, sizeof errbuf);
        }
        vramer_lease * ls = nullptr; void * dp = nullptr; size_t nb = 0;
        if (vramer_acquire(g.pager, pg, &ls, &dp, &nb, errbuf, sizeof errbuf)) {
            fprintf(stderr, "[VRAMER] acquire failed: %s\n", errbuf); return false;
        }
        if (vramer_lease_wait_on(ls, (void*) g.stream, errbuf, sizeof errbuf)) {
            fprintf(stderr, "[VRAMER] wait_on failed: %s\n", errbuf); return false;
        }
        const uint64_t ps = pg * g.page_size;
        const uint64_t s0 = (off > ps) ? off : ps;
        const uint64_t s1 = (off + size < ps + g.page_size) ? (off + size) : (ps + g.page_size);
        cudaError_t e = cudaMemcpyAsync((char*) dst + (s0 - off),
                                        (const char*) dp + (s0 - ps),
                                        s1 - s0, cudaMemcpyDeviceToDevice, g.stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "[VRAMER] D2D copy failed: %s\n", cudaGetErrorString(e));
            vramer_lease_release_on(ls, (void*) g.stream, errbuf, sizeof errbuf);
            return false;
        }
        if (vramer_lease_release_on(ls, (void*) g.stream, errbuf, sizeof errbuf)) {
            fprintf(stderr, "[VRAMER] release_on failed: %s\n", errbuf); return false;
        }
    }
    // Make the gathered bytes globally visible before the compute kernel (on a
    // different stream) reads dst.
    cudaError_t e = cudaStreamSynchronize(g.stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[VRAMER] sync failed: %s\n", cudaGetErrorString(e)); return false;
    }
    return true;
}

} // namespace

extern "C" int vramer_llama_copy_hook(ggml_backend_t /*backend*/,
                                      const struct ggml_tensor * src,
                                      struct ggml_tensor * dst) {
    if (!g.inited) {
        std::lock_guard<std::mutex> lk(g.mu);
        init_locked();
    }
    if (!g.enabled) return 0;
    if (!src || !dst || !dst->data || !src->name[0]) return 0;

    auto it = g.tensors.find(std::string(src->name));
    if (it == g.tensors.end()) return 0; // not a file-backed weight we know

    const uint64_t size = (uint64_t) ggml_nbytes(src);
    const uint64_t fsize = it->second.size;
    const uint64_t copy_size = size < fsize ? size : fsize;

    std::lock_guard<std::mutex> lk(g.mu);
    if (g.served_bytes.load() == 0) {
        // record the resident paged set once (tensors actually routed as host
        // weights are the ones the scheduler asks us to copy)
    }
    if (!gather_into(dst->data, it->second.off, copy_size)) {
        return 0; // fall back to default copy on any error
    }
    // account this tensor into the "paged set" the first time we see it
    static thread_local bool unused = false; (void) unused;
    {
        // track distinct tensors paged -> the real resident set we replaced
        static std::unordered_map<std::string,bool> seen;
        if (!seen.count(src->name)) { seen[src->name] = true; g.paged_set_bytes += fsize; g.paged_count++; }
    }
    g.served.fetch_add(1);
    g.served_bytes.fetch_add(copy_size);
    return 1;
}
