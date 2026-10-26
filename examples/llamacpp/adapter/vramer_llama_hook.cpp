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
//
// For a MoE expert stack (a 3D tensor, ne[2] = n_expert, e.g.
// blk.N.ffn_down_exps.weight) the hook pages ONLY the experts that are active
// for the current token. It recovers the active set the same way llama.cpp's
// own llama_context::sched_copy_experts does: the split graph's first node is
// the GGML_OP_MUL_MAT_ID whose src[0] is the (device) expert buffer and whose
// src[2] is the int32 routing ids; the distinct ids are the active experts.
// Only those expert slices are paged from the file, so the bytes that stream
// per token are ~ (n_expert_used / n_expert) of the stack.

#include "vramer_llama_hook.h"

#include "vramer/vramer_c.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    std::unordered_set<std::string> seen;

    // telemetry
    std::atomic<uint64_t> served{0};      // #copies served from VRAMER
    std::atomic<uint64_t> served_bytes{0};// bytes actually streamed (active only)
    std::atomic<uint64_t> expert_copies{0};  // #MoE expert-stack copies served
    std::atomic<uint64_t> experts_paged{0};  // total active-expert slices paged
    std::atomic<uint64_t> expert_full_bytes{0}; // what a full-stack copy WOULD cost
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
    const uint64_t ef = g.expert_full_bytes.load();
    const uint64_t sb = g.served_bytes.load();
    fprintf(stderr,
        "\n[VRAMER] ===== weight paging summary =====\n"
        "[VRAMER] paged weight set (host, NOT resident in VRAM): %.2f MiB across %llu tensors\n"
        "[VRAMER] VRAM budget: %.2f MiB (page %.2f MiB, %llu slots) -- %s than paged set\n"
        "[VRAMER] copies served from VRAMER: %llu  (of which %llu were MoE expert stacks)\n"
        "[VRAMER] bytes streamed (active only): %.2f MiB\n"
        "[VRAMER] MoE: %llu active-expert slices paged; a full-stack copy would have moved %.2f MiB"
        " -> streamed %.1f%% of that\n"
        "[VRAMER] pager: hits=%llu misses=%llu hit_rate=%.1f%% evictions=%llu\n"
        "[VRAMER] bytes_h2d=%.2f MiB  <- real host->device paging traffic\n"
        "[VRAMER] ==================================\n",
        g.paged_set_bytes / 1048576.0, (unsigned long long) g.paged_count,
        g.budget / 1048576.0, g.page_size / 1048576.0,
        (unsigned long long) vramer_pager_resident_capacity_pages(g.pager),
        (g.budget < g.paged_set_bytes ? "SMALLER" : "NOT smaller"),
        (unsigned long long) g.served.load(), (unsigned long long) g.expert_copies.load(),
        sb / 1048576.0,
        (unsigned long long) g.experts_paged.load(), ef / 1048576.0,
        ef ? 100.0 * sb / ef : 0.0,
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
    // MRU suits the dense cyclic-layer scan; LRU suits MoE, where a few experts
    // are "popular" across tokens and worth keeping resident. Default MRU.
    cfg.eviction = env_u64("VRAMER_EVICT", VRAMER_EVICT_MRU) ? VRAMER_EVICT_MRU
                                                             : VRAMER_EVICT_LRU;
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

// Gather the file range [off,off+size) from VRAMER into contiguous device dst,
// issuing all acquire/wait_on/D2D/release work on g.stream WITHOUT a CPU sync.
// The caller must sync_stream() once before the gathered bytes are consumed.
// Each page's lease is released_on(g.stream) immediately after its D2D is queued
// (ordered on the stream), so at most the pager's slot depth is ever in flight --
// identical residency to the old per-call path, only without the blocking sync
// between slices.
bool gather_range_async(void * dst, uint64_t off, uint64_t size) {
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
    return true;
}

// Make everything queued on g.stream globally visible before the compute kernel
// (launched by the scheduler on a different stream, after this hook returns)
// reads dst.
bool sync_stream() {
    cudaError_t e = cudaStreamSynchronize(g.stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[VRAMER] sync failed: %s\n", cudaGetErrorString(e)); return false;
    }
    return true;
}

// Page the file range [off,off+size) and block until it is visible. Used by the
// dense whole-tensor path (a single contiguous range -- nothing to overlap).
bool gather_into(void * dst, uint64_t off, uint64_t size) {
    if (!gather_range_async(dst, off, size)) return false;
    return sync_stream();
}

void account_distinct(const char * name, uint64_t fsize) {
    if (!g.seen.count(name)) { g.seen.insert(name); g.paged_set_bytes += fsize; g.paged_count++; }
}

// Dense path: page the whole tensor into dst->data.
int serve_whole(const char * name, const TInfo & ti, uint64_t src_bytes, void * dst) {
    const uint64_t copy_size = src_bytes < ti.size ? src_bytes : ti.size;
    if (!gather_into(dst, ti.off, copy_size)) return 0; // fall back on error
    account_distinct(name, ti.size);
    g.served.fetch_add(1);
    g.served_bytes.fetch_add(copy_size);
    return 1;
}

// MoE path: page ONLY the experts active for this token. Returns 1 if handled
// (including the no-active-experts case), 0 to fall back to the default copy.
int serve_experts(ggml_backend_t backend, const struct ggml_tensor * src,
                  struct ggml_tensor * dst, struct ggml_cgraph * graph,
                  const char * name, const TInfo & ti) {
    if (!graph || ggml_graph_n_nodes(graph) == 0) return 0;

    // The active experts for this split are computed from the MUL_MAT_ID that
    // consumes `dst`. Only the split's first node is guaranteed to carry the ids.
    const struct ggml_tensor * node = ggml_graph_node(graph, 0);
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) return 0;

    const struct ggml_tensor * ids = node->src[2];
    if (!ids) return 0;

    const int64_t n_expert    = src->ne[2];
    const uint64_t expert_size = (uint64_t) src->nb[2];

    account_distinct(name, ti.size); // the whole stack is host-resident
    g.expert_full_bytes.fetch_add(ti.size);

    if (ggml_nelements(ids) == 0) { g.expert_copies.fetch_add(1); return 1; }

    // read the routing ids to host and mark the used experts.
    std::vector<int32_t> ids_data(ggml_nbytes(ids) / sizeof(int32_t));
    ggml_backend_tensor_get_async(backend, ids, ids_data.data(), 0, ggml_nbytes(ids));
    ggml_backend_synchronize(backend);

    std::vector<char> used(n_expert, 0);
    for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
        for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
            const int32_t id = ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
            if (id >= 0 && id < n_expert) used[id] = 1;
        }
    }

    // Collect the consecutive runs of used experts (matches llama.cpp's
    // grouping, including the small trailing padding it copies to keep MMQ
    // NaN-free). The active set for one token is usually a few scattered experts
    // -> several short runs; the old path paged each run with its own blocking
    // sync, so a run's H2D could not begin until the previous run was fully
    // resident. We now overlap the runs: queue every run's acquire/D2D on
    // g.stream back-to-back and block once at the end.
    struct Run { uint64_t off; uint64_t size; uint64_t dst_off; };
    std::vector<Run> runs;
    uint64_t streamed = 0, slices = 0;
    for (int64_t first = 0; first < n_expert; ) {
        if (!used[first]) { first++; continue; }
        int64_t last = first;
        while (last + 1 < n_expert && used[last + 1]) last++;

        const uint64_t offset  = (uint64_t) first * expert_size;
        const uint64_t padding = last < n_expert - 1 ? std::min<uint64_t>(expert_size, 512) : 0;
        const uint64_t gsize   = (uint64_t)(last + 1 - first) * expert_size + padding;

        runs.push_back({ ti.off + offset, gsize, offset });
        streamed += gsize;
        slices   += (uint64_t)(last + 1 - first);
        first = last + 1;
    }

    // Queue every run's acquire/wait_on/D2D on g.stream back-to-back with NO
    // CPU sync between runs, then block once at the end. The slice H2Ds (driven
    // by the pager worker as each page is acquired) and the D2D gathers of the
    // several runs now pipeline on the stream instead of each run stalling the
    // CPU before the next can start -- the only behavioural change vs the old
    // synchronous path, and it leaves the delivered bytes identical.
    //
    // We deliberately do NOT speculatively prefetch the whole active set up
    // front: at a tight budget that floods the cache and evicts pages the very
    // next acquire still needs (measured: hit-rate 60% -> 4%, and a net loss at
    // roomy budgets). gather_range_async's one-ahead prefetch already pipelines
    // within each run; the win here is purely from removing the inter-run sync.
    for (const Run & r : runs) {
        if (!gather_range_async((char*) dst->data + r.dst_off, r.off, r.size)) return 0;
    }
    if (!runs.empty() && !sync_stream()) return 0;

    g.served.fetch_add(1);
    g.expert_copies.fetch_add(1);
    g.experts_paged.fetch_add(slices);
    g.served_bytes.fetch_add(streamed);
    return 1;
}

} // namespace

extern "C" int vramer_llama_copy_hook(ggml_backend_t backend,
                                      const struct ggml_tensor * src,
                                      struct ggml_tensor * dst,
                                      struct ggml_cgraph * graph) {
    if (!g.inited) {
        std::lock_guard<std::mutex> lk(g.mu);
        init_locked();
    }
    if (!g.enabled) return 0;
    if (!src || !dst || !dst->data || !src->name[0]) return 0;

    auto it = g.tensors.find(std::string(src->name));
    if (it == g.tensors.end()) return 0; // not a file-backed weight we know

    std::lock_guard<std::mutex> lk(g.mu);

    // 3D expert stack -> page only the active experts; 2D weight -> page whole.
    if (src->ne[2] > 1) {
        return serve_experts(backend, src, dst, graph, src->name, it->second);
    }
    return serve_whole(src->name, it->second, (uint64_t) ggml_nbytes(src), dst->data);
}
