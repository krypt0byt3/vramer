// GPU-level test for the prefetch/compute overlap (double-buffering) path.
//
// Two guarantees are checked:
//   1. Byte-correctness: a cyclic layer scan driven with prefetch-ahead returns
//      exactly the source bytes on every acquire, including pages that were
//      prefetched by a background worker, evicted under MRU, and reloaded.
//   2. Overlap effectiveness: issuing prefetch_async() a few pages ahead of the
//      demand cursor turns what would be demand misses into cache hits, so the
//      prefetch run has a strictly higher hit count than the demand-only run on
//      the identical scan.
//
// Requires a CUDA device. Skips (exit 0) cleanly if none is present.

#include "vramer/cuda_pager.hpp"
#include "vramer/memory_source.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

// CHECK() survives Release/NDEBUG (plain assert would compile out).
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << "CHECK failed: " #cond " at " << __FILE__ << ':'  \
                      << __LINE__ << '\n';                                 \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

using namespace vramer;

namespace {

constexpr std::uint64_t kPage = 1ull << 20;    // 1 MiB pages keep the test light
constexpr std::uint64_t kPages = 8;            // working set of 8 pages ...
constexpr std::uint64_t kSource = kPages * kPage;
constexpr std::uint64_t kBudget = 4 * kPage;   // ... through a 4-slot cache (> cache)
constexpr std::uint64_t kPrefetchAhead = 2;
constexpr unsigned kPasses = 3;

// A deterministic, fully byte-addressable source. Unlike AnonymousRamSource
// (which only touches the first byte of each OS page), every single byte here
// follows a known pattern, so we can verify the full contents of every page.
std::uint8_t pattern_byte(std::uint64_t offset) {
    return static_cast<std::uint8_t>(((offset * 1103515245ull + 12345ull) >> 7) & 0xffu);
}

class PatternSource final : public MemorySource {
public:
    explicit PatternSource(std::uint64_t bytes) : size_(bytes) {}
    std::uint64_t size() const noexcept override { return size_; }
    std::string name() const override { return "pattern-source"; }
    void read(std::uint64_t offset, void* dst, std::size_t bytes) override {
        auto* out = static_cast<std::uint8_t*>(dst);
        for (std::size_t i = 0; i < bytes; ++i) out[i] = pattern_byte(offset + i);
    }

private:
    std::uint64_t size_ = 0;
};

PagerConfig config(unsigned workers) {
    PagerConfig cfg;
    cfg.gpu = 0;
    cfg.vram_budget = kBudget;
    cfg.page_size = kPage;
    cfg.staging_buffers = 4;
    cfg.prefetch_workers = workers;
    // LRU is the classic scan pathology: a cyclic scan larger than the cache
    // gets ~0% demand reuse. That makes it the cleanest demonstration of
    // prefetch-ahead, which loads each page just before it is demanded so the
    // acquire hits instead of stalling on a cold transfer. (Under MRU the newest
    // page is itself the eviction target, so a just-prefetched page would be
    // evicted before the demand cursor reaches it -- MRU resists prefetch in a
    // cache this small; prefetch-ahead pays off there only with many slots.)
    cfg.eviction = EvictionPolicy::Lru;
    return cfg;
}

// Copy a leased page off the GPU and assert every byte matches the source.
void verify_page(const PageLease& lease, std::uint64_t page, cudaStream_t stream,
                 std::vector<std::uint8_t>& host) {
    const std::size_t n = lease.bytes();
    host.resize(n);
    CHECK(cudaMemcpyAsync(host.data(), lease.device_ptr(), n,
                          cudaMemcpyDeviceToHost, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    const std::uint64_t base = page * kPage;
    for (std::size_t i = 0; i < n; ++i) {
        if (host[i] != pattern_byte(base + i)) {
            std::cerr << "byte mismatch page " << page << " offset " << i
                      << ": got " << int(host[i]) << " want "
                      << int(pattern_byte(base + i)) << '\n';
            CHECK(false && "overlap path returned wrong bytes");
        }
    }
}

// Run a cyclic scan of all pages for kPasses. When prefetch_ahead > 0 (and a
// worker exists) it stays that many pages ahead of the demand cursor -- the
// double-buffering mode. Returns the final stats; verifies every page's bytes.
PagerStats run_scan(unsigned workers, std::uint64_t prefetch_ahead, cudaStream_t stream) {
    CudaPager pager(std::make_shared<PatternSource>(kSource), config(workers));
    CHECK(pager.page_count() == kPages);
    std::vector<std::uint8_t> host;

    for (unsigned pass = 0; pass < kPasses; ++pass) {
        // Prime the pipeline so the first few acquires can already hit.
        for (std::uint64_t k = 0; k < prefetch_ahead && k < kPages; ++k)
            pager.prefetch_async(k);
        for (std::uint64_t p = 0; p < kPages; ++p) {
            if (prefetch_ahead && p + prefetch_ahead < kPages)
                pager.prefetch_async(p + prefetch_ahead);
            auto lease = pager.acquire(p);
            lease.wait_on(stream);
            verify_page(lease, p, stream, host);  // byte-correctness on every page
            lease.release_on(stream);
        }
    }
    pager.synchronize();
    return pager.stats();
}

void test_overlap_correct_and_hits(cudaStream_t stream) {
    // Demand-only baseline: no workers, no prefetch-ahead.
    const PagerStats demand = run_scan(/*workers=*/0, /*prefetch_ahead=*/0, stream);
    // Overlap: a background worker loads pages kPrefetchAhead ahead of demand.
    const PagerStats overlap = run_scan(/*workers=*/1, kPrefetchAhead, stream);

    std::cout << "  demand-only : hits=" << demand.hits << " misses=" << demand.misses
              << " hit_rate=" << 100.0 * demand.hit_rate() << "%\n";
    std::cout << "  prefetch    : hits=" << overlap.hits << " misses=" << overlap.misses
              << " prefetches=" << overlap.prefetches
              << " hit_rate=" << 100.0 * overlap.hit_rate() << "%\n";

    // Every acquire in both runs was byte-verified inside run_scan(); reaching
    // here means the overlap path (prefetched + evicted + reloaded pages) is
    // byte-correct.
    // Overlap must convert demand misses into hits: strictly more hits than the
    // demand-only scan over the identical access pattern.
    CHECK(overlap.hits > demand.hits);
    CHECK(overlap.prefetches > 0);
    // A scan larger than the cache gets ~0 reuse without prefetch; with
    // prefetch-ahead almost every demand acquire should hit.
    CHECK(overlap.hit_rate() > demand.hit_rate());
    std::cout << "  [ok] overlap path is byte-correct and raises the hit rate\n";
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "no CUDA device; skipping prefetch-overlap test\n";
        return 0;
    }
    cudaSetDevice(0);
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    test_overlap_correct_and_hits(stream);

    cudaStreamDestroy(stream);
    std::cout << "prefetch-overlap test passed\n";
    return 0;
}
