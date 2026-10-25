// GPU-level tests for CudaPager: residency, LRU eviction, lease safety,
// demand hit/miss accounting, and the async prefetch worker.
//
// Requires a CUDA device. Skips (exit 0) cleanly if none is present so the
// suite is harmless on CPU-only CI.

#include "vramer/cuda_pager.hpp"
#include "vramer/memory_source.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <unistd.h>

// CHECK() compiles out under NDEBUG (Release), which would silently disable
// every check. Use a macro that always runs and aborts on failure.
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

constexpr std::uint64_t kPage = 1ull << 20;   // 1 MiB pages keep the test light
constexpr std::uint64_t kSource = 8 * kPage;  // 8 pages of source
constexpr std::uint64_t kBudget = 4 * kPage;  // cache holds 4 pages

PagerConfig base_config(unsigned workers = 0) {
    PagerConfig cfg;
    cfg.gpu = 0;
    cfg.vram_budget = kBudget;
    cfg.page_size = kPage;
    cfg.staging_buffers = 2;
    cfg.prefetch_workers = workers;
    return cfg;
}

MemorySourcePtr make_source() {
    return std::make_shared<AnonymousRamSource>(kSource, /*touch=*/true);
}

// Touch a page on the compute stream so the lease lifecycle is exercised the
// way a real consumer would use it.
void touch(CudaPager& pager, std::uint64_t page, cudaStream_t stream) {
    auto lease = pager.acquire(page);
    lease.wait_on(stream);
    lease.release_on(stream);
}

void test_basic_hit_miss(cudaStream_t stream) {
    CudaPager pager(make_source(), base_config());
    CHECK(pager.page_count() == 8);
    CHECK(pager.resident_capacity_pages() == 4);

    touch(pager, 0, stream);   // miss
    touch(pager, 0, stream);   // hit (still resident)
    touch(pager, 1, stream);   // miss
    pager.synchronize();

    auto s = pager.stats();
    CHECK(s.misses == 2);
    CHECK(s.hits == 1);
    CHECK(s.evictions == 0);
    std::cout << "  [ok] basic hit/miss accounting\n";
}

void test_lru_eviction(cudaStream_t stream) {
    CudaPager pager(make_source(), base_config());
    // Fill the 4-slot cache, then force eviction with 4 more distinct pages.
    for (std::uint64_t p = 0; p < 8; ++p) touch(pager, p, stream);
    pager.synchronize();

    auto s = pager.stats();
    CHECK(s.misses == 8);
    CHECK(s.evictions == 4);   // 8 pages through a 4-slot cache

    // Page 7 was most recent; it must still be resident (a hit, no new eviction).
    touch(pager, 7, stream);
    pager.synchronize();
    auto s2 = pager.stats();
    CHECK(s2.hits == 1);
    CHECK(s2.evictions == 4);
    std::cout << "  [ok] LRU eviction keeps the most-recent page\n";
}

void test_lease_prevents_eviction(cudaStream_t stream) {
    CudaPager pager(make_source(), base_config());
    // Hold leases on the whole cache, then demand one more page. The victim
    // search must never evict a leased page; with every slot leased it must
    // throw rather than corrupt an in-use page.
    std::vector<PageLease> held;
    for (std::uint64_t p = 0; p < 4; ++p) {
        auto l = pager.acquire(p);
        l.wait_on(stream);
        held.push_back(std::move(l));
    }
    bool threw = false;
    try {
        auto l = pager.acquire(4);  // no free/unleased slot available
        l.wait_on(stream);
        held.push_back(std::move(l));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw && "acquiring past a fully-leased cache must throw, not evict");
    for (auto& l : held) l.release_on(stream);
    pager.synchronize();
    std::cout << "  [ok] live leases are never evicted\n";
}

void test_prefetch_worker(cudaStream_t stream) {
    CudaPager pager(make_source(), base_config(/*workers=*/1));
    // Prefetch 3 pages, let the worker drain, then demand them: all must hit.
    pager.prefetch_async(0);
    pager.prefetch_async(1);
    pager.prefetch_async(2);
    pager.drain_prefetch_queue();
    pager.synchronize();

    auto before = pager.stats();
    CHECK(before.prefetches == 3);
    CHECK(before.misses == 0);   // prefetch path must not count as demand miss

    for (std::uint64_t p = 0; p < 3; ++p) touch(pager, p, stream);
    pager.synchronize();

    auto after = pager.stats();
    CHECK(after.hits == 3);      // prefetched pages served every demand acquire
    CHECK(after.misses == 0);
    std::cout << "  [ok] async prefetch worker makes demand acquires hit\n";
}

// A pinned source must take the direct-DMA path (no staging copy) and still
// deliver correct bytes.
void test_pinned_direct(cudaStream_t stream) {
    auto src = std::make_shared<PinnedHostSource>(kSource, /*touch=*/true);
    CudaPager pager(src, base_config());
    const long syspage = ::sysconf(_SC_PAGESIZE) > 0 ? ::sysconf(_SC_PAGESIZE) : 4096;
    for (std::uint64_t p = 0; p < pager.page_count(); ++p) {
        auto lease = pager.acquire(p);
        lease.wait_on(stream);
        std::uint8_t got = 0xff;
        cudaMemcpyAsync(&got, lease.device_ptr(), 1, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        const std::uint8_t want =
            static_cast<std::uint8_t>(((p * kPage) / static_cast<std::uint64_t>(syspage)) & 0xffu);
        CHECK(got == want);
        lease.release_on(stream);
    }
    pager.synchronize();
    auto s = pager.stats();
    CHECK(s.direct_loads == s.misses);  // every miss used the direct DMA path
    CHECK(s.direct_loads > 0);
    std::cout << "  [ok] pinned source uses direct DMA (" << s.direct_loads
              << " loads, no staging copy), data correct\n";
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "no CUDA device; skipping pager tests\n";
        return 0;
    }
    cudaSetDevice(0);
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    test_basic_hit_miss(stream);
    test_lru_eviction(stream);
    test_lease_prevents_eviction(stream);
    test_prefetch_worker(stream);
    test_pinned_direct(stream);

    cudaStreamDestroy(stream);
    std::cout << "pager tests passed\n";
    return 0;
}
