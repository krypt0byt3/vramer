// Multithreaded data-integrity stress test for CudaPager.
//
// Hammers a heavily oversubscribed cache (64 source pages through 8 slots)
// from many threads at once and verifies, on every acquire, that the resident
// page's first byte matches the deterministic content invariant of
// AnonymousRamSource. Any race -- a wrong page in a slot, a premature
// eviction, or staging-buffer reuse corruption -- surfaces as a byte mismatch.
//
// Requires a CUDA device. Skips (exit 0) cleanly if none is present so the
// suite is harmless on CPU-only CI.

#include "vramer/cuda_pager.hpp"
#include "vramer/memory_source.hpp"

#include <cuda_runtime.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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

constexpr std::uint64_t kPage = 1ull << 20;    // 1 MiB pages
constexpr std::uint64_t kPages = 64;           // 64 pages of source (64 MiB)
constexpr std::uint64_t kSource = kPages * kPage;
constexpr std::uint64_t kBudget = 8 * kPage;   // 8 slots -> 8x oversubscribed
constexpr unsigned kStaging = 4;

constexpr int kThreads = 8;
constexpr int kIters = 2000;

// Total transient std::runtime_error("cache exhausted") retries observed,
// aggregated across all threads and both scenarios.
std::atomic<std::uint64_t> g_retries{0};

PagerConfig make_config(unsigned prefetch_workers) {
    PagerConfig cfg;
    cfg.gpu = 0;
    cfg.vram_budget = kBudget;
    cfg.page_size = kPage;
    cfg.staging_buffers = kStaging;
    cfg.prefetch_workers = prefetch_workers;
    return cfg;
}

MemorySourcePtr make_source() {
    return std::make_shared<AnonymousRamSource>(kSource, /*touch=*/true);
}

// The deterministic content anchor: the first byte of VRAMER page p lives at
// host offset p * page_size, which (page_size being a multiple of the system
// page size) is a multiple of syspage, so AnonymousRamSource fills it with
// (offset / syspage) & 0xff.
unsigned char expected_first_byte(std::uint64_t page, std::uint64_t syspage) {
    const std::uint64_t offset = page * kPage;
    return static_cast<unsigned char>((offset / syspage) & 0xffull);
}

// One worker: repeatedly acquire a random page, copy its first byte D2H on an
// owned stream, and verify it against the invariant. Optionally prefetches the
// next page first. Returns the number of successful (verified) iterations.
std::uint64_t worker(CudaPager& pager, unsigned seed, std::uint64_t syspage,
                     bool prefetch) {
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
          cudaSuccess);

    const std::uint64_t pages = pager.page_count();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::uint64_t> pick(0, pages - 1);

    std::uint64_t ok = 0;
    for (int i = 0; i < kIters; ++i) {
        const std::uint64_t page = pick(rng);

        if (prefetch) {
            pager.prefetch_async((page + 1) % pages);
        }

        // acquire may throw the specific "cache exhausted" runtime_error when
        // every slot is momentarily leased by peers. Treat that as transient:
        // yield and retry the same page. Anything else is a real failure.
        PageLease lease;
        for (;;) {
            try {
                lease = pager.acquire(page);
                break;
            } catch (const std::runtime_error& e) {
                if (std::string(e.what()).find("cache exhausted") !=
                    std::string::npos) {
                    g_retries.fetch_add(1, std::memory_order_relaxed);
                    std::this_thread::yield();
                    continue;
                }
                CHECK(false && "unexpected runtime_error from acquire");
            } catch (...) {
                CHECK(false && "unexpected exception from acquire");
            }
        }

        CHECK(lease.page_id() == page);
        CHECK(lease.bytes() == kPage);
        lease.wait_on(stream);

        unsigned char got = 0xff;
        CHECK(cudaMemcpyAsync(&got, lease.device_ptr(), 1,
                              cudaMemcpyDeviceToHost, stream) == cudaSuccess);
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

        const unsigned char want = expected_first_byte(page, syspage);
        if (got != want) {
            std::cerr << "page " << page << " first byte " << (int)got
                      << " != expected " << (int)want << '\n';
            CHECK(false && "resident page content mismatch (data race)");
        }

        lease.release_on(stream);
        ++ok;
    }

    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    return ok;
}

void run_scenario(const char* label, unsigned prefetch_workers, bool prefetch,
                  std::uint64_t syspage) {
    CudaPager pager(make_source(), make_config(prefetch_workers));
    CHECK(pager.page_count() == kPages);
    CHECK(pager.resident_capacity_pages() == 8);

    std::vector<std::thread> threads;
    std::vector<std::uint64_t> counts(kThreads, 0);
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            counts[t] = worker(pager, 0x1234u + 97u * static_cast<unsigned>(t),
                               syspage, prefetch);
        });
    }
    for (auto& th : threads) th.join();

    pager.synchronize();

    std::uint64_t total = 0;
    for (auto c : counts) total += c;

    auto s = pager.stats();
    // Every successful iteration went through acquire and counted as a hit or a
    // miss, so the demand counters must cover at least that many acquires.
    CHECK(s.hits + s.misses >= total);

    std::cout << "  [ok] " << label << ": " << total
              << " verified acquires across " << kThreads << " threads"
              << " (hits=" << s.hits << " misses=" << s.misses
              << " evictions=" << s.evictions << ")\n";
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "no CUDA device; skipping pager concurrency tests\n";
        return 0;
    }
    cudaSetDevice(0);

    const long sp = ::sysconf(_SC_PAGESIZE);
    const std::uint64_t syspage = (sp > 0) ? static_cast<std::uint64_t>(sp)
                                           : 4096ull;

    run_scenario("concurrent demand (no prefetch workers)",
                 /*prefetch_workers=*/0, /*prefetch=*/false, syspage);
    run_scenario("concurrent demand + prefetch workers",
                 /*prefetch_workers=*/2, /*prefetch=*/true, syspage);

    std::cout << "transient cache-exhaustion retries: "
              << g_retries.load() << '\n';
    std::cout << "pager concurrency tests passed\n";
    return 0;
}
