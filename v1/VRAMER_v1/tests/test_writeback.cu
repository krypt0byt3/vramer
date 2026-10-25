// Write-back / dirty-page tests for CudaPager::acquire_mutable() + flush().
//
// Writes a distinct pattern into each page on the GPU, forces both
// eviction-triggered write-back (cache smaller than the working set) and an
// explicit flush() of the still-resident pages, then verifies every page's
// bytes persisted back into the WritableMemorySource. Skips cleanly with no GPU.

#include "vramer/cuda_pager.hpp"
#include "vramer/memory_source.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

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

constexpr std::uint64_t kPage = 1ull << 20;   // 1 MiB
constexpr std::uint64_t kPages = 8;
constexpr std::uint64_t kSource = kPages * kPage;
constexpr std::uint64_t kBudget = 4 * kPage;  // 4 slots -> forces eviction of dirty pages

__global__ void fill_page(std::uint8_t* d, std::size_t n, std::uint8_t v) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t grid = blockDim.x * gridDim.x;
    for (; i < n; i += grid) d[i] = v;
}

std::uint8_t pattern_for(std::uint64_t page) {
    return static_cast<std::uint8_t>((page * 37u + 101u) & 0xffu);  // distinct from source fill
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "no CUDA device; skipping write-back tests\n";
        return 0;
    }
    cudaSetDevice(0);
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // Keep our own handle to the source so we can read it back after flush.
    auto source = std::make_shared<AnonymousRamSource>(kSource, /*touch=*/true);

    PagerConfig cfg;
    cfg.gpu = 0;
    cfg.vram_budget = kBudget;
    cfg.page_size = kPage;
    cfg.staging_buffers = 2;
    {
        CudaPager pager(source, cfg);
        CHECK(pager.page_count() == kPages);

        // Write a distinct pattern into every page. Because the cache holds only
        // 4 of 8 pages, loading pages 4..7 evicts dirty pages 0..3 -> this
        // exercises eviction-triggered write-back. Pages 4..7 stay resident.
        for (std::uint64_t p = 0; p < kPages; ++p) {
            auto lease = pager.acquire_mutable(p);
            lease.wait_on(stream);
            fill_page<<<32, 256, 0, stream>>>(
                static_cast<std::uint8_t*>(lease.device_ptr()), lease.bytes(), pattern_for(p));
            CHECK(cudaGetLastError() == cudaSuccess);
            lease.release_on(stream);
        }
        cudaStreamSynchronize(stream);
        // Flush the pages still resident (the last ~4).
        pager.flush();

        auto s = pager.stats();
        CHECK(s.bytes_d2h >= kBudget);          // at least the resident set was written back
        CHECK(s.evictions >= kPages - kBudget / kPage);  // dirty evictions happened
        std::cout << "  [ok] acquire_mutable + eviction/flush wrote back "
                  << s.bytes_d2h << " bytes\n";
    }  // pager destroyed; source persists

    // Verify every page's bytes made it back into the source.
    std::vector<std::uint8_t> buf(kPage);
    for (std::uint64_t p = 0; p < kPages; ++p) {
        source->read(p * kPage, buf.data(), kPage);
        const std::uint8_t want = pattern_for(p);
        CHECK(buf[0] == want);
        CHECK(buf[kPage / 2] == want);
        CHECK(buf[kPage - 1] == want);
    }
    std::cout << "  [ok] all " << kPages << " pages persisted to the source\n";

    // A read-only source must reject acquire_mutable.
    {
        auto ro = std::make_shared<MmapFileSource>("/etc/hostname");  // small read-only file
        PagerConfig c2 = cfg;
        c2.page_size = 4096;
        c2.vram_budget = 4 * 4096;
        CudaPager pager(ro, c2);
        bool threw = false;
        try {
            auto l = pager.acquire_mutable(0);
            (void)l;
        } catch (const std::logic_error&) {
            threw = true;
        }
        CHECK(threw);
        std::cout << "  [ok] acquire_mutable rejected on a read-only source\n";
    }

    cudaStreamDestroy(stream);
    std::cout << "write-back tests passed\n";
    return 0;
}
