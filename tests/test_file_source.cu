// Byte-exactness of the mmap file source paged through CudaPager.
//
// This is the small-file CI guard for the "larger than RAM" disk tier
// (benchmarks/larger_than_ram + docs/LARGER_THAN_RAM.md): the >RAM run can only
// sample one byte per 4 KiB via the bench kernel, so here -- where we can afford
// it -- we page a small file through a deliberately undersized VRAM budget and
// verify EVERY byte that lands on the device, including after eviction and
// reload. Uses the same deterministic pattern the benchmark file uses:
//
//     byte_at(file_offset) = (file_offset / 4096) & 0xff
//
// Requires a CUDA device; skips cleanly (exit 0) if none is present.

#include "vramer/cuda_pager.hpp"
#include "vramer/memory_source.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

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

constexpr std::uint64_t kBlock = 4096;
constexpr std::uint64_t kPage = 1ull << 20;    // 1 MiB pages
constexpr std::uint64_t kPages = 8;            // 8 MiB file
constexpr std::uint64_t kSize = kPages * kPage;
constexpr std::uint64_t kBudget = 3 * kPage;   // only 3 of 8 pages fit -> eviction

inline unsigned char pattern_at(std::uint64_t off) {
    return static_cast<unsigned char>((off / kBlock) & 0xffull);
}

std::string make_pattern_file() {
    char tmpl[] = "/tmp/vramer_file_src_XXXXXX";
    int fd = ::mkstemp(tmpl);
    CHECK(fd >= 0);
    std::vector<unsigned char> buf(kPage);
    for (std::uint64_t p = 0; p < kPages; ++p) {
        for (std::uint64_t i = 0; i < kPage; ++i) buf[i] = pattern_at(p * kPage + i);
        std::uint64_t off = 0;
        while (off < kPage) {
            ssize_t w = ::write(fd, buf.data() + off, kPage - off);
            CHECK(w > 0);
            off += static_cast<std::uint64_t>(w);
        }
    }
    CHECK(::fsync(fd) == 0);
    ::close(fd);
    return std::string(tmpl);
}

void verify_device_page(void* dptr, std::uint64_t page_id, std::size_t bytes) {
    std::vector<unsigned char> host(bytes);
    CHECK(cudaMemcpy(host.data(), dptr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::size_t i = 0; i < bytes; ++i)
        CHECK(host[i] == pattern_at(page_id * kPage + i));
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "file_source: no CUDA device, skipping\n";
        return 0;
    }
    CHECK(cudaSetDevice(0) == cudaSuccess);
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreate(&stream) == cudaSuccess);

    const std::string path = make_pattern_file();

    PagerConfig cfg;
    cfg.gpu = 0;
    cfg.vram_budget = kBudget;
    cfg.page_size = kPage;
    cfg.staging_buffers = 2;

    auto source = std::make_shared<MmapFileSource>(path);
    CHECK(source->size() == kSize);

    {
        CudaPager pager(source, cfg);
        CHECK(pager.page_count() == kPages);

        // Two full sequential scans: the second forces reload of evicted pages
        // (budget holds only 3 of 8), so we check bytes on cold load AND reload.
        for (int pass = 0; pass < 2; ++pass) {
            for (std::uint64_t p = 0; p < kPages; ++p) {
                auto lease = pager.acquire(p);
                lease.wait_on(stream);
                CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
                verify_device_page(lease.device_ptr(), p, lease.bytes());
                lease.release_on(stream);
            }
        }
        pager.synchronize();

        auto s = pager.stats();
        CHECK(s.evictions > 0);               // the budget really was exceeded
        CHECK(s.bytes_h2d >= kSize);          // at least one full file paged in
    }

    ::unlink(path.c_str());
    cudaStreamDestroy(stream);
    std::cout << "file source byte-correctness tests passed\n";
    return 0;
}
