#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace vramer {

// VRAMER deliberately models backing storage as bytes, not as "DDR4".
// Any host-accessible tier can implement MemorySource: DDR3/4/5, LPDDR,
// CXL memory exposed by the OS, PMEM, mmap'd files, network-backed stores, etc.
class MemorySource {
public:
    virtual ~MemorySource() = default;
    virtual std::uint64_t size() const noexcept = 0;
    virtual std::string name() const = 0;
    virtual void read(std::uint64_t offset, void* dst, std::size_t bytes) = 0;

    // Fast-path hook: if the source's bytes live in page-locked (pinned) host
    // memory, return a pointer the pager can hand straight to cudaMemcpyAsync,
    // skipping the CPU source->staging copy entirely. Default: not directly
    // DMA-able (the pager stages through its pinned ring). Measured telemetry
    // showed that staging copy is the dominant per-page cost on pageable RAM, so
    // a pinned direct source removes the bottleneck and approaches the raw PCIe
    // ceiling. `bytes` lets the source validate the span.
    virtual const void* pinned_host_ptr(std::uint64_t /*offset*/,
                                        std::size_t /*bytes*/) const noexcept {
        return nullptr;
    }
};

// A source whose pages can be written back from the GPU. Required for paging
// *mutable* tensor data (e.g. training state): the pager flushes dirty device
// pages here on eviction or flush(). Read-only sources (mmap MAP_PRIVATE) do
// not implement this, and the pager rejects acquire_mutable() against them.
class WritableMemorySource : public MemorySource {
public:
    virtual void write(std::uint64_t offset, const void* src, std::size_t bytes) = 0;
};

class MmapFileSource final : public MemorySource {
public:
    explicit MmapFileSource(std::string path);
    ~MmapFileSource() override;

    MmapFileSource(const MmapFileSource&) = delete;
    MmapFileSource& operator=(const MmapFileSource&) = delete;

    std::uint64_t size() const noexcept override { return size_; }
    std::string name() const override;
    void read(std::uint64_t offset, void* dst, std::size_t bytes) override;

private:
    std::string path_;
    int fd_ = -1;
    std::uint64_t size_ = 0;
    const std::byte* mapping_ = nullptr;
};

class AnonymousRamSource final : public WritableMemorySource {
public:
    // numa_node >= 0 binds the allocation to that NUMA node (MPOL_BIND) before
    // first-touch, so the backing pages land on the memory closest to the GPU.
    // -1 leaves placement to the kernel's default policy. On non-NUMA systems
    // (numa_node reported as -1 by `detect`) this is a no-op.
    explicit AnonymousRamSource(std::uint64_t bytes, bool touch = true, int numa_node = -1);
    ~AnonymousRamSource() override;

    AnonymousRamSource(const AnonymousRamSource&) = delete;
    AnonymousRamSource& operator=(const AnonymousRamSource&) = delete;

    std::uint64_t size() const noexcept override { return size_; }
    std::string name() const override;
    void read(std::uint64_t offset, void* dst, std::size_t bytes) override;
    void write(std::uint64_t offset, const void* src, std::size_t bytes) override;

private:
    std::uint64_t size_ = 0;
    std::byte* mapping_ = nullptr;
};

// Page-locked (pinned) host memory source. Its bytes can be DMA'd to the GPU
// directly, so the pager skips its staging copy and hits the raw PCIe ceiling.
// Trade-off: pinning is a scarce resource -- NVIDIA recommends against pinning
// an entire large model. Use this to measure the ceiling or for a hot subset;
// for the general case keep AnonymousRamSource (pageable) + the staging ring.
class PinnedHostSource final : public WritableMemorySource {
public:
    explicit PinnedHostSource(std::uint64_t bytes, bool touch = true);
    ~PinnedHostSource() override;

    PinnedHostSource(const PinnedHostSource&) = delete;
    PinnedHostSource& operator=(const PinnedHostSource&) = delete;

    std::uint64_t size() const noexcept override { return size_; }
    std::string name() const override { return "pinned-host-memory"; }
    void read(std::uint64_t offset, void* dst, std::size_t bytes) override;
    void write(std::uint64_t offset, const void* src, std::size_t bytes) override;
    const void* pinned_host_ptr(std::uint64_t offset, std::size_t bytes) const noexcept override;

private:
    std::uint64_t size_ = 0;
    std::byte* mapping_ = nullptr;  // cudaHostAlloc'd
};

// Adapter for memory owned by another runtime. This is the hook for future
// framework integrations and for OS-exposed memory tiers such as CXL. Set
// `pinned` when the borrowed region is page-locked (e.g. a pinned PyTorch
// tensor) so the pager DMAs from it directly.
class BorrowedMemorySource final : public MemorySource {
public:
    BorrowedMemorySource(const void* base, std::uint64_t bytes, std::string label,
                         bool pinned = false)
        : base_(static_cast<const std::byte*>(base)), size_(bytes),
          label_(std::move(label)), pinned_(pinned) {}

    std::uint64_t size() const noexcept override { return size_; }
    std::string name() const override { return label_; }
    void read(std::uint64_t offset, void* dst, std::size_t bytes) override;
    const void* pinned_host_ptr(std::uint64_t offset, std::size_t bytes) const noexcept override {
        if (!pinned_ || offset > size_ || bytes > size_ - offset) return nullptr;
        return base_ + offset;
    }

private:
    const std::byte* base_ = nullptr;
    std::uint64_t size_ = 0;
    std::string label_;
    bool pinned_ = false;
};

using MemorySourcePtr = std::shared_ptr<MemorySource>;

} // namespace vramer
