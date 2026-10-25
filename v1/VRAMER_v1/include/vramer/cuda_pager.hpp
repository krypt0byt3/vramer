#pragma once

#include "vramer/memory_source.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vramer {

struct PagerConfig {
    int gpu = 0;
    std::uint64_t vram_budget = 2ull << 30;   // 2 GiB
    std::uint64_t page_size = 64ull << 20;    // 64 MiB
    unsigned staging_buffers = 2;
};

struct PagerStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t bytes_h2d = 0;
};

class CudaPager;

class PageLease {
public:
    PageLease() = default;
    ~PageLease();

    PageLease(const PageLease&) = delete;
    PageLease& operator=(const PageLease&) = delete;
    PageLease(PageLease&& other) noexcept;
    PageLease& operator=(PageLease&& other) noexcept;

    void* device_ptr() const noexcept { return ptr_; }
    std::size_t bytes() const noexcept { return bytes_; }
    std::uint64_t page_id() const noexcept { return page_id_; }

    // Make subsequent work on compute_stream wait until this page is resident.
    void wait_on(cudaStream_t compute_stream);

    // Record completion of all work that uses this page. Required for
    // asynchronous operation. If omitted, the destructor falls back to a
    // conservative device synchronization for safety.
    void release_on(cudaStream_t compute_stream);

private:
    friend class CudaPager;
    PageLease(CudaPager* owner, std::size_t slot, std::uint64_t page_id,
              void* ptr, std::size_t bytes);

    CudaPager* owner_ = nullptr;
    std::size_t slot_ = 0;
    std::uint64_t page_id_ = 0;
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
    bool waited_ = false;
    bool released_ = true;
};

class CudaPager {
public:
    CudaPager(MemorySourcePtr source, PagerConfig config);
    ~CudaPager();

    CudaPager(const CudaPager&) = delete;
    CudaPager& operator=(const CudaPager&) = delete;

    PageLease acquire(std::uint64_t page_id);
    void prefetch(std::uint64_t page_id);
    void synchronize();

    std::uint64_t page_count() const noexcept;
    std::uint64_t resident_capacity_pages() const noexcept { return slots_.size(); }
    std::uint64_t page_size() const noexcept { return config_.page_size; }
    std::uint64_t vram_budget() const noexcept { return config_.vram_budget; }
    PagerStats stats() const;

private:
    friend class PageLease;

    struct DeviceSlot {
        std::uint64_t page_id = UINT64_MAX;
        std::size_t valid_bytes = 0;
        std::uint64_t last_touch = 0;
        unsigned leases = 0;
        cudaEvent_t load_done = nullptr;
        std::vector<cudaEvent_t> pending_uses;
    };

    struct StagingSlot {
        void* host = nullptr;
        cudaEvent_t h2d_done = nullptr;
        bool in_flight = false;
    };

    void wait_slot(std::size_t slot, cudaStream_t stream);
    void release_slot_async(std::size_t slot, cudaStream_t stream);
    void release_slot_sync(std::size_t slot);
    void reap_completed_events(DeviceSlot& slot, bool block);
    bool reusable(DeviceSlot& slot);
    std::size_t choose_victim();
    StagingSlot& next_staging();

    MemorySourcePtr source_;
    PagerConfig config_;
    void* device_cache_ = nullptr;
    cudaStream_t transfer_stream_ = nullptr;
    std::vector<DeviceSlot> slots_;
    std::vector<StagingSlot> staging_;
    std::unordered_map<std::uint64_t, std::size_t> resident_;
    std::uint64_t clock_ = 0;
    std::size_t staging_cursor_ = 0;
    PagerStats stats_{};
    mutable std::mutex mutex_;
};

} // namespace vramer
