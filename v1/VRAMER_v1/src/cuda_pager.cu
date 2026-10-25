#include "vramer/common.hpp"
#include "vramer/cuda_common.hpp"
#include "vramer/cuda_pager.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vramer {

PageLease::PageLease(CudaPager* owner, std::size_t slot, std::uint64_t page_id,
                     void* ptr, std::size_t bytes)
    : owner_(owner), slot_(slot), page_id_(page_id), ptr_(ptr), bytes_(bytes), released_(false) {}

PageLease::~PageLease() {
    if (owner_ && !released_) {
        try { owner_->release_slot_sync(slot_); } catch (...) { /* never throw from destructor */ }
    }
}

PageLease::PageLease(PageLease&& other) noexcept { *this = std::move(other); }

PageLease& PageLease::operator=(PageLease&& other) noexcept {
    if (this == &other) return *this;
    if (owner_ && !released_) {
        try { owner_->release_slot_sync(slot_); } catch (...) {}
    }
    owner_ = other.owner_;
    slot_ = other.slot_;
    page_id_ = other.page_id_;
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    waited_ = other.waited_;
    released_ = other.released_;
    other.owner_ = nullptr;
    other.released_ = true;
    return *this;
}

void PageLease::wait_on(cudaStream_t compute_stream) {
    if (!owner_ || released_) throw std::logic_error("wait_on called on released PageLease");
    owner_->wait_slot(slot_, compute_stream);
    waited_ = true;
}

void PageLease::release_on(cudaStream_t compute_stream) {
    if (!owner_ || released_) return;
    owner_->release_slot_async(slot_, compute_stream);
    released_ = true;
    owner_ = nullptr;
}

CudaPager::CudaPager(MemorySourcePtr source, PagerConfig config)
    : source_(std::move(source)), config_(config) {
    if (!source_) throw std::invalid_argument("CudaPager requires a MemorySource");
    if (config_.page_size == 0 || config_.vram_budget < config_.page_size)
        throw std::invalid_argument("invalid page size / VRAM budget");
    if (config_.staging_buffers == 0)
        throw std::invalid_argument("staging_buffers must be >= 1");

    VRAMER_CUDA(cudaSetDevice(config_.gpu));
    VRAMER_CUDA(cudaStreamCreateWithFlags(&transfer_stream_, cudaStreamNonBlocking));

    const std::uint64_t slot_count = config_.vram_budget / config_.page_size;
    config_.vram_budget = slot_count * config_.page_size;
    VRAMER_CUDA(cudaMalloc(&device_cache_, config_.vram_budget));

    slots_.resize(static_cast<std::size_t>(slot_count));
    for (auto& slot : slots_) {
        VRAMER_CUDA(cudaEventCreateWithFlags(&slot.load_done, cudaEventDisableTiming));
        // Mark an initial event complete so a never-loaded slot has sane state.
        VRAMER_CUDA(cudaEventRecord(slot.load_done, transfer_stream_));
    }

    staging_.resize(config_.staging_buffers);
    for (auto& stage : staging_) {
        // A small pinned ring is intentional. The backing MemorySource itself
        // remains pageable/swappable and memory-technology agnostic.
        VRAMER_CUDA(cudaHostAlloc(&stage.host, config_.page_size, cudaHostAllocPortable));
        VRAMER_CUDA(cudaEventCreateWithFlags(&stage.h2d_done, cudaEventDisableTiming));
    }
    VRAMER_CUDA(cudaStreamSynchronize(transfer_stream_));
}

CudaPager::~CudaPager() {
    try {
        if (config_.gpu >= 0) cudaSetDevice(config_.gpu);
        if (transfer_stream_) cudaStreamSynchronize(transfer_stream_);
        for (auto& slot : slots_) {
            for (auto ev : slot.pending_uses) cudaEventDestroy(ev);
            if (slot.load_done) cudaEventDestroy(slot.load_done);
        }
        for (auto& stage : staging_) {
            if (stage.in_flight && stage.h2d_done) cudaEventSynchronize(stage.h2d_done);
            if (stage.h2d_done) cudaEventDestroy(stage.h2d_done);
            if (stage.host) cudaFreeHost(stage.host);
        }
        if (device_cache_) cudaFree(device_cache_);
        if (transfer_stream_) cudaStreamDestroy(transfer_stream_);
    } catch (...) {}
}

std::uint64_t CudaPager::page_count() const noexcept {
    return (source_->size() + config_.page_size - 1) / config_.page_size;
}

PagerStats CudaPager::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void CudaPager::reap_completed_events(DeviceSlot& slot, bool block) {
    auto it = slot.pending_uses.begin();
    while (it != slot.pending_uses.end()) {
        cudaError_t rc = block ? cudaEventSynchronize(*it) : cudaEventQuery(*it);
        if (rc == cudaSuccess) {
            cudaEventDestroy(*it);
            it = slot.pending_uses.erase(it);
        } else if (!block && rc == cudaErrorNotReady) {
            // Clear CUDA's thread-local last-error state produced by query.
            (void)cudaGetLastError();
            ++it;
        } else {
            VRAMER_CUDA(rc);
        }
    }
}

bool CudaPager::reusable(DeviceSlot& slot) {
    if (slot.leases != 0) return false;
    reap_completed_events(slot, false);
    return slot.pending_uses.empty();
}

std::size_t CudaPager::choose_victim() {
    // Free slot first.
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].page_id == UINT64_MAX && reusable(slots_[i])) return i;
    }

    // LRU among immediately reusable slots.
    std::size_t best = slots_.size();
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (reusable(slots_[i]) && slots_[i].last_touch < oldest) {
            oldest = slots_[i].last_touch;
            best = i;
        }
    }
    if (best != slots_.size()) return best;

    // All candidates are still in flight. Block on the oldest unleased slot,
    // but never evict a page that still has a live lease.
    oldest = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].leases == 0 && slots_[i].last_touch < oldest) {
            oldest = slots_[i].last_touch;
            best = i;
        }
    }
    if (best == slots_.size())
        throw std::runtime_error("VRAMER cache exhausted: every device slot is leased");
    reap_completed_events(slots_[best], true);
    return best;
}

CudaPager::StagingSlot& CudaPager::next_staging() {
    auto& stage = staging_[staging_cursor_++ % staging_.size()];
    if (stage.in_flight) {
        VRAMER_CUDA(cudaEventSynchronize(stage.h2d_done));
        stage.in_flight = false;
    }
    return stage;
}

PageLease CudaPager::acquire(std::uint64_t page_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    VRAMER_CUDA(cudaSetDevice(config_.gpu));
    if (page_id >= page_count()) throw std::out_of_range("page_id exceeds MemorySource");

    auto found = resident_.find(page_id);
    if (found != resident_.end()) {
        auto& slot = slots_[found->second];
        ++slot.leases;
        slot.last_touch = ++clock_;
        ++stats_.hits;
        auto* ptr = static_cast<std::byte*>(device_cache_) + found->second * config_.page_size;
        return PageLease(this, found->second, page_id, ptr, slot.valid_bytes);
    }

    ++stats_.misses;
    const std::size_t index = choose_victim();
    auto& slot = slots_[index];
    if (slot.page_id != UINT64_MAX) {
        resident_.erase(slot.page_id);
        ++stats_.evictions;
    }

    auto& stage = next_staging();
    const std::uint64_t offset = page_id * config_.page_size;
    const std::size_t valid = static_cast<std::size_t>(
        std::min<std::uint64_t>(config_.page_size, source_->size() - offset));

    // CPU/source -> small pinned ring. This preserves asynchronous H2D without
    // pinning the entire host-side model and without assuming any RAM type.
    source_->read(offset, stage.host, valid);

    auto* device_ptr = static_cast<std::byte*>(device_cache_) + index * config_.page_size;
    VRAMER_CUDA(cudaMemcpyAsync(device_ptr, stage.host, valid,
                                cudaMemcpyHostToDevice, transfer_stream_));
    VRAMER_CUDA(cudaEventRecord(slot.load_done, transfer_stream_));
    VRAMER_CUDA(cudaEventRecord(stage.h2d_done, transfer_stream_));
    stage.in_flight = true;

    slot.page_id = page_id;
    slot.valid_bytes = valid;
    slot.last_touch = ++clock_;
    slot.leases = 1;
    resident_[page_id] = index;
    stats_.bytes_h2d += valid;

    return PageLease(this, index, page_id, device_ptr, valid);
}

void CudaPager::prefetch(std::uint64_t page_id) {
    auto lease = acquire(page_id);
    // Release after the transfer stream reaches the page load. This keeps the
    // page resident but unleased, ready for a later cache hit.
    lease.wait_on(transfer_stream_);
    lease.release_on(transfer_stream_);
}

void CudaPager::wait_slot(std::size_t slot_index, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    VRAMER_CUDA(cudaSetDevice(config_.gpu));
    VRAMER_CUDA(cudaStreamWaitEvent(stream, slots_.at(slot_index).load_done, 0));
}

void CudaPager::release_slot_async(std::size_t slot_index, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    VRAMER_CUDA(cudaSetDevice(config_.gpu));
    auto& slot = slots_.at(slot_index);
    if (slot.leases == 0) throw std::logic_error("PageLease double release");

    cudaEvent_t done = nullptr;
    VRAMER_CUDA(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    VRAMER_CUDA(cudaEventRecord(done, stream));
    slot.pending_uses.push_back(done);
    --slot.leases;
}

void CudaPager::release_slot_sync(std::size_t slot_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    VRAMER_CUDA(cudaSetDevice(config_.gpu));
    auto& slot = slots_.at(slot_index);
    if (slot.leases == 0) return;
    // Safety fallback for callers that forget release_on(). The fast path is
    // always release_on(compute_stream), which never synchronizes the device.
    VRAMER_CUDA(cudaDeviceSynchronize());
    --slot.leases;
    reap_completed_events(slot, true);
}

void CudaPager::synchronize() {
    std::lock_guard<std::mutex> lock(mutex_);
    VRAMER_CUDA(cudaSetDevice(config_.gpu));
    VRAMER_CUDA(cudaStreamSynchronize(transfer_stream_));
    for (auto& slot : slots_) reap_completed_events(slot, true);
}

} // namespace vramer
