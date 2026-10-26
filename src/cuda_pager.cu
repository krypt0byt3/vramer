#include "vramer/common.hpp"
#include "vramer/cuda_common.hpp"
#include "vramer/cuda_pager.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace vramer {

namespace {
inline std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}
} // namespace

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
    // Enables acquire_mutable()/flush() write-back when the source supports it.
    writable_source_ = dynamic_cast<WritableMemorySource*>(source_.get());
    if (config_.page_size == 0 || config_.vram_budget < config_.page_size)
        throw std::invalid_argument("invalid page size / VRAM budget");
    if (config_.staging_buffers == 0)
        throw std::invalid_argument("staging_buffers must be >= 1");

    VRAMER_CUDA(cudaSetDevice(config_.gpu));
    const unsigned stream_count = std::max(1u, config_.transfer_streams);
    config_.transfer_streams = stream_count;
    transfer_streams_.resize(stream_count, nullptr);
    for (auto& s : transfer_streams_)
        VRAMER_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    const std::uint64_t slot_count = config_.vram_budget / config_.page_size;
    config_.vram_budget = slot_count * config_.page_size;
    VRAMER_CUDA(cudaMalloc(&device_cache_, config_.vram_budget));

    slots_.resize(static_cast<std::size_t>(slot_count));
    for (auto& slot : slots_) {
        VRAMER_CUDA(cudaEventCreateWithFlags(&slot.load_done, cudaEventDisableTiming));
        // Mark an initial event complete so a never-loaded slot has sane state.
        VRAMER_CUDA(cudaEventRecord(slot.load_done, transfer_streams_[0]));
    }

    staging_.resize(config_.staging_buffers);
    for (auto& stage : staging_) {
        // A small pinned ring is intentional. The backing MemorySource itself
        // remains pageable/swappable and memory-technology agnostic.
        VRAMER_CUDA(cudaHostAlloc(&stage.host, config_.page_size, cudaHostAllocPortable));
        VRAMER_CUDA(cudaEventCreateWithFlags(&stage.h2d_done, cudaEventDisableTiming));
    }
    VRAMER_CUDA(cudaStreamSynchronize(transfer_streams_[0]));

    // Launch background prefetch workers last, once all CUDA state is live.
    for (unsigned i = 0; i < config_.prefetch_workers; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

CudaPager::~CudaPager() {
    // Stop workers before tearing down any CUDA state they might touch.
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_workers_ = true;
    }
    queue_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }

    try {
        if (config_.gpu >= 0) cudaSetDevice(config_.gpu);
        for (auto s : transfer_streams_) if (s) cudaStreamSynchronize(s);
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
        for (auto s : transfer_streams_) if (s) cudaStreamDestroy(s);
    } catch (...) {}
}

std::uint64_t CudaPager::page_count() const noexcept {
    return (source_->size() + config_.page_size - 1) / config_.page_size;
}

PagerStats CudaPager::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

cudaStream_t CudaPager::next_transfer_stream() noexcept {
    if (transfer_streams_.size() == 1) return transfer_streams_[0];
    const std::size_t i = stream_cursor_.fetch_add(1, std::memory_order_relaxed);
    return transfer_streams_[i % transfer_streams_.size()];
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
    // Under MRU we prefer to evict the most-recently-used page; under LRU the
    // least-recently-used. `better` says whether candidate `a` should beat the
    // current best `b` for eviction given the policy.
    const bool mru = (config_.eviction == EvictionPolicy::Mru);
    auto better = [mru](std::uint64_t a, std::uint64_t b) {
        return mru ? (a > b) : (a < b);
    };

    // Free slot first, regardless of policy.
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].page_id == UINT64_MAX && reusable(slots_[i])) return i;
    }

    // Best eviction candidate among immediately reusable slots.
    std::size_t best = slots_.size();
    std::uint64_t best_touch = 0;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (reusable(slots_[i]) && (best == slots_.size() || better(slots_[i].last_touch, best_touch))) {
            best_touch = slots_[i].last_touch;
            best = i;
        }
    }
    if (best != slots_.size()) return best;

    // All candidates are still in flight. Block on the best-by-policy unleased
    // slot, but never evict a page that still has a live lease.
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].leases == 0 && (best == slots_.size() || better(slots_[i].last_touch, best_touch))) {
            best_touch = slots_[i].last_touch;
            best = i;
        }
    }
    if (best == slots_.size())
        throw std::runtime_error("VRAMER cache exhausted: every device slot is leased");
    const std::uint64_t t0 = now_ns();
    reap_completed_events(slots_[best], true);
    stats_.stall_ns += now_ns() - t0;
    return best;
}

std::size_t CudaPager::reserve_staging(std::unique_lock<std::mutex>& lock) {
    // Find an unreserved staging buffer. If every buffer is mid-load (more
    // concurrent loaders than buffers), wait for one to be released. Set
    // staging_buffers >= prefetch_workers + 1 to avoid this stall.
    std::size_t idx = staging_.size();
    for (;;) {
        for (std::size_t n = 0; n < staging_.size(); ++n) {
            const std::size_t i = (staging_cursor_ + n) % staging_.size();
            if (!staging_[i].reserved) { idx = i; break; }
        }
        if (idx != staging_.size()) break;
        const std::uint64_t t0 = now_ns();
        staging_cv_.wait(lock, [this] {
            for (auto& s : staging_) if (!s.reserved) return true;
            return false;
        });
        stats_.stall_ns += now_ns() - t0;
    }
    staging_cursor_ = (idx + 1) % staging_.size();
    auto& stage = staging_[idx];
    if (stage.in_flight) {
        // The buffer's previous H2D must finish before we overwrite the pinned
        // bytes. Rare when there are enough buffers.
        const std::uint64_t t0 = now_ns();
        VRAMER_CUDA(cudaEventSynchronize(stage.h2d_done));
        stats_.stall_ns += now_ns() - t0;
        stage.in_flight = false;
    }
    stage.reserved = true;
    return idx;
}

PageLease CudaPager::acquire(std::uint64_t page_id) {
    return acquire_impl(page_id, /*is_prefetch=*/false, /*mutate=*/false);
}

PageLease CudaPager::acquire_mutable(std::uint64_t page_id) {
    if (!writable_source_)
        throw std::logic_error("acquire_mutable requires a WritableMemorySource");
    return acquire_impl(page_id, /*is_prefetch=*/false, /*mutate=*/true);
}

PageLease CudaPager::acquire_impl(std::uint64_t page_id, bool is_prefetch, bool mutate) {
    std::unique_lock<std::mutex> lock(mutex_);
    VRAMER_CUDA(cudaSetDevice(config_.gpu));
    if (page_id >= page_count()) throw std::out_of_range("page_id exceeds MemorySource");

    for (;;) {
        auto found = resident_.find(page_id);
        if (found != resident_.end()) {
            const std::size_t idx = found->second;
            if (slots_[idx].loading) {
                // Coalesce: another caller is already loading this exact page.
                // Wait until its H2D has been issued (load_done recorded) rather
                // than starting a duplicate load.
                load_cv_.wait(lock, [this, idx, page_id] {
                    return !slots_[idx].loading || slots_[idx].page_id != page_id;
                });
                if (slots_[idx].page_id != page_id) continue;  // rare: slot repurposed
            }
            auto& slot = slots_[idx];
            ++slot.leases;
            slot.last_touch = ++clock_;
            if (mutate) slot.dirty = true;
            // A prefetch that lands on an already-resident page is not a demand hit.
            if (!is_prefetch) ++stats_.hits;
            auto* ptr = static_cast<std::byte*>(device_cache_) + idx * config_.page_size;
            return PageLease(this, idx, page_id, ptr, slot.valid_bytes);
        }

        // Miss. choose_victim() may throw if every slot is leased; it runs
        // before any state mutation so a throw leaves the pager unchanged.
        const std::size_t index = choose_victim();
        // Reserve the victim slot immediately (leases=1) so no concurrent
        // victim search or same-page acquire can touch it while we evict/flush.
        slots_[index].leases = 1;
        const bool victim_dirty = slots_[index].dirty;
        const std::uint64_t old_page = slots_[index].page_id;
        const std::size_t old_valid = slots_[index].valid_bytes;
        if (old_page != UINT64_MAX) {
            resident_.erase(old_page);   // remove first: no one acquires the old page now
            ++stats_.evictions;
        }
        // Write the evicted page back if it holds unsaved GPU writes. The device
        // still holds the old bytes (the new H2D happens further below). This
        // releases the lock during the D2H + CPU copy.
        if (victim_dirty) flush_slot_data(index, old_page, old_valid, lock);
        slots_[index].dirty = false;

        const std::uint64_t offset = page_id * config_.page_size;
        const std::size_t valid = static_cast<std::size_t>(
            std::min<std::uint64_t>(config_.page_size, source_->size() - offset));
        auto* device_ptr = static_cast<std::byte*>(device_cache_) + index * config_.page_size;

        // Publish the slot for the new page BEFORE dropping the lock. leases is
        // already 1 (reserved above); loading=true makes concurrent acquires of
        // this page coalesce above instead of launching a duplicate load.
        {
            auto& slot = slots_[index];
            slot.page_id = page_id;
            slot.valid_bytes = valid;
            slot.last_touch = ++clock_;
            slot.loading = true;
            slot.dirty = mutate;
        }
        resident_[page_id] = index;
        if (is_prefetch) ++stats_.prefetches; else ++stats_.misses;

        const cudaEvent_t load_done = slots_[index].load_done;

        // Fast path: if the source exposes a pinned host pointer, DMA straight
        // from it -- no staging buffer, no CPU source->pinned copy (the measured
        // bottleneck), approaching the raw PCIe ceiling.
        const void* direct = source_->pinned_host_ptr(offset, valid);
        if (direct) {
            // Route this load onto a stream from the pool so independent direct
            // loads overlap on the wire. load_done records on the SAME stream.
            cudaStream_t stream = next_transfer_stream();
            lock.unlock();
            VRAMER_CUDA(cudaMemcpyAsync(device_ptr, direct, valid,
                                        cudaMemcpyHostToDevice, stream));
            VRAMER_CUDA(cudaEventRecord(load_done, stream));
            lock.lock();
            slots_[index].loading = false;
            stats_.bytes_h2d += valid;
            ++stats_.direct_loads;
            load_cv_.notify_all();
            return PageLease(this, index, page_id, device_ptr, valid);
        }

        // Staging path: pageable source -> pinned ring -> async H2D.
        const std::size_t stage_idx = reserve_staging(lock);
        void* host = staging_[stage_idx].host;
        const cudaEvent_t h2d_done = staging_[stage_idx].h2d_done;
        // One stream for this load's H2D; both load_done and this staging slot's
        // h2d_done record on it, so reserve_staging's h2d_done wait and the
        // compute stream's load_done wait both observe the right copy.
        cudaStream_t stream = next_transfer_stream();

        // ---- unlocked region: the expensive source->pinned copy and the async
        // H2D run here so they overlap other threads' compute and transfers.
        // Only stable handles (device_ptr, host, events, stream) are touched;
        // the slot and staging buffer are reserved, so no other thread writes
        // them. CUDA runtime calls are themselves thread-safe.
        lock.unlock();
        const std::uint64_t t_read = now_ns();
        source_->read(offset, host, valid);
        const std::uint64_t read_ns = now_ns() - t_read;
        VRAMER_CUDA(cudaMemcpyAsync(device_ptr, host, valid,
                                    cudaMemcpyHostToDevice, stream));
        VRAMER_CUDA(cudaEventRecord(load_done, stream));
        VRAMER_CUDA(cudaEventRecord(h2d_done, stream));
        lock.lock();
        // ---- end unlocked region ----

        staging_[stage_idx].reserved = false;
        staging_[stage_idx].in_flight = true;
        slots_[index].loading = false;
        stats_.source_read_ns += read_ns;
        stats_.bytes_h2d += valid;
        staging_cv_.notify_all();  // a staging buffer is free again
        load_cv_.notify_all();     // coalesced waiters may proceed

        return PageLease(this, index, page_id, device_ptr, valid);
    }
}

void CudaPager::flush_slot_data(std::size_t index, std::uint64_t page_id,
                                std::size_t valid, std::unique_lock<std::mutex>& lock) {
    if (!writable_source_) throw std::logic_error("flush on a non-writable source");
    // All compute that read/wrote this page must finish before we read it back.
    reap_completed_events(slots_[index], true);
    const std::size_t stage_idx = reserve_staging(lock);
    void* host = staging_[stage_idx].host;
    auto* device_ptr = static_cast<std::byte*>(device_cache_) + index * config_.page_size;
    const std::uint64_t offset = page_id * config_.page_size;

    // Unlocked: D2H into the pinned buffer, then the CPU copy back to the source.
    // Issue and drain on one pool stream; the writeback is synchronous so no
    // cross-stream ordering with concurrent H2D loads is needed.
    cudaStream_t stream = next_transfer_stream();
    lock.unlock();
    VRAMER_CUDA(cudaMemcpyAsync(host, device_ptr, valid, cudaMemcpyDeviceToHost, stream));
    VRAMER_CUDA(cudaStreamSynchronize(stream));
    writable_source_->write(offset, host, valid);
    lock.lock();

    staging_[stage_idx].reserved = false;
    staging_[stage_idx].in_flight = false;  // stream was synchronized; nothing pending
    staging_cv_.notify_all();
    stats_.bytes_d2h += valid;
}

void CudaPager::flush() {
    std::unique_lock<std::mutex> lock(mutex_);
    VRAMER_CUDA(cudaSetDevice(config_.gpu));
    // Contract: call when no leases are live (e.g. between training steps). We
    // flush unleased dirty pages; a leased page is skipped (it is still in use).
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (!slots_[i].dirty || slots_[i].page_id == UINT64_MAX || slots_[i].leases != 0)
            continue;
        const std::uint64_t pid = slots_[i].page_id;
        const std::size_t valid = slots_[i].valid_bytes;
        slots_[i].leases = 1;  // reserve so a concurrent miss cannot reuse it mid-flush
        flush_slot_data(i, pid, valid, lock);
        slots_[i].dirty = false;
        slots_[i].leases = 0;
    }
    load_cv_.notify_all();
}

void CudaPager::prefetch(std::uint64_t page_id) {
    auto lease = acquire_impl(page_id, /*is_prefetch=*/true, /*mutate=*/false);
    // Release after a transfer stream reaches the page load. This keeps the
    // page resident but unleased, ready for a later cache hit. Any pool stream
    // works: wait_on makes it wait on the slot's load_done (recorded on whatever
    // stream issued the copy), so the pending-use event is ordered after the
    // load regardless of which stream we pick here.
    cudaStream_t stream = next_transfer_stream();
    lease.wait_on(stream);
    lease.release_on(stream);
}

void CudaPager::prefetch_async(std::uint64_t page_id) {
    if (workers_.empty()) {
        // No worker pool configured: honor the request synchronously so callers
        // get correct behavior regardless of configuration.
        prefetch(page_id);
        return;
    }
    {
        std::lock_guard<std::mutex> qlock(queue_mutex_);
        std::lock_guard<std::mutex> slock(mutex_);

        // Skip work that is pointless or actively harmful:
        //  - already queued (dedup),
        //  - already resident (demand loaded it first; re-loading just churns),
        //  - queue + in-flight already at the cap. An unbounded queue lets a
        //    slow worker fall behind the demand cursor and prefetch pages that
        //    were already consumed and evicted -- pure thrashing that doubles
        //    H2D traffic. The cap can never usefully exceed cache capacity.
        const std::size_t cap = slots_.size();
        const bool queued_already = queued_.count(page_id) != 0;
        const bool resident_already = resident_.count(page_id) != 0;
        const std::size_t depth = prefetch_queue_.size() + in_flight_prefetch_;
        if (queued_already || resident_already || depth >= cap) {
            ++stats_.prefetch_queue_skips;
            return;
        }
        queued_.insert(page_id);
        prefetch_queue_.push_back(page_id);
    }
    queue_cv_.notify_one();
}

void CudaPager::worker_loop() {
    for (;;) {
        std::uint64_t page_id;
        {
            std::unique_lock<std::mutex> qlock(queue_mutex_);
            queue_cv_.wait(qlock, [this] { return stop_workers_ || !prefetch_queue_.empty(); });
            if (stop_workers_ && prefetch_queue_.empty()) return;
            page_id = prefetch_queue_.front();
            prefetch_queue_.pop_front();
            queued_.erase(page_id);
            ++in_flight_prefetch_;
        }
        try {
            prefetch(page_id);
        } catch (const std::exception& e) {
            // A prefetch is best-effort; a demand acquire() will retry and
            // surface any real error. Log once at low volume.
            std::cerr << "VRAMER prefetch worker: " << e.what() << '\n';
        }
        {
            std::lock_guard<std::mutex> qlock(queue_mutex_);
            --in_flight_prefetch_;
            if (prefetch_queue_.empty() && in_flight_prefetch_ == 0) drained_cv_.notify_all();
        }
    }
}

void CudaPager::drain_prefetch_queue() {
    std::unique_lock<std::mutex> qlock(queue_mutex_);
    drained_cv_.wait(qlock, [this] {
        return prefetch_queue_.empty() && in_flight_prefetch_ == 0;
    });
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
    ++stats_.lease_sync_fallbacks;
    if (stats_.lease_sync_fallbacks == 1) {
        std::cerr << "VRAMER warning: PageLease destroyed without release_on(); "
                     "falling back to cudaDeviceSynchronize(). Call release_on(stream) "
                     "on the hot path to avoid device-wide stalls.\n";
    }
    VRAMER_CUDA(cudaDeviceSynchronize());
    --slot.leases;
    reap_completed_events(slot, true);
}

void CudaPager::synchronize() {
    std::lock_guard<std::mutex> lock(mutex_);
    VRAMER_CUDA(cudaSetDevice(config_.gpu));
    for (auto s : transfer_streams_) VRAMER_CUDA(cudaStreamSynchronize(s));
    for (auto& slot : slots_) reap_completed_events(slot, true);
}

} // namespace vramer
