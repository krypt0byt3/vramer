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

class AnonymousRamSource final : public MemorySource {
public:
    explicit AnonymousRamSource(std::uint64_t bytes, bool touch = true);
    ~AnonymousRamSource() override;

    AnonymousRamSource(const AnonymousRamSource&) = delete;
    AnonymousRamSource& operator=(const AnonymousRamSource&) = delete;

    std::uint64_t size() const noexcept override { return size_; }
    std::string name() const override;
    void read(std::uint64_t offset, void* dst, std::size_t bytes) override;

private:
    std::uint64_t size_ = 0;
    std::byte* mapping_ = nullptr;
};

// Adapter for memory owned by another runtime. This is the hook for future
// framework integrations and for OS-exposed memory tiers such as CXL.
class BorrowedMemorySource final : public MemorySource {
public:
    BorrowedMemorySource(const void* base, std::uint64_t bytes, std::string label)
        : base_(static_cast<const std::byte*>(base)), size_(bytes), label_(std::move(label)) {}

    std::uint64_t size() const noexcept override { return size_; }
    std::string name() const override { return label_; }
    void read(std::uint64_t offset, void* dst, std::size_t bytes) override;

private:
    const std::byte* base_ = nullptr;
    std::uint64_t size_ = 0;
    std::string label_;
};

using MemorySourcePtr = std::shared_ptr<MemorySource>;

} // namespace vramer
