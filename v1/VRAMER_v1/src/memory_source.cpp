#include "vramer/common.hpp"
#include "vramer/memory_source.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vramer {

static void checked_range(std::uint64_t total, std::uint64_t offset, std::size_t bytes) {
    if (offset > total || bytes > total - offset) {
        throw std::out_of_range("MemorySource read exceeds source bounds");
    }
}

std::uint64_t parse_bytes(const std::string& input) {
    if (input.empty()) throw std::invalid_argument("empty byte size");

    std::size_t pos = 0;
    double value = std::stod(input, &pos);
    std::string suffix = input.substr(pos);
    suffix.erase(std::remove_if(suffix.begin(), suffix.end(), [](unsigned char c) {
        return std::isspace(c);
    }), suffix.end());
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });

    std::uint64_t mul = 1;
    if (suffix.empty() || suffix == "B") mul = 1;
    else if (suffix == "K" || suffix == "KB" || suffix == "KIB") mul = 1ull << 10;
    else if (suffix == "M" || suffix == "MB" || suffix == "MIB") mul = 1ull << 20;
    else if (suffix == "G" || suffix == "GB" || suffix == "GIB") mul = 1ull << 30;
    else if (suffix == "T" || suffix == "TB" || suffix == "TIB") mul = 1ull << 40;
    else throw std::invalid_argument("unknown size suffix: " + suffix);

    if (value < 0.0) throw std::invalid_argument("negative byte size");
    return static_cast<std::uint64_t>(value * static_cast<double>(mul));
}

MmapFileSource::MmapFileSource(std::string path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("open(" + path_ + "): " + std::strerror(errno));

    struct stat st{};
    if (::fstat(fd_, &st) != 0) {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("fstat(" + path_ + "): " + std::strerror(e));
    }
    if (st.st_size <= 0) throw std::runtime_error("source file is empty: " + path_);
    size_ = static_cast<std::uint64_t>(st.st_size);

    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) {
        int e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("mmap(" + path_ + "): " + std::strerror(e));
    }
    mapping_ = static_cast<const std::byte*>(p);
    (void)::madvise(const_cast<std::byte*>(mapping_), size_, MADV_SEQUENTIAL);
}

MmapFileSource::~MmapFileSource() {
    if (mapping_) ::munmap(const_cast<std::byte*>(mapping_), size_);
    if (fd_ >= 0) ::close(fd_);
}

std::string MmapFileSource::name() const { return "mmap:" + path_; }

void MmapFileSource::read(std::uint64_t offset, void* dst, std::size_t bytes) {
    checked_range(size_, offset, bytes);
    std::memcpy(dst, mapping_ + offset, bytes);
}

AnonymousRamSource::AnonymousRamSource(std::uint64_t bytes, bool touch) : size_(bytes) {
    if (bytes == 0) throw std::invalid_argument("AnonymousRamSource size must be > 0");
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("anonymous mmap failed: " + std::string(std::strerror(errno)));
    mapping_ = static_cast<std::byte*>(p);
    (void)::madvise(mapping_, bytes, MADV_HUGEPAGE);

    if (touch) {
        // Touch every base page so the benchmark represents real host memory,
        // not merely Linux virtual-memory overcommit. Values are deterministic.
        long page = ::sysconf(_SC_PAGESIZE);
        if (page <= 0) page = 4096;
        for (std::uint64_t i = 0; i < bytes; i += static_cast<std::uint64_t>(page)) {
            mapping_[i] = static_cast<std::byte>((i / page) & 0xffu);
        }
        mapping_[bytes - 1] = std::byte{0x5a};
    }
}

AnonymousRamSource::~AnonymousRamSource() {
    if (mapping_) ::munmap(mapping_, size_);
}

std::string AnonymousRamSource::name() const { return "anonymous-host-memory"; }

void AnonymousRamSource::read(std::uint64_t offset, void* dst, std::size_t bytes) {
    checked_range(size_, offset, bytes);
    std::memcpy(dst, mapping_ + offset, bytes);
}

void BorrowedMemorySource::read(std::uint64_t offset, void* dst, std::size_t bytes) {
    checked_range(size_, offset, bytes);
    std::memcpy(dst, base_ + offset, bytes);
}

} // namespace vramer
