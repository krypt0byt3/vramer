#include "vramer/common.hpp"
#include "vramer/memory_source.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>

int main() {
    assert(vramer::parse_bytes("1GiB") == (1ull << 30));
    assert(vramer::parse_bytes("64M") == (64ull << 20));

    auto src = vramer::AnonymousRamSource(1ull << 20, true);
    std::array<std::byte, 16> b{};
    src.read(0, b.data(), b.size());
    // First touched page has deterministic byte zero at offset 0.
    assert(b[0] == std::byte{0});

    std::cout << "memory source tests passed\n";
    return 0;
}
