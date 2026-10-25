#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

namespace vramer {

inline std::string human_bytes(std::uint64_t bytes) {
    static const char* suffix[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(bytes);
    int i = 0;
    while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(v >= 100.0 ? 0 : (v >= 10.0 ? 1 : 2));
    os << v << ' ' << suffix[i];
    return os.str();
}

std::uint64_t parse_bytes(const std::string& text);

} // namespace vramer
