#pragma once

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace vramer {

inline void cuda_check(cudaError_t rc, const char* expr, const char* file, int line) {
    if (rc == cudaSuccess) return;
    std::ostringstream os;
    os << "CUDA error: " << cudaGetErrorString(rc)
       << " (" << static_cast<int>(rc) << ") at " << file << ':' << line
       << " in " << expr;
    throw std::runtime_error(os.str());
}

} // namespace vramer

#define VRAMER_CUDA(expr) ::vramer::cuda_check((expr), #expr, __FILE__, __LINE__)
