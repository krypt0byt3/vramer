#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

__global__
void touch_memory(float *data, size_t count, float add)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (; i < count; i += stride)
        data[i] += add;
}

int main(int argc, char **argv)
{
    int device = 0;
    double requested_gb = 24.0;

    if (argc >= 2)
        requested_gb = std::atof(argv[1]);

    if (argc >= 3)
        device = std::atoi(argv[2]);

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int managedMemory = 0;
    int concurrentManagedAccess = 0;
    int pageableMemoryAccess = 0;

    CUDA_CHECK(cudaDeviceGetAttribute(
        &managedMemory,
        cudaDevAttrManagedMemory,
        device));

    CUDA_CHECK(cudaDeviceGetAttribute(
        &concurrentManagedAccess,
        cudaDevAttrConcurrentManagedAccess,
        device));

    CUDA_CHECK(cudaDeviceGetAttribute(
        &pageableMemoryAccess,
        cudaDevAttrPageableMemoryAccess,
        device));

    size_t freeVRAM = 0;
    size_t totalVRAM = 0;

    CUDA_CHECK(cudaMemGetInfo(&freeVRAM, &totalVRAM));

    std::cout << "\n=== GPU ===\n";
    std::cout << "Device:                 " << device << "\n";
    std::cout << "Name:                   " << prop.name << "\n";
    std::cout << "Physical VRAM:          "
              << totalVRAM / 1024.0 / 1024.0 / 1024.0 << " GiB\n";
    std::cout << "Free VRAM:              "
              << freeVRAM / 1024.0 / 1024.0 / 1024.0 << " GiB\n";

    std::cout << "\n=== Unified Memory capabilities ===\n";
    std::cout << "ManagedMemory:           " << managedMemory << "\n";
    std::cout << "ConcurrentManagedAccess: " << concurrentManagedAccess << "\n";
    std::cout << "PageableMemoryAccess:    " << pageableMemoryAccess << "\n";

    size_t bytes =
        static_cast<size_t>(requested_gb * 1024.0 * 1024.0 * 1024.0);

    size_t count = bytes / sizeof(float);

    std::cout << "\n=== Test ===\n";
    std::cout << "Requesting:              "
              << requested_gb << " GiB\n";

    if (bytes > totalVRAM) {
        std::cout << "Oversubscription:        YES\n";
        std::cout << "Allocation / VRAM:       "
                  << std::fixed << std::setprecision(2)
                  << ((double)bytes / totalVRAM)
                  << "x\n";
    } else {
        std::cout << "Oversubscription:        NO\n";
    }

    float *data = nullptr;

    std::cout << "\nAllocating managed memory...\n";

    CUDA_CHECK(cudaMallocManaged(&data, bytes));

    std::cout << "Allocation successful.\n";

    //
    // Touch it from the CPU first.
    //
    // This deliberately establishes the pages in system RAM.
    //

    std::cout << "\nInitializing from CPU...\n";

    auto cpuStart = std::chrono::steady_clock::now();

    const size_t floatsPerPage = 4096 / sizeof(float);

    for (size_t i = 0; i < count; i += floatsPerPage)
        data[i] = 1.0f;

    auto cpuEnd = std::chrono::steady_clock::now();

    double cpuSeconds =
        std::chrono::duration<double>(cpuEnd - cpuStart).count();

    std::cout << "CPU initialization:      "
              << cpuSeconds << " sec\n";

    //
    // Have the GPU walk the ENTIRE allocation.
    //
    // If requested size exceeds VRAM, CUDA must use
    // managed-memory migration/eviction.
    //

    int block = 256;
    int grid = prop.multiProcessorCount * 32;

    for (int pass = 1; pass <= 3; ++pass)
    {
        std::cout << "\nGPU pass " << pass << "...\n";

        auto start = std::chrono::steady_clock::now();

        touch_memory<<<grid, block>>>(
            data,
            count,
            static_cast<float>(pass));

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::steady_clock::now();

        double seconds =
            std::chrono::duration<double>(end - start).count();

        double logicalBandwidth =
            requested_gb / seconds;

        std::cout << "Time:                   "
                  << seconds << " sec\n";

        std::cout << "Logical scan rate:      "
                  << logicalBandwidth << " GiB/s\n";
    }

    //
    // Verify that the allocation survived and remains
    // accessible from the CPU.
    //

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "\nVerification value:      "
              << data[0] << "\n";

    CUDA_CHECK(cudaFree(data));

    std::cout << "\nSUCCESS\n";
    std::cout << requested_gb
              << " GiB logical CUDA allocation operated on a "
              << totalVRAM / 1024.0 / 1024.0 / 1024.0
              << " GiB GPU.\n\n";

    return 0;
}
