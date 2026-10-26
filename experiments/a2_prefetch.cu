#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

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
void touch_memory(float *data, size_t count, float value)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (; i < count; i += stride)
        data[i] += value;
}

int main(int argc, char **argv)
{
    double totalGB = 48.0;
    double chunkGB = 8.0;
    int device = 0;

    if (argc >= 2)
        totalGB = std::atof(argv[1]);

    if (argc >= 3)
        chunkGB = std::atof(argv[2]);

    if (argc >= 4)
        device = std::atoi(argv[3]);

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    size_t freeVRAM = 0;
    size_t totalVRAM = 0;

    CUDA_CHECK(cudaMemGetInfo(&freeVRAM, &totalVRAM));

    size_t totalBytes =
        static_cast<size_t>(totalGB * 1024.0 * 1024.0 * 1024.0);

    size_t chunkBytes =
        static_cast<size_t>(chunkGB * 1024.0 * 1024.0 * 1024.0);

    // Keep float alignment.
    chunkBytes -= chunkBytes % sizeof(float);

    std::cout << "\n=== Explicit Unified Memory Prefetch Test ===\n\n";

    std::cout << "GPU:              " << prop.name << "\n";
    std::cout << "Physical VRAM:    "
              << totalVRAM / 1024.0 / 1024.0 / 1024.0
              << " GiB\n";

    std::cout << "Logical memory:   "
              << totalGB << " GiB\n";

    std::cout << "Prefetch window:  "
              << chunkGB << " GiB\n";

    std::cout << "Oversubscription: "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(totalBytes) / totalVRAM
              << "x\n";

    float *data = nullptr;

    std::cout << "\nAllocating " << totalGB
              << " GiB managed memory...\n";

    CUDA_CHECK(cudaMallocManaged(&data, totalBytes));

    std::cout << "Initializing from CPU...\n";

    auto cpuStart = std::chrono::steady_clock::now();

    std::memset(data, 0, totalBytes);

    auto cpuEnd = std::chrono::steady_clock::now();

    std::cout << "CPU initialization: "
              << std::chrono::duration<double>(
                     cpuEnd - cpuStart).count()
              << " sec\n";

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int block = 256;
    int grid = prop.multiProcessorCount * 32;

    for (int pass = 1; pass <= 3; ++pass)
    {
        std::cout << "\n===================================\n";
        std::cout << "PASS " << pass << "\n";
        std::cout << "===================================\n";

        auto passStart = std::chrono::steady_clock::now();

        double totalPrefetchSeconds = 0.0;
        double totalKernelSeconds = 0.0;

        size_t chunkNumber = 0;

        for (size_t offset = 0;
             offset < totalBytes;
             offset += chunkBytes)
        {
            size_t currentBytes =
                std::min(chunkBytes, totalBytes - offset);

            float *chunkPtr =
                reinterpret_cast<float *>(
                    reinterpret_cast<char *>(data) + offset);

            size_t count =
                currentBytes / sizeof(float);

            ++chunkNumber;

            //
            // Explicitly move the next chunk to the A2.
            //
            auto prefetchStart =
                std::chrono::steady_clock::now();

            CUDA_CHECK(cudaMemPrefetchAsync(
                chunkPtr,
                currentBytes,
                device,
                stream));

            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto prefetchEnd =
                std::chrono::steady_clock::now();

            double prefetchSeconds =
                std::chrono::duration<double>(
                    prefetchEnd - prefetchStart).count();

            totalPrefetchSeconds += prefetchSeconds;

            //
            // Now operate on memory that should already
            // be resident on the GPU.
            //
            auto kernelStart =
                std::chrono::steady_clock::now();

            touch_memory<<<grid, block, 0, stream>>>(
                chunkPtr,
                count,
                static_cast<float>(pass));

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto kernelEnd =
                std::chrono::steady_clock::now();

            double kernelSeconds =
                std::chrono::duration<double>(
                    kernelEnd - kernelStart).count();

            totalKernelSeconds += kernelSeconds;

            double migrationRate =
                (currentBytes /
                 1024.0 / 1024.0 / 1024.0) /
                prefetchSeconds;

            std::cout
                << "Chunk "
                << chunkNumber
                << ": prefetch "
                << std::fixed
                << std::setprecision(2)
                << migrationRate
                << " GiB/s"
                << ", kernel "
                << kernelSeconds
                << " sec\n";
        }

        auto passEnd =
            std::chrono::steady_clock::now();

        double totalSeconds =
            std::chrono::duration<double>(
                passEnd - passStart).count();

        std::cout << "\nPASS SUMMARY\n";

        std::cout << "Prefetch time:       "
                  << totalPrefetchSeconds
                  << " sec\n";

        std::cout << "Kernel time:         "
                  << totalKernelSeconds
                  << " sec\n";

        std::cout << "Total time:          "
                  << totalSeconds
                  << " sec\n";

        std::cout << "Logical throughput:  "
                  << totalGB / totalSeconds
                  << " GiB/s\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(data));

    std::cout << "\nSUCCESS\n";

    return 0;
}
