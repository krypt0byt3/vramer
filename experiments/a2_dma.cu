#include <cuda_runtime.h>

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

double benchmark_copy(
    void *dst,
    void *src,
    size_t bytes,
    cudaMemcpyKind direction,
    cudaStream_t stream,
    int iterations)
{
    // Warmup
    CUDA_CHECK(cudaMemcpyAsync(
        dst, src, bytes, direction, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(
            dst, src, bytes, direction, stream));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double seconds = ms / 1000.0;

    double totalGiB =
        ((double)bytes * iterations) /
        (1024.0 * 1024.0 * 1024.0);

    return totalGiB / seconds;
}

int main(int argc, char **argv)
{
    int device = 0;
    size_t bufferMiB = 1024;
    int iterations = 8;

    if (argc >= 2)
        device = std::atoi(argv[1]);

    if (argc >= 3)
        bufferMiB = std::strtoull(argv[2], nullptr, 10);

    if (argc >= 4)
        iterations = std::atoi(argv[3]);

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    size_t bytes =
        bufferMiB * 1024ULL * 1024ULL;

    std::cout << "\n=== A2 RAW PCIe DMA TEST ===\n\n";

    std::cout << "GPU:          " << device << "\n";
    std::cout << "Name:         " << prop.name << "\n";
    std::cout << "Buffer:       " << bufferMiB << " MiB\n";
    std::cout << "Iterations:   " << iterations << "\n";

    void *host = nullptr;
    void *deviceMem = nullptr;

    std::cout << "\nAllocating pinned host RAM...\n";

    CUDA_CHECK(cudaHostAlloc(
        &host,
        bytes,
        cudaHostAllocDefault));

    std::memset(host, 0x5a, bytes);

    std::cout << "Allocating real VRAM...\n";

    CUDA_CHECK(cudaMalloc(
        &deviceMem,
        bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::cout << "\nTesting Host -> Device...\n";

    double h2d =
        benchmark_copy(
            deviceMem,
            host,
            bytes,
            cudaMemcpyHostToDevice,
            stream,
            iterations);

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "H2D bandwidth: "
              << h2d
              << " GiB/s\n";

    std::cout << "\nTesting Device -> Host...\n";

    double d2h =
        benchmark_copy(
            host,
            deviceMem,
            bytes,
            cudaMemcpyDeviceToHost,
            stream,
            iterations);

    std::cout << "D2H bandwidth: "
              << d2h
              << " GiB/s\n";

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceMem));
    CUDA_CHECK(cudaFreeHost(host));

    std::cout << "\nSUCCESS\n";

    return 0;
}
