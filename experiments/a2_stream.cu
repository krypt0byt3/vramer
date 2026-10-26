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
void fake_transformer_work(float *data, size_t count, int work)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (; i < count; i += stride) {
        float x = data[i];

        for (int j = 0; j < work; ++j) {
            x = x * 1.000001f + 0.000001f;
        }

        data[i] = x;
    }
}

double run_serial(
    float **host,
    float **device,
    size_t bytes,
    int chunks,
    int work,
    int grid,
    int block)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < chunks; ++n) {
        int b = n & 1;

        CUDA_CHECK(cudaMemcpyAsync(
            device[b],
            host[b],
            bytes,
            cudaMemcpyHostToDevice,
            stream));

        fake_transformer_work<<<grid, block, 0, stream>>>(
            device[b],
            bytes / sizeof(float),
            work);

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto end = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaStreamDestroy(stream));

    return std::chrono::duration<double>(end - start).count();
}

double run_pipeline(
    float **host,
    float **device,
    size_t bytes,
    int chunks,
    int work,
    int grid,
    int block)
{
    cudaStream_t streams[2];

    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < chunks; ++n) {
        int b = n & 1;

        //
        // Operations within this stream remain ordered:
        //
        //     COPY chunk N
        //          ↓
        //     COMPUTE chunk N
        //
        // But the other stream can execute concurrently.
        //
        CUDA_CHECK(cudaMemcpyAsync(
            device[b],
            host[b],
            bytes,
            cudaMemcpyHostToDevice,
            streams[b]));

        fake_transformer_work<<<grid, block, 0, streams[b]>>>(
            device[b],
            bytes / sizeof(float),
            work);

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));

    auto end = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));

    return std::chrono::duration<double>(end - start).count();
}

int main(int argc, char **argv)
{
    int device = 0;
    size_t chunkMiB = 256;
    int chunks = 32;
    int work = 1024;

    if (argc >= 2)
        device = std::atoi(argv[1]);

    if (argc >= 3)
        chunkMiB = std::strtoull(argv[2], nullptr, 10);

    if (argc >= 4)
        chunks = std::atoi(argv[3]);

    if (argc >= 5)
        work = std::atoi(argv[4]);

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    size_t bytes = chunkMiB * 1024ULL * 1024ULL;

    std::cout << "\n=== VRAMER DOUBLE BUFFER TEST ===\n\n";

    std::cout << "GPU:                 " << device << "\n";
    std::cout << "Name:                " << prop.name << "\n";
    std::cout << "Chunk size:          " << chunkMiB << " MiB\n";
    std::cout << "Chunks:              " << chunks << "\n";
    std::cout << "Logical stream size: "
              << (chunkMiB * chunks / 1024.0)
              << " GiB\n";

    std::cout << "Compute work:        " << work << "\n";

    std::cout << "\nCUDA capabilities\n";
    std::cout << "Async engines:       "
              << prop.asyncEngineCount << "\n";
    std::cout << "Device overlap:      "
              << prop.deviceOverlap << "\n";
    std::cout << "Concurrent kernels:  "
              << prop.concurrentKernels << "\n";

    float *host[2]{};
    float *deviceMem[2]{};

    //
    // Two pinned host buffers.
    //
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaHostAlloc(
            &host[i],
            bytes,
            cudaHostAllocDefault));

        std::memset(host[i], i + 1, bytes);
    }

    //
    // Two actual GDDR6 buffers.
    //
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(
            &deviceMem[i],
            bytes));
    }

    int block = 256;
    int grid = prop.multiProcessorCount * 32;

    //
    // Warm up CUDA.
    //
    CUDA_CHECK(cudaMemcpy(
        deviceMem[0],
        host[0],
        bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "\nRunning SERIAL version...\n";

    double serial =
        run_serial(
            host,
            deviceMem,
            bytes,
            chunks,
            work,
            grid,
            block);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Serial time:          "
              << std::fixed << std::setprecision(3)
              << serial << " sec\n";

    std::cout << "\nRunning PIPELINED version...\n";

    double pipeline =
        run_pipeline(
            host,
            deviceMem,
            bytes,
            chunks,
            work,
            grid,
            block);

    CUDA_CHECK(cudaDeviceSynchronize());

    double logicalGiB =
        ((double)bytes * chunks) /
        (1024.0 * 1024.0 * 1024.0);

    std::cout << "Pipeline time:        "
              << pipeline << " sec\n";

    std::cout << "\n=== RESULTS ===\n";

    std::cout << "Serial throughput:    "
              << logicalGiB / serial
              << " GiB/s\n";

    std::cout << "Pipeline throughput:  "
              << logicalGiB / pipeline
              << " GiB/s\n";

    std::cout << "Speedup:              "
              << serial / pipeline
              << "x\n";

    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaFree(deviceMem[i]));
        CUDA_CHECK(cudaFreeHost(host[i]));
    }

    std::cout << "\nSUCCESS\n";

    return 0;
}
