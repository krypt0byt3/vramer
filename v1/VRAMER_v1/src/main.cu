#include "vramer/common.hpp"
#include "vramer/cuda_common.hpp"
#include "vramer/cuda_pager.hpp"
#include "vramer/memory_source.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/sysinfo.h>

namespace {

using Args = std::unordered_map<std::string, std::string>;

Args parse_args(int argc, char** argv, int start) {
    Args out;
    for (int i = start; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) != 0) throw std::invalid_argument("expected option, got: " + a);
        a.erase(0, 2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) out[a] = argv[++i];
        else out[a] = "1";
    }
    return out;
}

std::string get(const Args& a, const std::string& k, const std::string& def = {}) {
    auto it = a.find(k); return it == a.end() ? def : it->second;
}

std::uint64_t host_ram_bytes() {
    struct sysinfo si{};
    if (::sysinfo(&si) != 0) return 0;
    return static_cast<std::uint64_t>(si.totalram) * si.mem_unit;
}

int gpu_numa_node(const char* pci) {
    std::string p = "/sys/bus/pci/devices/" + std::string(pci) + "/numa_node";
    std::ifstream f(p);
    int node = -1;
    if (f) f >> node;
    return node;
}

void detect() {
    int count = 0;
    VRAMER_CUDA(cudaGetDeviceCount(&count));
    std::cout << "VRAMER 1.0 hardware detect\n";
    std::cout << "Host RAM: " << vramer::human_bytes(host_ram_bytes()) << "\n";
    std::cout << "CUDA devices: " << count << "\n\n";

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        VRAMER_CUDA(cudaGetDeviceProperties(&p, i));
        VRAMER_CUDA(cudaSetDevice(i));
        std::size_t free_b = 0, total_b = 0;
        VRAMER_CUDA(cudaMemGetInfo(&free_b, &total_b));
        char pci[32]{};
        VRAMER_CUDA(cudaDeviceGetPCIBusId(pci, sizeof(pci), i));
        std::cout << "GPU " << i << ": " << p.name << "\n"
                  << "  SM: " << p.major << '.' << p.minor << "\n"
                  << "  VRAM total/free: " << vramer::human_bytes(total_b)
                  << " / " << vramer::human_bytes(free_b) << "\n"
                  << "  PCI: " << pci << "  NUMA node: " << gpu_numa_node(pci) << "\n"
                  << "  async engines: " << p.asyncEngineCount << "\n"
                  << "  unified addressing: " << p.unifiedAddressing << "\n"
                  << "  managed memory: " << p.managedMemory << "\n"
                  << "  concurrent managed access: " << p.concurrentManagedAccess << "\n\n";
    }
}

__global__ void sample_kernel(const std::uint8_t* data, std::size_t n,
                              unsigned long long* sink, int reps) {
    unsigned long long local = 0;
    const std::size_t sample_stride = 4096;
    std::size_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t samples = (n + sample_stride - 1) / sample_stride;
    std::size_t grid = blockDim.x * gridDim.x;
    // `reps` simulates heavier per-page compute (real GEMM/attention work), so
    // the prefetch worker has something to overlap the next page's load with.
    for (int r = 0; r < reps; ++r) {
        for (std::size_t s = sample; s < samples; s += grid) {
            std::size_t i = s * sample_stride;
            if (i < n) local += data[i] + static_cast<unsigned long long>(r);
        }
    }
    if (local) atomicAdd(sink, local);
}

void plan(const Args& args) {
    const auto model = vramer::parse_bytes(get(args, "model-bytes"));
    const auto budget = vramer::parse_bytes(get(args, "vram-budget"));
    const auto page = vramer::parse_bytes(get(args, "page-size", "64MiB"));
    if (!model || !budget || !page) throw std::invalid_argument("sizes must be non-zero");
    const auto slots = budget / page;
    const auto pages = (model + page - 1) / page;
    const double resident = std::min(100.0, 100.0 * static_cast<double>(slots * page) / model);
    std::cout << "Model/source: " << vramer::human_bytes(model) << "\n"
              << "VRAM budget:  " << vramer::human_bytes(budget) << "\n"
              << "Page size:    " << vramer::human_bytes(page) << "\n"
              << "Source pages: " << pages << "\n"
              << "VRAM slots:   " << slots << "\n"
              << "Max resident: " << std::fixed << std::setprecision(1) << resident << "%\n";
}

struct RunResult {
    double sec = 0.0;
    vramer::PagerStats stats{};
    unsigned long long checksum = 0;
};

// Run the full paging benchmark loop on one GPU with its own independent cache,
// transfer stream, compute stream, and sink. Self-contained so it can be driven
// from a dedicated per-GPU thread for multi-GPU runs.
RunResult run_pager_bench(vramer::MemorySourcePtr source, vramer::PagerConfig cfg,
                          unsigned passes, std::uint64_t prefetch_ahead, int compute_reps) {
    VRAMER_CUDA(cudaSetDevice(cfg.gpu));
    vramer::CudaPager pager(std::move(source), cfg);
    cudaStream_t compute = nullptr;
    VRAMER_CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    unsigned long long* sink = nullptr;
    VRAMER_CUDA(cudaMalloc(reinterpret_cast<void**>(&sink), sizeof(*sink)));
    VRAMER_CUDA(cudaMemsetAsync(sink, 0, sizeof(*sink), compute));

    const std::uint64_t pages = pager.page_count();
    auto start = std::chrono::steady_clock::now();
    for (unsigned pass = 0; pass < passes; ++pass) {
        for (std::uint64_t k = 0; k < prefetch_ahead && k < pages; ++k) pager.prefetch_async(k);
        for (std::uint64_t p = 0; p < pages; ++p) {
            if (prefetch_ahead && p + prefetch_ahead < pages)
                pager.prefetch_async(p + prefetch_ahead);
            auto lease = pager.acquire(p);
            lease.wait_on(compute);
            sample_kernel<<<64, 256, 0, compute>>>(
                static_cast<const std::uint8_t*>(lease.device_ptr()), lease.bytes(), sink, compute_reps);
            VRAMER_CUDA(cudaGetLastError());
            lease.release_on(compute);
        }
    }
    VRAMER_CUDA(cudaStreamSynchronize(compute));
    pager.synchronize();
    auto stop = std::chrono::steady_clock::now();

    RunResult r;
    VRAMER_CUDA(cudaMemcpy(&r.checksum, sink, sizeof(r.checksum), cudaMemcpyDeviceToHost));
    r.stats = pager.stats();
    r.sec = std::chrono::duration<double>(stop - start).count();
    cudaFree(sink);
    cudaStreamDestroy(compute);
    return r;
}

void print_result(const std::string& label, const RunResult& r) {
    const double gib = static_cast<double>(r.stats.bytes_h2d) / (1024.0 * 1024.0 * 1024.0);
    std::cout << label << "\n"
              << "  elapsed:     " << std::fixed << std::setprecision(3) << r.sec << " s\n"
              << "  H2D bytes:   " << vramer::human_bytes(r.stats.bytes_h2d) << "\n"
              << "  H2D rate*:   " << std::setprecision(2) << (r.sec ? gib / r.sec : 0.0) << " GiB/s\n"
              << "  hits:        " << r.stats.hits << "\n"
              << "  misses:      " << r.stats.misses << "\n"
              << "  hit rate:    " << std::setprecision(1) << 100.0 * r.stats.hit_rate() << "%\n"
              << "  prefetches:  " << r.stats.prefetches << " (queue skips " << r.stats.prefetch_queue_skips << ")\n"
              << "  evictions:   " << r.stats.evictions << "\n"
              << "  src read:    " << std::setprecision(3)
              << static_cast<double>(r.stats.source_read_ns) / 1e6 << " ms\n"
              << "  stall:       " << static_cast<double>(r.stats.stall_ns) / 1e6 << " ms\n"
              << "  D2H bytes:   " << vramer::human_bytes(r.stats.bytes_d2h) << " (write-back)\n"
              << "  sync fallbk: " << r.stats.lease_sync_fallbacks << "\n"
              << "  checksum:    " << r.checksum << "\n";
}

std::vector<int> parse_gpu_list(const std::string& spec) {
    std::vector<int> gpus;
    std::size_t pos = 0;
    while (pos < spec.size()) {
        std::size_t comma = spec.find(',', pos);
        std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) gpus.push_back(std::stoi(tok));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return gpus;
}

void bench(const Args& args) {
    const auto budget = vramer::parse_bytes(get(args, "vram-budget", "2GiB"));
    const auto page = vramer::parse_bytes(get(args, "page-size", "64MiB"));
    const unsigned staging = static_cast<unsigned>(std::stoul(get(args, "staging", "2")));
    const unsigned passes = static_cast<unsigned>(std::stoul(get(args, "passes", "1")));
    const unsigned prefetch_workers = static_cast<unsigned>(std::stoul(get(args, "prefetch-workers", "0")));
    const std::uint64_t prefetch_ahead = std::stoull(get(args, "prefetch-ahead", "0"));
    const std::string source_kind = get(args, "source", "ram");
    const std::string evict = get(args, "eviction", "lru");
    const int compute_reps = std::stoi(get(args, "compute-reps", "1"));
    const int numa_node = std::stoi(get(args, "numa-node", "-1"));
    // --gpus 0,1 runs an independent cache per GPU concurrently; --gpu N is the
    // single-GPU shorthand (default 0).
    const std::vector<int> gpus = parse_gpu_list(get(args, "gpus", get(args, "gpu", "0")));
    if (gpus.empty()) throw std::invalid_argument("no GPUs specified");

    vramer::MemorySourcePtr source;
    if (source_kind == "file") {
        auto path = get(args, "file");
        if (path.empty()) throw std::invalid_argument("--source file requires --file PATH");
        source = std::make_shared<vramer::MmapFileSource>(path);
    } else if (source_kind == "ram") {
        auto bytes = vramer::parse_bytes(get(args, "bytes", "4GiB"));
        source = std::make_shared<vramer::AnonymousRamSource>(bytes, true, numa_node);
    } else {
        throw std::invalid_argument("--source must be ram or file");
    }

    vramer::PagerConfig base;
    base.vram_budget = budget;
    base.page_size = page;
    base.staging_buffers = staging;
    base.prefetch_workers = prefetch_workers;
    if (evict == "mru") base.eviction = vramer::EvictionPolicy::Mru;
    else if (evict == "lru") base.eviction = vramer::EvictionPolicy::Lru;
    else throw std::invalid_argument("--eviction must be lru or mru");

    std::cout << "Source:       " << source->name() << "\n"
              << "Source size:  " << vramer::human_bytes(source->size()) << "\n"
              << "GPUs:         ";
    for (std::size_t i = 0; i < gpus.size(); ++i) std::cout << (i ? "," : "") << gpus[i];
    std::cout << " (independent cache each)\n"
              << "VRAM budget:  " << vramer::human_bytes(budget) << " per GPU\n"
              << "Page size:    " << vramer::human_bytes(page) << "\n"
              << "Staging ring: " << staging << " x " << vramer::human_bytes(page) << " pinned\n"
              << "Eviction:     " << evict << "\n"
              << "Prefetch:     " << prefetch_workers << " worker(s), ahead=" << prefetch_ahead << "\n";

    // The RAM/file source is read-only during the run; concurrent const reads
    // from multiple per-GPU pagers are safe, so all GPUs share one host source.
    if (gpus.size() == 1) {
        vramer::PagerConfig cfg = base;
        cfg.gpu = gpus[0];
        RunResult r = run_pager_bench(source, cfg, passes, prefetch_ahead, compute_reps);
        std::cout << "\n";
        print_result("Results", r);
    } else {
        std::vector<RunResult> results(gpus.size());
        std::vector<std::exception_ptr> errors(gpus.size());
        std::vector<std::thread> threads;
        auto wall_start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < gpus.size(); ++i) {
            threads.emplace_back([&, i] {
                try {
                    vramer::PagerConfig cfg = base;
                    cfg.gpu = gpus[i];
                    results[i] = run_pager_bench(source, cfg, passes, prefetch_ahead, compute_reps);
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            });
        }
        for (auto& t : threads) t.join();
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();
        for (auto& e : errors) if (e) std::rethrow_exception(e);

        std::uint64_t total_bytes = 0;
        double sum_rate = 0.0;  // concurrent throughput = sum of per-GPU rates
        for (std::size_t i = 0; i < gpus.size(); ++i) {
            total_bytes += results[i].stats.bytes_h2d;
            const double gib = static_cast<double>(results[i].stats.bytes_h2d) / (1024.0 * 1024.0 * 1024.0);
            if (results[i].sec) sum_rate += gib / results[i].sec;
            std::cout << "\n";
            print_result("GPU " + std::to_string(gpus[i]), results[i]);
        }
        std::cout << "\nAggregate (" << gpus.size() << " GPUs)\n"
                  << "  H2D total:   " << vramer::human_bytes(total_bytes) << "\n"
                  << "  H2D rate*:   " << std::fixed << std::setprecision(2) << sum_rate
                  << " GiB/s concurrent (sum of per-GPU rates)\n"
                  << "  wall:        " << std::setprecision(3) << wall
                  << " s (includes serialized per-GPU CUDA context/alloc setup)\n";
    }
    std::cout << "\n*End-to-end pager rate: includes source->pinned staging copy, H2D,\n"
              << " lightweight GPU work, and cache management.\n";
}

void usage() {
    std::cout << R"(VRAMER 1.0 - explicit host-memory <-> VRAM pager

Commands:
  vramer detect
  vramer plan --model-bytes 12GiB --vram-budget 6GiB [--page-size 64MiB]
  vramer bench [options]

bench options:
  --gpu N                 CUDA device (default 0)
  --gpus 0,1              run an independent cache per GPU, concurrently
  --source ram|file       backing source (default ram)
  --bytes SIZE            RAM source size (default 4GiB)
  --file PATH             file source for mmap mode
  --vram-budget SIZE      device-cache budget (default 2GiB)
  --page-size SIZE        paging granularity (default 64MiB)
  --staging N             pinned staging buffers (default 2)
  --passes N              sequential passes (default 1)
  --prefetch-workers N    background prefetch worker threads (default 0)
  --prefetch-ahead N      stay N pages ahead of the demand cursor (needs workers)
  --eviction lru|mru      eviction policy (default lru; mru is scan-resistant)
  --compute-reps N        per-page kernel work multiplier (default 1)
  --numa-node N           bind RAM source to NUMA node N (default -1 = off)

Examples:
  vramer detect
  vramer bench --source ram --bytes 12GiB --vram-budget 6GiB --page-size 64MiB
  vramer bench --source file --file model.gguf --vram-budget 6GiB --passes 2
  vramer bench --source ram --bytes 12GiB --vram-budget 6GiB --page-size 64MiB \
    --prefetch-workers 1 --prefetch-ahead 2
)";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 1; }
        const std::string cmd = argv[1];
        const Args args = parse_args(argc, argv, 2);
        if (cmd == "detect") detect();
        else if (cmd == "plan") plan(args);
        else if (cmd == "bench") bench(args);
        else { usage(); return 1; }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "VRAMER error: " << e.what() << '\n';
        return 2;
    }
}
