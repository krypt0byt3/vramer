/*
 * vramer_ggml_mulmat.cpp — REAL ggml-cuda kernel consuming a VRAMER-resident
 * GGUF weight tensor, numerically verified against a CPU reference.
 *
 * This is the P2 deliverable from docs/LLAMACPP_ADAPTER.md: it proves the core
 * mechanism — a weight tensor's bytes are made resident in a *fixed VRAM budget
 * that is smaller than the tensor itself* via the VRAMER pager, and a genuine
 * ggml CUDA matmul kernel then reads those bytes and produces the correct
 * result.
 *
 * Pipeline:
 *   1. Open the GGUF with llama.cpp's gguf API (no_alloc), enumerate tensors,
 *      and pick the largest 2-D weight tensor that is valid for ggml_mul_mat
 *      (ne[0] a multiple of the type's block size).
 *   2. Choose a page size so the tensor spans several pages, and a VRAM budget
 *      HALF the tensor size — the tensor cannot be fully resident at once.
 *   3. vramer_source_file(gguf) + a MRU pager with 1 prefetch worker. For each
 *      page the tensor occupies: prefetch(next) / acquire -> devptr /
 *      lease_wait_on(stream) / D2D-copy that page's slice into a CONTIGUOUS
 *      device scratch buffer / lease_release_on(stream). (Contiguity handling:
 *      D2D gather — see the "CONTIGUITY" note below.)
 *   4. Wrap that scratch buffer in a ggml_tensor (type/shape/strides from the
 *      GGUF) by setting ->data to the gathered device pointer, and run a REAL
 *      ggml_mul_mat on the CUDA backend (W x B).
 *   5. Reference: read the same tensor bytes straight from the file with pread
 *      (no VRAMER) and run the identical ggml_mul_mat on the CPU backend.
 *   6. Assert the GPU-via-VRAMER result matches the CPU result within fp
 *      tolerance, and print VRAMER stats (hits/misses/bytes_h2d/evictions)
 *      proving the weight was physically paged through the budget.
 *
 * CONTIGUITY (docs/LLAMACPP_ADAPTER.md §3): a weight that is larger than one
 * page spans several pages, and VRAMER's fixed-slot cache does NOT guarantee
 * those pages land in contiguous device slots. We take option (ii) from the
 * design: after acquiring each page we D2D-copy its slice of the tensor into a
 * single contiguous cudaMalloc'd scratch region, then point the kernel at that.
 * This is what lets the VRAM budget be *smaller than the tensor* (we only ever
 * hold a few pages resident at once, releasing each after its slice is copied).
 */

#include "vramer/vramer_c.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <vector>

static char err[512];

#define V_FAIL(what) do { fprintf(stderr, "VRAMER %s: %s\n", (what), err); std::exit(1); } while (0)
#define CU_OK(expr, what) do { cudaError_t _e=(expr); if(_e!=cudaSuccess){ \
    fprintf(stderr, "CUDA %s: %s\n",(what),cudaGetErrorString(_e)); std::exit(1);} } while(0)

static uint64_t round_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <gguf> [budget_bytes] [page_bytes]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        printf("no CUDA device; skipping\n");
        return 0;
    }
    CU_OK(cudaSetDevice(0), "setDevice");

    /* ---- 1. GGUF metadata (no tensor data allocated) -------------------- */
    struct gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx      = nullptr;
    struct gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx) { fprintf(stderr, "gguf_init_from_file failed: %s\n", path); return 1; }

    const size_t data_off = gguf_get_data_offset(gctx);
    const int64_t nt = gguf_get_n_tensors(gctx);

    /* Pick the largest 2-D tensor valid for ggml_mul_mat (ne0 % block == 0). */
    int64_t best = -1; size_t best_sz = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const int64_t* ne = gguf_get_tensor_ne(gctx, i);
        if (ne[2] != 1 || ne[3] != 1) continue;          /* 2-D only          */
        if (ne[0] < 2 || ne[1] < 2)   continue;
        enum ggml_type t = gguf_get_tensor_type(gctx, i);
        if (ne[0] % ggml_blck_size(t) != 0) continue;    /* mul_mat validity  */
        size_t sz = gguf_get_tensor_size(gctx, i);
        if (sz > best_sz) { best_sz = sz; best = i; }
    }
    if (best < 0) { fprintf(stderr, "no suitable 2-D tensor found\n"); return 1; }

    const char*   wname = gguf_get_tensor_name(gctx, best);
    enum ggml_type wtype = gguf_get_tensor_type(gctx, best);
    const int64_t* wne = gguf_get_tensor_ne(gctx, best);
    const int64_t ne0 = wne[0], ne1 = wne[1];
    const uint64_t w_off  = (uint64_t)data_off + gguf_get_tensor_offset(gctx, best);
    const uint64_t w_size = (uint64_t)best_sz;

    printf("=== chosen weight tensor ===\n");
    printf("name=%s type=%s shape=[%lld x %lld] size=%.2f MiB\n",
           wname, ggml_type_name(wtype), (long long)ne0, (long long)ne1,
           w_size / 1048576.0);
    printf("file_offset=%llu (data_off=%zu)\n", (unsigned long long)w_off, data_off);

    /* ---- 2. page size so the tensor spans ~4 pages; budget = 2 pages ---- */
    uint64_t page_size = round_up(w_size / 4, 2ull << 20);   /* multiple of 2 MiB */
    if (page_size < (2ull << 20)) page_size = 2ull << 20;
    uint64_t budget    = 2 * page_size;                       /* HALF the tensor  */
    if (argc >= 3) { unsigned long long v = strtoull(argv[2], 0, 10); if (v) budget = v; }
    if (argc >= 4) { unsigned long long v = strtoull(argv[3], 0, 10); if (v) page_size = v; }

    /* ---- 3. pager over the GGUF ---------------------------------------- */
    vramer_source* src = vramer_source_file(path, err, sizeof err);
    if (!src) V_FAIL("source_file");

    vramer_config cfg; memset(&cfg, 0, sizeof cfg);
    cfg.gpu = 0;
    cfg.vram_budget = budget;
    cfg.page_size = page_size;
    cfg.staging_buffers = 2;
    cfg.prefetch_workers = 1;
    cfg.eviction = VRAMER_EVICT_MRU;

    vramer_pager* pager = vramer_pager_create(src, &cfg, err, sizeof err);
    if (!pager) V_FAIL("pager_create");
    vramer_source_free(src);

    const uint64_t first_pg = w_off / page_size;
    const uint64_t last_pg  = (w_off + w_size - 1) / page_size;
    const uint64_t npages   = last_pg - first_pg + 1;
    printf("=== VRAMER paging ===\n");
    printf("page_size=%.1f MiB budget=%.1f MiB (resident_capacity=%llu pages)\n",
           page_size / 1048576.0, budget / 1048576.0,
           (unsigned long long)vramer_pager_resident_capacity_pages(pager));
    printf("weight spans pages [%llu..%llu] = %llu pages; budget holds %llu -> "
           "tensor is PAGED through a budget %s than itself\n",
           (unsigned long long)first_pg, (unsigned long long)last_pg,
           (unsigned long long)npages,
           (unsigned long long)vramer_pager_resident_capacity_pages(pager),
           (budget < w_size ? "SMALLER" : "not smaller"));

    /* contiguous device scratch to gather the tensor into */
    void* d_scratch = nullptr;
    CU_OK(cudaMalloc(&d_scratch, w_size), "malloc scratch");

    cudaStream_t stream;
    CU_OK(cudaStreamCreate(&stream), "streamCreate");

    for (uint64_t pg = first_pg; pg <= last_pg; ++pg) {
        if (pg + 1 <= last_pg)
            vramer_pager_prefetch_async(pager, pg + 1, err, sizeof err);

        vramer_lease* lease = nullptr; void* devptr = nullptr; size_t pbytes = 0;
        if (vramer_acquire(pager, pg, &lease, &devptr, &pbytes, err, sizeof err))
            V_FAIL("acquire");
        if (vramer_lease_wait_on(lease, (void*)stream, err, sizeof err)) V_FAIL("wait_on");

        /* slice of this page that belongs to the tensor */
        uint64_t pg_start = pg * page_size;
        uint64_t s0 = (w_off > pg_start) ? w_off : pg_start;             /* file   */
        uint64_t s1 = (w_off + w_size < pg_start + page_size)
                          ? (w_off + w_size) : (pg_start + page_size);
        uint64_t len = s1 - s0;
        uint64_t src_off_in_page = s0 - pg_start;
        uint64_t dst_off         = s0 - w_off;
        CU_OK(cudaMemcpyAsync((char*)d_scratch + dst_off,
                              (const char*)devptr + src_off_in_page, len,
                              cudaMemcpyDeviceToDevice, stream), "D2D gather");

        if (vramer_lease_release_on(lease, (void*)stream, err, sizeof err)) V_FAIL("release_on");
    }
    CU_OK(cudaStreamSynchronize(stream), "gather sync");
    vramer_pager_drain_prefetch_queue(pager, err, sizeof err);
    vramer_pager_synchronize(pager, err, sizeof err);
    printf("gathered %.2f MiB of weight into contiguous device scratch\n",
           w_size / 1048576.0);

    /* ---- small deterministic input B (f32, [ne0 x M]) ------------------ */
    const int64_t M = 8;
    std::vector<float> hostB((size_t)ne0 * M);
    for (int64_t j = 0; j < M; ++j)
        for (int64_t k = 0; k < ne0; ++k)
            hostB[(size_t)j * ne0 + k] = 0.001f * (float)((k * 7 + j * 13) % 101 - 50);

    /* ---- 4. REAL ggml_mul_mat on CUDA backend reading d_scratch -------- */
    ggml_backend_t be_cuda = ggml_backend_cuda_init(0);
    if (!be_cuda) { fprintf(stderr, "ggml_backend_cuda_init failed\n"); return 1; }

    /* W lives in its own ctx; we do NOT let the allocator allocate it — its
     * ->data is the VRAMER-gathered device pointer. B and C are allocated in a
     * real CUDA buffer. */
    struct ggml_init_params ip_w { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context* ctxW = ggml_init(ip_w);
    ggml_tensor* W = ggml_new_tensor_2d(ctxW, wtype, ne0, ne1);
    ggml_set_name(W, "W_vramer");

    struct ggml_init_params ip_io { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context* ctxIO = ggml_init(ip_io);
    ggml_tensor* B = ggml_new_tensor_2d(ctxIO, GGML_TYPE_F32, ne0, M);
    ggml_set_name(B, "B");
    ggml_tensor* C = ggml_mul_mat(ctxIO, W, B);   /* C: [ne1 x M] */
    ggml_set_name(C, "C");
    ggml_set_output(C);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctxIO, be_cuda);
    if (!buf) { fprintf(stderr, "alloc_ctx_tensors (cuda) failed\n"); return 1; }

    /* bind W to the VRAMER-resident bytes */
    W->buffer = buf;            /* mark as a CUDA-resident tensor            */
    W->data   = d_scratch;      /* <-- ggml kernel reads VRAMER-paged bytes  */

    ggml_backend_tensor_set(B, hostB.data(), 0, hostB.size() * sizeof(float));

    struct ggml_init_params ip_g { ggml_graph_overhead(), nullptr, true };
    ggml_context* ctxG = ggml_init(ip_g);
    ggml_cgraph* gf = ggml_new_graph(ctxG);
    ggml_build_forward_expand(gf, C);
    if (ggml_backend_graph_compute(be_cuda, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cuda graph_compute failed\n"); return 1;
    }

    std::vector<float> outGPU((size_t)ne1 * M);
    ggml_backend_tensor_get(C, outGPU.data(), 0, outGPU.size() * sizeof(float));

    /* ---- 5. CPU reference: same op, weight read straight from file ----- */
    std::vector<uint8_t> hostW(w_size);
    {
        int fd = open(path, O_RDONLY);
        if (fd < 0) { perror("open"); return 1; }
        size_t done = 0;
        while (done < w_size) {
            ssize_t r = pread(fd, hostW.data() + done, w_size - done, w_off + done);
            if (r <= 0) { perror("pread"); return 1; }
            done += (size_t)r;
        }
        close(fd);
    }

    ggml_backend_t be_cpu = ggml_backend_cpu_init();
    struct ggml_init_params ip_w2 { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context* ctxW2 = ggml_init(ip_w2);
    ggml_tensor* W2 = ggml_new_tensor_2d(ctxW2, wtype, ne0, ne1);

    struct ggml_init_params ip_io2 { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context* ctxIO2 = ggml_init(ip_io2);
    ggml_tensor* B2 = ggml_new_tensor_2d(ctxIO2, GGML_TYPE_F32, ne0, M);
    ggml_tensor* C2 = ggml_mul_mat(ctxIO2, W2, B2);
    ggml_set_output(C2);

    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(ctxIO2, be_cpu);
    W2->buffer = buf2;
    W2->data   = hostW.data();   /* host bytes straight from the GGUF file   */
    ggml_backend_tensor_set(B2, hostB.data(), 0, hostB.size() * sizeof(float));

    struct ggml_init_params ip_g2 { ggml_graph_overhead(), nullptr, true };
    ggml_context* ctxG2 = ggml_init(ip_g2);
    ggml_cgraph* gf2 = ggml_new_graph(ctxG2);
    ggml_build_forward_expand(gf2, C2);
    if (ggml_backend_graph_compute(be_cpu, gf2) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cpu graph_compute failed\n"); return 1;
    }
    std::vector<float> outCPU((size_t)ne1 * M);
    ggml_backend_tensor_get(C2, outCPU.data(), 0, outCPU.size() * sizeof(float));

    /* ---- 6. compare ---------------------------------------------------- */
    double max_abs = 0, max_rel = 0, sum_ref = 0;
    size_t n = outGPU.size();
    for (size_t i = 0; i < n; ++i) {
        double a = outGPU[i], b = outCPU[i];
        double d = std::fabs(a - b);
        if (d > max_abs) max_abs = d;
        double denom = std::fabs(b) + 1e-6;
        if (d / denom > max_rel) max_rel = d / denom;
        sum_ref += std::fabs(b);
    }
    printf("=== numerical verification (GPU-via-VRAMER vs CPU-from-file) ===\n");
    printf("elements=%zu  mean|ref|=%.4f  max_abs_diff=%.3e  max_rel_diff=%.3e\n",
           n, sum_ref / n, max_abs, max_rel);
    printf("sample ref[0..3]=%.4f %.4f %.4f %.4f\n",
           outCPU[0], outCPU[1], outCPU[2], outCPU[3]);
    printf("sample gpu[0..3]=%.4f %.4f %.4f %.4f\n",
           outGPU[0], outGPU[1], outGPU[2], outGPU[3]);

    /* ---- VRAMER stats -------------------------------------------------- */
    vramer_stats st; memset(&st, 0, sizeof st);
    vramer_pager_stats(pager, &st);
    uint64_t demand = st.hits + st.misses;
    printf("=== VRAMER stats (proof the weight went through the pager) ===\n");
    printf("hits=%llu misses=%llu hit_rate=%.1f%% evictions=%llu prefetches=%llu\n",
           (unsigned long long)st.hits, (unsigned long long)st.misses,
           demand ? 100.0 * st.hits / demand : 0.0,
           (unsigned long long)st.evictions, (unsigned long long)st.prefetches);
    printf("bytes_h2d=%llu (%.2f MiB)  <- host->device bytes the pager moved\n",
           (unsigned long long)st.bytes_h2d, st.bytes_h2d / 1048576.0);

    const double tol = 5e-2;   /* quantized mul_mat: CUDA vs CPU accumulation */
    bool ok = (max_rel < tol) && (st.bytes_h2d >= w_size) && (max_abs < 1.0);
    printf("=== RESULT: %s ===\n", ok ? "PASS" : "FAIL");
    if (!ok) {
        printf("  (max_rel<%.2g? %d ; bytes_h2d>=weight? %d ; max_abs<1? %d)\n",
               tol, max_rel < tol, st.bytes_h2d >= w_size, max_abs < 1.0);
    }

    cudaStreamDestroy(stream);
    cudaFree(d_scratch);
    vramer_pager_destroy(pager);
    ggml_backend_free(be_cuda);
    ggml_backend_free(be_cpu);
    gguf_free(gctx);
    return ok ? 0 : 1;
}
