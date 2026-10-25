/*
 * vramer_ggml_layer.cpp — P3 (stretch): LAYER-GRANULAR residency.
 *
 * Pages the FULL set of 2-D weight tensors of ONE real transformer block
 * ("blk.0.*") of a GGUF model through a SINGLE fixed VRAM budget that is
 * smaller than the layer's total weight size, and runs a REAL ggml_mul_mat on
 * the CUDA backend for every one of those weights — twice, to simulate two
 * tokens' worth of the cyclic per-layer scan. This is Strategy (b) from
 * docs/LLAMACPP_ADAPTER.md (acquire / hold / release per tensor, prefetch the
 * next, MRU eviction on the cyclic scan), exercised on actual model weights.
 *
 * What it proves:
 *   - every weight of a transformer layer can be served from VRAMER through a
 *     budget < the layer, feeding genuine ggml CUDA kernels;
 *   - pass-1 results are numerically verified against a CPU reference that reads
 *     the same bytes straight from the file;
 *   - the second pass (next "token") shows VRAMER hits from residency — the MRU
 *     cyclic-scan behavior the adapter design relies on.
 *
 * What it is NOT (honest scope): this does not yet hook llama.cpp's own token
 * loop / llama_model_loader. It drives the exact VRAMER call sequence an adapter
 * would run per layer, on the real layer tensors, with real kernels — the
 * missing piece for a full adapter is a custom ggml_backend_buffer_type whose
 * tensors' ->data are patched to lease devptrs inside the scheduler hook (§2,§4).
 *
 * CONTIGUITY: same as the P2 demo — each weight's pages are D2D-gathered into a
 * per-tensor contiguous scratch buffer after acquire, so the budget can be far
 * smaller than any single tensor or the whole layer.
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
#define V_FAIL(what) do { fprintf(stderr, "VRAMER %s: %s\n", (what), err); std::exit(1);} while(0)
#define CU_OK(e,w) do{ cudaError_t _e=(e); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",(w),cudaGetErrorString(_e));std::exit(1);} }while(0)
static uint64_t round_up(uint64_t x,uint64_t a){return (x+a-1)/a*a;}

struct WInfo { std::string name; ggml_type type; int64_t ne0, ne1; uint64_t off, size; };

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <gguf> [layer_prefix]\n", argv[0]); return 2; }
    const char* path = argv[1];
    const char* prefix = (argc >= 3) ? argv[2] : "blk.0.";

    int ndev=0; if (cudaGetDeviceCount(&ndev)!=cudaSuccess||ndev<1){printf("no CUDA device\n");return 0;}
    CU_OK(cudaSetDevice(0), "setDevice");

    struct gguf_init_params gp; gp.no_alloc=true; gp.ctx=nullptr;
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx){ fprintf(stderr,"gguf open failed\n"); return 1; }
    const size_t data_off = gguf_get_data_offset(gctx);
    const int64_t nt = gguf_get_n_tensors(gctx);

    std::vector<WInfo> ws;
    uint64_t layer_bytes = 0;
    for (int64_t i=0;i<nt;++i){
        const char* nm = gguf_get_tensor_name(gctx, i);
        if (strncmp(nm, prefix, strlen(prefix)) != 0) continue;
        const int64_t* ne = gguf_get_tensor_ne(gctx, i);
        if (ne[2]!=1||ne[3]!=1||ne[0]<2||ne[1]<2) continue;  /* 2-D weights only */
        ggml_type t = gguf_get_tensor_type(gctx, i);
        if (ne[0] % ggml_blck_size(t) != 0) continue;
        WInfo w; w.name=nm; w.type=t; w.ne0=ne[0]; w.ne1=ne[1];
        w.off=(uint64_t)data_off+gguf_get_tensor_offset(gctx,i);
        w.size=(uint64_t)gguf_get_tensor_size(gctx,i);
        ws.push_back(w); layer_bytes += w.size;
    }
    if (ws.empty()){ fprintf(stderr,"no 2-D weights with prefix %s\n", prefix); return 1; }

    printf("=== layer '%s': %zu 2-D weight tensors, total %.2f MiB ===\n",
           prefix, ws.size(), layer_bytes/1048576.0);
    for (auto& w: ws)
        printf("  %-28s %-7s [%5lld x %6lld] %.2f MiB\n", w.name.c_str(),
               ggml_type_name(w.type), (long long)w.ne0,(long long)w.ne1, w.size/1048576.0);

    /* one shared budget, genuinely SMALLER than the layer's total weight set */
    uint64_t page_size = 4ull<<20;
    uint64_t budget    = round_up(layer_bytes*3/4, page_size);  /* < layer total */
    if (budget >= layer_bytes) budget = layer_bytes - page_size;
    if (budget < page_size)    budget = page_size;

    vramer_source* src = vramer_source_file(path, err, sizeof err);
    if(!src) V_FAIL("source_file");
    vramer_config cfg; memset(&cfg,0,sizeof cfg);
    cfg.gpu=0; cfg.vram_budget=budget; cfg.page_size=page_size;
    cfg.staging_buffers=2; cfg.prefetch_workers=1; cfg.eviction=VRAMER_EVICT_MRU;
    vramer_pager* pager = vramer_pager_create(src,&cfg,err,sizeof err);
    if(!pager) V_FAIL("pager_create");
    vramer_source_free(src);
    printf("=== pager: page=%.0f MiB budget=%.0f MiB (%llu slots) < layer %.0f MiB ===\n",
           page_size/1048576.0, budget/1048576.0,
           (unsigned long long)vramer_pager_resident_capacity_pages(pager),
           layer_bytes/1048576.0);

    ggml_backend_t be_cuda = ggml_backend_cuda_init(0);
    ggml_backend_t be_cpu  = ggml_backend_cpu_init();
    cudaStream_t stream; CU_OK(cudaStreamCreate(&stream),"stream");

    const int64_t M = 4;
    auto run_mulmat = [&](ggml_backend_t be, ggml_type type, int64_t ne0, int64_t ne1,
                          void* wdata, const std::vector<float>& B, std::vector<float>& out){
        struct ggml_init_params ipw{2*ggml_tensor_overhead(),nullptr,true};
        ggml_context* cw=ggml_init(ipw);
        ggml_tensor* W=ggml_new_tensor_2d(cw,type,ne0,ne1);
        struct ggml_init_params ipio{4*ggml_tensor_overhead(),nullptr,true};
        ggml_context* cio=ggml_init(ipio);
        ggml_tensor* Bt=ggml_new_tensor_2d(cio,GGML_TYPE_F32,ne0,M);
        ggml_tensor* C =ggml_mul_mat(cio,W,Bt); ggml_set_output(C);
        ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(cio,be);
        W->buffer=buf; W->data=wdata;
        ggml_backend_tensor_set(Bt,B.data(),0,B.size()*sizeof(float));
        struct ggml_init_params ipg{ggml_graph_overhead(),nullptr,true};
        ggml_context* cg=ggml_init(ipg);
        ggml_cgraph* gf=ggml_new_graph(cg); ggml_build_forward_expand(gf,C);
        ggml_backend_graph_compute(be,gf);
        out.resize((size_t)ne1*M);
        ggml_backend_tensor_get(C,out.data(),0,out.size()*sizeof(float));
        ggml_free(cg); ggml_free(cio); ggml_free(cw); ggml_backend_buffer_free(buf);
    };

    /* gather one weight's pages (D2D) into a contiguous device scratch */
    auto gather = [&](const WInfo& w, void* d_scratch){
        uint64_t fp=w.off/page_size, lp=(w.off+w.size-1)/page_size;
        for(uint64_t pg=fp; pg<=lp; ++pg){
            if(pg+1<=lp) vramer_pager_prefetch_async(pager,pg+1,err,sizeof err);
            vramer_lease* ls=nullptr; void* dp=nullptr; size_t nb=0;
            if(vramer_acquire(pager,pg,&ls,&dp,&nb,err,sizeof err)) V_FAIL("acquire");
            if(vramer_lease_wait_on(ls,(void*)stream,err,sizeof err)) V_FAIL("wait_on");
            uint64_t ps=pg*page_size;
            uint64_t s0=(w.off>ps)?w.off:ps;
            uint64_t s1=(w.off+w.size<ps+page_size)?(w.off+w.size):(ps+page_size);
            CU_OK(cudaMemcpyAsync((char*)d_scratch+(s0-w.off),(const char*)dp+(s0-ps),
                                  s1-s0,cudaMemcpyDeviceToDevice,stream),"D2D");
            if(vramer_lease_release_on(ls,(void*)stream,err,sizeof err)) V_FAIL("release");
        }
        CU_OK(cudaStreamSynchronize(stream),"gather sync");
    };

    int n_pass = 2, verified = 0, failed = 0;
    vramer_stats prev; memset(&prev,0,sizeof prev);
    for (int pass=0; pass<n_pass; ++pass){
        printf("\n--- pass %d (\"token %d\": cyclic scan of layer weights) ---\n", pass, pass);
        for (auto& w: ws){
            std::vector<float> B((size_t)w.ne0*M);
            for(size_t k=0;k<B.size();++k) B[k]=0.001f*((float)((k*7+pass*3)%101)-50.f);

            void* d_scratch=nullptr; CU_OK(cudaMalloc(&d_scratch,w.size),"scratch");
            gather(w, d_scratch);
            std::vector<float> gpu;
            run_mulmat(be_cuda, w.type, w.ne0, w.ne1, d_scratch, B, gpu);
            cudaFree(d_scratch);

            if (pass==0){ /* verify vs CPU-from-file */
                std::vector<uint8_t> hw(w.size);
                int fd=open(path,O_RDONLY); size_t done=0;
                while(done<w.size){ssize_t r=pread(fd,hw.data()+done,w.size-done,w.off+done);
                    if(r<=0){perror("pread");return 1;} done+=r;} close(fd);
                std::vector<float> cpu;
                run_mulmat(be_cpu, w.type, w.ne0, w.ne1, hw.data(), B, cpu);
                /* robust metric: relative L2 (Frobenius) norm ||gpu-cpu||/||cpu||.
                 * per-element relative error is meaningless where ref ~ 0, which
                 * is common in matmul outputs. sub-% L2 error is expected for
                 * quantized matmul (ggml quantizes activations to 8-bit). */
                double num=0,den=0,ma=0; for(size_t i=0;i<cpu.size();++i){
                    double d=(double)gpu[i]-cpu[i]; num+=d*d; den+=(double)cpu[i]*cpu[i];
                    if(std::fabs(d)>ma) ma=std::fabs(d);}
                double rel_l2 = den>0 ? std::sqrt(num/den) : 0.0;
                bool ok=(rel_l2<2e-2)&&(ma<0.5);
                printf("  %-28s verify max_abs=%.2e rel_L2=%.2e %s\n",
                       w.name.c_str(), ma, rel_l2, ok?"OK":"MISMATCH");
                ok?++verified:++failed;
            } else {
                printf("  %-28s recomputed (gpu[0]=%.4f)\n", w.name.c_str(), gpu[0]);
            }
        }
        vramer_pager_drain_prefetch_queue(pager,err,sizeof err);
        vramer_pager_synchronize(pager,err,sizeof err);
        vramer_stats st; memset(&st,0,sizeof st); vramer_pager_stats(pager,&st);
        printf("  [pass %d delta] hits=+%llu misses=+%llu evictions=+%llu bytes_h2d=+%.1f MiB\n",
               pass,(unsigned long long)(st.hits-prev.hits),
               (unsigned long long)(st.misses-prev.misses),
               (unsigned long long)(st.evictions-prev.evictions),
               (st.bytes_h2d-prev.bytes_h2d)/1048576.0);
        prev=st;
    }

    vramer_stats st; memset(&st,0,sizeof st); vramer_pager_stats(pager,&st);
    uint64_t dem=st.hits+st.misses;
    printf("\n=== P3 summary ===\n");
    printf("weights verified vs CPU: %d ok / %d fail\n", verified, failed);
    printf("VRAMER total: hits=%llu misses=%llu hit_rate=%.1f%% evictions=%llu bytes_h2d=%.1f MiB\n",
           (unsigned long long)st.hits,(unsigned long long)st.misses,
           dem?100.0*st.hits/dem:0.0,(unsigned long long)st.evictions, st.bytes_h2d/1048576.0);
    printf("=== RESULT: %s ===\n", (failed==0)?"PASS":"FAIL");

    cudaStreamDestroy(stream);
    vramer_pager_destroy(pager);
    ggml_backend_free(be_cuda); ggml_backend_free(be_cpu); gguf_free(gctx);
    return failed==0?0:1;
}
