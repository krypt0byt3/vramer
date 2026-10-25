// VRAMER <-> llama.cpp host-weight paging hook.
//
// Minimal injection point: llama.cpp's scheduler, when a WEIGHT tensor lives in
// host memory but its op runs on a GPU backend, copies the weight H2D into a
// bounded per-split device buffer before the op (see
// ggml_backend_sched_copy_input + callback_copy in ggml-backend.cpp). This hook
// REPLACES that default H2D copy with a VRAMER paged read of the SAME bytes out
// of the GGUF file, through a fixed VRAM budget SMALLER than the paged weight
// set. The bytes delivered are identical, so generated text is unchanged, but
// the weights never reside in VRAM in full -- they are paged on demand.
//
// Enabled only when env VRAMER_GGUF is set (otherwise the hook is a no-op and
// llama.cpp behaves exactly as upstream).

#pragma once

struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;

#ifdef __cplusplus
extern "C" {
#endif

// Returns 1 if this host weight was served from VRAMER (caller must NOT do the
// default copy), 0 otherwise (caller falls through to its normal path).
int vramer_llama_copy_hook(ggml_backend_t backend,
                           const struct ggml_tensor * src,
                           struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
