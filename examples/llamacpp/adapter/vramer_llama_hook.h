// VRAMER <-> llama.cpp host-weight paging hook.
//
// Minimal injection point: llama.cpp's scheduler, when a WEIGHT tensor lives in
// host memory but its op runs on a GPU backend, copies the weight H2D into a
// bounded per-split device buffer before the op (see
// ggml_backend_sched_copy_input + callback_copy in ggml-backend.cpp, routed
// through llama_context::sched_copy_experts). This hook REPLACES that default
// H2D copy with a VRAMER paged read of the SAME bytes out of the GGUF file,
// through a fixed VRAM budget SMALLER than the paged weight set.
//
// Two cases, both correct (bytes delivered are identical to the file):
//   * Dense 2D weight (e.g. ffn_down.weight): the whole tensor is paged.
//   * MoE 3D expert stack (e.g. ffn_down_exps.weight): ONLY the experts that
//     are active for this token -- read from the MUL_MAT_ID routing ids in the
//     split graph -- are paged. This is the property that lets a MoE model keep
//     most of its weight on host while only the ~active fraction streams per
//     token.
//
// Enabled only when env VRAMER_GGUF is set (otherwise the hook is a no-op and
// llama.cpp behaves exactly as upstream).

#pragma once

struct ggml_tensor;
struct ggml_cgraph;
typedef struct ggml_backend * ggml_backend_t;

#ifdef __cplusplus
extern "C" {
#endif

// Returns 1 if this host weight was served from VRAMER (caller must NOT do the
// default copy), 0 otherwise (caller falls through to its normal path).
// `graph` is the split's cgraph (may be null); it is used to recover the active
// expert ids for MoE expert stacks. Passing null forces whole-tensor paging.
int vramer_llama_copy_hook(ggml_backend_t backend,
                           const struct ggml_tensor * src,
                           struct ggml_tensor * dst,
                           struct ggml_cgraph * graph);

#ifdef __cplusplus
}
#endif
