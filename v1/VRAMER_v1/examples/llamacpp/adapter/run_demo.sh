#!/bin/bash
# =============================================================================
# VRAMER x llama.cpp end-to-end weight-paging demo.
#
# Runs the SAME greedy prompt twice through a forked llama.cpp:
#   1. BASELINE  : full GPU offload, weights resident in VRAM (VRAMER off).
#   2. VRAMER    : the FFN weights live in host memory (-ot ...=CPU), their
#                  mul_mats run on the GPU, and every H2D weight upload is served
#                  by the VRAMER pager out of the GGUF through a fixed VRAM budget
#                  that is SMALLER than the paged weight set.
# Prints both generations + the VRAMER paging stats so the paging is evident.
#
# The only fork change to llama.cpp is a ~5 line call into src/vramer_llama_hook.*
# from llama_context::sched_copy_experts (the existing host-weight copy callback).
# =============================================================================
set -u

# ---- config (override via env) ----------------------------------------------
GPU=${GPU:-1}                      # free GPU (check nvidia-smi)
MODEL=${MODEL:-/tmp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}
LLAMA=${LLAMA:-/tmp/llama-vramer/build/bin/llama-completion}
PROMPT=${PROMPT:-"The capital of France is"}
NPRED=${NPRED:-32}
BUDGET=${BUDGET:-$((32*1024*1024))}   # 32 MiB VRAM budget for paged weights
PAGE=${PAGE:-$((4*1024*1024))}        # 4 MiB page
FILTER='ffn_(down|gate|up)\.weight=CPU'
OUT=${OUT:-/tmp/claude-0}
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=$GPU

echo "############################################################"
echo "# 1) BASELINE  (full GPU offload, VRAMER OFF)"
echo "############################################################"
"$LLAMA" -m "$MODEL" -ngl 99 -p "$PROMPT" -n "$NPRED" --temp 0 -no-cnv --seed 1 \
    2>"$OUT/demo_baseline.err" | tee "$OUT/demo_baseline.out"
echo

echo "############################################################"
echo "# 2) VRAMER-PAGED  (FFN weights paged from GGUF via VRAMER)"
echo "#    budget=$((BUDGET/1024/1024)) MiB  page=$((PAGE/1024/1024)) MiB"
echo "############################################################"
VRAMER_GGUF="$MODEL" VRAMER_BUDGET="$BUDGET" VRAMER_PAGE="$PAGE" VRAMER_GPU=0 \
GGML_OP_OFFLOAD_MIN_BATCH=1 \
"$LLAMA" -m "$MODEL" -ngl 99 -ot "$FILTER" \
    -p "$PROMPT" -n "$NPRED" --temp 0 -no-cnv --seed 1 \
    2>"$OUT/demo_vramer.err" | tee "$OUT/demo_vramer.out"
echo
echo "----- VRAMER paging stats -----"
grep "\[VRAMER\]" "$OUT/demo_vramer.err"
echo
echo "############################################################"
echo "# RESULT"
echo "############################################################"
if diff -q "$OUT/demo_baseline.out" "$OUT/demo_vramer.out" >/dev/null; then
    echo "MATCH: VRAMER-paged generation is byte-identical to the baseline."
else
    echo "DIFFERENCE between baseline and VRAMER output:"
    diff "$OUT/demo_baseline.out" "$OUT/demo_vramer.out" || true
fi
