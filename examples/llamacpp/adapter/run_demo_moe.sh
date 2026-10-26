#!/bin/bash
# =============================================================================
# VRAMER x llama.cpp MoE EXPERT-PAGING demo.
#
# A Mixture-of-Experts model (Qwen1.5-MoE-A2.7B-Chat: 60 experts/layer, 4 active
# per token) is run with its big per-layer expert stacks
# (blk.N.ffn_{down,gate,up}_exps.weight, ~7.7 GiB of the ~8.8 GiB model) living
# in HOST memory. Their MUL_MAT_ID runs on the GPU, and the per-token expert
# upload is served by the VRAMER pager out of the GGUF through a fixed VRAM
# budget SMALLER than the expert set. Crucially, VRAMER pages ONLY the experts
# that are active for each token (read from the routing ids), so only a small
# fraction of the model streams per token -- the property that does NOT hold for
# a dense model.
#
# Runs three ways and reports VRAM, decode tok/s, output text, pager stats:
#   1. RESIDENT  : full GPU offload, everything in VRAM (VRAMER off).
#   2. PAGED     : experts on host, paged per token via VRAMER small budget.
#   (compare the two outputs for quality; compare tok/s for the MoE thesis.)
# =============================================================================
set -u

GPU=${GPU:-0}
MODEL=${MODEL:-/tmp/models/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf}
LLAMA=${LLAMA:-/tmp/llama-vramer/build/bin/llama-completion}
PROMPT=${PROMPT:-"Explain what a mixture-of-experts model is in two sentences."}
NPRED=${NPRED:-96}
BUDGET=${BUDGET:-$((512*1024*1024))}   # 512 MiB VRAM budget for paged experts
PAGE=${PAGE:-$((4*1024*1024))}         # 4 MiB page
FILTER='ffn_(down|gate|up)_exps\.weight=CPU'
OUT=${OUT:-/tmp/claude-0}
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=$GPU

echo "############################################################"
echo "# 1) RESIDENT baseline  (full GPU offload, VRAMER OFF)"
echo "############################################################"
"$LLAMA" -m "$MODEL" -ngl 99 -p "$PROMPT" -n "$NPRED" --temp 0 -no-cnv --seed 1 \
    2>"$OUT/moe_resident.err" | tee "$OUT/moe_resident.out"
echo
grep -iE "CUDA0 model buffer|model buffer size|tokens per second" "$OUT/moe_resident.err" | tail -6

echo
echo "############################################################"
echo "# 2) EXPERT-PAGED  (experts on host, paged via VRAMER)"
echo "#    budget=$((BUDGET/1024/1024)) MiB  page=$((PAGE/1024/1024)) MiB"
echo "############################################################"
VRAMER_GGUF="$MODEL" VRAMER_BUDGET="$BUDGET" VRAMER_PAGE="$PAGE" VRAMER_GPU=0 \
GGML_OP_OFFLOAD_MIN_BATCH=1 \
"$LLAMA" -m "$MODEL" -ngl 99 -ot "$FILTER" \
    -p "$PROMPT" -n "$NPRED" --temp 0 -no-cnv --seed 1 \
    2>"$OUT/moe_paged.err" | tee "$OUT/moe_paged.out"
echo
echo "----- VRAM model buffer (paged run) -----"
grep -iE "CUDA0 model buffer|model buffer size" "$OUT/moe_paged.err" | tail -6
echo "----- decode tok/s (paged run) -----"
grep -iE "tokens per second" "$OUT/moe_paged.err" | tail -3
echo "----- VRAMER paging stats -----"
grep "\[VRAMER\]" "$OUT/moe_paged.err"

echo
echo "############################################################"
echo "# OUTPUT QUALITY"
echo "############################################################"
if diff -q "$OUT/moe_resident.out" "$OUT/moe_paged.out" >/dev/null; then
    echo "MATCH: expert-paged generation is byte-identical to the resident baseline."
else
    echo "NOTE: outputs differ (see diff). For MoE, small routing/accumulation"
    echo "nondeterminism between a resident MUL_MAT_ID and a per-token gathered"
    echo "expert buffer can diverge after enough tokens; inspect coherence:"
    diff "$OUT/moe_resident.out" "$OUT/moe_paged.out" || true
fi
