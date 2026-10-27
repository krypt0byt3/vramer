#!/bin/bash
# =============================================================================
# VRAMER x llama.cpp -- the "TOO BIG TO FIT" demo.
#
# This is the regime where VRAMER is not competing with full GPU offload, it is
# ENABLING something full offload cannot do at all: running a model whose weights
# are LARGER than the GPU's entire VRAM.
#
#   Model : Qwen2.5-32B-Instruct, Q4_K_M GGUF  (~18.5 GiB of weights)
#   GPU   : 1x NVIDIA A2, 15356 MiB (~15 GiB) VRAM  -- the model does NOT fit.
#
# We run the SAME model on the SAME GPU four ways and report decode tok/s, peak
# VRAM, and (for the paging rows) real H2D bytes + pager hit-rate:
#
#   (a) full-offload   -ngl 99             -> OOMs. The "impossible" baseline.
#   (b) native-partial -ngl K              -> largest K that fits; rest on CPU.
#                                             llama.cpp's honest native fallback.
#   (b') native-cpuoff -ngl 99 -ot FFN=CPU -> FFN weights on host, llama's own
#                                             per-token H2D copy (no VRAMER).
#   (c) vramer/<B>     -ngl 99 -ot FFN=CPU -> the SAME FFN set, but every H2D
#                                             upload served by the VRAMER pager
#                                             through a FIXED VRAM budget <B>
#                                             (well under 15 GiB, YOUR choice).
#
# VRAMER rows are diffed against the native-cpuoff output for byte-identical
# greedy generation (same math, same resident attention/embeddings).
#
# Every background VRAM sampler is tracked by PID and killed when its row ends.
# =============================================================================
set -u

# ---- config (override via env) ----------------------------------------------
GPU=${GPU:-0}                       # MUST be 0 here (GPU 1 is a parallel effort)
MODEL=${MODEL:-/tmp/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf}
LLAMA=${LLAMA:-/tmp/llama-vramer-batched/build/bin/llama-completion}
PROMPT=${PROMPT:-"The capital of France is"}
NPRED=${NPRED:-48}
CTX=${CTX:-2048}                    # fixed context (keeps KV cache modest & comparable)
NGL_FIT=${NGL_FIT:-49}              # largest -ngl that fits in ~15 GiB (probed: 50 OOMs on KV)
PAGE=${PAGE:-$((4*1024*1024))}      # 4 MiB page
REPEATS=${REPEATS:-2}               # repeat each timed row (VM tok/s varies)
FILTER='ffn_(down|gate|up)\.weight=CPU'
# VRAMER budgets to sweep (bytes). Default 4 GiB and 8 GiB -- both < 15 GiB.
BUDGETS=(${BUDGETS:-$((4*1024*1024*1024)) $((8*1024*1024*1024))})
OUT=${OUT:-/vramer/benchmarks/raw/bigmodel}
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=$GPU
SEED=1
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SUMMARY="$OUT/summary.txt"

# ---- VRAM sampler: tracked PID, always killed ------------------------------
SAMPLER_PID=""
start_vram() {   # $1 = outfile
    : > "$1"
    ( while :; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 \
            >> "$1" 2>/dev/null
        sleep 0.1
      done ) &
    SAMPLER_PID=$!
}
stop_vram() {
    [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null
    wait "$SAMPLER_PID" 2>/dev/null
    SAMPLER_PID=""
}
# kill any sampler if the script is interrupted
trap 'stop_vram' EXIT INT TERM

peak_of() { sort -n "$1" 2>/dev/null | tail -1; }

# ---- run one row: label, logbase, [extra llama args...]; VRAMER_* env by caller
run_row() {
    local label="$1" logbase="$2"; shift 2
    local err="$OUT/$logbase.err" out="$OUT/$logbase.out" vram="$OUT/$logbase.vram"
    start_vram "$vram"
    "$LLAMA" -m "$MODEL" -p "$PROMPT" -n "$NPRED" -c "$CTX" --temp 0 -no-cnv --seed "$SEED" \
        "$@" >"$out" 2>"$err"
    local rc=$?
    stop_vram
    local toks peak h2d hit
    toks=$(grep "eval time" "$err" | grep -v "prompt eval" \
           | grep -oE "[0-9.]+ tokens per second" | head -1 | grep -oE "^[0-9.]+")
    [ -z "$toks" ] && toks="n/a"
    peak=$(peak_of "$vram"); [ -z "$peak" ] && peak="n/a"
    h2d=$(grep -oE "bytes_h2d=[0-9.]+" "$err" | head -1 | sed 's/.*=//'); [ -z "$h2d" ] && h2d="-"
    hit=$(grep -oE "hit_rate=[0-9.]+%" "$err" | grep -oE "[0-9.]+" | head -1); [ -z "$hit" ] && hit="-"
    printf '%-18s rc=%-3s tok/s=%-8s peakVRAM=%-7s MiB  H2D=%-10s MiB  hit=%s%%\n' \
        "$label" "$rc" "$toks" "$peak" "$h2d" "$hit" | tee -a "$SUMMARY"
    # detect OOM / allocation failure in the log
    if [ "$rc" != "0" ]; then
        echo "    -> non-zero exit. failure tail:" | tee -a "$SUMMARY"
        grep -iE "out of memory|failed to allocate|cudaMalloc|ggml_backend|error" "$err" \
            | tail -4 | sed 's/^/       /' | tee -a "$SUMMARY"
    fi
    echo "$out"
}

{
echo "############################################################"
echo "# VRAMER 'too big to fit' demo   $STAMP"
echo "#   model   : $MODEL"
echo "#   size    : $(du -h "$MODEL" 2>/dev/null | cut -f1)  (weights > 15 GiB A2 VRAM)"
echo "#   gpu     : $GPU (A2, 15356 MiB)   n_predict=$NPRED   prompt=\"$PROMPT\""
echo "#   binary  : $LLAMA"
echo "############################################################"
} | tee "$SUMMARY"

# -----------------------------------------------------------------------------
echo | tee -a "$SUMMARY"
echo "=== (a) FULL OFFLOAD  -ngl 99  (expected: OOM -- the model does not fit) ===" | tee -a "$SUMMARY"
run_row "full-offload/ngl99" "a_full_ngl99" -ngl 99 >/dev/null

# -----------------------------------------------------------------------------
echo | tee -a "$SUMMARY"
echo "=== (b) NATIVE PARTIAL OFFLOAD  -ngl $NGL_FIT  (honest native fallback) ===" | tee -a "$SUMMARY"
for r in $(seq 1 "$REPEATS"); do
    run_row "native-partial/ngl$NGL_FIT (r$r)" "b_ngl${NGL_FIT}_r$r" -ngl "$NGL_FIT" >/dev/null
done

# -----------------------------------------------------------------------------
echo | tee -a "$SUMMARY"
echo "=== (b') NATIVE CPU-OFFLOAD  -ngl 99 -ot FFN=CPU  (llama's own H2D, no VRAMER) ===" | tee -a "$SUMMARY"
cpuoff_out=""
for r in $(seq 1 "$REPEATS"); do
    o=$(GGML_OP_OFFLOAD_MIN_BATCH=1 \
        run_row "native-cpuoff (r$r)" "bp_cpuoff_r$r" -ngl 99 -ot "$FILTER" | tail -1)
    cpuoff_out="$o"
done

# -----------------------------------------------------------------------------
echo | tee -a "$SUMMARY"
echo "=== (c) VRAMER-PAGED  -ngl 99 -ot FFN=CPU, FFN served through a FIXED budget ===" | tee -a "$SUMMARY"
for b in "${BUDGETS[@]}"; do
    mib=$((b/1024/1024))
    for r in $(seq 1 "$REPEATS"); do
        vout=$(VRAMER_GGUF="$MODEL" VRAMER_BUDGET="$b" VRAMER_PAGE="$PAGE" VRAMER_GPU=0 \
               GGML_OP_OFFLOAD_MIN_BATCH=1 \
               run_row "vramer/${mib}MiB (r$r)" "c_vramer${mib}_r$r" \
                       -ngl 99 -ot "$FILTER" | tail -1)
    done
    # correctness: last vramer run of this budget vs native cpu-offload
    if [ -n "$cpuoff_out" ] && [ -f "$vout" ]; then
        if diff -q "$cpuoff_out" "$vout" >/dev/null 2>&1; then
            echo "    [correctness] vramer/${mib}MiB output BYTE-IDENTICAL to native-cpuoff" | tee -a "$SUMMARY"
        else
            echo "    [correctness] vramer/${mib}MiB DIFFERS from native-cpuoff:" | tee -a "$SUMMARY"
            diff "$cpuoff_out" "$vout" | head -12 | tee -a "$SUMMARY"
        fi
    fi
    grep "\[VRAMER\]" "$OUT/c_vramer${mib}_r${REPEATS}.err" | sed 's/^/    /' | tee -a "$SUMMARY"
done

echo | tee -a "$SUMMARY"
echo "raw logs + summary under: $OUT" | tee -a "$SUMMARY"
# belt-and-suspenders: ensure no sampler leaked
stop_vram
