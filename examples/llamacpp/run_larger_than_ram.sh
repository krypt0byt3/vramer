#!/bin/bash
# =============================================================================
# VRAMER x llama.cpp -- the "model LARGER THAN HOST RAM" demo.
#
# Runs a real LLM whose GGUF weights are LARGER than the machine's host RAM,
# paged on demand from the on-disk mmap'd GGUF through BOTH a bounded host-RAM
# ceiling (a cgroup memory cap) AND a bounded VRAM budget, with byte-identical
# output to a native reference.
#
#   Model : single-file GGUF whose on-disk size EXCEEDS host RAM (see MODEL).
#           e.g. Qwen2.5-72B-Instruct F16 (~135.9 GiB) on a 78 GiB-RAM box.
#   GPU   : 1x NVIDIA A2, 15356 MiB  (CUDA_VISIBLE_DEVICES=0)
#
# Three configs, same prompt / seed / n-predict, greedy (--temp 0):
#
#   (1) native-cpu      -ngl 0                 : honesty control. Plain
#                                                llama.cpp, default mmap, no
#                                                VRAMER, all matmuls on CPU. Does
#                                                native run a >RAM model at all?
#   (2) native-cpuoff   -ngl 99 -ot FILTER=CPU : BYTE REFERENCE. Big weights on
#                                                host (mmap), their matmuls on
#                                                GPU via llama's own per-split
#                                                H2D copy. VRAMER OFF.
#   (3) vramer/<B>      -ngl 99 -ot FILTER=CPU : the SAME offload, but every H2D
#                       + VRAMER_* + RAM cap     weight upload is served by the
#                                                VRAMER pager out of the GGUF
#                                                through a fixed VRAM budget <B>,
#                                                the whole process run under a
#                                                cgroup host-RAM cap <<model.
#
# (3) output is diffed against (2): same GPU math, identical delivered bytes =>
# byte-identical generation. The headline is the bounded footprint (host RAM
# capped at CAP, VRAM capped at the budget), NOT speed -- past the RAM boundary
# this tier is DISK-BOUND on a SATA SSD and slow by design.
#
# Every background sampler is tracked by PID and killed when its row ends.
# =============================================================================
set -u

# ---- config (override via env) ----------------------------------------------
GPU=${GPU:-0}                        # MUST be 0 (GPU 1 is a parallel effort)
MODEL=${MODEL:?set MODEL to a single-file GGUF larger than host RAM}
LLAMA=${LLAMA:-/tmp/llama-vramer-ltr/build/bin/llama-completion}
PROMPT=${PROMPT:-"The capital of France is"}
NPRED=${NPRED:-8}                    # decode tokens (keep small: disk-bound tier)
NPRED_CPU=${NPRED_CPU:-4}            # fewer for the pure-CPU control (slowest)
CTX=${CTX:-512}                      # small context => modest KV, comparable
PAGE=${PAGE:-$((4*1024*1024))}       # 4 MiB page
SEED=${SEED:-1}
# big-matrix weights routed to host so their matmuls fire the copy hook. For a
# 72B on a 15 GiB card the attention + output projections must ALSO be paged
# (FFN alone leaves ~24 GiB of attention resident -> will not fit the card).
FILTER=${FILTER:-'(ffn_(down|gate|up)|attn_(q|k|v|output))\.weight=CPU'}
# VRAMER budgets to sweep (bytes). Default just 4 GiB (add 8 GiB via env).
BUDGETS=(${BUDGETS:-$((4*1024*1024*1024))})
# host-RAM cap for the VRAMER row (cgroup memory.max). Must exceed the process's
# anonymous working set (KV + compute + pinned staging + CUDA host) but be far
# below the model size. 0 disables the cap.
RAMCAP=${RAMCAP:-24G}
PHASES=${PHASES:-"native_cpu native_cpuoff vramer"}   # which rows to run
OUT=${OUT:-/vramer/benchmarks/raw/llama_ltr}
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=$GPU
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SUMMARY="$OUT/summary.txt"
MODEL_BYTES=$(stat -c%s "$MODEL")
RAM_BYTES=$(awk '/MemTotal/{print $2*1024}' /proc/meminfo)

# ---- samplers: tracked PIDs, always killed ---------------------------------
VPID=""; IPID=""; FPID=""
start_samplers() {  # $1 = logbase
    : > "$OUT/$1.vram"; : > "$OUT/$1.iostat"; : > "$OUT/$1.free"
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 >> "$OUT/$1.vram" 2>/dev/null; sleep 0.2; done ) & VPID=$!
    ( iostat -x sda 2 >> "$OUT/$1.iostat" 2>/dev/null ) & IPID=$!
    ( while :; do date +%s | tr '\n' ' '; free -m | awk '/Mem:/{print "used="$3" free="$4" buffcache="$6" avail="$7}'; sleep 1; done >> "$OUT/$1.free" 2>/dev/null ) & FPID=$!
}
stop_samplers() {
    for p in "$VPID" "$IPID" "$FPID"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
    wait "$VPID" "$IPID" "$FPID" 2>/dev/null; VPID=""; IPID=""; FPID=""
}
trap 'stop_samplers' EXIT INT TERM
peak_of() { sort -n "$1" 2>/dev/null | tail -1; }

# ---- run one row ------------------------------------------------------------
# run_row LABEL LOGBASE NPRED USE_CAP -- <llama args...>   (VRAMER_* env by caller)
run_row() {
    local label="$1" logbase="$2" npred="$3" usecap="$4"; shift 4
    [ "$1" = "--" ] && shift
    local err="$OUT/$logbase.err" out="$OUT/$logbase.out" tm="$OUT/$logbase.time"
    echo ">>> $label  (npred=$npred cap=$usecap)  $(date -u +%H:%M:%S)" | tee -a "$SUMMARY"
    start_samplers "$logbase"
    local pre=( /usr/bin/time -v )
    if [ "$usecap" != "0" ]; then
        pre=( systemd-run --scope -q -p MemoryMax="$usecap" -p MemorySwapMax=0 /usr/bin/time -v )
    fi
    # drop caches so every row starts cold (fair, and forces the disk tier)
    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    "${pre[@]}" \
        "$LLAMA" -m "$MODEL" -p "$PROMPT" -n "$npred" -c "$CTX" --temp 0 -no-cnv --seed "$SEED" \
        "$@" >"$out" 2>"$err"
    local rc=$?
    stop_samplers
    # extract metrics
    local toks peakvram peakrss capkb
    toks=$(grep "eval time" "$err" | grep -v "prompt eval" | grep -oE "[0-9.]+ tokens per second" | head -1 | grep -oE "^[0-9.]+")
    [ -z "$toks" ] && toks="n/a"
    peakvram=$(peak_of "$OUT/$logbase.vram"); [ -z "$peakvram" ] && peakvram="n/a"
    peakrss=$(grep "Maximum resident set size" "$err" | grep -oE "[0-9]+" | head -1)
    [ -z "$peakrss" ] && peakrss="n/a"
    printf '    rc=%s  tok/s=%s  peakVRAM=%s MiB  peakRSS=%s KiB (%.2f GiB)\n' \
        "$rc" "$toks" "$peakvram" "$peakrss" "$(awk "BEGIN{print ${peakrss:-0}/1048576}")" | tee -a "$SUMMARY"
    if [ "$rc" != 0 ]; then
        echo "    NONZERO EXIT. tail:" | tee -a "$SUMMARY"
        grep -iE "oom|out of memory|killed|failed to allocate|cudaMalloc|error" "$err" | tail -4 | sed 's/^/      /' | tee -a "$SUMMARY"
    fi
    grep "\[VRAMER\]" "$err" | sed 's/^/    /' | tee -a "$SUMMARY"
    echo "$out"
}

{
echo "############################################################"
echo "# VRAMER 'model larger than host RAM' demo   $STAMP"
echo "#   model : $MODEL"
echo "#   size  : $MODEL_BYTES B ($(awk "BEGIN{printf \"%.1f\", $MODEL_BYTES/2^30}") GiB)"
echo "#   RAM   : $RAM_BYTES B ($(awk "BEGIN{printf \"%.1f\", $RAM_BYTES/2^30}") GiB)  -> model > RAM: $([ $MODEL_BYTES -gt $RAM_BYTES ] && echo YES || echo NO)"
echo "#   gpu   : $GPU   npred=$NPRED ctx=$CTX prompt=\"$PROMPT\"  ramcap=$RAMCAP"
echo "#   binary: $LLAMA"
echo "############################################################"
} | tee "$SUMMARY"

ref_out=""
for ph in $PHASES; do
  case "$ph" in
    native_cpu)
      echo | tee -a "$SUMMARY"; echo "=== (1) NATIVE CPU  -ngl 0  (honesty control: can native run a >RAM model?) ===" | tee -a "$SUMMARY"
      run_row "native-cpu/ngl0" "n1_native_cpu" "$NPRED_CPU" 0 -- -ngl 0 >/dev/null ;;
    native_cpuoff)
      echo | tee -a "$SUMMARY"; echo "=== (2) NATIVE CPU-OFFLOAD  -ngl 99 -ot FILTER=CPU  (byte REFERENCE, VRAMER off) ===" | tee -a "$SUMMARY"
      ref_out=$(GGML_OP_OFFLOAD_MIN_BATCH=1 run_row "native-cpuoff" "n2_native_cpuoff" "$NPRED" 0 -- -ngl 99 -ot "$FILTER" | tail -1) ;;
    vramer)
      for b in "${BUDGETS[@]}"; do
        mib=$((b/1024/1024))
        echo | tee -a "$SUMMARY"; echo "=== (3) VRAMER  budget=${mib}MiB  host-RAM cap=$RAMCAP  (headline: bounded footprint, >RAM) ===" | tee -a "$SUMMARY"
        vout=$(VRAMER_GGUF="$MODEL" VRAMER_BUDGET="$b" VRAMER_PAGE="$PAGE" VRAMER_GPU=0 GGML_OP_OFFLOAD_MIN_BATCH=1 \
               run_row "vramer/${mib}MiB" "v3_vramer_${mib}" "$NPRED" "$RAMCAP" -- -ngl 99 -ot "$FILTER" | tail -1)
        if [ -n "$ref_out" ] && [ -f "$vout" ] && [ -f "$ref_out" ]; then
          if diff -q "$ref_out" "$vout" >/dev/null 2>&1; then
            echo "    [correctness] vramer/${mib}MiB output BYTE-IDENTICAL to native-cpuoff reference" | tee -a "$SUMMARY"
          else
            echo "    [correctness] vramer/${mib}MiB DIFFERS from reference:" | tee -a "$SUMMARY"
            diff "$ref_out" "$vout" | head -16 | sed 's/^/      /' | tee -a "$SUMMARY"
          fi
        fi
      done ;;
  esac
done

echo | tee -a "$SUMMARY"
echo "raw logs under: $OUT" | tee -a "$SUMMARY"
stop_samplers
