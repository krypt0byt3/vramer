#!/bin/bash
# =============================================================================
# VRAMER comparative benchmark driver (Roadmap #5).
#
# Produces an honest, reproducible comparison of four ways to run the SAME model
# on the SAME GPU, with identical prompt / greedy decode / token count:
#
#   resident     full GPU offload, every weight in VRAM            (100% baseline)
#   ngl=K        llama.cpp native PARTIAL offload (K layers on GPU, rest on CPU)
#   cpu-offload  the SAME tensors kept on host with `-ot ...=CPU`, llama.cpp's
#                OWN per-token host->device copy (VRAMER NOT involved)
#   vramer/<B>   the SAME `-ot ...=CPU` tensor set, but every host->device upload
#                is served by the VRAMER pager through a fixed VRAM budget <B>
#                that is SMALLER than the paged set
#
# Per row we collect: decode tok/s (llama_perf eval line), peak VRAM on the GPU
# (nvidia-smi sampled), and for paging rows the real H2D bytes + pager hit-rate
# (from the adapter's [VRAMER] exit summary). VRAMER rows are diffed against the
# resident baseline for byte-identical greedy output.
#
# This script ONLY measures. It does not touch the adapter or the VRAMER library.
#
# Usage:  bash benchmarks/run_comparison.sh [dense|moe|both]
# Env:    GPU=0  NPRED=64  OUTDIR=benchmarks/raw
# =============================================================================
set -u

GPU=${GPU:-0}
NPRED=${NPRED:-64}
WHICH=${1:-both}
export CUDA_VISIBLE_DEVICES=$GPU

UP=${UP:-/tmp/llama.cpp/build/bin/llama-completion}            # upstream (baselines)
FK=${FK:-/tmp-llama-vramer-bench/build/bin/llama-completion}   # placeholder, overridden below
FK=${FK_OVERRIDE:-/tmp/llama-vramer-bench/build/bin/llama-completion}

DENSE_MODEL=/tmp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
MOE_MODEL=/tmp/models/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf

PROMPT=${PROMPT:-"The capital of France is"}
SEED=1
PAGE=${PAGE:-$((4*1024*1024))}

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUTDIR=${OUTDIR:-$HERE/raw}
mkdir -p "$OUTDIR"

STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SUMMARY="$OUTDIR/comparison.txt"

# ---------------------------------------------------------------------------
# sample_peak_vram <pidfile> <outfile> : background sampler of GPU mem.used
# ---------------------------------------------------------------------------
sample_vram() {
    local out="$1"
    : > "$out"
    while :; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 \
            >> "$out" 2>/dev/null
        sleep 0.1
    done
}

# ---------------------------------------------------------------------------
# run_row <label> <binary> <model> <logbase> <extra llama args...>
#   VRAMER_* env (if any) must be exported by the caller.
# Emits one row to stdout:  label \t tok/s \t peakVRAM \t h2d \t hitrate \t served
# ---------------------------------------------------------------------------
run_row() {
    local label="$1" bin="$2" model="$3" logbase="$4"; shift 4
    local err="$OUTDIR/$logbase.err" out="$OUTDIR/$logbase.out" vram="$OUTDIR/$logbase.vram"

    sample_vram "$vram" &
    local sampler=$!

    "$bin" -m "$model" -p "$PROMPT" -n "$NPRED" --temp 0 -no-cnv --seed "$SEED" \
        "$@" >"$out" 2>"$err"
    local rc=$?

    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null

    local toks peak h2d hit served
    # DECODE tok/s only: the "eval time" line, NOT "prompt eval time".
    toks=$(grep "eval time" "$err" | grep -v "prompt eval" | grep -oE "[0-9.]+ tokens per second" | head -1 | grep -oE "^[0-9.]+")
    [ -z "$toks" ] && toks="n/a"
    peak=$(sort -n "$vram" 2>/dev/null | tail -1)
    [ -z "$peak" ] && peak="n/a"
    h2d=$(grep -oE "bytes_h2d=[0-9.]+" "$err" | head -1 | sed 's/.*=//')
    [ -z "$h2d" ] && h2d="-"
    hit=$(grep -oE "hit_rate=[0-9.]+%" "$err" | grep -oE "[0-9.]+" | head -1)
    [ -z "$hit" ] && hit="-"
    served=$(grep -oE "bytes streamed \(active only\): [0-9.]+ MiB" "$err" | grep -oE "[0-9.]+" | head -1)
    [ -z "$served" ] && served="-"

    printf '%-16s rc=%s  tok/s=%-8s peakVRAM=%-7s MiB  H2D=%-9s MiB  hit=%-6s%%  streamed=%s MiB\n' \
        "$label" "$rc" "$toks" "$peak" "$h2d" "$hit" "$served" >&2
    # machine-readable line for the table builder
    printf 'ROW\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$rc" "$toks" "$peak" "$h2d" "$hit" "$served" >> "$OUTDIR/$logbase.row"
    echo "$out"   # return the stdout path (for diffing)
}

bench_model() {
    local tag="$1" model="$2" filter="$3" ngl_a="$4" ngl_b="$5"; shift 5
    local budgets=("$@")   # list of VRAMER budgets in bytes

    echo
    echo "############################################################"
    echo "# MODEL: $tag   ($model)"
    echo "# filter(host tensors): $filter"
    echo "############################################################"

    # 1. resident baseline (warm: run once discarded, then measured)
    "$UP" -m "$model" -p "$PROMPT" -n 8 --temp 0 -no-cnv --seed "$SEED" -ngl 99 \
        >/dev/null 2>&1   # warmup
    local resident_out
    resident_out=$(run_row "resident" "$UP" "$model" "${tag}_resident" -ngl 99 | tail -1)

    # 2. native partial offload, two K points
    run_row "ngl=$ngl_a" "$UP" "$model" "${tag}_ngl${ngl_a}" -ngl "$ngl_a" >/dev/null
    run_row "ngl=$ngl_b" "$UP" "$model" "${tag}_ngl${ngl_b}" -ngl "$ngl_b" >/dev/null

    # 3. llama.cpp own CPU-offload of the SAME tensor set (no VRAMER)
    GGML_OP_OFFLOAD_MIN_BATCH=1 \
    run_row "cpu-offload" "$UP" "$model" "${tag}_cpuoff" -ngl 99 -ot "$filter" >/dev/null

    # 4. VRAMER-paged, several budgets
    for b in "${budgets[@]}"; do
        local mib=$((b/1024/1024))
        local vout
        vout=$(VRAMER_GGUF="$model" VRAMER_BUDGET="$b" VRAMER_PAGE="$PAGE" VRAMER_GPU=0 \
               GGML_OP_OFFLOAD_MIN_BATCH=1 \
               run_row "vramer/${mib}MiB" "$FK" "$model" "${tag}_vramer${mib}" \
                       -ngl 99 -ot "$filter" | tail -1)
        if diff -q "$resident_out" "$vout" >/dev/null 2>&1; then
            echo "    [sanity] vramer/${mib}MiB output BYTE-IDENTICAL to resident"
        else
            echo "    [sanity] vramer/${mib}MiB output DIFFERS from resident:"
            diff "$resident_out" "$vout" | head -20
        fi
    done
}

{
echo "VRAMER comparative benchmark   gpu=$GPU  n_predict=$NPRED  prompt=\"$PROMPT\"  $STAMP"
echo "upstream llama.cpp : $UP"
echo "vramer fork        : $FK"
echo "host: QEMU/KVM VM, 2x NVIDIA A2 (SM86), H2D ~5.7 GiB/s (~half nominal PCIe)"

if [ "$WHICH" = dense ] || [ "$WHICH" = both ]; then
    # dense: 24 layers. partial offload at ~half and ~quarter. FFN weights paged.
    bench_model "dense" "$DENSE_MODEL" 'ffn_(down|gate|up)\.weight=CPU' 12 6 \
        $((16*1024*1024)) $((32*1024*1024)) $((64*1024*1024))
fi
if [ "$WHICH" = moe ] || [ "$WHICH" = both ]; then
    # moe: 24 layers, 60 experts/layer, 4 active. expert stacks paged.
    bench_model "moe" "$MOE_MODEL" 'ffn_(down|gate|up)_exps\.weight=CPU' 12 6 \
        $((256*1024*1024)) $((512*1024*1024)) $((1024*1024*1024))
fi
} 2>&1 | tee "$SUMMARY"

echo
echo "raw logs + summary under: $OUTDIR"
