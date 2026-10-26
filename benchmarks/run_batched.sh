#!/bin/bash
# =============================================================================
# VRAMER batched-serving benchmark driver.
#
# The single-stream comparison (run_comparison.sh / docs/BENCHMARKS.md) shows
# VRAMER tying or losing the PCIe wall vs native llama.cpp CPU-offload: every
# host weight streams per token and a single token stream can't hide it. The
# batched-serving thesis is that when B sequences decode in PARALLEL, one decode
# step uploads a weight once and that upload is AMORTISED across all B tokens of
# that step -- so aggregate tok/s should scale with B, and H2D bytes per
# generated token should FALL as B rises. For MoE the catch is selectivity: a
# larger batch routes to the UNION of more experts, so the streamed volume climbs
# again. This driver measures whether batching rescues VRAMER's throughput case.
#
# Tool: llama-batched-bench with the default TG pattern (0123 0123 ...), i.e. all
# B sequences decode one token together per step (B tokens / decode step). Its
# `S_TG t/s` column = pl*tg / t_tg is the AGGREGATE decode throughput across the
# B sequences, from a cleanly isolated TG timer. We use a shared prompt (-pps) so
# prompt processing is a single B-independent pass.
#
# For each (model, config, B) we run ONE process (VRAMER stats are cumulative, so
# one B per process) at two TG lengths:
#   TG_MAIN  -> throughput (S_TG), peak VRAM, pager hit-rate, cumulative H2D.
#   TG_LOW   -> a second H2D point.
# The warmup (16 tok) + shared-prompt pass is a fixed, B-independent H2D floor;
# subtracting the two H2D points cancels it, giving a floor-free MARGINAL TG H2D
# per generated token:  (h2d_main - h2d_low) / (B*(TG_MAIN-TG_LOW)).
# For VRAMER the H2D is the adapter's own counter; for native cpu-offload (no such
# counter) it is nvidia-smi `dmon` PCIe-rx integrated over the run (approximate).
#
# This script ONLY measures. It builds nothing and touches neither the adapter
# nor the VRAMER library.
#
# Usage:  bash benchmarks/run_batched.sh [moe|dense|both]
# Env:    GPU=0  TG_MAIN=64  TG_LOW=16  REPS=2  NPLS="1 2 4 8 16"
#         OUTDIR=benchmarks/raw/batched
# =============================================================================
set -u

GPU=${GPU:-0}
WHICH=${1:-moe}
export CUDA_VISIBLE_DEVICES=$GPU

TG_MAIN=${TG_MAIN:-64}
TG_LOW=${TG_LOW:-16}
REPS=${REPS:-2}
PP=${PP:-32}
NCTX=${NCTX:-4096}
PAGE=${PAGE:-$((4*1024*1024))}

UP=${UP:-/tmp/llama.cpp/build/bin/llama-batched-bench}
FK=${FK:-/tmp/llama-vramer-batched/build/bin/llama-batched-bench}

DENSE_MODEL=/tmp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
MOE_MODEL=/tmp/models/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf

HERE="$(cd "$(dirname "$0")" && pwd)"
OUTDIR=${OUTDIR:-$HERE/raw/batched}
mkdir -p "$OUTDIR"
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SUMMARY="$OUTDIR/batched.txt"

# ---------------------------------------------------------------------------
# run_cell <label> <bin> <model> <B> <tg> <logbase> <extra llama args...>
#   Caller exports any VRAMER_* env. Runs one batched-bench process with a dmon
#   PCIe-rx sampler alongside. Emits to stderr a human line and appends a
#   machine-readable ROW line to <logbase>.row:
#     ROW <label> <B> <tg> <rc> <s_tg> <peakVRAM> <h2d_vramer_MiB> <hit%> <pcie_rx_MiB>
# ---------------------------------------------------------------------------
run_cell() {
    local label="$1" bin="$2" model="$3" B="$4" tg="$5" logbase="$6"; shift 6
    local err="$OUTDIR/$logbase.err" out="$OUTDIR/$logbase.out"
    local vram="$OUTDIR/$logbase.vram" pcie="$OUTDIR/$logbase.pcie"

    # VRAM sampler (10 Hz)
    ( : > "$vram"; while :; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" >> "$vram" 2>/dev/null
        sleep 0.1; done ) &
    local vs=$!
    # PCIe rx sampler (1 Hz, MB/s). dmon prints one line per second.
    ( : > "$pcie"; nvidia-smi dmon -s t -i "$GPU" -d 1 >> "$pcie" 2>/dev/null ) &
    local ps=$!

    "$bin" -m "$model" -c "$NCTX" -b 2048 -ub 512 -ngl 99 -pps \
        -npp "$PP" -ntg "$tg" -npl "$B" "$@" >"$out" 2>"$err"
    local rc=$?

    kill "$vs" "$ps" 2>/dev/null; wait "$vs" "$ps" 2>/dev/null

    local s_tg peak h2d hit rx
    # Single B per process => exactly one data row (the '^|' line that is neither
    # the header nor the dashed separator). S_TG is the 9th |-delimited field.
    s_tg=$(grep -E "^\|" "$out" | grep -vE "PP|---" | tail -1 | awk -F'|' '{gsub(/ /,"",$9); print $9}')
    [ -z "$s_tg" ] && s_tg="n/a"
    peak=$(sort -n "$vram" 2>/dev/null | tail -1); [ -z "$peak" ] && peak="n/a"
    h2d=$(grep -oE "bytes_h2d=[0-9.]+" "$err" | head -1 | sed 's/.*=//'); [ -z "$h2d" ] && h2d="-"
    hit=$(grep -oE "hit_rate=[0-9.]+%" "$err" | grep -oE "[0-9.]+" | head -1); [ -z "$hit" ] && hit="-"
    # integrate PCIe rx MB/s over 1s samples -> MiB (MB*0.953674). numeric col 2.
    rx=$(awk '/^ *[0-9]/{s+=$2} END{printf "%.0f", s*0.953674}' "$pcie" 2>/dev/null)
    [ -z "$rx" ] && rx="-"

    printf '  %-14s B=%-3s tg=%-4s rc=%s  S_TG=%-8s tok/s  peakVRAM=%-7s MiB  H2D(vramer)=%-10s MiB  hit=%-6s%%  pcieRx=%s MiB\n' \
        "$label" "$B" "$tg" "$rc" "$s_tg" "$peak" "$h2d" "$hit" "$rx" >&2
    printf 'ROW\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$B" "$tg" "$rc" "$s_tg" "$peak" "$h2d" "$hit" "$rx" >> "$OUTDIR/$logbase.row"
}

# run a config across the B sweep: REPS at TG_MAIN (throughput+range) + 1 at TG_LOW (H2D slope)
sweep_config() {
    local tag="$1" label="$2" bin="$3" model="$4" lb="$5"; shift 5
    # remaining args: the B list, then "--", then extra llama args (+VRAMER_* via env already set)
    local bs=() ; while [ "$1" != "--" ]; do bs+=("$1"); shift; done; shift
    echo "  -- $label --" >&2
    local B
    for B in "${bs[@]}"; do
        local r
        for r in $(seq 1 "$REPS"); do
            run_cell "$label" "$bin" "$model" "$B" "$TG_MAIN" "${tag}_${lb}_B${B}_main_r${r}" "$@"
        done
        run_cell "$label" "$bin" "$model" "$B" "$TG_LOW" "${tag}_${lb}_B${B}_low" "$@"
    done
}

bench_model() {
    local tag="$1" model="$2" filter="$3"; shift 3
    local bs=() ; while [ "$1" != "--" ]; do bs+=("$1"); shift; done; shift
    local budgets=("$@")

    echo
    echo "############################################################"
    echo "# MODEL: $tag   ($model)"
    echo "# host-paged tensors: $filter   B sweep: ${bs[*]}"
    echo "############################################################"

    # resident (full offload) -- the throughput ceiling
    sweep_config "$tag" "resident" "$UP" "$model" "resident" "${bs[@]}" --

    # native cpu-offload of the SAME tensor set (no VRAMER)
    GGML_OP_OFFLOAD_MIN_BATCH=1 \
        sweep_config "$tag" "cpu-offload" "$UP" "$model" "cpuoff" "${bs[@]}" -- -ot "$filter"

    # VRAMER paged, at each budget
    local b mib
    for b in "${budgets[@]}"; do
        mib=$((b/1024/1024))
        VRAMER_GGUF="$model" VRAMER_BUDGET="$b" VRAMER_PAGE="$PAGE" VRAMER_GPU="$GPU" VRAMER_EVICT=0 \
        GGML_OP_OFFLOAD_MIN_BATCH=1 \
            sweep_config "$tag" "vramer/${mib}MiB" "$FK" "$model" "vramer${mib}" "${bs[@]}" -- -ot "$filter"
    done
}

{
echo "VRAMER batched-serving benchmark   gpu=$GPU  TG_MAIN=$TG_MAIN TG_LOW=$TG_LOW REPS=$REPS pp=$PP(shared)  $STAMP"
echo "tool: llama-batched-bench (default TG pattern; S_TG = pl*tg/t_tg = aggregate decode tok/s)"
echo "upstream (resident/native): $UP"
echo "vramer fork               : $FK"
echo "host: QEMU/KVM VM, 2x NVIDIA A2 (SM86), H2D ~5.7 GiB/s (~half nominal PCIe)"

MOE_NPLS=${MOE_NPLS:-"1 2 4 8 16"}
DENSE_NPLS=${DENSE_NPLS:-"1 2 4 8"}
if [ "$WHICH" = moe ] || [ "$WHICH" = both ]; then
    bench_model "moe" "$MOE_MODEL" 'ffn_(down|gate|up)_exps\.weight=CPU' \
        $MOE_NPLS -- $((512*1024*1024)) $((2048*1024*1024))
fi
if [ "$WHICH" = dense ] || [ "$WHICH" = both ]; then
    bench_model "dense" "$DENSE_MODEL" 'ffn_(down|gate|up)\.weight=CPU' \
        $DENSE_NPLS -- $((32*1024*1024)) $((64*1024*1024))
fi
} 2>&1 | tee "$SUMMARY"

echo
echo "raw logs + summary under: $OUTDIR"
