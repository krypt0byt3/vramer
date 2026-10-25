#!/usr/bin/env bash
# VRAMER residency / page-size characterization sweep.
#
# Measures the PCIe spill penalty: how end-to-end pager throughput and demand
# hit rate change as the resident fraction (VRAM cache / source size) shrinks.
# Uses multiple passes so pass 2+ exercises cache reuse -- that is where a
# partial resident set actually costs you.
#
# Usage: scripts/resident_sweep.sh [gpu] [source_gib]
set -euo pipefail

GPU="${1:-0}"
SRC_GIB="${2:-8}"
BIN="$(dirname "$0")/../build/vramer"
PASSES=3

run() {  # budget_gib page passes extra...
    local budget="$1" page="$2" passes="$3"; shift 3
    "$BIN" bench --gpu "$GPU" --source ram --bytes "${SRC_GIB}GiB" \
        --vram-budget "${budget}GiB" --page-size "$page" --passes "$passes" "$@" 2>&1
}

field() { sed -n "s/.*$1:[[:space:]]*//p" <<<"$2" | head -1; }

echo "VRAMER residency sweep  gpu=$GPU  source=${SRC_GIB}GiB  passes=$PASSES  $(date -u +%FT%TZ)"
echo
printf '%-10s %-8s %-9s %-10s %-9s %-10s %-10s\n' \
    resident budget page elapsed hit_rate H2D_rate stall_ms

echo "-- resident-fraction sweep (page=64MiB) --"
for budget in "$SRC_GIB" 6 4 2 1; do
    out="$(run "$budget" 64MiB "$PASSES")"
    frac="$(awk "BEGIN{printf \"%.0f%%\", 100*$budget/$SRC_GIB}")"
    printf '%-10s %-8s %-9s %-10s %-9s %-10s %-10s\n' \
        "$frac" "${budget}GiB" "64MiB" \
        "$(field elapsed "$out")" "$(field 'hit rate' "$out")" \
        "$(field 'H2D rate\*' "$out")" "$(field stall "$out")"
done

echo
echo "-- page-size sweep (resident=50%, budget=$((SRC_GIB/2))GiB) --"
for page in 16MiB 32MiB 64MiB 128MiB 256MiB; do
    out="$(run "$((SRC_GIB/2))" "$page" "$PASSES")"
    printf '%-10s %-8s %-9s %-10s %-9s %-10s %-10s\n' \
        "50%" "$((SRC_GIB/2))GiB" "$page" \
        "$(field elapsed "$out")" "$(field 'hit rate' "$out")" \
        "$(field 'H2D rate\*' "$out")" "$(field stall "$out")"
done

echo
echo "-- prefetch A/B (resident=50%, page=64MiB, single pass) --"
for cfg in "baseline::" "prefetch:--prefetch-workers 1 --prefetch-ahead 3:--staging 4"; do
    name="${cfg%%:*}"; rest="${cfg#*:}"; flags="${rest%%:*}"; extra="${rest#*:}"
    out="$(run "$((SRC_GIB/2))" 64MiB 1 $flags $extra)"
    printf '%-10s %-8s %-9s %-10s %-9s %-10s %-10s\n' \
        "$name" "$((SRC_GIB/2))GiB" "64MiB" \
        "$(field elapsed "$out")" "$(field 'hit rate' "$out")" \
        "$(field 'H2D rate\*' "$out")" "$(field stall "$out")"
done
