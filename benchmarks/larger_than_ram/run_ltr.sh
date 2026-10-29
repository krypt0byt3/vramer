#!/usr/bin/env bash
# run_ltr.sh -- "larger than host RAM" disk-tier probe for VRAMER.
#
# Pages a backing file that EXCEEDS host RAM through a small VRAM budget and
# reports effective bandwidth + PagerStats. Because the file is larger than RAM,
# a linear scan cannot be served from the OS page cache -> it forces real disk
# I/O (the cache-cold / disk-bound regime). A smaller-than-RAM file re-scanned
# gives the cache-warm contrast.
#
# This path is: disk -> OS page cache -> pinned staging -> GPU (cudaMemcpyAsync).
# It is NOT GPUDirect Storage / cuFile (not installed here, and this QEMU VM
# would not do true peer-to-peer DMA anyway). GDS is the documented next step.
#
# Env:
#   VRAMER   path to the vramer CLI      (default ../../build/vramer)
#   FILE     backing file to scan        (required)
#   BUDGET   VRAM budget per run         (default 4GiB)
#   GPU      CUDA device index           (default 0; caller usually sets
#            CUDA_VISIBLE_DEVICES=1 and leaves GPU=0)
#   DROP     1 = drop page cache before each cell (needs root)   (default 1)
#   REPS     repeats per cell            (default 3)
#   RAWDIR   where to tee raw logs       (default ../raw)
#
# Usage: FILE=/var/tmp/vramer-ltr-data/ltr120.bin DROP=1 bash run_ltr.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRAMER="${VRAMER:-$HERE/../../build/vramer}"
FILE="${FILE:?set FILE=/path/to/backing/file}"
BUDGET="${BUDGET:-4GiB}"
GPU="${GPU:-0}"
DROP="${DROP:-1}"
REPS="${REPS:-3}"
RAWDIR="${RAWDIR:-$HERE/../raw}"
mkdir -p "$RAWDIR"

drop_caches() {
    [ "$DROP" = "1" ] || return 0
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || {
        echo "WARN: could not drop caches (need root); results may be warm" >&2; }
}

# run_cell <label> <extra bench args...>
run_cell() {
    local label="$1"; shift
    local log="$RAWDIR/ltr_${label}.log"
    : > "$log"
    echo "### cell=$label  budget=$BUDGET  args: $*" | tee -a "$log"
    for r in $(seq 1 "$REPS"); do
        drop_caches
        echo "--- rep $r ($(date -u +%H:%M:%S)) ---" | tee -a "$log"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
            "$VRAMER" bench --source file --file "$FILE" \
            --vram-budget "$BUDGET" --gpu "$GPU" "$@" 2>&1 \
            | tee -a "$log" \
            | grep -E "elapsed|H2D rate|H2D bytes|hit rate|src read|stall|checksum|misses"
        echo | tee -a "$log"
    done
}

echo "VRAMER larger-than-RAM probe"
echo "  cli:    $VRAMER"
echo "  file:   $FILE  ($(stat -c %s "$FILE") bytes)"
echo "  RAM:    $(free -b | awk '/Mem:/{print $2}') bytes"
echo "  drop:   $DROP   reps: $REPS   budget: $BUDGET"
echo

# Page-size sweep on the disk-bound path, no prefetch (demand only).
run_cell p64_pf0  --page-size 64MiB  --prefetch-workers 0
run_cell p16_pf0  --page-size 16MiB  --prefetch-workers 0
run_cell p256_pf0 --page-size 256MiB --prefetch-workers 0

# Prefetch effect on the disk-bound path (1 worker, stay 2 pages ahead).
run_cell p64_pf1  --page-size 64MiB  --prefetch-workers 1 --prefetch-ahead 2
run_cell p64_pf2  --page-size 64MiB  --prefetch-workers 2 --prefetch-ahead 4

echo "done. raw logs in $RAWDIR"
