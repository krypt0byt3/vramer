#!/usr/bin/env bash
# One-command build + verify for VRAMER. Builds the library, runs the test
# suite, and runs the demos whose dependencies are present (skipping the rest).
#
# Usage: scripts/smoke.sh [cuda_arch]   (default arch 86; use 120 for RTX 50xx)
set -euo pipefail

ARCH="${1:-86}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== configure + build (sm_${ARCH}) =="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" >/dev/null
cmake --build build -j

echo "== ctest =="
ctest --test-dir build --output-on-failure

echo "== CLI sanity =="
./build/vramer detect >/dev/null && echo "  detect ok"
./build/vramer plan --model-bytes 40GiB --vram-budget 16GiB >/dev/null && echo "  plan ok"

if command -v python3 >/dev/null && python3 -c "import numpy" 2>/dev/null; then
    echo "== python paging demo =="
    python3 examples/python_paging_demo.py
else
    echo "== python paging demo: SKIP (needs python3 + numpy) =="
fi

if [ -n "${LLAMA:-}" ] && [ -d "${LLAMA}/build/bin" ]; then
    echo "== llama.cpp ggml demos (LLAMA=$LLAMA) =="
    make -C examples/llamacpp LLAMA="$LLAMA" >/dev/null
    if [ -n "${GGUF:-}" ]; then
        "${CUDA_VISIBLE_DEVICES:+env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES}" \
            examples/llamacpp/vramer_ggml_mulmat "$GGUF" | grep -E "RESULT" || true
    else
        echo "  built; set GGUF=<model.gguf> to run"
    fi
else
    echo "== llama.cpp ggml demos: SKIP (set LLAMA=/path/to/llama.cpp with a CUDA build) =="
fi

echo "== smoke OK =="
