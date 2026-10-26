#!/bin/bash
# Reproduce the VRAMER<->llama.cpp fork from a clean upstream checkout.
#
#   build_fork.sh  [UPSTREAM_LLAMA]  [FORK_DST]
#
# UPSTREAM_LLAMA : an existing llama.cpp checkout built with GGML_CUDA (default
#                  /tmp/llama.cpp). FORK_DST : where to put the fork (default
#                  /tmp/llama-vramer). Requires VRAMER already built at
#                  $VRAMER_ROOT/build/libvramer.a (see top-level README).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VRAMER_ROOT="$(cd "$HERE/../../.." && pwd)"          # repo root (examples/llamacpp/adapter -> ../../..)
UPSTREAM="${1:-/tmp/llama.cpp}"
FORK="${2:-/tmp/llama-vramer}"

echo "VRAMER_ROOT = $VRAMER_ROOT"
echo "upstream    = $UPSTREAM"
echo "fork dst    = $FORK"

# 0. VRAMER static lib must exist.
if [ ! -f "$VRAMER_ROOT/build/libvramer.a" ]; then
    echo "building VRAMER..."
    cmake -S "$VRAMER_ROOT" -B "$VRAMER_ROOT/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
    cmake --build "$VRAMER_ROOT/build" -j
fi

# 1. fresh copy of upstream.
rm -rf "$FORK"
cp -r "$UPSTREAM" "$FORK"
rm -rf "$FORK/build"

# 2. drop in the new adapter source files.
cp "$HERE/vramer_llama_hook.cpp" "$FORK/src/vramer_llama_hook.cpp"
cp "$HERE/vramer_llama_hook.h"   "$FORK/src/vramer_llama_hook.h"

# 3. apply the two-file patch (llama-context.cpp call site + src/CMakeLists.txt).
#    The patch carries a @VRAMER_ROOT@ token; rewrite it to this checkout's path.
sed "s#@VRAMER_ROOT@#$VRAMER_ROOT#g" \
    "$HERE/llama_cpp_fork.patch" > "$FORK/.vramer.patch"
( cd "$FORK" && patch -p1 < .vramer.patch )

# 4. build.
cmake -S "$FORK" -B "$FORK/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF
cmake --build "$FORK/build" -j --target llama-completion

echo "OK: $FORK/build/bin/llama-completion"
