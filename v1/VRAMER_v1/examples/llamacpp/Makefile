# Standalone build for the VRAMER <-> ggml-cuda weight-paging demos.
#
# Prereqs:
#   - VRAMER built:   cd ../.. && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#                         -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j
#     => ../../build/libvramer.a
#   - llama.cpp built with CUDA (shared ggml libs), default /tmp/llama.cpp:
#       git clone --depth 1 https://github.com/ggml-org/llama.cpp $(LLAMA)
#       cmake -S $(LLAMA) -B $(LLAMA)/build -DGGML_CUDA=ON \
#             -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
#       cmake --build $(LLAMA)/build -j
#
# Run (pick the GPU with the most free VRAM; keep budgets small):
#   CUDA_VISIBLE_DEVICES=1 ./vramer_ggml_mulmat <model.gguf>
#   CUDA_VISIBLE_DEVICES=1 ./vramer_ggml_layer  <model.gguf>

LLAMA   ?= /tmp/llama.cpp
VRAMER  ?= ../..
CXX     ?= g++
CXXFLAGS = -std=c++17 -O2 -I$(VRAMER)/include -I$(LLAMA)/ggml/include
LDFLAGS  = $(VRAMER)/build/libvramer.a \
           -L$(LLAMA)/build/bin -lggml -lggml-base -lggml-cpu -lggml-cuda \
           -L/usr/lib/x86_64-linux-gnu -lcudart \
           -Wl,-rpath,$(LLAMA)/build/bin -lpthread

all: vramer_ggml_mulmat vramer_ggml_layer

vramer_ggml_mulmat: vramer_ggml_mulmat.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

vramer_ggml_layer: vramer_ggml_layer.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f vramer_ggml_mulmat vramer_ggml_layer
