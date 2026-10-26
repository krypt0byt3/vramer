#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

TIMEFORMAT='WALL=%3R sec | USER=%3U sec | SYS=%3S sec'

for RUN in 1 2; do
    echo
    echo "========== PP2 80% / RUN $RUN =========="

    time mpirun \
        --allow-run-as-root \
        --oversubscribe \
        --bind-to none \
        -n 2 \
        -x CUDA_VISIBLE_DEVICES \
        -x NCCL_P2P_DISABLE \
        -x NCCL_DEBUG \
        python3 examples/run.py \
            --engine_dir /workspace/vramer/engine/qwen2.5-14b-native-ws-pp2 \
            --tokenizer_dir /workspace/vramer/models/Qwen2.5-14B-Instruct \
            --max_output_len 16 \
            --input_text "Explain PCIe bandwidth in one short paragraph." \
            --gpu_weights_percent 0.80 \
            --kv_cache_free_gpu_memory_fraction 0.03
done 2>&1 | tee /workspace/vramer/pp2-final-validation.txt
