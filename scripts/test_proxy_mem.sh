#!/bin/bash

# Proxy-memory configs can be overridden by env vars to match training.
PYTHON_BIN=${PYTHON_BIN:-python}
MEMORY_SIZE=${MEMORY_SIZE:-512}
MEMORY_TOPK=${MEMORY_TOPK:-8}
MEMORY_MOMENTUM=${MEMORY_MOMENTUM:-0.1}
MEMORY_CONF_THRESH=${MEMORY_CONF_THRESH:-0.6}
MEMORY_DEDUP_THRESH=${MEMORY_DEDUP_THRESH:-0.95}
MEMORY_TEMPERATURE=${MEMORY_TEMPERATURE:-0.07}
MEMORY_ALPHA=${MEMORY_ALPHA:-0.7}
MEMORY_FUSE_MODE=${MEMORY_FUSE_MODE:-dynamic}
MEMORY_WARMUP_EPOCH=${MEMORY_WARMUP_EPOCH:-2}

if [ "$3" == "mvtec" ]; then
    CUDA_VISIBLE_DEVICES=$1 "$PYTHON_BIN" test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset MVTec \
        --n_shots 1 2 4 \
        --a_shots 1 \
        --num_learnable_proxies 25 \
        --enable_proxy_memory \
        --memory_size "$MEMORY_SIZE" \
        --memory_topk "$MEMORY_TOPK" \
        --memory_momentum "$MEMORY_MOMENTUM" \
        --memory_conf_thresh "$MEMORY_CONF_THRESH" \
        --memory_dedup_thresh "$MEMORY_DEDUP_THRESH" \
        --memory_temperature "$MEMORY_TEMPERATURE" \
        --memory_alpha "$MEMORY_ALPHA" \
        --memory_fuse_mode "$MEMORY_FUSE_MODE" \
        --memory_warmup_epoch "$MEMORY_WARMUP_EPOCH" \
        --num_seeds 3 \
        --eval_segm \
        --tag proxy_mem \
        --data_root /data2/zhangheyao/PromptAD-master/data/mvtec
elif [ "$3" == "visa" ]; then
    CUDA_VISIBLE_DEVICES=$1 "$PYTHON_BIN" test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset VisA \
        --n_shots 1 2 4 \
        --a_shots 1 \
        --num_learnable_proxies 25 \
        --enable_proxy_memory \
        --memory_size "$MEMORY_SIZE" \
        --memory_topk "$MEMORY_TOPK" \
        --memory_momentum "$MEMORY_MOMENTUM" \
        --memory_conf_thresh "$MEMORY_CONF_THRESH" \
        --memory_dedup_thresh "$MEMORY_DEDUP_THRESH" \
        --memory_temperature "$MEMORY_TEMPERATURE" \
        --memory_alpha "$MEMORY_ALPHA" \
        --memory_fuse_mode "$MEMORY_FUSE_MODE" \
        --memory_warmup_epoch "$MEMORY_WARMUP_EPOCH" \
        --num_seeds 3 \
        --eval_segm \
        --tag proxy_mem \
        --data_root /data2/zhangheyao/PromptAD-master/data/visa
elif [ "$3" == "btad" ]; then
    CUDA_VISIBLE_DEVICES=$1 "$PYTHON_BIN" test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset BTAD \
        --n_shots 1 2 4 \
        --a_shots 1 \
        --num_learnable_proxies 25 \
        --enable_proxy_memory \
        --memory_size "$MEMORY_SIZE" \
        --memory_topk "$MEMORY_TOPK" \
        --memory_momentum "$MEMORY_MOMENTUM" \
        --memory_conf_thresh "$MEMORY_CONF_THRESH" \
        --memory_dedup_thresh "$MEMORY_DEDUP_THRESH" \
        --memory_temperature "$MEMORY_TEMPERATURE" \
        --memory_alpha "$MEMORY_ALPHA" \
        --memory_fuse_mode "$MEMORY_FUSE_MODE" \
        --memory_warmup_epoch "$MEMORY_WARMUP_EPOCH" \
        --num_seeds 3 \
        --eval_segm \
        --tag proxy_mem \
        --data_root /data2/zhangheyao/PromptAD-master/data/btad
elif [ "$3" == "brats" ]; then
    CUDA_VISIBLE_DEVICES=$1 "$PYTHON_BIN" test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset BraTS \
        --n_shots 1 2 4 \
        --a_shots 1 \
        --num_learnable_proxies 25 \
        --enable_proxy_memory \
        --memory_size "$MEMORY_SIZE" \
        --memory_topk "$MEMORY_TOPK" \
        --memory_momentum "$MEMORY_MOMENTUM" \
        --memory_conf_thresh "$MEMORY_CONF_THRESH" \
        --memory_dedup_thresh "$MEMORY_DEDUP_THRESH" \
        --memory_temperature "$MEMORY_TEMPERATURE" \
        --memory_alpha "$MEMORY_ALPHA" \
        --memory_fuse_mode "$MEMORY_FUSE_MODE" \
        --memory_warmup_epoch "$MEMORY_WARMUP_EPOCH" \
        --num_seeds 3 \
        --eval_segm \
        --tag proxy_mem \
        --data_root /data2/zhangheyao/PromptAD-master/data/brats
fi
