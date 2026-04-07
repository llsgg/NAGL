#!/bin/bash

# Define arrays for n_shot and a_shot
n_shot=(1 2 4)
a_shot=(1)

# Get command line arguments
gpu=$1
save_dir=$2
fold=$3

# Data/model paths can be overridden by env vars.
DATA_ROOT=${DATA_ROOT:-/data2/zhangheyao/PromptAD-master/data}
META_ROOT=${META_ROOT:-./dataset/meta_json}
DINOV2_LOCAL_DIR=${DINOV2_LOCAL_DIR:-/data2/zhangheyao/AAA/NAGL/NAGL/dinov2}

# Proxy-memory configs can be overridden by env vars.
MEMORY_SIZE=${MEMORY_SIZE:-512}
MEMORY_TOPK=${MEMORY_TOPK:-8}
MEMORY_MOMENTUM=${MEMORY_MOMENTUM:-0.1}
MEMORY_CONF_THRESH=${MEMORY_CONF_THRESH:-0.6}
MEMORY_DEDUP_THRESH=${MEMORY_DEDUP_THRESH:-0.95}
MEMORY_TEMPERATURE=${MEMORY_TEMPERATURE:-0.07}
MEMORY_ALPHA=${MEMORY_ALPHA:-0.7}
MEMORY_FUSE_MODE=${MEMORY_FUSE_MODE:-dynamic}
MEMORY_WARMUP_EPOCH=${MEMORY_WARMUP_EPOCH:-2}

# Create the save directory if it doesn't exist
mkdir -p "$save_dir"

# Loop through each combination of n_shot and a_shot
for n in "${n_shot[@]}"; do
  for a in "${a_shot[@]}"; do
    # Generate a random port for torchrun
    port=$((RANDOM % 64512 + 1024))

    echo "Starting proxy-memory training for n_shot=${n}, a_shot=${a} on GPU ${gpu}, saving to ${save_dir}"

    # Run the training command
    CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$port train.py \
      --data_root "$DATA_ROOT" \
      --meta_root "$META_ROOT" \
      --dinov2_local_dir "$DINOV2_LOCAL_DIR" \
      --fold "$fold" \
      --epoch 20 \
      --batch_size 8 \
      --image_size 448 \
      --print_freq 50 \
      --n_shot "$n" \
      --a_shot "$a" \
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
      --save_path "$save_dir" \
      | tee "${save_dir}/proxy_mem_n_${n}_a_${a}.log"
  done
done

echo "All proxy-memory training settings completed."
