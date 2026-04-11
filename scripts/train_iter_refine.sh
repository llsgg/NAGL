#!/bin/bash

# Define arrays for n_shot and a_shot (overridable by env vars).
N_SHOTS_STR=${N_SHOTS:-"1 2 4"}
A_SHOTS_STR=${A_SHOTS:-"1"}
read -r -a n_shot <<< "$N_SHOTS_STR"
read -r -a a_shot <<< "$A_SHOTS_STR"

# Get command line arguments
gpu=$1
save_dir=$2
fold=$3

# Data/model paths can be overridden by env vars.
DATA_ROOT=${DATA_ROOT:-/data2/zhangheyao/PromptAD-master/data}
META_ROOT=${META_ROOT:-./dataset/meta_json}
DINOV2_LOCAL_DIR=${DINOV2_LOCAL_DIR:-/data2/zhangheyao/AAA/NAGL/NAGL/dinov2}

# Iterative refinement configs can be overridden by env vars.
NUM_REFINE_ROUNDS=${NUM_REFINE_ROUNDS:-3}
REFINE_TEMPERATURE=${REFINE_TEMPERATURE:-5.0}

# Create the save directory if it doesn't exist
mkdir -p "$save_dir"

# Loop through each combination of n_shot and a_shot
for n in "${n_shot[@]}"; do
  for a in "${a_shot[@]}"; do
    # Generate a random port for torchrun
    port=$((RANDOM % 64512 + 1024))

    echo "Starting iter-refine training for n_shot=${n}, a_shot=${a} (rounds=${NUM_REFINE_ROUNDS}, tau=${REFINE_TEMPERATURE}) on GPU ${gpu}, saving to ${save_dir}"

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
      --num_refine_rounds "$NUM_REFINE_ROUNDS" \
      --refine_temperature "$REFINE_TEMPERATURE" \
      --save_path "$save_dir" \
      | tee "${save_dir}/iter_refine_n_${n}_a_${a}.log"
  done
done

echo "All iter-refine training settings completed."
