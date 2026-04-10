#!/bin/bash

# Run train->test sequentially for each shot setting.
# Usage:
#   bash scripts/train_test_proxy_mem_step.sh <gpu> <save_dir> <fold> <dataset> [tag]

gpu=$1
save_dir=$2
fold=$3
test_dataset=$4
tag=${5:-proxy_mem}

N_SHOTS_STR=${N_SHOTS:-"1 2 4"}
A_SHOTS_STR=${A_SHOTS:-"1"}
read -r -a n_shot <<< "$N_SHOTS_STR"
read -r -a a_shot <<< "$A_SHOTS_STR"

for n in "${n_shot[@]}"; do
  for a in "${a_shot[@]}"; do
    echo "==== Train then test: n_shot=${n}, a_shot=${a} ===="
    N_SHOTS="$n" A_SHOTS="$a" bash scripts/train_proxy_mem.sh "$gpu" "$save_dir" "$fold"
    N_SHOTS="$n" A_SHOTS="$a" bash scripts/test_proxy_mem.sh "$gpu" "$save_dir" "$test_dataset" "$tag"
  done
done

echo "All train->test steps completed."
