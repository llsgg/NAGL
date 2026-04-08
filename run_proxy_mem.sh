#!/bin/bash

mode=$1  # "train" or "test"
gpu=$2
save_dir=$3
fold=$4
test_dataset=$5
tag=${6:-proxy_mem}

if [ "$mode" == "train" ]; then
    bash scripts/train_proxy_mem.sh $gpu $save_dir $fold
    bash scripts/test_proxy_mem.sh $gpu $save_dir $test_dataset $tag
elif [ "$mode" == "train_step" ]; then
    bash scripts/train_test_proxy_mem_step.sh $gpu $save_dir $fold $test_dataset $tag
elif [ "$mode" == "test" ]; then
    bash scripts/test_proxy_mem.sh $gpu $save_dir $test_dataset $tag
else
    echo "Unknown mode: $mode"
    echo "Usage: bash run_proxy_mem.sh [train|train_step|test] <gpu> <save_dir> <fold> <dataset> [tag]"
    exit 1
fi
