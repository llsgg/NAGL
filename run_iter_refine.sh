#!/bin/bash

mode=$1  # "train" or "test"
gpu=$2
save_dir=$3
fold=$4
test_dataset=$5
tag=${6:-iter_refine}

if [ "$mode" == "train" ]; then
    bash scripts/train_iter_refine.sh $gpu $save_dir $fold
    bash scripts/test_iter_refine.sh $gpu $save_dir $test_dataset $tag
elif [ "$mode" == "train_step" ]; then
    bash scripts/train_test_iter_refine_step.sh $gpu $save_dir $fold $test_dataset $tag
elif [ "$mode" == "test" ]; then
    bash scripts/test_iter_refine.sh $gpu $save_dir $test_dataset $tag
else
    echo "Unknown mode: $mode"
    echo "Usage: bash run_iter_refine.sh [train|train_step|test] <gpu> <save_dir> <fold> <dataset> [tag]"
    exit 1
fi
