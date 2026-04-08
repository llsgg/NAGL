#!/bin/bash

mode=$1  # "train" or "test"
gpu=$2
save_dir=$3
fold=$4
test_dataset=$5
tag=${6:-proxy_mem}

if [ "$mode" == "train" ]; then
    bash scripts/train_proxy_mem.sh $gpu $save_dir $fold
fi

bash scripts/test_proxy_mem.sh $gpu $save_dir $test_dataset $tag
