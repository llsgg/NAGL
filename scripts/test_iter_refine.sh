#!/bin/bash

# Iterative refinement configs can be overridden by env vars.
PYTHON_BIN=${PYTHON_BIN:-python}
NUM_REFINE_ROUNDS=${NUM_REFINE_ROUNDS:-3}
REFINE_TEMPERATURE=${REFINE_TEMPERATURE:-5.0}
TAG=${4:-iter_refine}
N_SHOTS_STR=${5:-${N_SHOTS:-"1 2 4"}}
A_SHOTS_STR=${6:-${A_SHOTS:-"1"}}
read -r -a N_SHOTS_ARR <<< "$N_SHOTS_STR"
read -r -a A_SHOTS_ARR <<< "$A_SHOTS_STR"

if [ "$3" == "mvtec" ]; then
    CUDA_VISIBLE_DEVICES=$1 "$PYTHON_BIN" test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset MVTec \
        --n_shots "${N_SHOTS_ARR[@]}" \
        --a_shots "${A_SHOTS_ARR[@]}" \
        --num_learnable_proxies 25 \
        --num_refine_rounds "$NUM_REFINE_ROUNDS" \
        --refine_temperature "$REFINE_TEMPERATURE" \
        --num_seeds 3 \
        --eval_segm \
        --tag "$TAG" \
        --data_root /data2/zhangheyao/PromptAD-master/data/mvtec
elif [ "$3" == "visa" ]; then
    CUDA_VISIBLE_DEVICES=$1 "$PYTHON_BIN" test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset VisA \
        --n_shots "${N_SHOTS_ARR[@]}" \
        --a_shots "${A_SHOTS_ARR[@]}" \
        --num_learnable_proxies 25 \
        --num_refine_rounds "$NUM_REFINE_ROUNDS" \
        --refine_temperature "$REFINE_TEMPERATURE" \
        --num_seeds 3 \
        --eval_segm \
        --tag "$TAG" \
        --data_root /data2/zhangheyao/PromptAD-master/data/visa
elif [ "$3" == "btad" ]; then
    CUDA_VISIBLE_DEVICES=$1 "$PYTHON_BIN" test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset BTAD \
        --n_shots "${N_SHOTS_ARR[@]}" \
        --a_shots "${A_SHOTS_ARR[@]}" \
        --num_learnable_proxies 25 \
        --num_refine_rounds "$NUM_REFINE_ROUNDS" \
        --refine_temperature "$REFINE_TEMPERATURE" \
        --num_seeds 3 \
        --eval_segm \
        --tag "$TAG" \
        --data_root /data2/zhangheyao/PromptAD-master/data/btad
elif [ "$3" == "brats" ]; then
    CUDA_VISIBLE_DEVICES=$1 "$PYTHON_BIN" test.py \
        --save_path $2 \
        --image_size 448 \
        --dataset BraTS \
        --n_shots "${N_SHOTS_ARR[@]}" \
        --a_shots "${A_SHOTS_ARR[@]}" \
        --num_learnable_proxies 25 \
        --num_refine_rounds "$NUM_REFINE_ROUNDS" \
        --refine_temperature "$REFINE_TEMPERATURE" \
        --num_seeds 3 \
        --eval_segm \
        --tag "$TAG" \
        --data_root /data2/zhangheyao/PromptAD-master/data/brats
fi
