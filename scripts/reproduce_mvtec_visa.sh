#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/reproduce_mvtec_visa.sh <gpu_id>
# Example:
#   bash scripts/reproduce_mvtec_visa.sh 7

GPU_ID="${1:-0}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi
echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys, torch; print('python_exec:', sys.executable); print('torch_version:', torch.__version__)"

DATA_ROOT="${DATA_ROOT:-/data2/zhangheyao/PromptAD-master/data}"
META_ROOT="${META_ROOT:-./dataset/meta_json}"
DINOV2_LOCAL_DIR="${DINOV2_LOCAL_DIR:-/data2/zhangheyao/AAA/NAGL/NAGL/dinov2}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
PRINT_FREQ="${PRINT_FREQ:-50}"
N_SHOT="${N_SHOT:-1}"
A_SHOT="${A_SHOT:-1}"
NUM_LEARNABLE_PROXIES="${NUM_LEARNABLE_PROXIES:-25}"

OUT_BASE="${OUT_BASE:-./outputs/repro_tmux}"
LOG_BASE="${LOG_BASE:-./logs/repro_tmux}"
mkdir -p "$OUT_BASE" "$LOG_BASE"

# Resume/skip controls:
#   START_STAGE=1..5 to resume from a specific stage
#   SKIP_FOLD0_TRAIN=1 to force-skip fold0 training
START_STAGE="${START_STAGE:-1}"
SKIP_FOLD0_TRAIN="${SKIP_FOLD0_TRAIN:-0}"

run_train() {
  local fold="$1"
  local save_path="$2"
  local port="$3"
  local log_file="$4"

  mkdir -p "$save_path"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=1 --master_port="$port" train.py \
    --data_root "$DATA_ROOT" \
    --meta_root "$META_ROOT" \
    --dinov2_local_dir "$DINOV2_LOCAL_DIR" \
    --fold "$fold" \
    --epoch "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --image_size "$IMAGE_SIZE" \
    --print_freq "$PRINT_FREQ" \
    --n_shot "$N_SHOT" \
    --a_shot "$A_SHOT" \
    --num_learnable_proxies "$NUM_LEARNABLE_PROXIES" \
    --save_path "$save_path" | tee "$log_file"
}

run_test() {
  local dataset="$1"
  local data_root="$2"
  local save_path="$3"
  local tag="$4"
  local log_file="$5"

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" test.py \
    --save_path "$save_path" \
    --image_size "$IMAGE_SIZE" \
    --dataset "$dataset" \
    --n_shots 1 2 4 \
    --a_shots 1 \
    --num_learnable_proxies "$NUM_LEARNABLE_PROXIES" \
    --num_seeds 3 \
    --eval_segm \
    --tag "$tag" \
    --data_root "$data_root" | tee "$log_file"
}

if [ "$START_STAGE" -le 1 ]; then
  if [ "$SKIP_FOLD0_TRAIN" = "1" ] || [ -f "$OUT_BASE/fold0/n_${N_SHOT}_a_${A_SHOT}_best.pth" ]; then
    echo "[1/5] Skip Fold0 train (checkpoint exists or SKIP_FOLD0_TRAIN=1)"
  else
    echo "[1/5] Fold0 train (train on VisA, val on MVTec)"
    run_train 0 "$OUT_BASE/fold0" 29501 "$LOG_BASE/fold0_train.log"
  fi
else
  echo "[1/5] Skip Fold0 train (START_STAGE=$START_STAGE)"
fi

if [ "$START_STAGE" -le 2 ]; then
  echo "[2/5] Fold0 test on MVTec"
  run_test MVTec "$DATA_ROOT/mvtec" "$OUT_BASE/fold0" repro_fold0 "$LOG_BASE/fold0_test_mvtec.log"
else
  echo "[2/5] Skip Fold0 test (START_STAGE=$START_STAGE)"
fi

if [ "$START_STAGE" -le 3 ]; then
  echo "[3/5] Fold1 train (train on MVTec, val on VisA)"
  run_train 1 "$OUT_BASE/fold1" 29511 "$LOG_BASE/fold1_train.log"
else
  echo "[3/5] Skip Fold1 train (START_STAGE=$START_STAGE)"
fi

if [ "$START_STAGE" -le 4 ]; then
  echo "[4/5] Fold1 test on VisA"
  run_test VisA "$DATA_ROOT/visa" "$OUT_BASE/fold1" repro_fold1 "$LOG_BASE/fold1_test_visa.log"
else
  echo "[4/5] Skip Fold1 test (START_STAGE=$START_STAGE)"
fi

if [ "$START_STAGE" -le 5 ]; then
  echo "[5/5] Export xlsx summary"
  "$PYTHON_BIN" scripts/get_xlsx_result.py | tee "$LOG_BASE/export_xlsx.log"
else
  echo "[5/5] Skip export (START_STAGE=$START_STAGE)"
fi

echo "Reproduction finished."
echo "Logs: $LOG_BASE"
echo "Outputs: $OUT_BASE"
