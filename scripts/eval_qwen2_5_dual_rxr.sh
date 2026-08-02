#!/bin/bash
set -e

# === 0. Environment ===
# Activate the required Python environment before running this script.
export OMP_NUM_THREADS=1
PROJECT_ROOT=$(pwd)
cd $PROJECT_ROOT

# GPU and runtime environment variables.
export CUDA_VISIBLE_DEVICES="0,1"
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore"
export MODELSCOPE_OFFLINE=1
export MAX_PIXELS=1003520
export IMAGE_MAX_TOKEN_NUM=2048
export ROOT_IMAGE_DIR="data/StreamVLN-Trajectory-Data/R2R/"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# === 1. Model paths ===
# Set MODEL1_PATH and MODEL2_PATH for local loading. Leave them empty when using vLLM.
MODEL1_PATH=""
MODEL2_PATH=""
EXP_NAME="run"
CONFIG="configs/vln_rxr_dual.yaml"
RESULT_ROOT="eval_results/rxr"
NUM_WORKERS=6

# Recover token ablation. Default keeps original behavior.
ENABLE_M2_RECOVER_TOKEN_ON_M1_RECOVERING=false
M2_RECOVER_TOKEN="<|mode_recover|>"


# === 2. Optional vLLM servers ===
M1_SERVER_ARGS="--m1_server_url http://127.0.0.1:8081 --m1_server_model m1"
M2_SERVER_ARGS="--m2_server_url http://127.0.0.1:8080 --m2_server_model m2"

# The evaluation waits for the servers to become ready before starting Habitat.
wait_for_vllm_server() {
    local server_name="$1"
    local server_url="${2%/}"
    local models_url="${server_url}/v1/models"

    until curl --fail --silent --show-error --connect-timeout 5 --max-time 10 \
        "$models_url" >/dev/null 2>&1; do
        echo "$(date '+%F %T') ${server_name} vLLM server is unavailable at ${server_url}; retrying in 10 seconds..."
        sleep 10
    done

    echo "$(date '+%F %T') ${server_name} vLLM server is ready at ${server_url}."
}

wait_for_vllm_server "M1" "http://127.0.0.1:8081"
wait_for_vllm_server "M2" "http://127.0.0.1:8080"

# === 3. Run evaluation ===
# Additional arguments are forwarded to the Habitat evaluation entry point.
for SPLIT in "val_unseen"
do
    echo "--------------------------------------------------------"
    echo "Starting Evaluation on RxR $SPLIT (Exp: $EXP_NAME)..."
    echo "--------------------------------------------------------"

    python habitat_vln/habitat_eval_qwen2_5_dual_lm.py \
        --model1_path "$MODEL1_PATH" \
        --model2_path "$MODEL2_PATH" \
        --exp_name "$EXP_NAME" \
        --num_workers $NUM_WORKERS \
        --eval_split "$SPLIT" \
        --config_path "$CONFIG" \
        --result_dir "$RESULT_ROOT" \
        --model_type qwen2_5_vl \
        --rxr_language en \
        --enable_m1_recursive_split \
        $M1_SERVER_ARGS \
        $M2_SERVER_ARGS \
        "$@"
done
        # --enable_m1_recursive_split \

echo "All evaluations complete. Results saved in $RESULT_ROOT/$EXP_NAME"
