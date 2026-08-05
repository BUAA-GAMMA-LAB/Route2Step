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
# Transformers local loading is the default.
MODEL1_PATH="model_zoo/MIA"
MODEL2_PATH="model_zoo/MAG"
EXP_NAME="run"
CONFIG="configs/vln_r2r_dual.yaml"
RESULT_ROOT="eval/r2r_v1_3"
NUM_WORKERS=4

# Recover limiting. Keep both at 0 to disable. To enable, set both values, e.g. 5 and 3.
M2_RECOVER_MAX_CONSECUTIVE=0
M2_RECOVER_COOLDOWN=0

# Keep waiting for vLLM: 0 disables the per-request HTTP timeout and -1 retries
# transient vLLM connection/server errors indefinitely.
VLLM_HTTP_TIMEOUT_S=0
VLLM_HTTP_RETRY_COUNT=-1

# === 2. Optional vLLM servers ===
USE_VLLM="${USE_VLLM:-false}"
M1_SERVER_ARGS=""
M2_SERVER_ARGS=""
RECOVER_LIMIT_ARGS=""
if [ "$M2_RECOVER_MAX_CONSECUTIVE" -gt 0 ] || [ "$M2_RECOVER_COOLDOWN" -gt 0 ]; then
    if [ "$M2_RECOVER_MAX_CONSECUTIVE" -le 0 ] || [ "$M2_RECOVER_COOLDOWN" -le 0 ]; then
        echo "ERROR: M2_RECOVER_MAX_CONSECUTIVE and M2_RECOVER_COOLDOWN must both be > 0, or both be 0."
        exit 1
    fi
    RECOVER_LIMIT_ARGS="--m2_recover_max_consecutive $M2_RECOVER_MAX_CONSECUTIVE --m2_recover_cooldown $M2_RECOVER_COOLDOWN"
fi

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

if [ "$USE_VLLM" = "true" ]; then
    M1_SERVER_ARGS="--m1_server_url http://127.0.0.1:8081 --m1_server_model m1"
    M2_SERVER_ARGS="--m2_server_url http://127.0.0.1:8080 --m2_server_model m2"
    wait_for_vllm_server "M1" "http://127.0.0.1:8081"
    wait_for_vllm_server "M2" "http://127.0.0.1:8080"
fi

# === 3. Run evaluation ===
# Additional arguments are forwarded to the Habitat evaluation entry point.
for SPLIT in "val_unseen"
do
    echo "--------------------------------------------------------"
    echo "Starting Evaluation on R2R $SPLIT (Exp: $EXP_NAME)..."
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
        $M1_SERVER_ARGS \
        $M2_SERVER_ARGS \
        $RECOVER_LIMIT_ARGS \
        --http_timeout_s "$VLLM_HTTP_TIMEOUT_S" \
        --http_retry_count "$VLLM_HTTP_RETRY_COUNT" \
        "$@"
done
    # --strip_m1_recovering_prefix_for_m2 \

echo "All evaluations complete. Results saved in $RESULT_ROOT/$EXP_NAME"
