#!/usr/bin/env bash
set -euo pipefail

# Query an OpenAI-compatible vLLM service against a static VLN sample.
#
# TASK_TYPE=m1_subinstruction preserves the dual-system M1 benchmark: its
# History trajectory/Current view input, prompt, and text metrics are unchanged.
# TASK_TYPE=single_m1 evaluates the same sub-instruction output with the
# single model's global-uniform (up to eight) Observation history input.
# The external vLLM service is fixed at port 8086 and currently uses model m1.
TASK_TYPE="${TASK_TYPE:-single_m1}"  # "m1_subinstruction" or "single_m1"
DATASET="${DATASET:-rxr_deviation}"  # "r2r", "r2r_deviation", "rxr", or "rxr_deviation"
SAMPLING_SEED=42
SAMPLE_FOCUS="uniform_length"

case "$TASK_TYPE" in
  m1_subinstruction|single_m1) ;;
  *)
    echo "TASK_TYPE must be m1_subinstruction or single_m1, got: $TASK_TYPE" >&2
    exit 2
    ;;
esac

case "$DATASET" in
  r2r)
    SAMPLE_TARGET=15000
    SEGMENTATION="data/datasets/R2R_VLNCE_v1-3/val_unseen/fgr2r_val_unseen_continuous.json"
    IMAGE_ROOT="data/StreamVLN-Trajectory-Data/R2R/gt_val_unseen_images"
    ;;
  r2r_deviation)
    SAMPLE_TARGET=15000
    SAMPLE_FOCUS="perturbation"
    SEGMENTATION="data/evaluation/deviation/fgr2r_deviation_trajectories.json"
    IMAGE_ROOT="data/StreamVLN-Trajectory-Data/R2R/gt_val_unseen_deviation_images"
    FIXED_SAMPLED_FRAMES="data/evaluation/deviation/fgr2r_deviation_sampled_frames.json"
    ;;
  rxr)
    SAMPLE_TARGET=30000
    SEGMENTATION="data/datasets/rxr/val_unseen/landmark_rxr_val_unseen_en_continuous.json"
    IMAGE_ROOT="data/StreamVLN-Trajectory-Data/RxR"
    ;;
  rxr_deviation)
    SAMPLE_TARGET=30000
    SAMPLE_FOCUS="perturbation"
    SEGMENTATION="data/evaluation/deviation/landmark_rxr_deviation_trajectories.json"
    IMAGE_ROOT="data/StreamVLN-Trajectory-Data/RxR/gt_val_unseen_deviation_images"
    FIXED_SAMPLED_FRAMES="data/evaluation/deviation/landmark_rxr_deviation_sampled_frames.json"
    ;;
  *)
    echo "DATASET must be r2r, r2r_deviation, rxr, or rxr_deviation, got: $DATASET" >&2
    exit 2
    ;;
esac

if [[ -z "${RESULT_DIR:-}" ]]; then
  read -r -p "Result directory (for example: eval_results/m1/r2r/run): " RESULT_DIR
fi
RESULT_DIR="${RESULT_DIR%/}"
if [[ ! "$RESULT_DIR" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$ ]]; then
  echo "RESULT_DIR must be a relative path containing only letters, numbers, dot, underscore, hyphen, and slash" >&2
  exit 2
fi

OUT_PREFIX="${RESULT_DIR}/result"
SAMPLED_FRAMES="${RESULT_DIR}/sampled_${SAMPLE_TARGET}_frames.json"
SAMPLED_FRAMES="${FIXED_SAMPLED_FRAMES:-$SAMPLED_FRAMES}"
echo "Writing evaluation artifacts to: ${RESULT_DIR}"
echo "Task type: ${TASK_TYPE}; dataset: ${DATASET}"

wait_for_vllm_server() {
  local server_url="${1%/}"
  local models_url="${server_url}/v1/models"
  until curl --fail --silent --show-error --connect-timeout 5 --max-time 10 "$models_url" >/dev/null 2>&1; do
    echo "$(date '+%F %T') vLLM is unavailable at ${server_url}; retrying in 10 seconds..."
    sleep 10
  done
  echo "$(date '+%F %T') vLLM is ready at ${server_url}."
}

VLLM_URL="http://127.0.0.1:8086"
wait_for_vllm_server "$VLLM_URL"

python eval_m1_static_qa.py \
  --segmentation "$SEGMENTATION" \
  --image-root "$IMAGE_ROOT" \
  --vllm-url "$VLLM_URL" \
  --vllm-model m1 \
  --task-type "$TASK_TYPE" \
  --frame-policy random_per_segment \
  --sampled-frames "$SAMPLED_FRAMES" \
  --sampling-seed "$SAMPLING_SEED" \
  --sample-target "$SAMPLE_TARGET" \
  --sample-focus "$SAMPLE_FOCUS" \
  --vllm-workers 30 \
  --vllm-max-tokens 256 \
  --device cuda:4 \
  --batch-size 32 \
  --model model_zoo/all-MiniLM-L6-v2 \
  --top2-max-similarity-gap 0.1 \
  --output-prefix "$OUT_PREFIX"
