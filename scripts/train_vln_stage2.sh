#!/bin/bash
#
# train_vln_stage2.sh — Stage 2 M2 SFT via Slurm sbatch
#
# Usage:
#   bash scripts/train_vln_stage2.sh submit    # Submit Slurm job (default)
#   bash scripts/train_vln_stage2.sh status    # Show current user job status
#   bash scripts/train_vln_stage2.sh cancel    # Cancel jobs from job_id file
#   bash scripts/train_vln_stage2.sh logs      # Tail recent Slurm logs

set -euo pipefail

ACTION="${1:-submit}"

# ===================== CONFIG: modify only here =====================
# ---------- Slurm resource params ----------
PARTITION="gpu"
SLURM_TIME="16:00:00"
SLURM_GPU_COUNT=4
SLURM_CPUS_PER_TASK=$((SLURM_GPU_COUNT * 8))
SLURM_MEM="$((SLURM_GPU_COUNT * 64))G"
# Effective global batch: 4 GPUs x 8 samples x 4 accumulation steps = 128.

# ---------- Basic config ----------
CONDA_ENV="your_environment"

# ---------- Training config ----------
MODEL="path/to/m2_checkpoint"
TRAIN_DATA="your_m2_dataset"
OUTPUT_DIR="path/to/m2_output"

NUM_TRAIN_EPOCHS=2
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-6
MAX_LENGTH=8192
# ====================================================================

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT="$PWD"
LOG_DIR="$PROJECT_ROOT/slurm_logs/train_stage2"
JOB_ID_FILE="$LOG_DIR/.job_ids"
mkdir -p "$LOG_DIR"

# ---- sbatch script generation ----
JOB_NAME="s2-m2lm"

generate_sbatch() {
    cat > "$LOG_DIR/${JOB_NAME}.sbatch" << 'SBATCH_EOF'
#!/bin/bash
#SBATCH -J JOB_NAME_PLACEHOLDER
#SBATCH -p PARTITION_PLACEHOLDER
#SBATCH --gres=gpu:GPU_COUNT_PLACEHOLDER
#SBATCH --cpus-per-task=CPUS_PLACEHOLDER
#SBATCH --mem=MEM_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH -o LOG_DIR_PLACEHOLDER/JOB_NAME_PLACEHOLDER-%j.out
#SBATCH -e LOG_DIR_PLACEHOLDER/JOB_NAME_PLACEHOLDER-%j.err

set -euo pipefail

echo "============================================"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GPU count (gres): GPU_COUNT_PLACEHOLDER"
echo "============================================"
nvidia-smi

source ~/.bashrc
set +u
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
conda activate CONDA_ENV_PLACEHOLDER || { echo "ERROR: conda activate failed"; exit 1; }
set -u

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export MAX_PIXELS=1003520
export IMAGE_MAX_TOKEN_NUM=2048
export MODELSCOPE_OFFLINE=1
unset ROOT_IMAGE_DIR

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^veth,lo
export MASTER_PORT=29519

echo "=========================================="
echo "Stage 2 M2 SFT: MODEL_PATH_PLACEHOLDER"
echo "Dataset: DATASET_PLACEHOLDER"
echo "Output: OUTPUT_DIR_PLACEHOLDER"
echo "=========================================="

NNODES=1 \
NPROC_PER_NODE=GPU_COUNT_PLACEHOLDER \
python -m torch.distributed.run \
    --nproc_per_node=GPU_COUNT_PLACEHOLDER \
    --master_port=${MASTER_PORT} \
    --nnodes=1 \
    --node_rank=0 \
    scripts/train_with_aug.py \
    --model MODEL_PATH_PLACEHOLDER \
    --model_type qwen2_5_vl \
    --custom_dataset_info data/dataset_info.json \
    --dataset DATASET_PLACEHOLDER \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs NUM_EPOCHS_PLACEHOLDER \
    --per_device_train_batch_size BATCH_SIZE_PLACEHOLDER \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps GRAD_ACCUM_PLACEHOLDER \
    --learning_rate LR_PLACEHOLDER \
    --freeze_vit true \
    --freeze_aligner false \
    --attn_impl flash_attn \
    --gradient_checkpointing true \
    --packing false \
    --eval_steps 100000 \
    --save_steps 50 \
    --save_total_limit 500 \
    --save_only_model true \
    --logging_steps 10 \
    --max_length MAX_LEN_PLACEHOLDER \
    --output_dir OUTPUT_DIR_PLACEHOLDER \
    --warmup_ratio 0.03 \
    --deepspeed zero3 \
    --dataset_num_proc 32 \
    --dataloader_num_workers 32 \
    --load_from_cache_file false \
    --split_dataset_ratio 0 \
    --eval_strategy no

echo "Training done. Output: OUTPUT_DIR_PLACEHOLDER"
SBATCH_EOF

    sed -i \
        -e "s|JOB_NAME_PLACEHOLDER|$JOB_NAME|g" \
        -e "s|PARTITION_PLACEHOLDER|$PARTITION|g" \
        -e "s|GPU_COUNT_PLACEHOLDER|$SLURM_GPU_COUNT|g" \
        -e "s|CPUS_PLACEHOLDER|$SLURM_CPUS_PER_TASK|g" \
        -e "s|MEM_PLACEHOLDER|$SLURM_MEM|g" \
        -e "s|TIME_PLACEHOLDER|$SLURM_TIME|g" \
        -e "s|LOG_DIR_PLACEHOLDER|$LOG_DIR|g" \
        -e "s|CONDA_ENV_PLACEHOLDER|$CONDA_ENV|g" \
        -e "s|MODEL_PATH_PLACEHOLDER|$MODEL|g" \
        -e "s|DATASET_PLACEHOLDER|$TRAIN_DATA|g" \
        -e "s|OUTPUT_DIR_PLACEHOLDER|$OUTPUT_DIR|g" \
        -e "s|NUM_EPOCHS_PLACEHOLDER|$NUM_TRAIN_EPOCHS|g" \
        -e "s|BATCH_SIZE_PLACEHOLDER|$PER_DEVICE_TRAIN_BATCH_SIZE|g" \
        -e "s|GRAD_ACCUM_PLACEHOLDER|$GRADIENT_ACCUMULATION_STEPS|g" \
        -e "s|LR_PLACEHOLDER|$LEARNING_RATE|g" \
        -e "s|MAX_LEN_PLACEHOLDER|$MAX_LENGTH|g" \
        "$LOG_DIR/${JOB_NAME}.sbatch"

    echo "$LOG_DIR/${JOB_NAME}.sbatch"
}

# ---- Action dispatch ----
case "$ACTION" in
    status)
        echo "===== Current user Slurm jobs ($USER) ====="
        squeue -u "$USER" || echo "Cannot query job status"
        echo ""
        echo "===== Local job records ====="
        if [ -f "$JOB_ID_FILE" ]; then cat "$JOB_ID_FILE"; else echo "(none)"; fi
        exit 0
        ;;
    cancel)
        if [ ! -f "$JOB_ID_FILE" ] || [ ! -s "$JOB_ID_FILE" ]; then
            echo "No job records found"; exit 0
        fi
        echo "Cancelling jobs:"; cat "$JOB_ID_FILE"
        while read -r name jid; do
            [ -n "$jid" ] && scancel "$jid" 2>&1 || true
        done < "$JOB_ID_FILE"
        rm -f "$JOB_ID_FILE"
        echo "Done."
        exit 0
        ;;
    logs)
        echo "===== Recent Slurm logs ====="
        if [ -d "$LOG_DIR" ]; then
            ls -lt "$LOG_DIR"/*.out "$LOG_DIR"/*.err 2>/dev/null | head -10
            latest=$(ls -t "$LOG_DIR"/*.out 2>/dev/null | head -1)
            [ -n "$latest" ] && tail -f "$latest"
        fi
        exit 0
        ;;
    submit) ;;
    *)
        echo "ERROR: Unknown ACTION=$ACTION, available: submit | status | cancel | logs"
        exit 1
        ;;
esac

# ---- Submit ----
echo "============================================"
echo "  Stage 2 M2 SFT Slurm Launcher"
echo "  Dataset: $TRAIN_DATA"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs: $SLURM_GPU_COUNT | Epochs: $NUM_TRAIN_EPOCHS"
echo "============================================"

SBATCH_FILE=$(generate_sbatch)
echo "Generated sbatch: $SBATCH_FILE"
echo "Submitting $JOB_NAME ..."
output=$(sbatch "$SBATCH_FILE" 2>&1)
echo "  $output"
job_id=$(echo "$output" | grep -oP 'Submitted batch job \K\d+')
if [ -n "$job_id" ]; then
    echo "$JOB_NAME $job_id" >> "$JOB_ID_FILE"
    echo "  Job ID: $job_id"
fi
echo ""
echo "Done. Monitor with:"
echo "  bash scripts/train_vln_stage2.sh status"
echo "  bash scripts/train_vln_stage2.sh logs"
