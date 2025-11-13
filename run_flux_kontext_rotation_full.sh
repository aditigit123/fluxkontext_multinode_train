#!/bin/bash

################################################################################
# FLUX Kontext Full Fine-Tuning - PRODUCTION MODE
################################################################################
# Full dataset training with 10k step intervals for checkpoints and data viz
################################################################################

set -e  # Exit on error

# ============================================================================
# Environment Variables - Redirect all caches to /mnt/localssd
# ============================================================================
export HF_HOME="/mnt/localssd/.cache/huggingface"
export TRANSFORMERS_CACHE="/mnt/localssd/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/mnt/localssd/.cache/huggingface/datasets"
export TORCH_HOME="/mnt/localssd/.cache/torch"
export TMPDIR="/mnt/localssd/tmp"
export MODELSCOPE_CACHE="/mnt/localssd/.cache/modelscope"

# Force HuggingFace download instead of ModelScope (faster for most regions)
export HF_ENDPOINT="https://huggingface.co"
export HF_TOKEN=""
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"  # Many libraries expect this
export HF_HUB_DISABLE_TELEMETRY=1

# Install hf_transfer for faster downloads if not already installed
if ! python3 -c "import importlib.util; exit(0 if importlib.util.find_spec('hf_transfer') else 1)" 2>/dev/null; then
    echo "Installing hf_transfer for faster HuggingFace downloads..."
    python3 -m pip install --user hf_transfer || echo "⚠️  Failed to install hf_transfer, will use standard downloads"
fi

# Enable fast transfer if available
if python3 -c "import importlib.util; exit(0 if importlib.util.find_spec('hf_transfer') else 1)" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "✓ Fast HF downloads enabled (hf_transfer)"
else
    echo "⚠️  Using standard HF downloads"
fi

# Create cache directories if they don't exist
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$TORCH_HOME"
mkdir -p "$TMPDIR"
mkdir -p "$MODELSCOPE_CACHE"

echo "✓ Cache directories configured in /mnt/localssd"

# ============================================================================
# PRODUCTION Configuration
# ============================================================================

# Generate timestamp for run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${TIMESTAMP}_rotate_flux_kontext_full_objaverse_only"

# Data paths - FULL DATASET
METADATA_PATH="/sensei-fs/users/asinghan/DiffSynth-Studio/data/objaverse_rotation/metadata_kontext_full.json"
S3_BUCKET="s3://phidias/zchen/czq_objaverse_toon_1280"

# Output path (saved to local SSD, not /sensei-fs)
OUTPUT_PATH="/mnt/localssd/FLUX.1-Kontext-Rotation-FULL-PRODUCTION-${TIMESTAMP}"

# S3 backup path for checkpoints
S3_CHECKPOINT_PATH="s3://phidias/experiments/flux-kontext-rotation-full-production/${TIMESTAMP}"

# Visualization configuration - PRODUCTION (10k intervals)
DATA_BATCH_VIZ_STEPS="10000,20000,30000"  # Data batch viz at 10k, 20k, 30k

# Training hyperparameters - PRODUCTION
LEARNING_RATE="1e-5"  # Lower LR for full fine-tuning (vs 1e-4 for LoRA)
NUM_EPOCHS="10"       # Train for 10 epochs to match multinode config
DATASET_REPEAT="1"

# Learning rate scheduler configuration
LR_SCHEDULER="cosine"         # Options: constant, cosine, linear, cosine_with_restarts (cosine recommended for full fine-tuning)
LR_WARMUP_STEPS="500"         # Number of warmup steps (0 = no warmup) - increased for stability
LR_NUM_CYCLES="0.5"           # For cosine_with_restarts only

# Batch size configuration
# Effective batch size = BATCH_SIZE × GRADIENT_ACCUM × NUM_GPUS
# With 8 GPUs: 16 × 1 × 8 = 128 effective batch size (PRODUCTION MODE)
BATCH_SIZE="16"       # Per-GPU batch size (train_micro_batch_size_per_gpu)
GRADIENT_ACCUM="1"    # Gradient accumulation steps

SAVE_STEPS="2000"  # Save checkpoint every 2000 steps (PRODUCTION MODE)

# Full fine-tuning configuration
TRAINABLE_MODELS="dit"  # Train the entire diffusion transformer

# Image settings
MAX_PIXELS="262144"  # 512x512

# ============================================================================
# Print Configuration
# ============================================================================

echo "================================================================================"
echo "FLUX Kontext Full Fine-Tuning - PRODUCTION MODE"
echo "================================================================================"
echo ""
echo "🚀 PRODUCTION TRAINING - Full Objaverse Dataset"
echo ""
echo "Data Configuration:"
echo "  Metadata:        $METADATA_PATH"
echo "  S3 Bucket:       $S3_BUCKET"
echo ""
echo "Training Configuration:"
echo "  Learning Rate:   $LEARNING_RATE"
echo "  Epochs:          $NUM_EPOCHS"
echo "  Dataset Repeat:  ${DATASET_REPEAT}x"
echo ""
echo "Batch Size Configuration:"
echo "  Per-GPU Batch:   $BATCH_SIZE"
echo "  Gradient Accum:  $GRADIENT_ACCUM"
echo "  Num GPUs:        8 (from accelerate config)"
echo "  Effective Batch: $((BATCH_SIZE * GRADIENT_ACCUM * 8))"
echo ""
echo "Checkpoint Configuration:"
echo "  Save Interval:   Every $SAVE_STEPS steps"
echo "  Data Batch Viz:  Steps $DATA_BATCH_VIZ_STEPS"
echo "  Inference Viz:   DISABLED (run separately after training)"
echo ""
echo "Fine-Tuning Configuration:"
echo "  Training Mode:   FULL (entire DIT)"
echo "  Trainable:       $TRAINABLE_MODELS"
echo ""
echo "Output:"
echo "  Local Path:      $OUTPUT_PATH"
echo "  S3 Path:         $S3_CHECKPOINT_PATH"
echo ""
echo "WandB:"
echo "  Run Name:        $RUN_NAME"
echo ""
echo "================================================================================"
echo ""

# Confirm metadata exists
if [ ! -f "$METADATA_PATH" ]; then
    echo "❌ Error: Metadata file not found: $METADATA_PATH"
    exit 1
fi

# Count samples
SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$METADATA_PATH'))))")
NUM_GPUS=8  # From accelerate config
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRADIENT_ACCUM * NUM_GPUS))
TOTAL_STEPS=$((SAMPLE_COUNT * DATASET_REPEAT * NUM_EPOCHS / EFFECTIVE_BATCH_SIZE))
echo "✓ Found $SAMPLE_COUNT samples in metadata"
echo "✓ Effective batch size: $EFFECTIVE_BATCH_SIZE (${BATCH_SIZE} × ${GRADIENT_ACCUM} × ${NUM_GPUS})"
echo "✓ Estimated training steps: ~$TOTAL_STEPS"
echo "✓ Expected checkpoint saves: ~$((TOTAL_STEPS / SAVE_STEPS))"
echo ""

# ============================================================================
# Launch Training
# ============================================================================

echo "Starting production training..."
echo ""

# WandB Configuration (Adobe Sensei self-hosted instance)
export WANDB_API_KEY="local-30b57d5eb821ed35772f8799d7e6fc037a26aa1a"
export WANDB_BASE_URL="https://adobesensei.wandb.io"

# PyTorch Configuration
export PYTHONHASHSEED=0  # Deterministic hashing
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # Deterministic CUDA operations
export OMP_NUM_THREADS=8  # CPU thread management

accelerate launch --config_file examples/flux/model_training/full/accelerate_config_zero3.yaml examples/flux/model_training/train_kontext_rotation_s3.py \
  --dataset_base_path "." \
  --s3_bucket "$S3_BUCKET" \
  --dataset_metadata_path "$METADATA_PATH" \
  --data_file_keys "image,kontext_images" \
  --extra_inputs "kontext_images" \
  --max_pixels "$MAX_PIXELS" \
  --dataset_repeat "$DATASET_REPEAT" \
  --batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUM" \
  --save_steps "$SAVE_STEPS" \
  --data_batch_viz_steps "$DATA_BATCH_VIZ_STEPS" \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-Kontext-dev:flux1-kontext-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler "$LR_SCHEDULER" \
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --lr_num_cycles "$LR_NUM_CYCLES" \
  --num_epochs "$NUM_EPOCHS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_PATH" \
  --trainable_models "$TRAINABLE_MODELS" \
  --use_gradient_checkpointing \
  --use_wandb \
  --wandb_project "flux-kontext-rotation" \
  --wandb_run_name "$RUN_NAME" \
  --wandb_entity "asinghan" \
  --wandb_host "https://adobesensei.wandb.io" \
  --s3_checkpoint_path "$S3_CHECKPOINT_PATH"

echo ""
echo "================================================================================"
echo "Production Full Fine-Tuning Complete!"
echo "================================================================================"
echo ""
echo "Full model checkpoints saved at 10k step intervals to:"
echo "  Local: $OUTPUT_PATH"
echo "  S3:    $S3_CHECKPOINT_PATH"
echo ""
echo "Check WandB for training metrics:"
echo "  URL: https://adobesensei.wandb.io/asinghan/flux-kontext-rotation"
echo "  Run: $RUN_NAME"
echo ""
echo "Note: Full fine-tuned models are significantly larger than LoRA adapters"
echo "================================================================================"


