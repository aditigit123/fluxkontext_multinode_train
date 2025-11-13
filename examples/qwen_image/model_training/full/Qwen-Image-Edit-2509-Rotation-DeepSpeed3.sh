#!/bin/bash

# Rotation Task Training with Qwen-Image-Edit-2509 and DeepSpeed ZeRO-3
# Full fine-tuning with Wandb tracking and S3 data loading
#
# Data: Objaverse HQ filtered rotation pairs from S3
# Training: DeepSpeed ZeRO-3 for efficient multi-GPU training

# ============================================================================
# Wandb Configuration
# ============================================================================
export WANDB_BASE_URL="https://adobesensei.wandb.io/"
export WANDB_API_KEY="local-30b57d5eb821ed35772f8799d7e6fc037a26aa1a"
export WANDB_PROJECT="qwen-img-edit-rotate"
export WANDB_ENTITY="asinghan"
export WANDB_RUN_NAME="qwen2509_rotation_deepspeed3_$(date +%Y%m%d_%H%M%S)"
export WANDB_TAGS="rotation,objaverse,deepspeed3,qwen-image-edit-2509"

# Memory/FlashAttention tuning
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# To disable wandb:
# export WANDB_DISABLED="true"

# ============================================================================
# Data Paths
# ============================================================================
METADATA_PATH="/sensei-fs/users/asinghan/DiffSynth-Studio/data/objaverse_rotation/metadata_rotation.json"
S3_BUCKET="s3://phidias/zchen/czq_objaverse_toon_1280"

# Verify metadata exists
if [ ! -f "$METADATA_PATH" ]; then
    echo "ERROR: Metadata file not found: $METADATA_PATH"
    exit 1
fi

echo "============================================================================"
echo "Starting Qwen-Image-Edit-2509 Rotation Training with DeepSpeed ZeRO-3"
echo "============================================================================"
echo "Metadata: $METADATA_PATH"
echo "S3 Bucket: $S3_BUCKET"
echo "Wandb Project: $WANDB_PROJECT"
echo "Wandb Entity: $WANDB_ENTITY"
echo "============================================================================"

# ============================================================================
# Training Launch
# ============================================================================

accelerate launch \
  --config_file examples/qwen_image/model_training/full/accelerate_config_zero3_compatible.yaml \
  examples/qwen_image/model_training/train_s3_deepspeed.py \
  --dataset_base_path "" \
  --s3_bucket "$S3_BUCKET" \
  --dataset_metadata_path "$METADATA_PATH" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --batch_size 8 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --save_steps 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/mnt/localssd/Qwen-Image-Edit-2509-Rotation-DeepSpeed3" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --gradient_accumulation_steps 1

# ============================================================================
# Configuration Details
# ============================================================================
#
# Model: Qwen-Image-Edit-2509 (Latest version)
# Training: Full fine-tuning with DeepSpeed ZeRO-3
# 
# Dataset:
#   - Source: Objaverse HQ filtered rotation pairs
#   - Total: ~1.26M pairs
#   - Loading: Direct from S3 (no local download)
#   - Preprocessing: RGBA composition with backgrounds/edges
#
# DeepSpeed ZeRO-3:
#   - Stage: 3 (full model sharding)
#   - Mixed Precision: BF16
#   - GPUs: 8 (configured in accelerate_config_zero3.yaml)
#   - Gradient Checkpointing: Enabled
#
# Training Settings:
#   - Learning rate: 1e-5
#   - Epochs: 10
#   - Batch size: 1 per GPU
#   - Gradient accumulation: 1
#   - Max resolution: 1024x1024
#   - Dataset repeat: 1x (1.26M samples per epoch)
#
# Wandb:
#   - URL: https://adobesensei.wandb.io/
#   - Project: qwen-img-edit-rotate
#   - Entity: asinghan
#   - Logging: Loss, learning rate, epoch metrics, checkpoints
#
# Output:
#   - Checkpoints: ./models/train/Qwen-Image-Edit-2509-Rotation-DeepSpeed3/
#   - Format: SafeTensors (epoch_*.safetensors)
#
# ============================================================================

