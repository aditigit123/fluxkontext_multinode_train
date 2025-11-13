#!/bin/bash
################################################################################
# Pre-download FLUX Kontext models from HuggingFace (faster than ModelScope)
################################################################################

set -e

export HF_HOME="/mnt/localssd/.cache/huggingface"
export HF_HUB_CACHE="/mnt/localssd/.cache/huggingface/hub"
export HF_TOKEN=""
mkdir -p "$HF_HOME"
mkdir -p "$HF_HUB_CACHE"

echo "================================================================================"
echo "Downloading FLUX Models from HuggingFace"
echo "================================================================================"
echo ""

# Login to HuggingFace
echo "🔑 Logging in to HuggingFace..."
huggingface-cli login --token "$HF_TOKEN"
echo "✓ Logged in successfully"
echo ""

# Download FLUX.1-Kontext-dev (main model)
echo "📦 [1/2] Downloading FLUX.1-Kontext-dev..."
huggingface-cli download black-forest-labs/FLUX.1-Kontext-dev \
  flux1-kontext-dev.safetensors \
  --local-dir /mnt/localssd/.cache/huggingface/hub/black-forest-labs/FLUX.1-Kontext-dev \
  --cache-dir "$HF_HUB_CACHE"

echo "✓ FLUX.1-Kontext-dev downloaded"
echo ""

# Download FLUX.1-dev (text encoders and VAE)
echo "📦 [2/2] Downloading FLUX.1-dev components (text encoders + VAE)..."
huggingface-cli download black-forest-labs/FLUX.1-dev \
  text_encoder/model.safetensors \
  ae.safetensors \
  --local-dir /mnt/localssd/.cache/huggingface/hub/black-forest-labs/FLUX.1-dev \
  --cache-dir "$HF_HUB_CACHE"

# Download text_encoder_2 directory (CLIP model)
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --include "text_encoder_2/*" \
  --local-dir /mnt/localssd/.cache/huggingface/hub/black-forest-labs/FLUX.1-dev \
  --cache-dir "$HF_HUB_CACHE"

echo "✓ FLUX.1-dev components downloaded"
echo ""

echo "================================================================================"
echo "✅ All FLUX models downloaded successfully!"
echo "================================================================================"
echo ""
echo "Downloaded to: /mnt/localssd/.cache/huggingface/hub/"
echo ""
echo "You can now run training without waiting for ModelScope downloads"
echo ""

