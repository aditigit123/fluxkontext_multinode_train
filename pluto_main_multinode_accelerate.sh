#!/bin/bash

################################################################################
# FLUX Kontext Multinode Training - Pluto Main Script (PRODUCTION)
################################################################################
# This script runs FLUX Kontext multinode training using Accelerate + DeepSpeed ZeRO-3
# Compatible with Pluto job submission system
#
# Environment variables should be set by Pluto launcher:
#   - NUM_NODES, NUM_GPUS, RANK, MASTER_ADDR, MASTER_PORT
#   - EXPERIMENT_NAME, LOG_DIR, S3_EXPERIMENT_FOLDER
#   - WANDB_API_KEY, WANDB_RUN_NAME
################################################################################

# Enable debug mode and exit on error
set -x
set -e

# ============================================================================ #
# Environment Configuration
# ============================================================================ #
# User-defined variables (set by Pluto launcher or use defaults for manual testing)

export PROJECT_NAME="${PROJECT_NAME:-illustrator_pg1}"
export NUM_NODES="${NUM_NODES:-2}"
export NUM_GPUS="${NUM_GPUS:-8}"
export WANDB_API_KEY="${WANDB_API_KEY:-local-30b57d5eb821ed35772f8799d7e6fc037a26aa1a}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-training}"
export S3_EXPERIMENT_FOLDER="${S3_EXPERIMENT_FOLDER:-s3://phidias/experiments/flux-kontext-rotation-multinode}"
export LOG_DIR="${LOG_DIR:-/mnt/localssd/mori-logdir/flux-kontext-rotation}"
export GITHUB_BRANCH="${GITHUB_BRANCH:-main}"
export COMMIT_HASH="${COMMIT_HASH:-HEAD}"
export KEEP_CKPT_STEP_DIV_BY="${KEEP_CKPT_STEP_DIV_BY:-}"
export PREFLIGHT_TEST_TIMEOUT="${PREFLIGHT_TEST_TIMEOUT:-300}"

# Pluto sets RANK automatically - it's the node rank (0, 1, 2, ...)
export PLUTO_NODE_RANK=${RANK:-0}
export NODE_RANK=$PLUTO_NODE_RANK

# Reduce S3 torch connector throughput to avoid API throttling
export S3_CONNECTOR_THROUGHPUT_TARGET_GBPS=5

# ============================================================================ #
# Cleanup Handler (CRITICAL for Pluto)
# ============================================================================ #
# This cleanup process is critical for Pluto to show main script exit status
# If there are orphaned processes, the Pluto Job will hang indefinitely

# Array to hold background process IDs
BG_PIDS=()

# Cleanup function to terminate background processes
cleanup() {
    echo "Running cleanup..."
    # Use a copy of the PIDs array in case the trap gets called recursively or concurrently
    local pids_to_kill=("${BG_PIDS[@]}")
    for pid in "${pids_to_kill[@]}"; do
        # Check if the process exists using kill -0
        if kill -0 "$pid" 2>/dev/null; then
            echo "Attempting to terminate process $pid..."
            # Send SIGTERM first for graceful shutdown
            kill -TERM "$pid"
            # Wait briefly
            sleep 2
            # Check again and force kill if necessary
            if kill -0 "$pid" 2>/dev/null; then
                echo "Process $pid did not terminate gracefully, sending SIGKILL..."
                kill -KILL "$pid"
            else
                echo "Process $pid terminated successfully."
            fi
        else
            echo "Process $pid does not exist or already terminated."
        fi
    done
    # Clear the global array after cleanup attempt
    BG_PIDS=()
    echo "Cleanup finished."
}

# Register the cleanup function to run on EXIT signal
trap cleanup EXIT

# ============================================================================ #
# Shared TIMESTAMP across all nodes (CRITICAL for multinode)
# ============================================================================ #
# All nodes must use the same TIMESTAMP for consistent checkpoint paths and experiment names
# Node 0 generates it, other nodes wait and read from shared filesystem

TIMESTAMP_FILE="/sensei-fs/users/asinghan/.flux_kontext_timestamp"

if [ "$NODE_RANK" -eq 0 ]; then
    # Node 0 generates the timestamp and writes it to shared storage
    export JOB_LAUNCH_TIMESTAMP=$(date +%s)
    export TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
    echo "$TIMESTAMP" > "$TIMESTAMP_FILE"
    echo "✓ Node 0 wrote TIMESTAMP=$TIMESTAMP to $TIMESTAMP_FILE"
else
    # Other nodes wait until the timestamp file exists
    echo "Node $NODE_RANK waiting for TIMESTAMP_FILE=$TIMESTAMP_FILE..."
    WAIT_COUNT=0
    while [ ! -f "$TIMESTAMP_FILE" ]; do
        sleep 1
        WAIT_COUNT=$((WAIT_COUNT + 1))
        if [ $WAIT_COUNT -gt 60 ]; then
            echo "❌ ERROR: Timeout waiting for TIMESTAMP_FILE from Node 0"
            exit 1
        fi
    done
    export JOB_LAUNCH_TIMESTAMP=$(date +%s)
    export TIMESTAMP="$(cat "$TIMESTAMP_FILE")"
    echo "✓ Node $NODE_RANK read TIMESTAMP=$TIMESTAMP from $TIMESTAMP_FILE"
fi

# Now set experiment name and WandB run name using the shared TIMESTAMP
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-flux_kontext_rotation_multinode_${TIMESTAMP}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT_NAME}}"

echo "✓ Shared experiment configuration:"
echo "  TIMESTAMP:        $TIMESTAMP"
echo "  EXPERIMENT_NAME:  $EXPERIMENT_NAME"
echo "  WANDB_RUN_NAME:   $WANDB_RUN_NAME"

# ============================================================================ #
# Python & pip bootstrap (must run BEFORE any pip usage)
# ============================================================================ #
# This section ensures pip is available before any subsequent pip commands
echo "================================================================================"
echo "Python & pip Bootstrap"
echo "================================================================================"

# Prefer python3; alias 'python' for convenience and set PYTHON_BIN for subprocesses
if ! command -v python &> /dev/null; then
    echo "⚠️  'python' not found, creating alias to python3"
    alias python=python3
    export PYTHON_BIN=python3
else
    export PYTHON_BIN=python
fi

echo "Python version:"
${PYTHON_BIN} --version || true

# Ensure pip exists (images sometimes ship Python without pip)
echo ""
echo "Checking for pip module..."
if ! ${PYTHON_BIN} -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pip') else 1)" 2>/dev/null; then
    echo "🔧 pip module not found, bootstrapping..."
    
    # Try ensurepip first (built-in, no network required)
    echo "  Trying ensurepip..."
    ${PYTHON_BIN} -m ensurepip --upgrade 2>/dev/null || echo "  ensurepip not available or failed"
fi

# If still no pip, try get-pip.py (requires network egress)
if ! ${PYTHON_BIN} -m pip --version >/dev/null 2>&1; then
    echo "🔧 Fetching get-pip.py as fallback..."
    if curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py; then
        if [ -s /tmp/get-pip.py ]; then
            ${PYTHON_BIN} /tmp/get-pip.py --user || echo "⚠️  get-pip.py failed"
            rm -f /tmp/get-pip.py
            # Update PATH to include user site packages
            export PATH="$HOME/.local/bin:$PATH"
        fi
    else
        echo "⚠️  Failed to download get-pip.py (network issue?)"
    fi
fi

# Final check and report status
echo ""
if ${PYTHON_BIN} -m pip --version >/dev/null 2>&1; then
    echo "✓ Pip is available"
    ${PYTHON_BIN} -m pip --version
    export HAVE_PIP=1
    
    # Optionally upgrade pip (set UPGRADE_PIP=1 to enable)
    if [ "${UPGRADE_PIP:-0}" = "1" ]; then
        echo "Upgrading pip (UPGRADE_PIP is set)..."
        ${PYTHON_BIN} -m pip install --upgrade pip setuptools wheel || echo "⚠️  pip upgrade failed"
    fi
else
    echo "⚠️  Pip is unavailable; will skip package installations"
    export HAVE_PIP=0
fi

# Ensure user site packages are in PATH
export PATH="$HOME/.local/bin:$PATH"

echo "✓ Python bootstrap complete"
echo ""

# ============================================================================ #
# Disable FI_* Environment Variables for H100
# ============================================================================ #
# According to: https://github.com/aws-samples/awsome-distributed-training
is_h100() {
    # Check if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        # Use nvidia-smi to query GPU names and check for H100
        gpu_names=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)
        echo "$gpu_names" | grep -q "H100"
        return $?
    else
        echo "nvidia-smi not found. Please make sure NVIDIA drivers are installed."
        return 1
    fi
}

# Check if the GPU is H100
if is_h100; then
    echo "H100 GPU detected. Unsetting FI_* environment variables."
    for var in $(env | grep '^FI_' | awk -F= '{print $1}'); do
        echo "Unsetting $var"
        unset $var
    done
else
    echo "H100 GPU not detected. No changes made."
fi

# ============================================================================ #
# AWS CLI Setup
# ============================================================================ #
if ! command -v aws &> /dev/null; then
    if [ "${HAVE_PIP:-0}" = "1" ]; then
        echo "Installing AWS CLI via pip..."
        ${PYTHON_BIN} -m pip install --user awscli || echo "⚠️  Failed to install awscli; S3 log sync will be disabled."
    else
        echo "⚠️  AWS CLI not found and pip unavailable. S3 log uploads will be disabled."
    fi
else
    echo "✓ AWS CLI already installed."
fi

# Verify AWS CLI availability
if command -v aws &> /dev/null; then
    aws --version
    export HAVE_AWS_CLI=1
else
    echo "⚠️  AWS CLI not available; S3 operations will be skipped"
    export HAVE_AWS_CLI=0
fi

# ============================================================================ #
# Cache Directories Setup
# ============================================================================ #
# export HF_HOME="/mnt/localssd/.cache/huggingface"
# export TRANSFORMERS_CACHE="/mnt/localssd/.cache/huggingface/transformers"
# export HF_DATASETS_CACHE="/mnt/localssd/.cache/huggingface/datasets"
# export TORCH_HOME="/mnt/localssd/.cache/torch"
export HF_HOME="/sensei-fs/users/asinghan/cache_hf_model"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="/sensei-fs/users/asinghan/cache_torch"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME"
export TMPDIR="/mnt/localssd/tmp"
export MODELSCOPE_CACHE="/mnt/localssd/.cache/modelscope"

# Create directories
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR" "$MODELSCOPE_CACHE"
mkdir -p "$LOG_DIR"

# HuggingFace configuration
export HF_ENDPOINT="https://huggingface.co"
export HF_TOKEN=""
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"  # Many libraries expect this
# Note: HF_HUB_ENABLE_HF_TRANSFER requires 'hf_transfer' package - disabled by default
# export HF_HUB_ENABLE_HF_TRANSFER=1  # Uncomment if hf_transfer is installed
export HF_HUB_DISABLE_TELEMETRY=1

# WandB configuration (allow override from submit script/Pluto UI)
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://adobesensei.wandb.io}"

echo "✓ Cache directories configured in /mnt/localssd"

# ============================================================================ #
# Distributed Training Configuration
# ============================================================================ #
export TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))

# Master address - detect from Pluto/SLURM or set manually
if [ -n "$MASTER_ADDR" ]; then
    echo "✓ MASTER_ADDR already set: $MASTER_ADDR"
elif [ -n "$SLURM_JOB_NODELIST" ]; then
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    echo "✓ MASTER_ADDR from SLURM: $MASTER_ADDR"
else
    # CRITICAL: For multinode training, MASTER_ADDR MUST be set!
    if [ "$NUM_NODES" -gt 1 ]; then
        echo "❌ ERROR: MASTER_ADDR not set for multinode training ($NUM_NODES nodes)"
        echo "   Pluto should set this automatically. If running manually, you must set:"
        echo "   export MASTER_ADDR=<ip-address-of-node-0>"
        exit 1
    else
        # Single-node training: localhost is OK
        export MASTER_ADDR="localhost"
        echo "✓ MASTER_ADDR set to localhost (single-node training)"
    fi
fi

export MASTER_PORT=${MASTER_PORT:-29500}

# NCCL Configuration for multinode communication
# IMPORTANT: Verify your network interface!
#   - For InfiniBand: export NCCL_SOCKET_IFNAME=ib0
#   - For Ethernet: export NCCL_SOCKET_IFNAME=eth0 (current default)
#   - Check with: ip addr show | grep -E "ib0|eth0"
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}  # Set to 1 to disable InfiniBand
export NCCL_NET_GDR_LEVEL=0
export NCCL_ASYNC_ERROR_HANDLING=1  # Critical for catching errors early

# NCCL debug (uncomment for troubleshooting)
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# PyTorch Distributed Debug & Deadlock Detection (helps during S3 warmup)
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_BLOCKING_WAIT=1

# Deterministic Training - Global seed across all nodes
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# CPU Thread Management - prevents dataloader contention
export OMP_NUM_THREADS=8

echo "================================================================================"
echo "FLUX Kontext Multinode Training - Pluto/Accelerate/DeepSpeed ZeRO-3"
echo "================================================================================"
echo "Experiment:      $EXPERIMENT_NAME"
echo "Node Rank:       $NODE_RANK / $((NUM_NODES - 1))"
echo "Master Address:  $MASTER_ADDR:$MASTER_PORT"
echo "Total Nodes:     $NUM_NODES"
echo "GPUs per Node:   $NUM_GPUS"
echo "Total GPUs:      $TOTAL_GPUS"
echo "Timestamp:       $TIMESTAMP"
echo "WandB Run:       $WANDB_RUN_NAME"
echo "Log Directory:   $LOG_DIR"
echo "================================================================================"

# ============================================================================ #
# Training Configuration
# ============================================================================ #
# Data paths
# Use JSON metadata file
METADATA_PATH="${METADATA_PATH:-/sensei-fs/users/asinghan/DiffSynth-Studio/data/objaverse_rotation/metadata_kontext_full.json}"

S3_BUCKET="${S3_BUCKET:-s3://phidias/zchen/czq_objaverse_toon_1280}"

# Output paths
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/localssd/FLUX.1-Kontext-Rotation-MULTINODE-${TIMESTAMP}}"
S3_CHECKPOINT_PATH="${S3_CHECKPOINT_PATH:-${S3_EXPERIMENT_FOLDER}/${TIMESTAMP}}"

# Training hyperparameters (can be overridden by environment variables)
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"        # Per GPU - TESTING MODE (reduced from 12 to 8)
GRADIENT_ACCUM="${GRADIENT_ACCUM:-1}"
SAVE_STEPS="${SAVE_STEPS:-10}"      # Save checkpoint every 10 steps - TESTING MODE (reduced from 1000 to 10)
DATA_BATCH_VIZ_STEPS="${DATA_BATCH_VIZ_STEPS:-1000,2000,3000}"
TRAINABLE_MODELS="${TRAINABLE_MODELS:-dit}"
MAX_PIXELS="${MAX_PIXELS:-262144}"

# WandB overrides (entity/project/host)
export WANDB_ENTITY="${WANDB_ENTITY:-asinghan}"
export WANDB_PROJECT="${WANDB_PROJECT:-flux-kontext-rotation}"
export WANDB_HOST="${WANDB_HOST:-$WANDB_BASE_URL}"

# Learning Rate Scheduler Configuration
# Options: constant (no decay), cosine (cosine annealing), linear (linear decay), cosine_with_restarts
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"           # Using cosine for better convergence
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"       # Warmup for 500 steps to stabilize training
LR_NUM_CYCLES="${LR_NUM_CYCLES:-0.5}"           # For cosine_with_restarts only

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRADIENT_ACCUM * TOTAL_GPUS))

echo ""
echo "Training Configuration:"
echo "  Learning Rate:     $LEARNING_RATE"
echo "  LR Scheduler:      $LR_SCHEDULER"
echo "  LR Warmup Steps:   $LR_WARMUP_STEPS"
echo "  Batch Size:        $BATCH_SIZE (per GPU)"
echo "  Gradient Accum:    $GRADIENT_ACCUM"
echo "  Effective Batch:   $EFFECTIVE_BATCH_SIZE"
echo "  Save Interval:     Every $SAVE_STEPS steps"
echo "  Epochs:            $NUM_EPOCHS"
echo "  Dataset Repeat:    ${DATASET_REPEAT}x"
echo ""
echo "Paths:"
echo "  Metadata:          $METADATA_PATH"
echo "  S3 Data:           $S3_BUCKET"
echo "  Local Output:      $OUTPUT_PATH"
echo "  S3 Checkpoints:    $S3_CHECKPOINT_PATH"
echo ""

# ============================================================================ #
# Persist Job Config (only on rank 0)
# ============================================================================ #
if [ "$NODE_RANK" -eq 0 ]; then
    mkdir -p "$LOG_DIR"
    cp "$0" "${LOG_DIR}/pluto_main.sh" 2>/dev/null || true
    
    # Save environment variables for debugging
    env | sort > "${LOG_DIR}/environment_vars.txt"
    
    echo "✓ Job config saved to ${LOG_DIR}/pluto_main.sh"
fi

# ============================================================================ #
# Change to DiffSynth-Studio Directory
# ============================================================================ #
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-/sensei-fs/users/asinghan/DiffSynth-Studio}"
cd "$DIFFSYNTH_DIR"

echo "✓ Working directory: $(pwd)"

# ============================================================================ #
# Pre-flight Test (Optional)
# ============================================================================ #
# Basic GPU health check
echo "Running GPU health check on Node $NODE_RANK..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi || echo "⚠️  nvidia-smi failed"
    echo ""
    echo "GPU Summary:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits || true
    echo ""
else
    echo "⚠️  nvidia-smi not available"
fi

# Check if metadata file exists (only on rank 0)
if [ "$NODE_RANK" -eq 0 ]; then
    if [ ! -f "$METADATA_PATH" ]; then
        echo "❌ Error: Metadata file not found: $METADATA_PATH"
        exit 1
    fi
    
    # Count samples and estimate training time
    # Note: PYTHON_BIN is set later, so use python3 directly here
    SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$METADATA_PATH'))))" 2>/dev/null || echo "unknown")
    if [ "$SAMPLE_COUNT" != "unknown" ]; then
        TOTAL_STEPS=$((SAMPLE_COUNT * DATASET_REPEAT * NUM_EPOCHS / EFFECTIVE_BATCH_SIZE))
        echo "✓ Found $SAMPLE_COUNT samples in metadata"
        echo "✓ Estimated training steps: ~$TOTAL_STEPS"
        echo "✓ Expected checkpoint saves: ~$((TOTAL_STEPS / SAVE_STEPS))"
        echo ""
    fi
fi

# ============================================================================ #
# Log Uploader Setup (Background Process - Rank 0 Only)
# ============================================================================ #
if [ "$NODE_RANK" -eq 0 ] && [ "${HAVE_AWS_CLI:-0}" = "1" ]; then
    echo "Starting log uploader (uploads logs to S3 every 10 minutes)..."
    (
        while true; do
            sleep 600  # Every 10 minutes
            if [ -d "$LOG_DIR" ]; then
                aws s3 sync "$LOG_DIR" "${S3_EXPERIMENT_FOLDER}/logs/" \
                    --exclude "*.pyc" \
                    --exclude "__pycache__/*" \
                    --quiet 2>/dev/null || true
            fi
        done
    ) &
    LOG_UPLOADER_PID=$!
    BG_PIDS+=($LOG_UPLOADER_PID)
    echo "✓ Log uploader started (PID: $LOG_UPLOADER_PID)"
elif [ "$NODE_RANK" -eq 0 ]; then
    echo "⚠️  Log uploader disabled (AWS CLI not available)"
fi

# ============================================================================ #
# S3 Torch Connector Setup
# ============================================================================ #
# Collect s3torchconnector logs for debugging
# https://github.com/awslabs/s3-connector-for-pytorch/blob/main/DEVELOPMENT.md#debugging
export S3_TORCH_CONNECTOR_DEBUG_LOGS="trace,mountpoint_s3_client=debug,awscrt=error"
export S3_TORCH_CONNECTOR_LOGS_DIR_PATH="${LOG_DIR}/s3_torch_connector_logs"
mkdir -p "$S3_TORCH_CONNECTOR_LOGS_DIR_PATH"

# ============================================================================ #
# Python Environment Setup (Additional Checks)
# ============================================================================ #
echo "Checking Python environment configuration..."

# If CONDA_PREFIX is set, we're in a conda environment
if [ -n "$CONDA_PREFIX" ]; then
    echo "✓ Detected conda environment: $CONDA_PREFIX"
fi

# If VIRTUAL_ENV is set, we're in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ Detected virtual environment: $VIRTUAL_ENV"
fi

echo "✓ Python environment check complete"

# ============================================================================ #
# Install DiffSynth-Studio Package
# ============================================================================ #
echo ""
echo "Installing DiffSynth-Studio package and dependencies..."
cd "$DIFFSYNTH_DIR"

# Check if we should skip installation (set SKIP_PIP_INSTALL=1 to skip)
if [ "${SKIP_PIP_INSTALL:-0}" = "1" ]; then
    echo "⚠️  SKIP_PIP_INSTALL is set, skipping pip installation"
elif [ "${HAVE_PIP:-0}" != "1" ]; then
    echo "❌ ERROR: pip is not available and package installation is required."
    echo "   Cannot proceed without pip. Please ensure pip is installed in your Python environment."
    exit 1
else
    # Install the package in editable mode
    if [ -f "requirements.txt" ]; then
        echo "Installing from requirements.txt..."
        ${PYTHON_BIN} -m pip install -r requirements.txt || {
            echo "❌ ERROR: Failed to install requirements.txt"
            exit 1
        }
    else
        echo "⚠️  No requirements.txt found"
    fi

    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        echo "Installing DiffSynth-Studio in editable mode..."
        ${PYTHON_BIN} -m pip install -e . || {
            echo "❌ ERROR: Failed to install DiffSynth-Studio"
            exit 1
        }
    else
        echo "⚠️  No setup.py or pyproject.toml found"
    fi

    # Install additional training dependencies if not already installed
    echo ""
    echo "Installing critical training dependencies..."
    echo "  - accelerate (for distributed training)"
    echo "  - deepspeed (for ZeRO-3 optimization)"
    echo "  - wandb (for logging)"
    echo "  - hf_transfer (for faster HuggingFace downloads)"
    echo ""
    
    ${PYTHON_BIN} -m pip install accelerate deepspeed wandb hf_transfer || {
        echo "⚠️  Failed to install some dependencies in one command, trying individually..."
        ${PYTHON_BIN} -m pip install accelerate || echo "⚠️  Failed to install accelerate"
        ${PYTHON_BIN} -m pip install deepspeed || echo "⚠️  Failed to install deepspeed"
        ${PYTHON_BIN} -m pip install wandb || echo "⚠️  Failed to install wandb"
        ${PYTHON_BIN} -m pip install hf_transfer || echo "⚠️  Failed to install hf_transfer"
    }
    
    # Verify hf_transfer installation and enable if available
    echo ""
    echo "Checking hf_transfer availability..."
    if ${PYTHON_BIN} -c "import importlib.util; exit(0 if importlib.util.find_spec('hf_transfer') else 1)" 2>/dev/null; then
        echo "✓ hf_transfer is installed, enabling fast downloads"
        export HF_HUB_ENABLE_HF_TRANSFER=1
    else
        echo "⚠️  hf_transfer not available, using standard downloads"
    fi
fi

echo ""
echo "✓ Package installation complete"
echo ""

# ============================================================================ #
# Verify Installation
# ============================================================================ #
echo "Verifying installation..."
echo ""
echo "PyTorch version:"
${PYTHON_BIN} -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || echo "⚠️  PyTorch not available"

echo ""
echo "Accelerate version:"
${PYTHON_BIN} -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" || echo "⚠️  Accelerate not installed"

echo ""
echo "DeepSpeed version:"
${PYTHON_BIN} -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')" || echo "⚠️  DeepSpeed not installed"

echo ""
echo "DiffSynth package:"
${PYTHON_BIN} -c "import diffsynth; print(f'DiffSynth: OK')" || echo "⚠️  DiffSynth not available"

echo ""

# ============================================================================ #
# Wait for All Nodes to be Ready (Synchronization Point)
# ============================================================================ #
echo "Node $NODE_RANK is ready. Waiting for all nodes to synchronize..."

if [ "$NUM_NODES" -gt 1 ]; then
    # For multinode: give more time for all nodes to reach this point
    echo "  Multinode training detected - waiting 15 seconds for all nodes..."
    sleep 15
    
    # Additional validation: try to ping master node (except from master itself)
    if [ "$NODE_RANK" -ne 0 ]; then
        echo "  Worker node $NODE_RANK attempting to verify connectivity to master ($MASTER_ADDR)..."
        if ping -c 1 -W 5 "$MASTER_ADDR" &>/dev/null; then
            echo "  ✓ Master node is reachable from worker node $NODE_RANK"
        else
            echo "  ⚠️  Warning: Cannot ping master node $MASTER_ADDR (may be normal if ICMP is blocked)"
        fi
    fi
else
    # Single-node: quick sync
    sleep 5
fi

echo "✓ Node $NODE_RANK proceeding to training launch"

# ============================================================================ #
# Training Execution - Launch with Accelerate + DeepSpeed
# ============================================================================ #

echo ""
echo "================================================================================"
echo "🚀 LAUNCHING MULTINODE TRAINING ON NODE $NODE_RANK"
echo "================================================================================"
echo ""

# Configuration file path
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-examples/flux/model_training/full/accelerate_config_multinode.yaml}"

if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "❌ Error: Accelerate config not found: $ACCELERATE_CONFIG"
    exit 1
fi

# Training script path
TRAINING_SCRIPT="${TRAINING_SCRIPT:-examples/flux/model_training/train_kontext_rotation_s3.py}"

if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "❌ Error: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

echo "Configuration:"
echo "  Accelerate Config: $ACCELERATE_CONFIG"
echo "  Training Script:   $TRAINING_SCRIPT"
echo "  DeepSpeed:         ZeRO-3 (enabled in config)"
echo "  Mixed Precision:   BF16"
echo ""

# Launch training and capture exit code
# IMPORTANT: --num_processes is TOTAL WORLD SIZE, so use TOTAL_GPUS (NUM_NODES × NUM_GPUS)
accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_machines "$NUM_NODES" \
  --num_processes "$TOTAL_GPUS" \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  --rdzv_backend static \
  "$TRAINING_SCRIPT" \
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
  --num_epochs "$NUM_EPOCHS" \
  --lr_scheduler "$LR_SCHEDULER" \
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --lr_num_cycles "$LR_NUM_CYCLES" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_PATH" \
  --trainable_models "$TRAINABLE_MODELS" \
  --use_gradient_checkpointing \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --wandb_entity "$WANDB_ENTITY" \
  --wandb_host "$WANDB_HOST" \
  --s3_checkpoint_path "$S3_CHECKPOINT_PATH"

# Capture exit code
TRAIN_EXIT_CODE=$?

# ============================================================================ #
# Post-Training Status Report
# ============================================================================ #

echo ""
echo "================================================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED SUCCESSFULLY ON NODE $NODE_RANK"
    echo "================================================================================"
    echo ""
    
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "📊 Training Summary:"
        echo "  Experiment:        $EXPERIMENT_NAME"
        echo "  WandB Run:         $WANDB_RUN_NAME"
        echo "  Total GPUs:        $TOTAL_GPUS ($NUM_NODES nodes × $NUM_GPUS GPUs)"
        echo "  Effective Batch:   $EFFECTIVE_BATCH_SIZE"
        echo ""
        echo "💾 Checkpoints:"
        echo "  Local Path:        $OUTPUT_PATH"
        echo "  S3 Path:           $S3_CHECKPOINT_PATH"
        echo ""
        echo "📈 Monitoring:"
        echo "  WandB Dashboard:   ${WANDB_BASE_URL}/${WANDB_ENTITY}/${WANDB_PROJECT}"
        echo "  Run Name:          $WANDB_RUN_NAME"
        echo ""
        echo "📝 Logs:"
        echo "  Local:             $LOG_DIR"
        echo "  S3:                ${S3_EXPERIMENT_FOLDER}/logs/"
        echo ""
        
        # Final log upload
        if [ -d "$LOG_DIR" ] && [ "${HAVE_AWS_CLI:-0}" = "1" ]; then
            echo "Uploading final logs to S3..."
            aws s3 sync "$LOG_DIR" "${S3_EXPERIMENT_FOLDER}/logs/" \
                --exclude "*.pyc" \
                --exclude "__pycache__/*" \
                2>/dev/null || true
            echo "✓ Logs uploaded"
        elif [ -d "$LOG_DIR" ]; then
            echo "⚠️  Skipping log upload (AWS CLI not available)"
        fi
        
        # List saved checkpoints
        if [ -d "$OUTPUT_PATH" ]; then
            echo ""
            echo "Saved checkpoints:"
            ls -lh "$OUTPUT_PATH" | tail -n +2 || true
        fi
        
        # Cleanup shared timestamp file
        if [ -f "$TIMESTAMP_FILE" ]; then
            rm -f "$TIMESTAMP_FILE"
            echo "✓ Cleaned up timestamp file: $TIMESTAMP_FILE"
        fi
    fi
    
    echo ""
    echo "================================================================================"
    echo "Job finished successfully."
    exit 0
    
else
    echo "❌ TRAINING FAILED ON NODE $NODE_RANK"
    echo "================================================================================"
    echo ""
    echo "Exit Code: $TRAIN_EXIT_CODE"
    echo ""
    
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "🔍 Troubleshooting Tips:"
        echo "  1. Check logs in: $LOG_DIR"
        echo "  2. Check WandB for error details"
        echo "  3. Verify NCCL connectivity between nodes"
        echo "  4. Check GPU memory usage: nvidia-smi"
        echo ""
        
        # Upload failure logs
        if [ -d "$LOG_DIR" ] && [ "${HAVE_AWS_CLI:-0}" = "1" ]; then
            echo "Uploading failure logs to S3..."
            aws s3 sync "$LOG_DIR" "${S3_EXPERIMENT_FOLDER}/logs_failed_${TIMESTAMP}/" \
                --exclude "*.pyc" \
                --exclude "__pycache__/*" \
                2>/dev/null || true
            echo "✓ Failure logs uploaded to: ${S3_EXPERIMENT_FOLDER}/logs_failed_${TIMESTAMP}/"
        elif [ -d "$LOG_DIR" ]; then
            echo "⚠️  Skipping failure log upload (AWS CLI not available)"
            echo "   Logs available locally at: $LOG_DIR"
        fi
        
        # Cleanup shared timestamp file
        if [ -f "$TIMESTAMP_FILE" ]; then
            rm -f "$TIMESTAMP_FILE"
            echo "✓ Cleaned up timestamp file: $TIMESTAMP_FILE"
        fi
    fi
    
    echo ""
    echo "================================================================================"
    echo "Job failed with exit code $TRAIN_EXIT_CODE"
    exit 1
fi
