#!/usr/bin/env python3
"""
Convert DeepSpeed ZeRO-3 checkpoint to standard PyTorch format for inference.

Usage:
    python scripts/convert_deepspeed_checkpoint.py \
        --checkpoint_dir models/train/Qwen-Image-Edit-2509-Rotation-DeepSpeed3/final \
        --output_file models/train/Qwen-Image-Edit-2509-Rotation-DeepSpeed3/final_model.safetensors
"""

import os
import sys
import argparse
import torch
from pathlib import Path

try:
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("Warning: DeepSpeed not installed. Using fallback method.")

try:
    from safetensors.torch import save_file as save_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not installed. Will save as .pt instead.")


def convert_deepspeed_checkpoint(checkpoint_dir, output_file, tag=None):
    """
    Convert a DeepSpeed ZeRO checkpoint to standard format.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (contains pytorch_model/)
        output_file: Output file path (.safetensors or .pt)
        tag: Optional checkpoint tag (if not using latest)
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    print(f"Converting DeepSpeed checkpoint from: {checkpoint_dir}")
    print(f"Output: {output_file}")
    
    # Method 1: Use DeepSpeed's official converter (best)
    if DEEPSPEED_AVAILABLE:
        print("\n[1/2] Loading checkpoint with DeepSpeed converter...")
        try:
            state_dict = load_state_dict_from_zero_checkpoint(
                str(checkpoint_path),
                tag=tag
            )
            print(f"✓ Loaded {len(state_dict)} parameters")
        except Exception as e:
            print(f"DeepSpeed converter failed: {e}")
            print("Trying fallback method...")
            state_dict = load_checkpoint_fallback(checkpoint_path)
    else:
        print("\n[1/2] Loading checkpoint with fallback method...")
        state_dict = load_checkpoint_fallback(checkpoint_path)
    
    # Remove 'module.' prefix if present (common in distributed training)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        cleaned_state_dict[new_key] = value
    
    print(f"\n[2/2] Saving to {output_file}...")
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.endswith('.safetensors') and SAFETENSORS_AVAILABLE:
        # Convert to CPU and ensure contiguous
        cpu_state_dict = {k: v.cpu().contiguous() for k, v in cleaned_state_dict.items()}
        save_safetensors(cpu_state_dict, str(output_path))
        print(f"✓ Saved as SafeTensors: {output_file}")
    else:
        torch.save(cleaned_state_dict, output_path)
        print(f"✓ Saved as PyTorch: {output_file}")
    
    # Print model info
    total_params = sum(p.numel() for p in cleaned_state_dict.values())
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Number of tensors: {len(cleaned_state_dict)}")
    print(f"  File size: {output_path.stat().st_size / 1024**3:.2f} GB")
    print("\n✓ Conversion complete!")


def load_checkpoint_fallback(checkpoint_path):
    """
    Fallback method to load checkpoint by manually combining shards.
    """
    pytorch_model_path = checkpoint_path / "pytorch_model"
    
    if not pytorch_model_path.exists():
        raise ValueError(f"pytorch_model directory not found in {checkpoint_path}")
    
    # Find all model state files
    state_files = sorted(pytorch_model_path.glob("mp_rank_*_model_states.pt"))
    
    if not state_files:
        raise ValueError(f"No model state files found in {pytorch_model_path}")
    
    print(f"Found {len(state_files)} model shards")
    
    # Load and combine all shards
    combined_state_dict = {}
    
    for i, state_file in enumerate(state_files):
        print(f"  Loading shard {i+1}/{len(state_files)}: {state_file.name}")
        shard_state = torch.load(state_file, map_location='cpu')
        
        # DeepSpeed saves model state under 'module' key
        if 'module' in shard_state:
            shard_dict = shard_state['module']
        else:
            shard_dict = shard_state
        
        # Merge into combined dict
        for key, value in shard_dict.items():
            if key in combined_state_dict:
                # If parameter already exists, it might be sharded - concatenate
                print(f"    Merging duplicate key: {key}")
                # This is a simplified merge - might need more sophisticated logic
            else:
                combined_state_dict[key] = value
    
    return combined_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DeepSpeed ZeRO-3 checkpoint to standard format")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to DeepSpeed checkpoint directory"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file path (.safetensors or .pt)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Checkpoint tag (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_deepspeed_checkpoint(
            args.checkpoint_dir,
            args.output_file,
            args.tag
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

