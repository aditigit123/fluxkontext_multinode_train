#!/usr/bin/env python3
"""
Multi-GPU Inference Script for Qwen-Image-Edit-2509
Processes 102 expression prompts across 15 input images
"""

import os
import torch
import torch.multiprocessing as mp
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from pathlib import Path
import time
from tqdm import tqdm

# Configuration
INPUT_DIR = "/mnt/localssd/consi_gen_output"
OUTPUT_DIR = "/mnt/localssd/consi_gen_output/qwen_edit"
EXPRESSIONS_FILE = "/mnt/localssd/consi_gen_output/expressions_low.txt"

# Input images (15 characters)
CHARACTERS = [
    'apple', 'ballon', 'boy', 'candle', 'cat', 'clock', 'dog', 'girl',
    'man', 'mohawk_man', 'monkey', 'old_woman', 'owl', 'penguine', 'teapot'
]

# Inference parameters
NUM_INFERENCE_STEPS = 40
TRUE_CFG_SCALE = 4.0
GUIDANCE_SCALE = 1.0
NUM_IMAGES_PER_PROMPT = 1
SEED = 42


def load_expressions(filepath):
    """Load expression prompts from file."""
    with open(filepath, 'r') as f:
        expressions = [line.strip() for line in f.readlines() if line.strip()]
    return expressions


def process_batch_on_gpu(gpu_id, prompt_indices, expressions, characters, input_dir, output_dir):
    """
    Process a batch of prompts on a specific GPU.
    
    Args:
        gpu_id: GPU device ID
        prompt_indices: List of prompt indices to process
        expressions: List of all expression prompts
        characters: List of character names
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
    """
    print(f"[GPU {gpu_id}] Starting processing {len(prompt_indices)} prompts...")
    
    # Set device
    device = f'cuda:{gpu_id}'
    
    # Load pipeline on this GPU
    print(f"[GPU {gpu_id}] Loading Qwen-Image-Edit-2509 pipeline...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    print(f"[GPU {gpu_id}] Pipeline loaded successfully!")
    
    # Process each prompt
    for prompt_idx in tqdm(prompt_indices, desc=f"GPU {gpu_id}", position=gpu_id):
        expression = expressions[prompt_idx]
        
        # Determine output folder structure (matching existing structure)
        if prompt_idx < 34:
            output_folder = os.path.join(output_dir, "outputs", f"output_{prompt_idx + 1}")
        else:
            output_folder = os.path.join(output_dir, "outputs_medium", f"output_{prompt_idx - 33}")
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Process each character
        for char in characters:
            input_path = os.path.join(input_dir, f"{char}.png")
            output_path = os.path.join(output_folder, f"{char}.png")
            
            # Skip if already processed
            if os.path.exists(output_path):
                continue
            
            try:
                # Load input image
                image = Image.open(input_path).convert('RGB')
                
                # Run inference
                with torch.inference_mode():
                    output = pipeline(
                        image=image,
                        prompt=expression,
                        generator=torch.manual_seed(SEED),
                        true_cfg_scale=TRUE_CFG_SCALE,
                        negative_prompt=" ",
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                    )
                    output_image = output.images[0]
                    output_image.save(output_path)
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing {char} with prompt {prompt_idx + 1}: {e}")
                continue
    
    print(f"[GPU {gpu_id}] Completed processing!")


def distribute_work(num_gpus, total_prompts):
    """
    Distribute prompts across GPUs evenly.
    
    Args:
        num_gpus: Number of available GPUs
        total_prompts: Total number of prompts to process
    
    Returns:
        List of prompt indices for each GPU
    """
    prompts_per_gpu = total_prompts // num_gpus
    remainder = total_prompts % num_gpus
    
    gpu_assignments = []
    start_idx = 0
    
    for i in range(num_gpus):
        # Add extra prompt to first 'remainder' GPUs
        end_idx = start_idx + prompts_per_gpu + (1 if i < remainder else 0)
        gpu_assignments.append(list(range(start_idx, end_idx)))
        start_idx = end_idx
    
    return gpu_assignments


def main():
    # Load expressions
    print("Loading expression prompts...")
    expressions = load_expressions(EXPRESSIONS_FILE)
    print(f"Loaded {len(expressions)} expression prompts")
    
    # Verify input images exist
    print("\nVerifying input images...")
    missing_chars = []
    for char in CHARACTERS:
        input_path = os.path.join(INPUT_DIR, f"{char}.png")
        if not os.path.exists(input_path):
            missing_chars.append(char)
    
    if missing_chars:
        print(f"WARNING: Missing input images for: {missing_chars}")
        return
    print(f"All {len(CHARACTERS)} input images found!")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Determine number of GPUs to use
    num_gpus = torch.cuda.device_count()
    print(f"\nUsing {num_gpus} GPUs for parallel processing")
    
    # Distribute work across GPUs
    gpu_assignments = distribute_work(num_gpus, len(expressions))
    
    print("\nWork distribution:")
    for gpu_id, prompt_indices in enumerate(gpu_assignments):
        print(f"  GPU {gpu_id}: {len(prompt_indices)} prompts (indices {prompt_indices[0]}-{prompt_indices[-1]})")
    
    total_images = len(expressions) * len(CHARACTERS)
    print(f"\nTotal images to generate: {total_images} ({len(expressions)} prompts × {len(CHARACTERS)} characters)")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Start time
    start_time = time.time()
    
    # Launch parallel processing on multiple GPUs
    print("\nStarting multi-GPU inference...\n")
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for gpu_id, prompt_indices in enumerate(gpu_assignments):
        p = mp.Process(
            target=process_batch_on_gpu,
            args=(gpu_id, prompt_indices, expressions, CHARACTERS, INPUT_DIR, OUTPUT_DIR)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"Total images generated: {total_images}")
    print(f"Output saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


