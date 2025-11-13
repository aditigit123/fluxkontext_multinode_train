#!/usr/bin/env python3
"""
Test script for FLUX Kontext Rotation LoRA checkpoint.
Usage: python test_rotation_checkpoint.py --checkpoint /path/to/checkpoint.safetensors
"""

import argparse
import os
import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Test FLUX Kontext Rotation LoRA checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/localssd/FLUX.1-Kontext-Rotation-LoRA-TEST/step-25.safetensors",
        help="Path to LoRA checkpoint (.safetensors file)"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Path to input image (if None, will generate a sample image first)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_outputs",
        help="Directory to save output images"
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=1.0,
        help="LoRA weight (0.0 to 1.0, higher = stronger LoRA effect)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Output image height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Output image width"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps (higher = better quality, slower)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--embedded_guidance",
        type=float,
        default=6.0,
        help="Embedded guidance for FLUX Kontext"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("FLUX Kontext Rotation LoRA Inference")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"Image size: {args.height}x{args.width}")
    print(f"Inference steps: {args.num_inference_steps}")
    print()
    
    # Load pipeline
    print("Loading FLUX.1-Kontext-dev pipeline...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="black-forest-labs/FLUX.1-Kontext-dev",
                origin_file_pattern="flux1-kontext-dev.safetensors"
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="text_encoder/model.safetensors"
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="text_encoder_2/"
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="ae.safetensors"
            ),
        ],
    )
    print("✅ Pipeline loaded")
    
    # Load LoRA checkpoint
    print(f"Loading LoRA checkpoint: {args.checkpoint}")
    pipe.load_lora(pipe.dit, args.checkpoint, alpha=args.lora_alpha)
    print(f"✅ LoRA loaded with alpha={args.lora_alpha}")
    print()
    
    # Get or generate input image
    if args.input_image is None:
        print("No input image provided, generating a sample image first...")
        context_image = pipe(
            prompt="A red cube on a wooden table",
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            embedded_guidance=args.embedded_guidance,
            seed=args.seed
        )
        context_image_path = os.path.join(args.output_dir, "context_image.jpg")
        context_image.save(context_image_path)
        print(f"✅ Generated context image: {context_image_path}")
    else:
        print(f"Loading input image: {args.input_image}")
        context_image = Image.open(args.input_image).convert("RGB")
        # Resize to match output dimensions
        context_image = context_image.resize((args.width, args.height))
        print("✅ Input image loaded")
    
    print()
    
    # Test different rotation angles
    rotation_angles = [90, 180, 270, 45, 135]
    
    print(f"Generating rotations for {len(rotation_angles)} angles...")
    print()
    
    for angle in rotation_angles:
        print(f"  Rotating by {angle} degrees...")
        
        prompt = f"Please rotate the image {angle} degrees"
        
        rotated_image = pipe(
            prompt=prompt,
            kontext_images=context_image,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            embedded_guidance=args.embedded_guidance,
            seed=args.seed + angle  # Different seed per angle
        )
        
        output_path = os.path.join(args.output_dir, f"rotated_{angle}_degrees.jpg")
        rotated_image.save(output_path)
        print(f"    ✅ Saved: {output_path}")
    
    print()
    print("=" * 80)
    print("✅ Inference complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print("=" * 80)
    print()
    print("Example outputs:")
    print(f"  - Context image: {args.output_dir}/context_image.jpg")
    for angle in rotation_angles:
        print(f"  - Rotated {angle}°: {args.output_dir}/rotated_{angle}_degrees.jpg")


if __name__ == "__main__":
    main()


