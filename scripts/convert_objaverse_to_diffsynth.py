#!/usr/bin/env python3
"""
Convert objaverse rotation pairs to DiffSynth metadata format.

This script reads the rotation pairs from your filtered data and creates
a metadata.json file compatible with DiffSynth training.

Input format: shape_name source_azimuth target_azimuth rotation_angle
Output format: JSON with image paths and prompts
"""

import json
import argparse
import random
from pathlib import Path


def convert_rotation_pairs_to_metadata(
    input_file: str,
    output_file: str,
    s3_bucket: str = "s3://phidias/zchen/czq_objaverse_toon_1280",
    max_samples: int = None,
    elevation: int = 0,
):
    """
    Convert rotation pairs txt to DiffSynth metadata JSON.
    
    Args:
        input_file: Path to complexcheck_filter_objaverse_aligned_keep.txt
        output_file: Path to output metadata.json
        s3_bucket: S3 bucket path
        max_samples: Maximum number of samples to include (None = all)
        elevation: Fixed elevation angle (0 for objaverse_toon_hq)
    """
    
    metadata = []
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Reading {len(lines)} rotation pairs from {input_file}")
    
    # Optionally limit samples for faster experimentation
    if max_samples and max_samples < len(lines):
        lines = random.sample(lines, max_samples)
        print(f"Randomly sampled {max_samples} pairs")
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) != 4:
            print(f"Skipping malformed line {line_idx}: {line}")
            continue
        
        shape_name, source_azimuth, target_azimuth, rotation_angle = parts
        
        # Construct image paths (relative to S3 bucket)
        # Format: {shape_name}/albedo/{elevation}_{azimuth}.png
        source_image = f"{shape_name}/albedo/{elevation}_{source_azimuth}.png"
        target_image = f"{shape_name}/albedo/{elevation}_{target_azimuth}.png"
        
        # Create prompt
        rotation_deg = int(rotation_angle)
        if rotation_deg == 0:
            prompt = "Rotate the image 0 degrees"
        else:
            prompt = f"Rotate the image {rotation_angle} degrees"
        
        metadata.append({
            "image": source_image,
            "edit_image": target_image,
            "prompt": prompt
        })
        
        if (line_idx + 1) % 10000 == 0:
            print(f"Processed {line_idx + 1} pairs...")
    
    print(f"Created {len(metadata)} metadata entries")
    
    # Save metadata
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {output_file}")
    print(f"\nNote: This metadata references S3 paths. Make sure your data is:")
    print(f"  1. Downloaded locally to dataset_base_path, OR")
    print(f"  2. S3 is mounted as local filesystem")
    
    # Print sample entries
    print(f"\nSample metadata entries:")
    for sample in metadata[:3]:
        print(f"  Source: {sample['image']}")
        print(f"  Target: {sample['edit_image']}")
        print(f"  Prompt: {sample['prompt']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Convert objaverse rotation pairs to DiffSynth metadata")
    parser.add_argument(
        "--input",
        type=str,
        default="/sensei-fs/users/asinghan/ConsiGenData/rotate-vector-data-filtering/complexcheck_filter_objaverse_aligned_keep.txt",
        help="Path to input rotation pairs file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/sensei-fs/users/asinghan/DiffSynth-Studio/data/objaverse_rotation/metadata_rotation.json",
        help="Path to output metadata JSON"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default="s3://phidias/zchen/czq_objaverse_toon_1280",
        help="S3 bucket path"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples (for testing, default: all)"
    )
    parser.add_argument(
        "--elevation",
        type=int,
        default=0,
        help="Elevation angle (default: 0)"
    )
    
    args = parser.parse_args()
    
    convert_rotation_pairs_to_metadata(
        input_file=args.input,
        output_file=args.output,
        s3_bucket=args.s3_bucket,
        max_samples=args.max_samples,
        elevation=args.elevation,
    )


if __name__ == "__main__":
    main()

