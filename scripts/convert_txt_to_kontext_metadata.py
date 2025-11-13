#!/usr/bin/env python3
"""
Convert rotation_pairs TXT format to FLUX Kontext metadata

Input TXT format (from complexcheck_filter_objaverse.txt):
  shape_name source_azimuth target_azimuth rotation_angle
  Example: 000-000-0002c6eafa154e8bb08ebafb715a8d46 28 8 135

Output JSON format (for Kontext):
  {
    "image": "shape_name/albedo/elevation_target_azimuth.png",  # Target (rotated)
    "kontext_images": "shape_name/albedo/elevation_source_azimuth.png",  # Source (original)
    "prompt": "Rotate the image X degrees"
  }

The paths are relative to S3 bucket: s3://phidias/zchen/czq_objaverse_toon_1280/
"""

import json
import sys
import argparse


def convert_txt_to_kontext_metadata(
    input_txt_path,
    output_json_path,
    elevation=0,
    max_samples=None
):
    """
    Convert rotation pairs txt to Kontext metadata format
    
    Args:
        input_txt_path: Path to input txt file
        output_json_path: Path to output JSON file
        elevation: Elevation ID to use (default: 0 for front views)
        max_samples: Maximum number of samples to convert (None = all)
    """
    
    print(f"Reading from: {input_txt_path}")
    print(f"Elevation ID: {elevation}")
    
    metadata = []
    
    with open(input_txt_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        
        if len(parts) == 4:
            # Format: shape_name source_azimuth target_azimuth rotation_angle
            shape_name, source_azimuth, target_azimuth, rotation_angle = parts
        elif len(parts) == 5:
            # Format with elevation: shape_name source_azimuth target_azimuth rotation_angle elevation
            shape_name, source_azimuth, target_azimuth, rotation_angle, elev = parts
            elevation = int(elev)  # Use elevation from file
        else:
            print(f"Skipping invalid line {idx+1}: {line}")
            continue
        
        # Construct paths (relative to S3 bucket)
        # These paths will be combined with s3://phidias/zchen/czq_objaverse_toon_1280/
        source_path = f"{shape_name}/albedo/{elevation}_{source_azimuth}.png"
        target_path = f"{shape_name}/albedo/{elevation}_{target_azimuth}.png"
        
        # Create prompt
        prompt = f"Rotate the image {rotation_angle} degrees"
        
        # Create metadata entry (Kontext format)
        entry = {
            "image": target_path,  # Target output (rotated)
            "kontext_images": source_path,  # Context input (original)
            "prompt": prompt
        }
        
        metadata.append(entry)
        
        # Check max samples
        if max_samples and len(metadata) >= max_samples:
            break
    
    print(f"Converted {len(metadata)} samples")
    
    # Save to JSON
    print(f"Writing to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Conversion complete!")
    print(f"\nExample entry:")
    print(json.dumps(metadata[0], indent=2))
    
    print(f"\nTo use this metadata:")
    print(f"  --dataset_metadata_path {output_json_path}")
    print(f"  --s3_bucket s3://phidias/zchen/czq_objaverse_toon_1280")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert rotation pairs TXT to Kontext metadata JSON"
    )
    parser.add_argument(
        "input_txt",
        help="Path to input TXT file (e.g., complexcheck_filter_objaverse.txt)"
    )
    parser.add_argument(
        "output_json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--elevation",
        type=int,
        default=0,
        help="Elevation ID to use (default: 0 for front views)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert (default: all)"
    )
    
    args = parser.parse_args()
    
    convert_txt_to_kontext_metadata(
        args.input_txt,
        args.output_json,
        elevation=args.elevation,
        max_samples=args.max_samples
    )


