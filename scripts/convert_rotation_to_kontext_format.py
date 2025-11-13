#!/usr/bin/env python3
"""
Convert rotation metadata to FLUX Kontext format

BEFORE (rotation format):
  "image": "original.png"        <- source image
  "edit_image": "rotated.png"    <- target rotated image
  "prompt": "Rotate 90 degrees"

AFTER (kontext format):
  "kontext_images": "original.png"  <- context/input image
  "image": "rotated.png"            <- target output image
  "prompt": "Rotate 90 degrees"
"""

import json
import sys


def convert_metadata(input_file, output_file):
    """Convert rotation metadata to kontext format"""
    
    print(f"Reading from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original samples: {len(data)}")
    
    # Convert each entry
    converted_data = []
    for item in data:
        converted_item = {
            "image": item["edit_image"],          # Target output
            "kontext_images": item["image"],      # Context input
            "prompt": item["prompt"]
        }
        converted_data.append(converted_item)
    
    print(f"Converted samples: {len(converted_data)}")
    
    # Save converted data
    print(f"Writing to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print("✓ Conversion complete!")
    print(f"\nExample converted entry:")
    print(json.dumps(converted_data[0], indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_rotation_to_kontext_format.py <input.json> <output.json>")
        print("\nExample:")
        print("  python scripts/convert_rotation_to_kontext_format.py \\")
        print("    data/objaverse_rotation/metadata_rotation.json \\")
        print("    data/objaverse_rotation/metadata_kontext.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_metadata(input_file, output_file)
