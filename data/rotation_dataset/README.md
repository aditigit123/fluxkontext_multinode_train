# Rotation Dataset

This directory should contain your rotation training data.

## Structure

```
rotation_dataset/
├── metadata_rotation.json (or metadata_rotation.csv)
├── image_001.jpg          # Original images
├── image_001_rotated.jpg  # Rotated versions
├── image_002.jpg
├── image_002_rotated.jpg
└── ...
```

## How to Prepare Your Data

1. **Collect original images**: Gather images you want to train on
2. **Create rotated versions**: Use your rotation pipeline/tool to create rotated versions
3. **Name files consistently**: Use a consistent naming pattern (e.g., `{base_name}.jpg` and `{base_name}_rotated.jpg`)
4. **Create metadata file**: List all image pairs with rotation prompts

## Metadata Format

### JSON Format (metadata_rotation.json):
```json
[
  {
    "image": "image_001.jpg",
    "edit_image": "image_001_rotated.jpg",
    "prompt": "Rotate the image 90 degrees clockwise"
  }
]
```

### CSV Format (metadata_rotation.csv):
```csv
image,edit_image,prompt
image_001.jpg,image_001_rotated.jpg,"Rotate the image 90 degrees clockwise"
```

## Prompt Examples

Use clear, descriptive prompts:
- "Rotate the image 90 degrees clockwise"
- "Rotate the image 180 degrees"
- "Rotate the image 90 degrees counterclockwise"
- "Rotate the image 45 degrees"
- "Turn the image 270 degrees"

## Image Requirements

- **Formats**: JPG, JPEG, PNG, WEBP
- **Resolution**: Any resolution (will be resized based on training config)
- **Quantity**: More data = better results (minimum 100 pairs recommended)

