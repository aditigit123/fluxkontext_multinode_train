"""
Custom data operators for loading from S3 and RGBA preprocessing
Following dataset_rotate_vector_i2i.py logic
"""
import boto3
import io
import numpy as np
import cv2
import random
from PIL import Image
from diffsynth.trainers.unified_dataset import DataProcessingOperator

# Edge kernels from reference
kernel_list = []
for size_n in range(3, 30):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size_n, size_n))
    kernel_list.append(kernel)


class LoadImageFromS3(DataProcessingOperator):
    """Load image directly from S3 path"""
    
    def __init__(self, s3_bucket="s3://phidias/zchen/czq_objaverse_toon_1280"):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        
        # Parse bucket name and prefix
        s3_path = s3_bucket.replace('s3://', '')
        parts = s3_path.split('/', 1)
        self.bucket_name = parts[0]
        self.prefix = parts[1] if len(parts) > 1 else ''
    
    def __call__(self, relative_path: str):
        """Load image from S3 given relative path"""
        # Combine prefix with relative path
        object_key = f"{self.prefix}/{relative_path}" if self.prefix else relative_path
        
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_key)
            img_data = response['Body'].read()
            img = Image.open(io.BytesIO(img_data))
            return img
        except Exception as e:
            print(f"Error loading {object_key} from S3: {e}")
            # Return a black image as fallback
            return Image.new('RGB', (1024, 1024), (0, 0, 0))


class RGBAPreprocessing(DataProcessingOperator):
    """
    Apply RGBA preprocessing with background composition
    Following logic from dataset_rotate_vector_i2i.py
    """
    
    def __init__(
        self, 
        target_size=(1024, 1024),
        bgcolors=None,
        edgecolors=None,
        edge_prob=0.3
    ):
        self.target_size = target_size
        
        # Background colors from reference (Black, Gray, White, Green)
        if bgcolors is None:
            self.bgcolors = [[0, 0, 0], [128, 128, 128], [255, 255, 255], [0, 255, 0]]
        else:
            self.bgcolors = bgcolors
        
        # Edge colors from reference
        if edgecolors is None:
            self.edgecolors = [[255, 255, 255], [0, 0, 0]]
        else:
            self.edgecolors = edgecolors
        
        self.edge_prob = edge_prob
    
    def _check_bgcolor_compat(self, img_np, bgcolor, use_edge):
        """
        Check if background color is compatible with image
        From dataset_rotate_vector_i2i.py lines 320-326
        """
        if use_edge:
            return True  # Skip check if using edges
        
        img_mask = img_np[..., 3] > 10
        colormask = (np.abs(img_np[..., 0].astype(np.int32) - bgcolor[0]) < 16) & \
                   (np.abs(img_np[..., 1].astype(np.int32) - bgcolor[1]) < 16) & \
                   (np.abs(img_np[..., 2].astype(np.int32) - bgcolor[2]) < 16) & img_mask
        
        # If >10% of visible pixels are too close to bgcolor, reject
        if np.sum(colormask) > np.sum(img_mask) * 0.1:
            return False
        return True
    
    def __call__(self, img: Image.Image):
        """Process RGBA image with background and optional edge"""
        # Convert to numpy
        img_np = np.array(img)
        
        # Ensure RGBA format
        if img_np.shape[2] == 3:
            # Add alpha channel if missing (fully opaque)
            alpha = np.ones((img_np.shape[0], img_np.shape[1], 1), dtype=np.uint8) * 255
            img_np = np.concatenate([img_np, alpha], axis=2)
        
        # Decide if using edge first (needed for bgcolor check)
        use_edge = random.random() < self.edge_prob
        
        # Select compatible background color (with retry logic)
        max_retries = 10
        for _ in range(max_retries):
            bgcolor = random.choice(self.bgcolors)
            if self._check_bgcolor_compat(img_np, bgcolor, use_edge):
                break
        # If all retries fail, just use the last bgcolor
        
        # Split color and alpha
        img_c = img_np[..., :3].astype(np.float32)
        img_a = img_np[..., 3:4].astype(np.float32) / 255
        
        # Alpha blend with background color
        bgcolor_array = np.array(bgcolor, dtype=np.float32).reshape(1, 1, 3)
        img_np = img_c * img_a + (1 - img_a) * bgcolor_array
        img_np = img_np.astype(np.uint8)
        
        # Optional edge processing (already decided above)
        if use_edge:
            edgecolor = random.choice(self.edgecolors)
            size_t = random.randint(3, min(10, len(kernel_list) - 1))
            
            # Create mask from alpha
            mask = (img_a.squeeze() > 0.5).astype(np.uint8)
            
            # Dilate to create edge
            kernel = kernel_list[size_t]
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            edge_mask = mask_dilated - mask
            
            # Apply edge color
            edgecolor_array = np.array(edgecolor, dtype=np.uint8).reshape(1, 1, 3)
            img_np = np.where(
                edge_mask[..., None] > 0,
                edgecolor_array,
                img_np
            )
        
        # Bounding box crop and pad (from reference)
        mask = (img_a.squeeze() > 0.5).astype(np.uint8)
        if mask.sum() > 0:
            y_indices, x_indices = np.where(mask > 0)
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            # Crop to bounding box
            img_np = img_np[y_min:y_max+1, x_min:x_max+1]
            
            # Pad to square with background color
            h, w = img_np.shape[:2]
            size = max(h, w)
            padded = np.ones((size, size, 3), dtype=np.uint8) * np.array(bgcolor, dtype=np.uint8)
            
            offset_y = (size - h) // 2
            offset_x = (size - w) // 2
            padded[offset_y:offset_y+h, offset_x:offset_x+w] = img_np
            img_np = padded
        
        # Resize to target resolution
        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.resize(self.target_size, Image.LANCZOS)
        
        # Convert to RGB (already composited)
        return img_pil.convert("RGB")

