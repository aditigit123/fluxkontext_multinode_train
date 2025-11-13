"""
Rotation Pair Dataset for FLUX Kontext Training
Follows the logic from dataset_rotate_vector_i2i.py
"""
import torch
import json
import random
import numpy as np
from PIL import Image
from diffsynth.trainers.s3_operators import LoadImageFromS3


class RotationPairDataset(torch.utils.data.Dataset):
    """
    Dataset for rotation image pairs with proper background handling.
    Ensures both source and target images use the SAME random background color.
    """
    
    def __init__(
        self,
        metadata_path,
        s3_bucket,
        target_size=(1024, 1024),
        max_pixels=262144,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        repeat=1,
        bgcolors=None,
        edgecolors=None,
        edge_prob=0.3,
    ):
        self.metadata_path = metadata_path
        self.s3_bucket = s3_bucket
        self.target_size = target_size
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.repeat = repeat
        
        # Background and edge colors from reference
        if bgcolors is None:
            self.bgcolors = [[0, 0, 0], [128, 128, 128], [255, 255, 255], [0, 255, 0]]
        else:
            self.bgcolors = bgcolors
            
        if edgecolors is None:
            self.edgecolors = [[255, 255, 255], [0, 0, 0]]
        else:
            self.edgecolors = edgecolors
            
        self.edge_prob = edge_prob
        
        # S3 loader
        self.s3_loader = LoadImageFromS3(s3_bucket=s3_bucket)
        
        # Load metadata from JSON file
        with open(metadata_path, "r") as f:
            self.data = json.load(f)
        
        # Compatibility with UnifiedDataset interface
        self.load_from_cache = False  # We don't use cached data
        
        print(f"Loaded {len(self.data)} rotation pairs from {metadata_path}")
    
    
    def _check_bgcolor_compat(self, img_np, bgcolor, use_edge):
        """Check if background color is compatible with image"""
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
    
    def _process_rgba_image(self, img, bgcolor, use_edge, edgecolor, edge_size):
        """
        Process RGBA image with background and optional edge.
        Matches logic from dataset_rotate_vector_i2i.py load_im() function.
        """
        # Convert to numpy
        img_np = np.array(img)
        
        # Ensure RGBA format
        if img_np.shape[2] == 3:
            alpha = np.ones((img_np.shape[0], img_np.shape[1], 1), dtype=np.uint8) * 255
            img_np = np.concatenate([img_np, alpha], axis=2)
        
        # Split color and alpha
        img_c = img_np[..., :3].astype(np.float32)
        img_a = img_np[..., 3:4].astype(np.float32) / 255
        
        # Alpha blend with background color
        bgcolor_array = np.array(bgcolor, dtype=np.float32).reshape(1, 1, 3)
        img_np = img_c * img_a + (1 - img_a) * bgcolor_array
        img_np = img_np.astype(np.uint8)
        
        # Optional edge processing
        if use_edge and edge_size > 0:
            try:
                import cv2
                # Create mask from alpha
                mask = (img_a.squeeze() > 0.5).astype(np.uint8)
                
                # Dilate to create edge
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_size, edge_size))
                mask_dilated = cv2.dilate(mask, kernel, iterations=1)
                edge_mask = mask_dilated - mask
                
                # Apply edge color
                edgecolor_array = np.array(edgecolor, dtype=np.uint8).reshape(1, 1, 3)
                img_np = np.where(
                    edge_mask[..., None] > 0,
                    edgecolor_array,
                    img_np
                )
            except Exception as e:
                print(f"Warning: Edge processing failed: {e}")
        
        # Bounding box crop and pad
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
        
        return img_pil.convert("RGB")
    
    def _resize_for_training(self, img):
        """Final resize for training with max_pixels constraint"""
        from diffsynth.trainers.unified_dataset import ImageCropAndResize
        
        resizer = ImageCropAndResize(
            height=self.height,
            width=self.width,
            max_pixels=self.max_pixels,
            height_division_factor=self.height_division_factor,
            width_division_factor=self.width_division_factor,
        )
        return resizer(img)
    
    def __getitem__(self, idx):
        """
        Load a rotation pair with SAME background color for both images.
        This is the key difference from the pipeline approach.
        """
        try:
            data = self.data[idx % len(self.data)].copy()
            
            # Extract paths
            source_path = data.get("kontext_images")
            target_path = data.get("image")
            
            # Handle list format for kontext_images
            if isinstance(source_path, list):
                source_path = source_path[0] if source_path else None
            
            if source_path is None or target_path is None:
                raise ValueError(f"Missing image paths in data: {data}")
            
            # Load raw images from S3
            source_img = self.s3_loader(source_path)
            target_img = self.s3_loader(target_path)
            
            # CRITICAL: Choose ONE background color for BOTH images
            use_edge = random.random() < self.edge_prob
            
            # Select compatible background color (with retry logic)
            max_retries = 10
            source_np = np.array(source_img)
            target_np = np.array(target_img)
            
            for _ in range(max_retries):
                bgcolor = random.choice(self.bgcolors)
                # Check if bgcolor is compatible with both images
                if (self._check_bgcolor_compat(source_np, bgcolor, use_edge) and
                    self._check_bgcolor_compat(target_np, bgcolor, use_edge)):
                    break
            # If all retries fail, just use the last bgcolor
            
            # Edge parameters (same for both)
            if use_edge:
                edgecolor = random.choice(self.edgecolors)
                edge_size = random.randint(3, min(10, 29))
            else:
                edgecolor = [255, 255, 255]  # Dummy
                edge_size = 0
            
            # Process both images with THE SAME parameters
            source_processed = self._process_rgba_image(source_img, bgcolor, use_edge, edgecolor, edge_size)
            target_processed = self._process_rgba_image(target_img, bgcolor, use_edge, edgecolor, edge_size)
            
            # Final resize for training
            source_final = self._resize_for_training(source_processed)
            target_final = self._resize_for_training(target_processed)
        
            # Return with proper keys
            result = data.copy()
            result["kontext_images"] = [source_final]  # List format as expected
            result["image"] = target_final
            
            return result
            
        except Exception as e:
            print(f"❌ Error loading sample {idx}: {e}")
            print(f"   Data keys: {list(data.keys()) if 'data' in locals() else 'N/A'}")
            if 'source_path' in locals():
                print(f"   Source path: {source_path}")
            if 'target_path' in locals():
                print(f"   Target path: {target_path}")
            import traceback
            traceback.print_exc()
            raise
    
    def __len__(self):
        return len(self.data) * self.repeat

