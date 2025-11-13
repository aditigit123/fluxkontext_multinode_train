"""
S3 Checkpoint Upload Utilities for DiffSynth Training

Based on foundation_exploration/utils/s3_io_utils.py
"""

import logging
import subprocess
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_to_s3_with_s5cmd(local_path: str, s3_path: str) -> None:
    """
    Upload a file or directory to S3 using s5cmd.
    
    s5cmd is a fast S3 CLI tool that's much faster than aws s3 sync.
    Install: pip install s5cmd OR download from https://github.com/peak/s5cmd
    
    Args:
        local_path: Local file or directory path to upload
        s3_path: S3 destination path (e.g., s3://bucket/path/to/checkpoint)
    
    Raises:
        RuntimeError: If upload fails
    """
    logger.info(f"📤 Uploading {local_path} to {s3_path}")
    
    try:
        # Use s5cmd cp for fast upload
        cmd = ["s5cmd", "cp", str(local_path), str(s3_path)]
        
        # Execute the upload command
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to upload to {s3_path}: {result.stderr}")
            raise RuntimeError(f"s5cmd upload failed with return code {result.returncode}")
        
        logger.info(f"✅ Successfully uploaded {local_path} to {s3_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"s5cmd upload failed: {e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        logger.error(f"Upload to {s3_path} timed out after 1 hour")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise


def upload_checkpoint_to_s3(
    local_checkpoint_path: str,
    s3_base_path: str,
    step_or_epoch: int,
    checkpoint_type: str = "step",
    keep_local: bool = False,
) -> str:
    """
    Upload a checkpoint to S3 with organized naming.
    
    Args:
        local_checkpoint_path: Path to local checkpoint file (e.g., step-5000.safetensors)
        s3_base_path: Base S3 path (e.g., s3://bucket/experiments/flux-rotation-lora)
        step_or_epoch: Step number or epoch number
        checkpoint_type: Either "step" or "epoch"
        keep_local: If False, delete local checkpoint after successful upload
        
    Returns:
        S3 path where checkpoint was uploaded
        
    Example:
        >>> upload_checkpoint_to_s3(
        ...     "/mnt/localssd/FLUX.1-Kontext-Rotation-LoRA/step-5000.safetensors",
        ...     "s3://phidias/experiments/flux-rotation-lora",
        ...     5000,
        ...     checkpoint_type="step"
        ... )
        "s3://phidias/experiments/flux-rotation-lora/step-5000/checkpoint.safetensors"
    """
    # Ensure s3_base_path doesn't end with /
    s3_base_path = s3_base_path.rstrip('/')
    
    # Create organized S3 path
    if checkpoint_type == "step":
        s3_checkpoint_dir = f"{s3_base_path}/step-{step_or_epoch:08d}"
    else:
        s3_checkpoint_dir = f"{s3_base_path}/epoch-{step_or_epoch}"
    
    # Get filename from local path
    filename = Path(local_checkpoint_path).name
    s3_checkpoint_path = f"{s3_checkpoint_dir}/{filename}"
    
    # Upload to S3
    try:
        upload_to_s3_with_s5cmd(local_checkpoint_path, s3_checkpoint_path)
        
        # Optionally delete local file to save space
        if not keep_local:
            logger.info(f"🗑️  Deleting local checkpoint: {local_checkpoint_path}")
            os.remove(local_checkpoint_path)
            logger.info(f"✅ Local checkpoint deleted")
            
        return s3_checkpoint_path
        
    except Exception as e:
        logger.error(f"❌ Failed to upload checkpoint: {e}")
        raise


def sync_directory_to_s3(local_dir: str, s3_path: str, delete_local: bool = False) -> None:
    """
    Sync an entire directory to S3 using s5cmd sync.
    
    Args:
        local_dir: Local directory to sync
        s3_path: S3 destination path
        delete_local: If True, delete local directory after successful sync
        
    Example:
        >>> sync_directory_to_s3(
        ...     "/mnt/localssd/FLUX.1-Kontext-Rotation-LoRA",
        ...     "s3://phidias/experiments/flux-rotation-lora/checkpoints"
        ... )
    """
    # Ensure paths end with / for directory sync
    if not local_dir.endswith('/'):
        local_dir += '/'
    if not s3_path.endswith('/'):
        s3_path += '/'
    
    logger.info(f"🔄 Syncing directory {local_dir} to {s3_path}")
    
    try:
        # Use s5cmd sync for efficient directory sync
        cmd = ["s5cmd", "sync", local_dir, s3_path]
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout for large directories
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to sync to {s3_path}: {result.stderr}")
            raise RuntimeError(f"s5cmd sync failed with return code {result.returncode}")
        
        logger.info(f"✅ Successfully synced {local_dir} to {s3_path}")
        
        # Optionally delete local directory
        if delete_local:
            import shutil
            logger.info(f"🗑️  Deleting local directory: {local_dir}")
            shutil.rmtree(local_dir)
            logger.info(f"✅ Local directory deleted")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"s5cmd sync failed: {e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        logger.error(f"Sync to {s3_path} timed out after 2 hours")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during sync: {e}")
        raise

