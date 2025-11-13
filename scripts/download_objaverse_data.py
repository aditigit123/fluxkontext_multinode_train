#!/usr/bin/env python3
"""
Download objaverse rotation data from S3 to local filesystem.

This script downloads the necessary images from S3 based on the metadata file.
Only downloads the specific images referenced in the training metadata.
"""

import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_s3_file(s3_path: str, local_path: str) -> bool:
    """Download a single file from S3 using aws s3 cp."""
    try:
        local_path_obj = Path(local_path)
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Use aws s3 cp for download
        result = subprocess.run(
            ["aws", "s3", "cp", s3_path, local_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"Failed to download {s3_path}: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error downloading {s3_path}: {e}")
        return False


def download_objaverse_data(
    metadata_file: str,
    s3_bucket: str,
    local_base_path: str,
    num_workers: int = 8,
    dry_run: bool = False,
):
    """
    Download images referenced in metadata from S3.
    
    Args:
        metadata_file: Path to metadata.json
        s3_bucket: S3 bucket URL
        local_base_path: Local directory to download to
        num_workers: Number of parallel download workers
        dry_run: If True, only print what would be downloaded
    """
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} entries from {metadata_file}")
    
    # Collect unique image paths
    image_paths = set()
    for entry in metadata:
        image_paths.add(entry["image"])
        image_paths.add(entry["edit_image"])
    
    print(f"Found {len(image_paths)} unique images to download")
    
    # Prepare download tasks
    download_tasks = []
    for rel_path in image_paths:
        s3_path = f"{s3_bucket}/{rel_path}"
        local_path = os.path.join(local_base_path, rel_path)
        
        # Skip if already exists
        if os.path.exists(local_path):
            continue
        
        download_tasks.append((s3_path, local_path))
    
    print(f"Need to download {len(download_tasks)} images")
    print(f"Already have {len(image_paths) - len(download_tasks)} images locally")
    
    if dry_run:
        print("\nDry run - would download:")
        for s3_path, local_path in download_tasks[:10]:
            print(f"  {s3_path} -> {local_path}")
        if len(download_tasks) > 10:
            print(f"  ... and {len(download_tasks) - 10} more")
        return
    
    # Download files
    print(f"\nDownloading with {num_workers} workers...")
    
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_s3_file, s3_path, local_path): (s3_path, local_path)
            for s3_path, local_path in download_tasks
        }
        
        with tqdm(total=len(download_tasks)) as pbar:
            for future in as_completed(futures):
                s3_path, local_path = futures[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"Error processing {s3_path}: {e}")
                    failed_count += 1
                pbar.update(1)
    
    print(f"\nDownload complete:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Already existed: {len(image_paths) - len(download_tasks)}")
    print(f"  Total images: {len(image_paths)}")


def main():
    parser = argparse.ArgumentParser(description="Download objaverse data from S3")
    parser.add_argument(
        "--metadata",
        type=str,
        default="/sensei-fs/users/asinghan/DiffSynth-Studio/data/objaverse_rotation/metadata_rotation.json",
        help="Path to metadata JSON file"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default="s3://phidias/zchen/czq_objaverse_toon_1280",
        help="S3 bucket URL"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default="/sensei-fs/users/asinghan/DiffSynth-Studio/data/objaverse_rotation",
        help="Local directory to download to"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be downloaded"
    )
    
    args = parser.parse_args()
    
    download_objaverse_data(
        metadata_file=args.metadata,
        s3_bucket=args.s3_bucket,
        local_base_path=args.local_path,
        num_workers=args.workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

