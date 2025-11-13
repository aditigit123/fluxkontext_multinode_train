"""
Visualization utilities for training monitoring.
Creates HTML visualizations of training batches and inference results, uploads to S3, and logs to WandB.

Adapted from foundation_exploration's visualization approach.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import dominate
    from dominate.tags import a, div, h3, img, meta, p, style, table, tbody, td, th, thead, tr
    from dominate.util import raw
    DOMINATE_AVAILABLE = True
except ImportError:
    DOMINATE_AVAILABLE = False
    logger.warning("dominate not installed. HTML visualization will be disabled. Install with: pip install dominate")

try:
    from wandb import Html as WandbHtml
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def s3_url_to_https(s3_url: str) -> str:
    """Convert S3 URL to HTTPS URL for browser access."""
    if not s3_url.startswith("s3://"):
        return s3_url
    
    # Remove s3:// prefix
    path = s3_url[5:]
    # Split into bucket and key
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    
    return f"https://{bucket}.s3.us-west-2.amazonaws.com/{key}"


def upload_directory_to_s3(local_dir: str, s3_path: str) -> bool:
    """
    Upload an entire directory to S3 using s5cmd for speed.
    
    Args:
        local_dir: Local directory path to upload
        s3_path: S3 destination path (e.g., s3://bucket/path/)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not s3_path.endswith("/"):
            s3_path += "/"
        
        # Use s5cmd sync for efficient directory upload
        cmd = ["s5cmd", "sync", f"{local_dir}/", s3_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully uploaded {local_dir} to {s3_path}")
            return True
        else:
            logger.error(f"❌ s5cmd sync failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ s5cmd sync failed with error: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("❌ s5cmd not found. Install with: pip install s5cmd")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during S3 upload: {e}")
        return False


class HTMLVisualizer:
    """Creates HTML visualization pages for training monitoring."""
    
    def __init__(self, title: str = "Training Visualization"):
        """
        Initialize HTML visualizer.
        
        Args:
            title: Page title
        """
        if not DOMINATE_AVAILABLE:
            raise ImportError("dominate is required for HTML visualization. Install with: pip install dominate")
        
        self.title = title
        self.doc = dominate.document(title=title)
        
        # Add default styling
        with self.doc.head:
            meta(charset="utf-8")
            style("""
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                h1 { 
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }
                h3 {
                    color: #555;
                    margin-top: 30px;
                }
                table { 
                    border-collapse: collapse; 
                    margin: 20px 0;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                th, td { 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: center;
                    vertical-align: top;
                }
                th { 
                    background-color: #4CAF50; 
                    color: white;
                    font-weight: bold;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                img { 
                    display: block; 
                    margin: 5px auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .info { 
                    background-color: #e7f3fe;
                    border-left: 4px solid #2196F3;
                    padding: 12px;
                    margin: 20px 0;
                }
                .caption {
                    font-size: 12px;
                    color: #666;
                    margin-top: 5px;
                    max-width: 400px;
                    word-wrap: break-word;
                }
                .metadata {
                    font-size: 11px;
                    color: #888;
                    font-family: monospace;
                }
            """)
    
    def add_info_section(self, info_lines: list[str]):
        """Add an information section at the top of the page."""
        with self.doc:
            info_div = div(cls="info")
            for line in info_lines:
                info_div.add(p(line))
    
    def add_image_table(
        self, 
        rows: list[dict[str, Any]], 
        headers: list[str] = None,
        image_width: int = 320,
        image_height: int = 320
    ):
        """
        Add a table of images to the HTML page.
        
        Args:
            rows: List of dicts, each containing:
                - 'images': list of image paths (relative to HTML file)
                - 'caption': optional caption for the row
                - 'metadata': optional dict of metadata to display
            headers: Column headers (one per image)
            image_width: Display width for images
            image_height: Display height for images
        """
        with self.doc:
            tbl = table()
            
            # Add headers if provided
            if headers:
                with tbl:
                    thead_row = thead()
                    with thead_row:
                        tr_header = tr()
                        for header in headers:
                            tr_header.add(th(header))
            
            # Add rows
            with tbl:
                tbody_section = tbody()
                with tbody_section:
                    for row_data in rows:
                        images = row_data.get('images', [])
                        caption_text = row_data.get('caption', '')
                        metadata = row_data.get('metadata', {})
                        
                        # Image row
                        img_row = tr()
                        with img_row:
                            for img_path in images:
                                cell = td()
                                with cell:
                                    # Create clickable image
                                    img_elem = img(
                                        src=img_path,
                                        width=str(image_width),
                                        height=str(image_height),
                                        style="object-fit: contain;"
                                    )
                                    a(img_elem, href=img_path, target="_blank")
                        
                        # Caption/metadata row (if any)
                        if caption_text or metadata:
                            caption_row = tr()
                            with caption_row:
                                for i, img_path in enumerate(images):
                                    cell = td()
                                    with cell:
                                        if i == 0 and caption_text:  # Show caption only in first column
                                            p(caption_text, cls="caption")
                                        if metadata:
                                            meta_div = div(cls="metadata")
                                            for key, value in metadata.items():
                                                meta_div.add(p(f"{key}: {value}"))
    
    def save(self, output_path: str) -> str:
        """
        Save HTML to file.
        
        Args:
            output_path: Local path or S3 path (s3://bucket/path/file.html)
        
        Returns:
            The URL to access the HTML (HTTPS URL if S3, local path otherwise)
        """
        html_content = self.doc.render(pretty=True)
        
        if output_path.startswith("s3://"):
            # Save to S3 directly
            try:
                # Write to temp file first
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                    f.write(html_content)
                    temp_path = f.name
                
                # Upload to S3
                cmd = ["s5cmd", "cp", temp_path, output_path]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Clean up temp file
                os.remove(temp_path)
                
                if result.returncode == 0:
                    https_url = s3_url_to_https(output_path)
                    logger.info(f"✅ HTML saved to S3: {https_url}")
                    return https_url
                else:
                    logger.error(f"❌ Failed to upload HTML to S3: {result.stderr}")
                    return output_path
                    
            except Exception as e:
                logger.error(f"❌ Error saving HTML to S3: {e}")
                return output_path
        else:
            # Save locally
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"✅ HTML saved locally: {output_path}")
            return output_path


def create_rotation_visualization(
    step: int,
    samples: list[dict[str, Any]],
    output_dir: str,
    s3_base_path: str = None,
    title: str = None
) -> tuple[str, str]:
    """
    Create HTML visualization for rotation task samples.
    
    Args:
        step: Training step number
        samples: List of dicts containing:
            - 'input_image': PIL Image or tensor (context image)
            - 'generated_image': PIL Image or tensor (generated rotated image)
            - 'target_image': PIL Image or tensor (ground truth, optional)
            - 'rotation_degrees': int (requested rotation)
            - 'prompt': str (the text prompt used)
        output_dir: Local directory to save visualization files
        s3_base_path: Optional S3 base path (e.g., s3://bucket/experiments/run-name)
        title: Optional custom title
    
    Returns:
        Tuple of (local_html_path, https_url)
        - local_html_path: Local path to HTML file
        - https_url: HTTPS URL to access (S3 URL if uploaded, local path otherwise)
    """
    if not DOMINATE_AVAILABLE:
        logger.error("Cannot create visualization: dominate not installed")
        return None, None
    
    # Create step-specific directory
    step_dir = os.path.join(output_dir, f"step-{step:08d}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Prepare visualization data
    rows = []
    headers = ["Context Image", "Generated (Rotated)", "Ground Truth", "Details"]
    
    for idx, sample in enumerate(samples):
        # Save images locally
        image_paths = []
        
        # Context image
        context_img = sample.get('input_image')
        if context_img is not None:
            context_path = os.path.join(step_dir, f"sample_{idx}_context.png")
            _save_image(context_img, context_path)
            image_paths.append(f"sample_{idx}_context.png")
        else:
            image_paths.append(None)
        
        # Generated image
        generated_img = sample.get('generated_image')
        if generated_img is not None:
            generated_path = os.path.join(step_dir, f"sample_{idx}_generated.png")
            _save_image(generated_img, generated_path)
            image_paths.append(f"sample_{idx}_generated.png")
        else:
            image_paths.append(None)
        
        # Target image (optional)
        target_img = sample.get('target_image')
        if target_img is not None:
            target_path = os.path.join(step_dir, f"sample_{idx}_target.png")
            _save_image(target_img, target_path)
            image_paths.append(f"sample_{idx}_target.png")
        else:
            image_paths.append(None)
        
        # Create a placeholder for details column (just text)
        image_paths.append(None)
        
        # Prepare row data
        rotation = sample.get('rotation_degrees', 'N/A')
        prompt = sample.get('prompt', 'N/A')
        
        rows.append({
            'images': image_paths,
            'caption': f"Rotation: {rotation}°",
            'metadata': {
                'Sample': idx,
                'Prompt': prompt[:100] + ('...' if len(prompt) > 100 else ''),
                'Rotation': f"{rotation}°"
            }
        })
    
    # Create HTML
    if title is None:
        title = f"FLUX Kontext Rotation - Step {step}"
    
    html_viz = HTMLVisualizer(title=title)
    
    # Add info section
    info_lines = [
        f"Training Step: {step}",
        f"Number of Samples: {len(samples)}",
        f"Task: Image Rotation with FLUX.1-Kontext-dev LoRA"
    ]
    html_viz.add_info_section(info_lines)
    
    # Add image table
    html_viz.add_image_table(rows, headers=headers, image_width=256, image_height=256)
    
    # Save HTML locally first
    local_html_path = os.path.join(step_dir, "visualization.html")
    html_viz.save(local_html_path)
    
    # Upload to S3 if requested
    https_url = local_html_path
    if s3_base_path:
        s3_step_path = os.path.join(s3_base_path, "visualizations", f"step-{step:08d}")
        if not s3_step_path.startswith("s3://"):
            s3_step_path = f"s3://{s3_step_path}"
        
        # Upload entire directory to S3
        if upload_directory_to_s3(step_dir, s3_step_path):
            # Construct HTTPS URL
            s3_html_path = os.path.join(s3_step_path, "visualization.html")
            https_url = s3_url_to_https(s3_html_path)
    
    return local_html_path, https_url


def log_visualization_to_wandb(
    wandb_tracker,
    html_url: str,
    step: int,
    run_name: str = None,
    panel_name: str = "visualization",
    metadata_url: str = None
) -> bool:
    """
    Log visualization HTML URL to WandB.
    Adapted from foundation_exploration's url_to_wandb_link pattern.
    
    Args:
        wandb_tracker: WandB run object
        html_url: HTTPS URL to the HTML visualization
        step: Training step number
        run_name: Optional name for the run (defaults to step number)
        panel_name: WandB panel name (e.g., "inference", "data_batches")
        metadata_url: Optional URL to metadata JSON file
    
    Returns:
        True if successful, False otherwise
    """
    if not WANDB_AVAILABLE:
        logger.warning("WandB not available, skipping visualization logging")
        return False
    
    if wandb_tracker is None:
        logger.warning("WandB tracker is None, skipping visualization logging")
        return False
    
    try:
        if run_name is None:
            run_name = f"Step {step}"
        
        # Create WandB HTML object with clickable link(s)
        # This format matches foundation_exploration's approach for better WandB UI integration
        html_content = f'<a href="{html_url}" target="_blank">{run_name}</a>'
        
        # Add metadata link if provided
        if metadata_url:
            html_content += f' (<a href="{metadata_url}" target="_blank">metadata</a>)'
        
        html_content += '<br>'  # Line break for multiple entries
        
        # Log to WandB with panel name
        wandb_log_dict = {panel_name: WandbHtml(html_content)}
        
        wandb_tracker.log(wandb_log_dict, step=step)
        logger.info(f"✅ Logged visualization to WandB panel '{panel_name}': {html_url}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to log visualization to WandB: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_data_batch_visualization(
    step: int,
    batch_data: dict[str, Any],
    output_dir: str,
    s3_base_path: str = None,
    title: str = None,
    max_samples: int = 4
) -> tuple[str, str]:
    """
    Create HTML visualization for a data batch (shows what's being fed into training).
    
    Args:
        step: Training step number
        batch_data: Dictionary containing batch data with keys:
            - 'image': Target images (PIL Images or list)
            - 'kontext_images': Context images (PIL Images or list of lists)
            - 'prompt' or 'text': Text prompts (str or list)
        output_dir: Local directory to save visualization files
        s3_base_path: Optional S3 base path
        title: Optional custom title
        max_samples: Maximum number of samples to visualize
    
    Returns:
        Tuple of (local_html_path, https_url)
    """
    if not DOMINATE_AVAILABLE:
        logger.error("Cannot create visualization: dominate not installed")
        return None, None
    
    # Create step-specific directory
    step_dir = os.path.join(output_dir, f"batch-step-{step:08d}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Extract data from batch
    target_images = batch_data.get('image', [])
    context_images = batch_data.get('kontext_images', [])
    prompts = batch_data.get('text', batch_data.get('prompt', []))
    
    # Ensure lists
    if not isinstance(target_images, list):
        target_images = [target_images]
    if not isinstance(context_images, list):
        context_images = [context_images]
    if not isinstance(prompts, list):
        prompts = [prompts]
    
    # Limit to max_samples
    num_samples = min(len(target_images), max_samples)
    
    # Prepare visualization data
    rows = []
    headers = ["Context Image (Input)", "Target Image (Ground Truth)", "Details"]
    
    for idx in range(num_samples):
        # Save images locally
        image_paths = []
        
        # Context image (input)
        if idx < len(context_images):
            context_img = context_images[idx]
            # Handle list of context images (take first one)
            if isinstance(context_img, list) and len(context_img) > 0:
                context_img = context_img[0]
            
            if context_img is not None:
                context_path = os.path.join(step_dir, f"sample_{idx}_context.png")
                _save_image(context_img, context_path)
                image_paths.append(f"sample_{idx}_context.png")
            else:
                image_paths.append(None)
        else:
            image_paths.append(None)
        
        # Target image
        if idx < len(target_images):
            target_img = target_images[idx]
            if target_img is not None:
                target_path = os.path.join(step_dir, f"sample_{idx}_target.png")
                _save_image(target_img, target_path)
                image_paths.append(f"sample_{idx}_target.png")
            else:
                image_paths.append(None)
        else:
            image_paths.append(None)
        
        # Placeholder for details column
        image_paths.append(None)
        
        # Get prompt
        prompt = prompts[idx] if idx < len(prompts) else "N/A"
        
        # Extract rotation from prompt if available
        rotation = 'N/A'
        if isinstance(prompt, str) and 'rotate' in prompt.lower():
            import re
            match = re.search(r'(\d+)\s*degrees?', prompt, re.IGNORECASE)
            if match:
                rotation = f"{match.group(1)}°"
        
        rows.append({
            'images': image_paths,
            'caption': f"Training Sample {idx}",
            'metadata': {
                'Prompt': prompt[:100] + ('...' if len(prompt) > 100 else '') if isinstance(prompt, str) else str(prompt),
                'Rotation': rotation,
            }
        })
    
    # Create HTML
    if title is None:
        title = f"Training Data Batch - Step {step}"
    
    html_viz = HTMLVisualizer(title=title)
    
    # Add info section
    info_lines = [
        f"Training Step: {step}",
        f"Batch Size: {num_samples}",
        f"Purpose: Verify data loading pipeline (S3 → RGBA preprocessing → model input)",
        f"Note: These are the ACTUAL images being fed into training, not generated outputs"
    ]
    html_viz.add_info_section(info_lines)
    
    # Add image table
    html_viz.add_image_table(rows, headers=headers, image_width=320, image_height=320)
    
    # Save HTML locally first
    local_html_path = os.path.join(step_dir, "batch_visualization.html")
    html_viz.save(local_html_path)
    
    # Upload to S3 if requested
    https_url = local_html_path
    if s3_base_path:
        s3_step_path = os.path.join(s3_base_path, "data_batches", f"batch-step-{step:08d}")
        if not s3_step_path.startswith("s3://"):
            s3_step_path = f"s3://{s3_step_path}"
        
        # Upload entire directory to S3
        if upload_directory_to_s3(step_dir, s3_step_path):
            # Construct HTTPS URL
            s3_html_path = os.path.join(s3_step_path, "batch_visualization.html")
            https_url = s3_url_to_https(s3_html_path)
    
    return local_html_path, https_url


def _save_image(image, output_path: str):
    """Helper function to save an image (handles both PIL Images and tensors)."""
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL Image
        # Assume tensor is in range [-1, 1] or [0, 1]
        img_tensor = image.detach().cpu()
        
        # Handle different tensor shapes
        if img_tensor.dim() == 4:  # B, C, H, W
            img_tensor = img_tensor[0]
        if img_tensor.dim() == 3:  # C, H, W
            if img_tensor.shape[0] == 1:  # Grayscale
                img_tensor = img_tensor.repeat(3, 1, 1)
            img_tensor = img_tensor.permute(1, 2, 0)  # H, W, C
        
        # Normalize to [0, 255]
        if img_tensor.min() < 0:  # Assuming [-1, 1]
            img_tensor = (img_tensor + 1) * 127.5
        else:  # Assuming [0, 1]
            img_tensor = img_tensor * 255
        
        img_tensor = img_tensor.clamp(0, 255).to(torch.uint8)
        image = Image.fromarray(img_tensor.numpy())
    
    # Save PIL Image
    image.save(output_path)

