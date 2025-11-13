"""
Modified training script for S3-based Objaverse data with DeepSpeed ZeRO-3 support
Uses custom S3 image loading with RGBA preprocessing
Adds wandb logging and DeepSpeed ZeRO-3 compatibility
"""
import torch, os, json, logging, sys
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.pipelines.flux_image_new import ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, qwen_image_parser, launch_training_task, launch_data_process_task
from diffsynth.trainers.unified_dataset import UnifiedDataset
from diffsynth.trainers.s3_operators import LoadImageFromS3, RGBAPreprocessing
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_metrics.log')
    ]
)
logger = logging.getLogger(__name__)

# Wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Training will proceed without wandb logging.")

# Import from train.py
from train import QwenImageTrainingModule


if __name__ == "__main__":
    # Parse arguments
    parser = qwen_image_parser()
    parser.add_argument("--s3_bucket", type=str, default="s3://phidias/zchen/czq_objaverse_toon_1280",
                       help="S3 bucket path for Objaverse data")
    args = parser.parse_args()
    
    logger.info(f"Creating dataset with S3 loading from: {args.dataset_metadata_path}")
    logger.info(f"S3 bucket: {args.s3_bucket}")
    
    # Create custom image operator: Load from S3 >> RGBA preprocessing
    s3_image_operator = LoadImageFromS3(s3_bucket=args.s3_bucket) >> RGBAPreprocessing(
        target_size=(1024, 1024),  # Will be resized based on max_pixels if needed
    )
    
    # Create dataset with custom operator
    dataset = UnifiedDataset(
        base_path="",  # Not used, paths are in metadata
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=s3_image_operator
    )
    
    logger.info(f"Dataset size: {len(dataset)} samples (with {args.dataset_repeat}x repeat)")
    
    # Check if wandb should be used (but don't init yet - wait for accelerator)
    use_wandb = (
        WANDB_AVAILABLE and os.environ.get("WANDB_DISABLED", "false").lower() != "true"
    )
    
    # Create model
    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        enable_fp8_training=args.enable_fp8_training,
        task=args.task,
    )
    
    # Create logger
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    
    # Custom collate function for batching PIL Images and other data
    def custom_collate_fn(batch):
        """Custom collate that keeps non-tensor data (like PIL Images) as lists"""
        if len(batch) == 0:
            return {}
        
        # Get all keys from first sample
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            # Collect all values for this key
            values = [item[key] for item in batch]
            # Keep as list (model's forward_preprocess will handle batching)
            collated[key] = values
        
        return collated
    
    # Custom training loop with wandb logging
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        shuffle=True, 
        collate_fn=custom_collate_fn,  # Custom collate for PIL Images
        batch_size=args.batch_size,
        num_workers=args.dataset_num_workers
    )
    
    # Set gradient_accumulation_steps from args
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    # Initialize wandb AFTER accelerator (so we know which process is main)
    if use_wandb and accelerator.is_main_process:
        wandb_project = os.environ.get("WANDB_PROJECT", "qwen-img-edit-rotate")
        wandb_run_name = os.environ.get("WANDB_RUN_NAME", None)
        wandb_tags = (
            os.environ.get("WANDB_TAGS", "").split(",")
            if os.environ.get("WANDB_TAGS")
            else []
        )
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=wandb_tags,
            dir="/mnt/localssd/wandb",
            config={
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_epochs": args.num_epochs,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes,
                "dataset_size": len(dataset),
                "save_steps": args.save_steps,
                "task": args.task,
                "trainable_models": args.trainable_models,
                "lora_rank": args.lora_rank,
                "max_pixels": args.max_pixels,
                "dataset_repeat": args.dataset_repeat,
                "s3_bucket": args.s3_bucket,
            },
        )
        logger.info(f"✓ Wandb initialized: project={wandb_project}, run_name={wandb_run_name}")
    
    global_step = 0
    for epoch_id in range(args.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(
            dataloader, 
            disable=not accelerator.is_main_process, 
            desc=f"Epoch {epoch_id + 1}/{args.num_epochs}"
        )
        
        for data in progress_bar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                
                accelerator.backward(loss)
                
                # Compute gradient norm BEFORE optimizer.step()
                total_norm = 0.0
                if use_wandb and accelerator.is_main_process:
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                
                optimizer.step()
                scheduler.step()
            
            # Save checkpoint if needed
            if args.save_steps is not None:
                model_logger.on_step_end(accelerator, model, args.save_steps)
            
            # Gather loss across GPUs
            loss_value = accelerator.gather_for_metrics(loss.detach()).mean().item()
            epoch_loss += loss_value
            num_batches += 1
            global_step += 1
            
            if accelerator.is_main_process:
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})
                
                # Log to file every 100 steps
                if global_step % 100 == 0:
                    logger.info(f"Step {global_step} | Loss: {loss_value:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | GradNorm: {total_norm:.4f}")
                
                # Log to wandb
                if use_wandb:
                    wandb.log({
                        "train/loss": loss_value,
                        "train/epoch": epoch_id,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/gradient_norm": total_norm,
                        "train/step": global_step,
                    }, step=global_step)
        
        # Epoch end
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            if use_wandb and accelerator.is_main_process:
                checkpoint_path = os.path.join(
                    model_logger.output_path, f"epoch-{epoch_id}.safetensors"
                )
                if os.path.exists(checkpoint_path):
                    artifact = wandb.Artifact(
                        name=f"model-epoch-{epoch_id}",
                        type="model",
                        description=f"Model checkpoint at epoch {epoch_id}",
                    )
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        if use_wandb and accelerator.is_main_process:
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "epoch": epoch_id + 1,
            }, step=global_step)
        
        if accelerator.is_main_process:
            logger.info("="*80)
            logger.info(f"Epoch {epoch_id + 1}/{args.num_epochs} Complete - Average Loss: {avg_epoch_loss:.4f}")
            logger.info(f"Total Steps: {global_step} | Batches: {num_batches}")
            logger.info("="*80)
    
    model_logger.on_training_end(accelerator, model, args.save_steps)
    
    # Finish wandb
    if use_wandb and accelerator.is_main_process:
        accelerator.wait_for_everyone()
        final_checkpoint_path = os.path.join(
            model_logger.output_path, "final.safetensors"
        )
        if os.path.exists(final_checkpoint_path):
            artifact = wandb.Artifact(
                name="model-final",
                type="model",
                description="Final trained model",
            )
            artifact.add_file(final_checkpoint_path)
            wandb.log_artifact(artifact)
        wandb.finish()
        logger.info("✓ Wandb run finished")

