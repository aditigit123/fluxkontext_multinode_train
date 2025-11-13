"""
Training script for S3-based Objaverse rotation data with DeepSpeed ZeRO-3
Properly uses accelerator.save_state() for ZeRO-3 checkpoint compatibility
"""
import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.pipelines.flux_image_new import ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, qwen_image_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset
from diffsynth.trainers.s3_operators import LoadImageFromS3, RGBAPreprocessing
from accelerate import Accelerator
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Training will proceed without wandb logging.")

# Import QwenImageTrainingModule from original train.py
from train import QwenImageTrainingModule


class DeepSpeedCheckpointManager:
    """
    Checkpoint manager that properly handles DeepSpeed ZeRO-3 saving.
    Uses accelerator.save_state() instead of get_state_dict() to avoid timeout.
    """
    def __init__(self, output_path, accelerator):
        self.output_path = output_path
        self.accelerator = accelerator
        self.num_steps = 0
        os.makedirs(output_path, exist_ok=True)

    def on_step_end(self, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_checkpoint(f"step-{self.num_steps}")

    def on_epoch_end(self, epoch_id):
        self.save_checkpoint(f"epoch-{epoch_id}")

    def on_training_end(self, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_checkpoint(f"step-{self.num_steps}")
        self.save_checkpoint("final")

    def save_checkpoint(self, checkpoint_name):
        """
        Save checkpoint using accelerator.save_state() for DeepSpeed ZeRO-3 compatibility.
        This saves the full training state (model + optimizer + scheduler) in distributed format.
        """
        checkpoint_dir = os.path.join(self.output_path, checkpoint_name)
        
        self.accelerator.print(f"💾 Saving checkpoint: {checkpoint_name}")
        
        # accelerator.save_state() handles DeepSpeed ZeRO-3 natively
        # No weight gathering needed - each GPU saves its shard
        self.accelerator.save_state(checkpoint_dir)
        
        self.accelerator.print(f"✓ Checkpoint saved: {checkpoint_dir}")
        
        return checkpoint_dir


if __name__ == "__main__":
    # Parse arguments
    parser = qwen_image_parser()
    parser.add_argument("--s3_bucket", type=str, default="s3://phidias/zchen/czq_objaverse_toon_1280",
                       help="S3 bucket path for Objaverse data")
    args = parser.parse_args()
    
    print(f"Creating dataset with S3 loading from: {args.dataset_metadata_path}")
    print(f"S3 bucket: {args.s3_bucket}")
    
    # Create custom image operator: Load from S3 >> RGBA preprocessing
    s3_image_operator = LoadImageFromS3(s3_bucket=args.s3_bucket) >> RGBAPreprocessing(
        target_size=(1024, 1024),
    )
    
    # Create dataset with custom operator
    dataset = UnifiedDataset(
        base_path="",
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=s3_image_operator
    )
    
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size} samples (with {args.dataset_repeat}x repeat)")
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Initialize wandb on main process
    use_wandb = (
        WANDB_AVAILABLE and os.environ.get("WANDB_DISABLED", "false").lower() != "true"
    )
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
                "dataset_size": dataset_size,
                "save_steps": args.save_steps,
                "task": args.task,
                "trainable_models": args.trainable_models,
                "lora_rank": args.lora_rank,
                "max_pixels": args.max_pixels,
                "dataset_repeat": args.dataset_repeat,
                "batch_size": args.batch_size,
                "s3_bucket": args.s3_bucket,
            },
        )
        accelerator.print(
            f"✓ Wandb initialized: project={wandb_project}, run_name={wandb_run_name}"
        )
    elif accelerator.is_main_process:
        accelerator.print("Wandb logging disabled")
    
    # Create model normally - DeepSpeed will wrap it in accelerator.prepare()
    # Per official docs: https://huggingface.co/docs/accelerate/usage_guides/deepspeed
    accelerator.print("Creating model...")
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
    
    # Setup optimizer, scheduler, dataloader
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        shuffle=True, 
        batch_size=args.batch_size, 
        num_workers=args.dataset_num_workers
    )
    
    # Prepare with accelerator (DeepSpeed ZeRO-3 wrapping happens here)
    accelerator.print("Preparing model with DeepSpeed ZeRO-3...")
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    accelerator.print("✓ DeepSpeed ZeRO-3 preparation complete!")
    accelerator.print("✓ Using accelerator.save_state() for checkpoint saving (no timeout!)")
    
    # Create checkpoint manager (uses save_state instead of get_state_dict)
    checkpoint_manager = DeepSpeedCheckpointManager(args.output_path, accelerator)
    
    # Training loop
    global_step = 0
    for epoch_id in range(args.num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(
            dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch_id + 1}/{args.num_epochs}"
        )
        for data in progress_bar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # Model's forward_preprocess() auto-detects and handles batched data
                # data = {'image': [img1,...,img8], 'prompt': [p1,...,p8]} with batch_size=8
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)  # Processes all 8 samples, returns averaged loss
                accelerator.backward(loss)
                optimizer.step()
                
                # Save checkpoint using DeepSpeed-compatible method
                checkpoint_manager.on_step_end(args.save_steps)
                
                scheduler.step()
            
            loss_value = (
                accelerator.gather_for_metrics(loss.detach()).mean().item()
            )
            epoch_loss += loss_value
            num_batches += 1
            global_step += 1

            if accelerator.is_main_process:
                progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

            if use_wandb and accelerator.is_main_process:
                wandb.log(
                    {
                        "train/loss": loss_value,
                        "train/epoch": epoch_id,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/step": global_step,
                    }
                )
        
        # Save at end of epoch if not saving by steps
        if args.save_steps is None:
            checkpoint_dir = checkpoint_manager.on_epoch_end(epoch_id)
            if use_wandb and accelerator.is_main_process:
                # Log checkpoint directory as artifact
                artifact = wandb.Artifact(
                    name=f"checkpoint-epoch-{epoch_id}",
                    type="model",
                    description=f"DeepSpeed ZeRO-3 checkpoint at epoch {epoch_id}",
                )
                artifact.add_dir(checkpoint_dir)
                wandb.log_artifact(artifact)

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        if use_wandb and accelerator.is_main_process:
            wandb.log(
                {
                    "train/epoch_loss": avg_epoch_loss,
                    "epoch": epoch_id + 1,
                }
            )
        accelerator.print(
            f"Epoch {epoch_id + 1}/{args.num_epochs} - Average Loss: {avg_epoch_loss:.4f}"
        )
    
    # Save final checkpoint
    checkpoint_manager.on_training_end(args.save_steps)

    if use_wandb and accelerator.is_main_process:
        accelerator.wait_for_everyone()
        final_checkpoint_dir = os.path.join(args.output_path, "final")
        if os.path.exists(final_checkpoint_dir):
            artifact = wandb.Artifact(
                name="checkpoint-final",
                type="model",
                description="Final DeepSpeed ZeRO-3 checkpoint",
            )
            artifact.add_dir(final_checkpoint_dir)
            wandb.log_artifact(artifact)
        wandb.finish()
        accelerator.print("✓ Wandb run finished")
    
    accelerator.print("✓ Training complete!")
    accelerator.print(f"✓ Checkpoints saved in: {args.output_path}")
    accelerator.print("\nTo convert checkpoint to standard format:")
    accelerator.print(f"  python zero_to_fp32.py {args.output_path}/final {args.output_path}/final_model.safetensors")
