"""
FLUX Kontext LoRA Training with S3 Support
===========================================

This script trains FLUX Kontext for rotation tasks using:
1. S3 data loading (no local storage needed)
2. RGBA background processing (random backgrounds for robustness)
3. LoRA finetuning (efficient, small model size)

Data Format Expected:
{
  "image": "path/to/rotated.png",      # Target output
  "kontext_images": "path/to/original.png",  # Context input  
  "prompt": "Rotate the image 90 degrees"
}
"""

import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig, ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, flux_parser
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.trainers.rotation_pair_dataset import RotationPairDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class FluxTrainingModule(DiffusionTrainingModule):
    """Training module for FLUX Kontext"""
    
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, 
        lora_target_modules="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp", 
        lora_rank=32, 
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
    ):
        super().__init__()
        
        # Load FLUX models (DiT, text encoders, VAE)
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths, enable_fp8_training=False
        )
        self.pipe = FluxImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs
        )
        
        # Switch to training mode + inject LoRA adapters
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        
    
    def forward_preprocess(self, data):
        """
        Prepare data for training forward pass - supports both single and batched inputs
        
        Pipeline units handle batching natively:
          - Single: {"image": img, "prompt": "text"}
          - Batched: {"image": [img1, img2], "prompt": ["text1", "text2"]}
        
        data contains:
          - "image": target output image(s) (rotated)
          - "kontext_images": context input image(s) (original)  
          - "prompt": text instruction(s)
        """
        # Text prompts (tokenizers handle both str and list[str])
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        
        # Get image dimensions (handle both single image and list of images)
        input_image = data["image"]
        if isinstance(input_image, list):
            height, width = input_image[0].size[1], input_image[0].size[0]
        else:
            height, width = input_image.size[1], input_image.size[0]
        
        # Image and model configs
        inputs_shared = {
            "input_image": input_image,  # Target output
            "height": height,
            "width": width,
            "cfg_scale": 1,  # No CFG during training
            "embedded_guidance": 1,
            "t5_sequence_length": 512,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        
        # Add extra inputs (kontext_images for Kontext model)
        controlnet_input = {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                # This is where "kontext_images" gets added
                inputs_shared[extra_input] = data[extra_input]
        
        if len(controlnet_input) > 0:
            inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]
        
        # Run through pipeline units (encode text, VAE encode, etc.)
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )
        
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        """
        Training forward pass - compute loss
        """
        if inputs is None:
            inputs = self.forward_preprocess(data)
        
        # Get models
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        
        # Compute diffusion loss
        loss = self.pipe.training_loss(**models, **inputs)
        return loss



if __name__ == "__main__":
    # Parse command line arguments
    parser = flux_parser()
    parser.add_argument(
        "--s3_bucket", 
        type=str, 
        default="s3://phidias/zchen/czq_objaverse_toon_1280",
        help="S3 bucket path where images are stored"
    )
    # Note: --batch_size is now defined in flux_parser() in utils.py
    parser.add_argument(
        "--visualization_steps",
        type=int,
        default=None,
        help="Generate visualization every N steps (e.g., 5000). If None, no visualizations are created."
    )
    parser.add_argument(
        "--visualization_num_samples",
        type=int,
        default=4,
        help="Number of samples to generate for each visualization"
    )
    parser.add_argument(
        "--data_batch_viz_steps",
        type=str,
        default="100",
        help="Comma-separated steps at which to visualize data batches (e.g., '100,1000'). Set to empty string to disable."
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("FLUX Kontext Rotation Training with S3")
    print("=" * 80)
    print(f"Metadata: {args.dataset_metadata_path}")
    print(f"S3 Bucket: {args.s3_bucket}")
    print(f"Output: {args.output_path}")
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print("=" * 80)
    
    # ========================================================================
    # Create Dataset with Proper Paired Processing
    # ========================================================================
    # Key difference: Both source and target images get the SAME random background
    # This matches the reference implementation from dataset_rotate_vector_i2i.py
    
    dataset = RotationPairDataset(
        metadata_path=args.dataset_metadata_path,
        s3_bucket=args.s3_bucket,
        target_size=(1024, 1024),
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=16,
        width_division_factor=16,
        repeat=args.dataset_repeat,
        bgcolors=[[0, 0, 0], [128, 128, 128], [255, 255, 255], [0, 255, 0]],  # Black, Gray, White, Green
        edgecolors=[[255, 255, 255], [0, 0, 0]],  # White, Black edges
        edge_prob=0.3,  # 30% chance of edge
    )
    
    print(f"Dataset created with paired processing:")
    print(f"  Samples: {len(dataset)} (with {args.dataset_repeat}x repeat)")
    print(f"  Background colors: Black, Gray, White, Green (SAME for both images in pair)")
    print(f"  Edge probability: 30%")
    print()
    
    # ========================================================================
    # Create Training Model
    # ========================================================================
    model = FluxTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,  # This should be "kontext_images"
    )
    
    # ========================================================================
    # NOTE: Inference visualization disabled for production training
    # ========================================================================
    # Inference visualization is DISABLED - run separate inference after training
    
    # ========================================================================
    # Create Model Logger (saves checkpoints)
    # ========================================================================
    
    # Prepare WandB config
    wandb_config = {
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_rank": args.lora_rank,
        "lora_target_modules": args.lora_target_modules,
        "max_pixels": args.max_pixels,
        "dataset_repeat": args.dataset_repeat,
        "s3_bucket": args.s3_bucket,
        "metadata_path": args.dataset_metadata_path,
    }
    
    # Parse data batch visualization steps
    data_batch_viz_steps = None
    if hasattr(args, 'data_batch_viz_steps') and args.data_batch_viz_steps:
        try:
            data_batch_viz_steps = [int(s.strip()) for s in args.data_batch_viz_steps.split(',') if s.strip()]
        except ValueError:
            print(f"⚠️  Warning: Could not parse data_batch_viz_steps '{args.data_batch_viz_steps}', disabling data batch visualization")
    
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=(
            FluxLoRAConverter.align_to_opensource_format 
            if args.align_to_opensource_format 
            else lambda x: x
        ),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config=wandb_config,
        wandb_entity=args.wandb_entity if hasattr(args, 'wandb_entity') else None,
        wandb_host=args.wandb_host if hasattr(args, 'wandb_host') else None,
        s3_upload_path=args.s3_checkpoint_path if hasattr(args, 's3_checkpoint_path') else None,
        visualization_steps=None,  # Inference visualization DISABLED
        visualization_callback=None,  # Inference visualization DISABLED
        data_batch_viz_steps=data_batch_viz_steps,
    )
    
    # ========================================================================
    # Launch Training!
    # ========================================================================
    print("Starting training...")
    launch_training_task(dataset, model, model_logger, args=args)

