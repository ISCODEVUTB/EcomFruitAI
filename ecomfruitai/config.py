import torch
import os

# Hardware Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Configuration
MODEL_CONFIG = {
    "clip_model": "openai/clip-vit-base-patch32",
    "vae_model": "stabilityai/sd-vae-ft-mse",
    "unet_config": {
        "sample_size": 16,  # 128//8 = 16 (VAE downsamples by 8)
        "in_channels": 4,   # VAE latent channels
        "out_channels": 4,
        "layers_per_block": 2,
        "block_out_channels": (128, 256, 512, 512),
        "down_block_types": (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ),
        "up_block_types": (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D"
        ),
        "cross_attention_dim": 512,  # CLIP embedding dim
    }
}

# Training Configuration
TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "num_epochs": 2,
    "batch_size": 16,  # Optimized batch size
    "gradient_accumulation_steps": 2,  # Reduced for efficiency
    "subset_size": 1000,  # Smaller subset for faster training
    "checkpoint_frequency": 100,  # More frequent checkpointing
    "test_generation_frequency": 50,  # Frequent generation testing
    "mixed_precision": True,  # Enable mixed precision training
    "freeze_pretrained": True  # Freeze text encoder and VAE
}

# Scheduler Configuration
SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear"
}

# Data Configuration
DATA_CONFIG = {
    "image_size": (128, 128),
    "normalize_mean": [0.5, 0.5, 0.5],
    "normalize_std": [0.5, 0.5, 0.5],
    "vae_scaling_factor": 0.18215
}

# Dataset Configuration
DATASET_CONFIG = {
    "kaggle_dataset": "moltean/fruits",
    "descriptive_keywords": [
        # Colors
        'red', 'green', 'yellow', 'white', 'pink', 'black', 'orange', 'blue', 'dark', 'marron',
        # States/Varieties
        'ripe', 'ripen', 'washed', 'peeled', 'flat', 'mini', 'sweet', 'sour',
        # Specific varieties
        'crimson', 'golden', 'granny', 'delicious', 'lady', 'braeburn', 'meyer',
        'rainier', 'forelle', 'williams', 'abate', 'kaiser', 'monster', 'stone', 'sapo',
        # Conditions
        'fresh', 'rotten', 'worm', 'core', 'heart', 'wedge', 'husk', 'long', 'washed'
    ]
}

# Generation Configuration
GENERATION_CONFIG = {
    "default_inference_steps": 50,
    "fast_inference_steps": 20,
    "default_prompt_suffix": ", whole fruit, realistic photo"
}

# Paths
PATHS = {
    "models": "models",
    "checkpoints": "models/checkpoints",
    "data": "data",
    "raw_data": "data/raw",
    "processed_data": "data/processed",
    "figures": "reports/figures"
}

# Create directories if they don't exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)
