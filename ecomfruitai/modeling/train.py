import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

from ..config import MODEL_CONFIG, TRAINING_CONFIG, SCHEDULER_CONFIG, DATA_CONFIG, DEVICE, PATHS

def setup_models():
    """Initialize and setup all models"""
    # Text encoder (CLIP)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_CONFIG["clip_model"])
    text_encoder = CLIPTextModel.from_pretrained(MODEL_CONFIG["clip_model"])
    text_encoder.to(DEVICE)
    text_encoder.eval()

    # VAE (for latent space)
    vae = AutoencoderKL.from_pretrained(MODEL_CONFIG["vae_model"])
    vae.to(DEVICE)
    vae.eval()

    # U-Net (this is what we'll train)
    unet = UNet2DConditionModel(**MODEL_CONFIG["unet_config"])
    unet.to(DEVICE)

    # Noise scheduler
    scheduler = DDPMScheduler(**SCHEDULER_CONFIG)

    # Freeze text encoder and VAE
    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False

    print("Models loaded successfully!")
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")
    
    return tokenizer, text_encoder, vae, unet, scheduler

def encode_text(texts, tokenizer, text_encoder):
    """Encode text prompts to embeddings"""
    inputs = tokenizer(texts, padding=True, truncation=True,
                      max_length=77, return_tensors="pt")

    with torch.no_grad():
        embeddings = text_encoder(inputs.input_ids.to(DEVICE))[0]

    return embeddings

def encode_images(images, vae):
    """Encode images to latent space"""
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * DATA_CONFIG["vae_scaling_factor"]
    return latents

def train_model(train_loader, models, generate_fn=None):
    """Main training loop"""
    tokenizer, text_encoder, vae, unet, scheduler = models
    
    # Training setup
    optimizer = AdamW(unet.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    scaler = GradScaler()
    
    unet.train()
    
    step = 0
    for epoch in range(TRAINING_CONFIG["num_epochs"]):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")

        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(DEVICE)
            texts = batch['text']

            # Mixed precision training
            with autocast(device_type=DEVICE.type):
                # Encode text and images
                text_embeddings = encode_text(texts, tokenizer, text_encoder)
                latents = encode_images(images, vae)

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                         (latents.shape[0],), device=DEVICE).long()

                # Add noise to latents
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample

                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / TRAINING_CONFIG["gradient_accumulation_steps"]

            # Backward pass with scaling
            scaler.scale(loss).backward()

            if (batch_idx + 1) % TRAINING_CONFIG["gradient_accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1

            epoch_loss += loss.item() * TRAINING_CONFIG["gradient_accumulation_steps"]
            progress_bar.set_postfix({'loss': loss.item() * TRAINING_CONFIG["gradient_accumulation_steps"]})

            # Save checkpoint
            if step % TRAINING_CONFIG["checkpoint_frequency"] == 0 and step > 0:
                save_checkpoint(unet, optimizer, step, loss.item())

            # Test generation with updated models
            if step % TRAINING_CONFIG["test_generation_frequency"] == 0 and step > 0 and generate_fn:
                # Create updated models tuple for testing
                updated_models = (tokenizer, text_encoder, vae, unet, scheduler)
                test_generation_with_models(updated_models, step)

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

    print("Training completed!")
    
    # Return the updated models tuple
    return (tokenizer, text_encoder, vae, unet, scheduler)

def save_checkpoint(unet, optimizer, step, loss):
    """Save model checkpoint"""
    # Use centralized path configuration
    checkpoint_path = os.path.join(PATHS["checkpoints"], f'checkpoint_step_{step}.pt')
    torch.save({
        'unet': unet.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'loss': loss
    }, checkpoint_path)
    print(f"Checkpoint saved at step {step}")

def test_generation_with_models(models, step):
    """Test generation during training with current models"""
    print(f"\nTesting generation at step {step}...")
    try:
        with torch.no_grad():
            from .predict import generate_image
            test_prompt = "red apple, whole fruit, realistic photo"
            generated = generate_image(test_prompt, models, num_inference_steps=20)
            print("Generation test successful!")
    except Exception as e:
        print(f"Generation test failed: {e}")
