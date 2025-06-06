import torch
from tqdm import tqdm
from ..config import DEVICE, DATA_CONFIG, GENERATION_CONFIG

def generate_image(prompt, models, num_inference_steps=None):
    """Generate image from text prompt"""
    if num_inference_steps is None:
        num_inference_steps = GENERATION_CONFIG["default_inference_steps"]
    
    tokenizer, text_encoder, vae, unet, scheduler = models
    unet.eval()

    # Encode prompt
    from .train import encode_text
    text_embeddings = encode_text([prompt], tokenizer, text_encoder)

    # Random latent
    latents = torch.randn(1, 4, 16, 16, device=DEVICE)

    # Set scheduler for inference
    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            noise_pred = unet(latents, t, text_embeddings).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents to image
    with torch.no_grad():
        latents = latents / DATA_CONFIG["vae_scaling_factor"]
        images = vae.decode(latents).sample
        images = (images + 1) / 2  # Convert from [-1,1] to [0,1]
        images = torch.clamp(images, 0, 1)

    return images

def generate_multiple_images(prompts, models, num_inference_steps=None):
    """Generate multiple images from list of prompts"""
    if num_inference_steps is None:
        num_inference_steps = GENERATION_CONFIG["fast_inference_steps"]
    
    generated_images = []
    for prompt in prompts:
        print(f"Generating: {prompt}")
        image = generate_image(prompt, models, num_inference_steps)
        generated_images.append(image)
    
    return generated_images

def load_checkpoint(checkpoint_path, unet):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    unet.load_state_dict(checkpoint['unet'])
    print(f"Loaded checkpoint from step {checkpoint['step']}")
    return unet
