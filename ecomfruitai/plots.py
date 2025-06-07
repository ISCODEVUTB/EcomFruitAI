import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image

def show_generated_image(generated_tensor, title="Generated Image", save_path=None):
    """Convert tensor to displayable image"""
    # Convert from tensor to numpy
    img = generated_tensor[0].cpu().numpy()  # Take first image from batch
    img = np.transpose(img, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]

    # Display
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()
    return img

def show_multiple_generated_images(generated_images, prompts, save_path=None):
    """Display multiple generated images in a grid"""
    num_images = len(generated_images)
    cols = 2
    rows = (num_images + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (generated, prompt) in enumerate(zip(generated_images, prompts)):
        # Convert to displayable format
        img = generated[0].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(prompt, fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()

def visualize_dataset_samples(dataset_path, fruit_classes, num_samples=10):
    """Visualize random samples from the dataset"""
    # Ensure we have the correct full path structure
    if not dataset_path.endswith("fruits-360"):
        dataset_path = os.path.join(dataset_path, "fruits-360_100x100", "fruits-360")
    
    train_path = os.path.join(dataset_path, "Training")
    
    # Verify path exists
    if not os.path.exists(train_path):
        print(f"Warning: Training path not found: {train_path}")
        return
    
    # Select random classes
    random_classes = random.sample(fruit_classes, min(num_samples, len(fruit_classes)))
    
    cols = 5
    rows = (num_samples + 4) // 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for i, fruit_class in enumerate(random_classes):
        class_path = os.path.join(train_path, fruit_class)
        if os.path.exists(class_path):
            img_files = os.listdir(class_path)
            if img_files:
                random_img = random.choice(img_files)
                img_path = os.path.join(class_path, random_img)
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(fruit_class, fontsize=8)
                axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(random_classes), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle('Random Dataset Samples', fontsize=16, y=1.02)
    plt.show()

def plot_training_loss(losses, save_path=None):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()
