import os
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from .config import DATA_CONFIG, DATASET_CONFIG, TRAINING_CONFIG

def download_fruit_dataset():
    """Download the fruit dataset from Kaggle"""
    path = kagglehub.dataset_download(DATASET_CONFIG["kaggle_dataset"])
    print("Path to dataset files:", path)
    return path

def has_descriptive_info(class_name):
    """Check if class name contains descriptive information"""
    name_lower = class_name.lower()
    return any(keyword in name_lower for keyword in DATASET_CONFIG["descriptive_keywords"])

def create_text_description(class_name):
    """Convert class name to natural text description"""
    name = class_name.lower().replace(' 1', '').replace(' 2', '').replace(' 3', '')

    # Base fruit/vegetable name mapping
    base_mapping = {
        'apple': 'apple', 'cherry': 'cherry', 'tomato': 'tomato',
        'pear': 'pear', 'grape': 'grape', 'cucumber': 'cucumber',
        'pepper': 'bell pepper', 'banana': 'banana'
    }
    
    base = next((v for k, v in base_mapping.items() if k in name), name.split()[0])

    # Extract descriptors
    descriptors = []

    # Colors
    colors = ['red', 'green', 'yellow', 'white', 'pink', 'black', 'orange', 'blue']
    descriptors.extend([color for color in colors if color in name])

    # States
    if 'ripe' in name: descriptors.append('ripe')
    elif 'not ripen' in name: descriptors.append('unripe')
    if 'fresh' in name: descriptors.append('fresh')
    if 'sweet' in name: descriptors.append('sweet')
    if 'flat' in name: descriptors.append('flat')
    if 'mini' in name: descriptors.append('small')

    # Varieties
    if 'golden' in name: descriptors.append('golden')
    if 'granny smith' in name: descriptors.append('granny smith')
    if 'delicious' in name: descriptors.append('red delicious')

    # Combine
    if descriptors:
        description = f"{' '.join(descriptors)} {base}, whole fruit, realistic photo"
    else:
        description = f"{base}, whole fruit, realistic photo"

    return description

def get_transforms():
    """Get image transforms for training"""
    return transforms.Compose([
        transforms.Resize(DATA_CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(DATA_CONFIG["normalize_mean"], DATA_CONFIG["normalize_std"])
    ])

class FruitDiffusionDataset(Dataset):
    """Dataset class for fruit diffusion training"""
    
    def __init__(self, data_path, fruit_classes, transform=None, split='Training'):
        self.data_path = data_path
        self.fruit_classes = fruit_classes
        self.transform = transform
        self.split = split

        # Collect all image paths and their descriptions
        self.image_paths = []
        self.text_descriptions = []

        split_path = os.path.join(data_path, split)

        for fruit_class in fruit_classes:
            class_path = os.path.join(split_path, fruit_class)
            if os.path.exists(class_path):
                image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

                for img_file in image_files:
                    img_path = os.path.join(class_path, img_file)
                    text_desc = create_text_description(fruit_class)

                    self.image_paths.append(img_path)
                    self.text_descriptions.append(text_desc)

        print(f"Dataset created with {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'text': self.text_descriptions[idx]
        }

def create_datasets_and_loaders(dataset_path):
    """Create train and test datasets with data loaders"""
    # Get all fruit classes
    train_path = os.path.join(dataset_path, "Training")
    fruit_classes = sorted(os.listdir(train_path))
    
    # Filter meaningful classes
    descriptive_classes = [cls for cls in fruit_classes if has_descriptive_info(cls)]
    
    print(f"Total classes: {len(fruit_classes)}")
    print(f"Descriptive classes: {len(descriptive_classes)}")
    
    # Get transforms
    transform = get_transforms()
    
    # Create datasets
    train_dataset = FruitDiffusionDataset(
        dataset_path, descriptive_classes, transform=transform, split='Training'
    )
    
    test_dataset = FruitDiffusionDataset(
        dataset_path, descriptive_classes, transform=transform, split='Test'
    )
    
    # Create subset for faster training
    subset_size = TRAINING_CONFIG["subset_size"]
    if len(train_dataset) > subset_size:
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    # Create data loaders
    batch_size = TRAINING_CONFIG["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, descriptive_classes
