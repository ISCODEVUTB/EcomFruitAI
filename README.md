# EcomFruitAI

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Creation of Synthetic Fruit Images with Diffusion Models**

EcomFruitAI is a deep learning project that generates synthetic fruit images using diffusion models. Trained on the Fruits-360 dataset, this system can create fruit images from text descriptions, enabling applications in e-commerce, computer vision research, and data augmentation.

## Features

- **Text-to-Image Generation**: Generate realistic fruit images from natural language descriptions
- **Pre-trained Models**: Uses CLIP text encoder and Stable Diffusion VAE components
- **Custom UNet Architecture**: Optimized for fruit image generation
- **Modular Design**: Clean, organized codebase for easy extension and maintenance
- **Google Colab Compatible**: Designed to run efficiently in cloud environments

## Quick Start

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ISCODEVUTB/EcomFruitAI
    cd EcomFruitAI
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install the package in development mode:
    ```bash
    pip install -e .
    ```

### Basic Usage

#### Using the Jupyter Notebook (Recommended)

Open and run the main notebook for a complete walkthrough:

```bash
jupyter lab notebooks/ecomfruitai.ipynb
```




### Training Configuration

Key training parameters can be modified in `ecomfruitai/config.py`:

```python
TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "num_epochs": 2,
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "subset_size": 1000,  # For faster training
    "checkpoint_frequency": 100,
    "test_generation_frequency": 50
}
```


## Generation Examples

### Single Image Generation

```python
from ecomfruitai.modeling.predict import generate_image
from ecomfruitai.plots import show_generated_image

# Generate image
image = generate_image("green apple, whole fruit, realistic photo", models)

# Display
show_generated_image(image, title="Generated Green Apple")
```

### Batch Generation

```python
from ecomfruitai.modeling.predict import generate_multiple_images
from ecomfruitai.plots import show_multiple_generated_images

# Define prompts
prompts = [
    "red apple, whole fruit, realistic photo",
    "yellow banana, whole fruit, realistic photo",
    "orange carrot, whole vegetable, realistic photo"
]

# Generate multiple images
images = generate_multiple_images(prompts, models)

# Display grid
show_multiple_generated_images(images, prompts)
```

## Notebooks visualization

[Notebooks](https://nbviewer.org/github/ISCODEVUTB/EcomFruitAI/tree/main/notebooks/)

## Dataset

The project uses the [Fruits-360 dataset](https://www.kaggle.com/datasets/moltean/fruits) from Kaggle, which contains:
- 137,104 total images of fruits, vegetables, nuts and seeds
- 201 classes (fruits, vegetables, nuts and seeds)
- 100x100 pixel resolution
- Training set: 102,790 images
- Test set: 34,314 images

The system automatically filters classes with descriptive information (colors, varieties, conditions) for better text-to-image alignment.

## Configuration

All project settings are centralized in `ecomfruitai/config.py`:

- **Model configurations**: Architecture parameters, pre-trained model paths
- **Training settings**: Learning rates, batch sizes, checkpointing
- **Data processing**: Image transforms, normalization parameters
- **Generation parameters**: Inference steps, sampling configurations

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         ecomfruitai and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
│
└── ecomfruitai   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ecomfruitai a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```




## Acknowledgments

- [Fruits-360 Dataset](https://www.kaggle.com/datasets/moltean/fruits) by Mihai Oltean
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/) library
- [Stability AI](https://stability.ai/) for VAE components
- [OpenAI CLIP](https://openai.com/blog/clip/) for text encoding

---
