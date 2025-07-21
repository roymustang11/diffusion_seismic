# Comprehensive Seismic Diffusion Model with Visualization

A complete implementation of diffusion models for seismic velocity model generation with enhanced visualization capabilities for understanding the noise addition and denoising process.

## Overview

This project provides a comprehensive toolkit for training and applying diffusion models to seismic data generation. It includes:

- **Robust UNet Architecture**: Enhanced with attention mechanisms and residual blocks
- **Complete Diffusion Framework**: Proper DDPM implementation with configurable noise scheduling  
- **Advanced Visualization**: Real-time monitoring and step-by-step process visualization
- **Training Pipeline**: Full training script with validation and checkpointing
- **Interactive Demos**: Comprehensive demonstrations of the diffusion process

## Features

### 🏗️ Model Architecture
- **Enhanced UNet**: Skip connections, attention mechanisms, and residual blocks
- **Flexible Configuration**: Configurable model size, attention resolutions, and channel multipliers
- **Efficient Design**: Optimized for seismic data with proper normalization

### 🎯 Diffusion Process
- **Forward Process**: Clean → Noisy with configurable noise scheduling
- **Reverse Process**: Noisy → Clean with DDPM sampling
- **Multiple Schedulers**: Linear and cosine noise schedules
- **Flexible Inference**: Adjustable number of denoising steps

### 📊 Visualization Capabilities
- **Forward Process**: Step-by-step noise addition visualization
- **Reverse Process**: Real-time denoising visualization  
- **Training Monitoring**: Loss curves and sample generation tracking
- **Comparative Analysis**: Ground truth vs generated samples
- **Parameter Studies**: Effect of different inference settings

### 🚀 Training Features
- **Synthetic Data Generation**: Layered and fault-based seismic models
- **Real-time Monitoring**: Training progress and sample visualization
- **Automatic Checkpointing**: Best model saving and recovery
- **Validation Pipeline**: Comprehensive evaluation metrics

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/roymustang11/diffusion_seismic.git
cd diffusion_seismic

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Training a Model

Train a model with default settings:
```bash
python train_with_visualization.py
```

Train with custom parameters:
```bash
python train_with_visualization.py \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_train_samples 2000 \
    --image_size 64
```

### 2. Running Demonstrations

Complete demonstration suite:
```bash
python demo_diffusion_process.py --model_path checkpoints/best_model.pt
```

Specific demonstration types:
```bash
# Forward process (clean → noisy)
python demo_diffusion_process.py --demo_type forward --model_path checkpoints/best_model.pt

# Reverse process (noisy → clean) 
python demo_diffusion_process.py --demo_type reverse --model_path checkpoints/best_model.pt

# Noise level comparison
python demo_diffusion_process.py --demo_type noise --model_path checkpoints/best_model.pt

# Generated vs ground truth comparison
python demo_diffusion_process.py --demo_type comparison --model_path checkpoints/best_model.pt
```

### 3. Using the Model Programmatically

```python
from simple_diffusion_seismic import create_seismic_diffusion_model
from visualization_utils import SeismicVisualizationUtils
import torch

# Create or load model
model = create_seismic_diffusion_model(image_size=64)
# model, checkpoint = SeismicDiffusionModel.load_checkpoint('path/to/model.pt')

# Generate samples
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
with torch.no_grad():
    samples = model.sample(batch_size=4, device=device, num_inference_steps=50)

# Visualize results
vis = SeismicVisualizationUtils()
fig = vis.plot_seismic_model(samples[0], title="Generated Seismic Model")
```

## File Structure

```
diffusion_seismic/
├── simple_diffusion_seismic.py      # Core diffusion model implementation
├── visualization_utils.py           # Comprehensive visualization utilities  
├── train_with_visualization.py      # Training script with monitoring
├── demo_diffusion_process.py        # Interactive demonstration script
├── requirements.txt                 # Project dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
├── outputs/                         # Training visualizations
├── checkpoints/                     # Model checkpoints
└── demo_outputs/                    # Demonstration outputs
```

## Model Architecture Details

### UNet Components
- **Time Embedding**: Sinusoidal position embeddings for timestep encoding
- **Residual Blocks**: GroupNorm + SiLU activation + skip connections
- **Attention Blocks**: Multi-head self-attention for feature refinement
- **Skip Connections**: Proper encoder-decoder connections for detail preservation

### Diffusion Process
- **Noise Schedule**: Linear beta schedule from 0.0001 to 0.02
- **Forward Process**: Progressive Gaussian noise addition
- **Reverse Process**: Learned denoising with UNet prediction
- **Sampling**: DDPM algorithm with configurable inference steps

## Training Parameters

### Default Configuration
```python
batch_size = 16
learning_rate = 1e-4
num_epochs = 100
image_size = 64
num_train_samples = 1000
num_val_samples = 200
```

### Synthetic Data Generation
- **Layered Models**: Horizontal velocity layers with random thicknesses
- **Fault Models**: Geological fault structures with realistic offsets
- **Noise Addition**: Smooth Gaussian noise for realistic variation
- **Normalization**: [-1, 1] range for optimal training

## Visualization Examples

### Forward Process
Shows gradual noise addition from clean seismic data to pure noise across timesteps [0, 200, 400, 600, 800, 1000].

### Reverse Process  
Demonstrates step-by-step denoising from random noise to coherent seismic structures.

### Training Progress
Real-time monitoring of training/validation loss with sample generation at regular intervals.

### Comparative Analysis
Side-by-side comparison of ground truth vs generated velocity models.

## Performance Notes

### Model Size
- **Parameters**: ~18M parameters for default configuration
- **Memory Usage**: ~2GB GPU memory for training (batch_size=16)
- **Training Time**: ~1-2 minutes per epoch (CPU), ~10-20 seconds (GPU)

### Inference Speed
- **50 steps**: ~15-20 seconds (CPU), ~2-3 seconds (GPU)
- **100 steps**: ~30-40 seconds (CPU), ~4-6 seconds (GPU)

## Advanced Usage

### Custom Data Loading
```python
from torch.utils.data import DataLoader, TensorDataset

# Load your seismic data
your_data = load_seismic_data()  # Shape: (N, 1, H, W)
dataset = TensorDataset(your_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train with custom data
trainer = DiffusionTrainer(config)
trainer.train_custom(dataloader)
```

### Model Customization
```python
# Create custom model
model = SeismicDiffusionModel(
    image_size=128,           # Larger images
    features=128,             # More model capacity  
    num_timesteps=2000        # More denoising steps
)
```

### Visualization Customization
```python
vis = SeismicVisualizationUtils(figsize=(15, 10), dpi=150)

# Custom colormap for seismic data
vis.colormap = 'seismic'

# Custom timesteps for visualization
custom_timesteps = [0, 100, 300, 500, 700, 900, 1000]
vis.visualize_forward_process(model, data, timesteps=custom_timesteps)
```

## Research Applications

This implementation is suitable for:

- **Seismic Data Augmentation**: Generate synthetic training data
- **Uncertainty Quantification**: Multiple realizations of subsurface models  
- **Inversion Studies**: Probabilistic seismic inversion
- **Educational Demonstrations**: Understanding diffusion processes
- **Method Development**: Baseline for advanced diffusion techniques

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{seismic_diffusion_2024,
  title={Comprehensive Seismic Diffusion Model with Visualization},
  author={roymustang11},
  year={2024},
  url={https://github.com/roymustang11/diffusion_seismic}
}
```

## Acknowledgments

- Based on the DDPM paper: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- UNet architecture inspired by "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)
- Seismic data generation techniques adapted from common geological modeling practices
