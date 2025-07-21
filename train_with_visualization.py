"""
Training Script with Visualization for Seismic Diffusion Model

This script provides a comprehensive training pipeline with real-time visualization,
monitoring, and sample generation capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
from pathlib import Path
import logging
import argparse
from typing import Optional, Dict, List, Tuple
import time
from tqdm import tqdm
import json

# Import our modules
from simple_diffusion_seismic import create_seismic_diffusion_model, SeismicDiffusionModel
from visualization_utils import SeismicVisualizationUtils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SeismicDataGenerator:
    """Generate synthetic seismic velocity models for training."""
    
    def __init__(self, image_size: int = 64):
        self.image_size = image_size
    
    def create_layered_model(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create layered velocity models."""
        models = []
        
        for _ in range(batch_size):
            # Create a layered model with random layer thicknesses and velocities
            model = torch.zeros(self.image_size, self.image_size, device=device)
            
            # Add horizontal layers with random velocities
            current_depth = 0
            while current_depth < self.image_size:
                layer_thickness = torch.randint(3, 15, (1,)).item()
                layer_velocity = torch.rand(1).item() * 0.6 + 0.2  # Velocity between 0.2 and 0.8
                
                end_depth = min(current_depth + layer_thickness, self.image_size)
                model[current_depth:end_depth, :] = layer_velocity
                current_depth = end_depth
            
            # Add some lateral velocity variations
            for _ in range(torch.randint(2, 5, (1,)).item()):
                x_start = torch.randint(0, self.image_size//2, (1,)).item()
                x_end = torch.randint(self.image_size//2, self.image_size, (1,)).item()
                y_start = torch.randint(0, self.image_size//2, (1,)).item()
                y_end = torch.randint(self.image_size//2, self.image_size, (1,)).item()
                
                variation = (torch.rand(1).item() - 0.5) * 0.2
                model[y_start:y_end, x_start:x_end] += variation
            
            # Add smooth noise
            noise = torch.randn(self.image_size, self.image_size, device=device) * 0.05
            # Apply Gaussian smoothing to the noise
            noise = torch.nn.functional.conv2d(
                noise.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 3, 3, device=device) / 9,
                padding=1
            ).squeeze()
            
            model += noise
            
            # Normalize to [-1, 1] range for training
            model = torch.clamp(model, 0, 1)
            model = (model - 0.5) * 2  # Scale to [-1, 1]
            
            models.append(model.unsqueeze(0))  # Add channel dimension
        
        return torch.stack(models)
    
    def create_fault_model(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create models with fault structures."""
        models = []
        
        for _ in range(batch_size):
            # Start with layered model
            model = self.create_layered_model(1, device).squeeze(0).squeeze(0)
            
            # Add fault
            fault_x = torch.randint(self.image_size//4, 3*self.image_size//4, (1,)).item()
            fault_dip = torch.rand(1).item() * 0.5 + 0.2  # Dip between 0.2 and 0.7
            offset = torch.randint(2, 8, (1,)).item()
            
            for y in range(self.image_size):
                fault_pos = int(fault_x + y * fault_dip)
                if 0 <= fault_pos < self.image_size:
                    # Create offset across fault
                    if fault_pos + offset < self.image_size:
                        model[y, fault_pos:] = torch.roll(model[y, fault_pos:], offset, dims=0)
            
            models.append(model.unsqueeze(0))  # Add channel dimension
        
        return torch.stack(models)
    
    def create_dataset(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Create a mixed dataset of different geological structures."""
        all_models = []
        
        # Create different types of models
        layered_models = self.create_layered_model(num_samples // 2, device)
        fault_models = self.create_fault_model(num_samples // 2, device)
        
        all_models.extend([layered_models, fault_models])
        
        # Combine and shuffle
        dataset = torch.cat(all_models, dim=0)
        
        # Add remaining samples if needed
        remaining = num_samples - dataset.shape[0]
        if remaining > 0:
            extra_models = self.create_layered_model(remaining, device)
            dataset = torch.cat([dataset, extra_models], dim=0)
        
        return dataset


class TrainingConfig:
    """Configuration for training."""
    
    def __init__(self):
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.save_interval = 10
        self.sample_interval = 5
        self.num_train_samples = 1000
        self.num_val_samples = 200
        self.image_size = 64
        self.num_workers = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = './outputs'
        self.checkpoint_dir = './checkpoints'
        self.log_dir = './logs'


class DiffusionTrainer:
    """Trainer for seismic diffusion model with visualization."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directories
        for dir_path in [config.output_dir, config.checkpoint_dir, config.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = create_seismic_diffusion_model(image_size=config.image_size)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # Initialize visualization utils
        self.visualizer = SeismicVisualizationUtils()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        logger.info(f"Initialized trainer with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        logger.info("Generating synthetic seismic data...")
        
        data_generator = SeismicDataGenerator(self.config.image_size)
        
        # Generate training data
        train_data = data_generator.create_dataset(self.config.num_train_samples, self.device)
        train_dataset = TensorDataset(train_data)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False
        )
        
        # Generate validation data
        val_data = data_generator.create_dataset(self.config.num_val_samples, self.device)
        val_dataset = TensorDataset(val_data)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"Created datasets: {len(train_dataset)} training, {len(val_dataset)} validation samples")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
            for batch_idx, (batch,) in enumerate(pbar):
                batch = batch.to(self.device)
                
                # Forward pass
                loss = self.model.training_step(batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(self.device)
                loss = self.model.training_step(batch)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def generate_samples(self, num_samples: int = 4) -> torch.Tensor:
        """Generate samples for visualization."""
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(
                batch_size=num_samples,
                device=self.device,
                num_inference_steps=50
            )
        return samples
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
        
        self.model.save_checkpoint(
            str(checkpoint_path),
            optimizer_state=self.optimizer.state_dict(),
            epoch=epoch,
            loss=val_loss
        )
        
        # Save training config
        config_path = Path(self.config.checkpoint_dir) / f"config_epoch_{epoch+1}.json"
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2, default=str)
    
    def visualize_training_progress(self, epoch: int, val_data: torch.Tensor):
        """Create and save training visualizations."""
        # Plot training curves
        if len(self.train_losses) > 1:
            losses = {
                'Training Loss': self.train_losses,
                'Validation Loss': self.val_losses
            }
            fig = self.visualizer.plot_training_progress(losses)
            plt.savefig(Path(self.config.output_dir) / f"training_progress_epoch_{epoch+1}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Generate and visualize samples
        generated_samples = self.generate_samples(num_samples=4)
        
        # Compare with validation data
        fig = self.visualizer.compare_samples(
            val_data[:4],
            generated_samples,
            num_comparisons=4
        )
        plt.savefig(Path(self.config.output_dir) / f"samples_epoch_{epoch+1}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Show forward process on validation data
        if epoch % (self.config.sample_interval * 2) == 0:
            fig = self.visualizer.visualize_forward_process(
                self.model,
                val_data[:2],
                timesteps=[0, 200, 400, 600, 800, 1000]
            )
            plt.savefig(Path(self.config.output_dir) / f"forward_process_epoch_{epoch+1}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        logger.info(f"Visualizations saved for epoch {epoch+1}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        
        # Get some validation data for visualization
        val_data_for_vis = next(iter(val_loader))[0][:8].to(self.device)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = Path(self.config.checkpoint_dir) / "best_model.pt"
                self.model.save_checkpoint(
                    str(best_checkpoint_path),
                    optimizer_state=self.optimizer.state_dict(),
                    epoch=epoch,
                    loss=val_loss
                )
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Periodic saving and visualization
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)
            
            if (epoch + 1) % self.config.sample_interval == 0:
                self.visualize_training_progress(epoch, val_data_for_vis)
        
        logger.info("Training completed!")
        
        # Final visualization
        self.visualize_training_progress(self.config.num_epochs - 1, val_data_for_vis)
        
        # Save final model
        final_checkpoint_path = Path(self.config.checkpoint_dir) / "final_model.pt"
        self.model.save_checkpoint(
            str(final_checkpoint_path),
            optimizer_state=self.optimizer.state_dict(),
            epoch=self.config.num_epochs - 1,
            loss=self.val_losses[-1]
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Seismic Diffusion Model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--num_train_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--num_val_samples", type=int, default=200, help="Number of validation samples")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    
    # Update config with command line arguments
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Auto-detect device
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Training configuration: {vars(config)}")
    
    # Create trainer and start training
    trainer = DiffusionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()