"""
Simple Seismic Diffusion Model Implementation

A working, simplified version to get started quickly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleResBlock(nn.Module):
    """Simple residual block with time embedding."""
    
    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = F.silu(self.time_mlp(time_emb))
        h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return x + h


class SimpleUNet(nn.Module):
    """Simple UNet for seismic diffusion."""
    
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: int = 64,
                 time_emb_dim: int = 256):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(features),
            nn.Linear(features, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, features, 3, padding=1)
        self.res1 = SimpleResBlock(features, time_emb_dim)
        self.down1 = nn.Conv2d(features, features*2, 3, stride=2, padding=1)
        
        self.res2 = SimpleResBlock(features*2, time_emb_dim)
        self.down2 = nn.Conv2d(features*2, features*4, 3, stride=2, padding=1)
        
        self.res3 = SimpleResBlock(features*4, time_emb_dim)
        self.down3 = nn.Conv2d(features*4, features*8, 3, stride=2, padding=1)
        
        # Middle
        self.res_mid = SimpleResBlock(features*8, time_emb_dim)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(features*8, features*4, 4, stride=2, padding=1)
        self.res4 = SimpleResBlock(features*8, time_emb_dim)  # features*4 + features*4 from skip
        
        self.up2 = nn.ConvTranspose2d(features*8, features*2, 4, stride=2, padding=1)
        self.res5 = SimpleResBlock(features*4, time_emb_dim)  # features*2 + features*2 from skip
        
        self.up1 = nn.ConvTranspose2d(features*4, features, 4, stride=2, padding=1)
        self.res6 = SimpleResBlock(features*2, time_emb_dim)  # features + features from skip
        
        # Output
        self.final_conv = nn.Conv2d(features*2, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Encoder
        x1 = self.conv1(x)
        x1 = self.res1(x1, time_emb)
        
        x2 = self.down1(x1)
        x2 = self.res2(x2, time_emb)
        
        x3 = self.down2(x2)
        x3 = self.res3(x3, time_emb)
        
        x4 = self.down3(x3)
        
        # Middle
        x4 = self.res_mid(x4, time_emb)
        
        # Decoder
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.res4(x, time_emb)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.res5(x, time_emb)
        
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.res6(x, time_emb)
        
        x = self.final_conv(x)
        return x


class NoiseScheduler:
    """Noise scheduler for the diffusion process."""
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to clean data according to the noise schedule."""
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(x_start.device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(x_start.device)
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[timesteps][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[timesteps][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class SeismicDiffusionModel(nn.Module):
    """Complete seismic diffusion model."""
    
    def __init__(self, 
                 image_size: int = 64,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: int = 64,
                 num_timesteps: int = 1000):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps
        
        # Initialize UNet
        self.unet = SimpleUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features
        )
        
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting noise."""
        return self.unet(x, timesteps)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Single training step computing the denoising loss."""
        batch_size = batch.shape[0]
        device = batch.device
        
        # Sample random timesteps
        timesteps = self.noise_scheduler.sample_timesteps(batch_size, device)
        
        # Sample noise
        noise = torch.randn_like(batch)
        
        # Add noise to the batch
        noisy_batch = self.noise_scheduler.add_noise(batch, noise, timesteps)
        
        # Predict noise
        predicted_noise = self.forward(noisy_batch, timesteps)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, 
               num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """Sample new data using the reverse diffusion process."""
        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps
            
        # Start from pure noise
        shape = (batch_size, self.in_channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)
        
        # Create timestep schedule
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
        
        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(x, t_tensor)
            
            # Compute previous sample using DDPM sampling
            alpha_t = self.noise_scheduler.alphas[t]
            alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
            
            # Mean of the reverse distribution
            pred_original_sample = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
            
            # Compute previous sample mean
            if t > 0:
                alpha_cumprod_prev = self.noise_scheduler.alphas_cumprod[t - 1]
                pred_sample_direction = torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise
                pred_prev_sample = torch.sqrt(alpha_cumprod_prev) * pred_original_sample + pred_sample_direction
                
                # Add noise
                variance = self.noise_scheduler.posterior_variance[t]
                if variance > 0:
                    noise = torch.randn_like(x)
                    pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise
            else:
                pred_prev_sample = pred_original_sample
            
            x = pred_prev_sample
        
        return x

    def save_checkpoint(self, filepath: str, optimizer_state: Optional[dict] = None, 
                       epoch: Optional[int] = None, loss: Optional[float] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'image_size': self.image_size,
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'num_timesteps': self.num_timesteps,
            }
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
            
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: str, device: torch.device = None):
        """Load model from checkpoint."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model with saved configuration
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from {filepath}")
        return model, checkpoint


def create_seismic_diffusion_model(image_size: int = 64, **kwargs) -> SeismicDiffusionModel:
    """Factory function to create a seismic diffusion model."""
    return SeismicDiffusionModel(
        image_size=image_size,
        in_channels=1,
        out_channels=1,
        features=64,
        num_timesteps=1000,
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_seismic_diffusion_model(image_size=64)
    model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 64, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    
    with torch.no_grad():
        output = model(x, timesteps)
        print(f"Forward pass: {x.shape} -> {output.shape}")
        
    # Test training step
    loss = model.training_step(x)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling
    with torch.no_grad():
        samples = model.sample(batch_size=2, device=device, num_inference_steps=50)
        print(f"Sampling: {samples.shape}")
        
    print("All tests passed!")