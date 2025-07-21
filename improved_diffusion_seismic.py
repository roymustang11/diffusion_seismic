"""
Improved Seismic Diffusion Model Implementation

This module implements a comprehensive diffusion model specifically designed for seismic data
generation with enhanced UNet architecture, attention mechanisms, and proper forward/reverse
diffusion processes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Union, List
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


class ResidualBlock(nn.Module):
    """Residual block with time embedding and group normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First convolution
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # Second convolution
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        
        # First convolution path
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = F.silu(self.time_mlp(time_emb))
        h = h + time_emb[:, :, None, None]
        
        # Second convolution path
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + residual


class AttentionBlock(nn.Module):
    """Self-attention block for the UNet."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        
        # Reshape for attention computation
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, batch, num_heads, height*width, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        
        out = self.proj_out(out)
        return out + residual


class DownBlock(nn.Module):
    """Downsampling block with residual connections and optional attention."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 has_attention: bool = False, num_layers: int = 2, downsample: bool = True):
        super().__init__()
        self.has_attention = has_attention
        self.downsample_enabled = downsample
        
        # Residual blocks
        self.resnets = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        
        # Optional attention
        if has_attention:
            self.attentions = nn.ModuleList([
                AttentionBlock(out_channels) for _ in range(num_layers)
            ])
        
        # Downsampling
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_connections = []
        
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, time_emb)
            if self.has_attention:
                x = self.attentions[i](x)
            skip_connections.append(x)
        
        if self.downsample_enabled:
            x = self.downsample(x)
        
        return x, skip_connections


class UpBlock(nn.Module):
    """Upsampling block with residual connections and optional attention."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 has_attention: bool = False, num_layers: int = 2, upsample: bool = True):
        super().__init__()
        self.has_attention = has_attention
        self.upsample_enabled = upsample
        
        # Upsampling
        if upsample:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        else:
            self.upsample = nn.Identity()
        
        # Residual blocks (note: input channels include skip connections)
        self.resnets = nn.ModuleList([
            ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim)
            if i == 0 else ResidualBlock(out_channels + out_channels, out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        
        # Optional attention
        if has_attention:
            self.attentions = nn.ModuleList([
                AttentionBlock(out_channels) for _ in range(num_layers)
            ])

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor], time_emb: torch.Tensor) -> torch.Tensor:
        if self.upsample_enabled:
            x = self.upsample(x)
        
        for i, resnet in enumerate(self.resnets):
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]  # Reverse order
                # Ensure spatial dimensions match
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            
            x = resnet(x, time_emb)
            if self.has_attention:
                x = self.attentions[i](x)
        
        return x


class ImprovedUNet(nn.Module):
    """Enhanced UNet architecture for seismic diffusion model."""
    
    def __init__(self, 
                 in_channels: int = 1, 
                 model_channels: int = 128,
                 out_channels: int = 1,
                 num_res_blocks: int = 2,
                 attention_resolutions: Tuple[int, ...] = (16, 32),
                 channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
                 dropout: float = 0.1,
                 time_embed_dim: Optional[int] = None):
        super().__init__()
        
        if time_embed_dim is None:
            time_embed_dim = model_channels * 4
            
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        ch = model_channels
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for j in range(num_res_blocks):
                has_attention = (64 // (2 ** i)) in attention_resolutions
                self.encoder_blocks.append(nn.ModuleDict({
                    'resnet': ResidualBlock(ch, out_ch, time_embed_dim),
                    'attention': AttentionBlock(out_ch) if has_attention else nn.Identity(),
                    'downsample': nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if j == num_res_blocks - 1 and i < len(channel_mult) - 1 else nn.Identity()
                }))
                ch = out_ch
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResidualBlock(ch, ch, time_embed_dim),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_embed_dim),
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for j in range(num_res_blocks + 1):
                # Include skip connection channels
                resnet_in_ch = ch + out_ch if j == 0 else out_ch + out_ch if j < num_res_blocks else out_ch
                has_attention = (64 // (2 ** i)) in attention_resolutions
                self.decoder_blocks.append(nn.ModuleDict({
                    'resnet': ResidualBlock(resnet_in_ch, out_ch, time_embed_dim),
                    'attention': AttentionBlock(out_ch) if has_attention else nn.Identity(),
                    'upsample': nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1) if j == num_res_blocks and i > 0 else nn.Identity()
                }))
                ch = out_ch
        
        # Output layers
        self.output_norm = nn.GroupNorm(8, model_channels)
        self.output_conv = nn.Conv2d(model_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            timesteps: Timestep tensor of shape (batch_size,)
            
        Returns:
            Output tensor of same shape as input
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Initial convolution
        h = self.input_conv(x)
        
        # Encoder with skip connections
        skip_connections = []
        for block in self.encoder_blocks:
            h = block['resnet'](h, time_emb)
            h = block['attention'](h)
            skip_connections.append(h)
            h = block['downsample'](h)
        
        # Middle block
        h = self.middle_block[0](h, time_emb)
        h = self.middle_block[1](h)
        h = self.middle_block[2](h, time_emb)
        
        # Decoder with skip connections
        skip_idx = len(skip_connections) - 1
        for i, block in enumerate(self.decoder_blocks):
            # Add skip connection if available
            if skip_idx >= 0 and i % (self.num_res_blocks + 1) < self.num_res_blocks:
                skip = skip_connections[skip_idx]
                # Ensure spatial dimensions match
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
                h = torch.cat([h, skip], dim=1)
                if i % (self.num_res_blocks + 1) == self.num_res_blocks - 1:
                    skip_idx -= 1
            
            h = block['resnet'](h, time_emb)
            h = block['attention'](h)
            h = block['upsample'](h)
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        
        return h


class NoiseScheduler:
    """Noise scheduler for the diffusion process."""
    
    def __init__(self, 
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 schedule_type: str = "linear"):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        # Create beta schedule
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            # Cosine schedule from Improved DDPM paper
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Precompute useful values
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
    """Complete seismic diffusion model combining UNet and noise scheduler."""
    
    def __init__(self, 
                 image_size: int = 64,
                 in_channels: int = 1,
                 model_channels: int = 128,
                 out_channels: int = 1,
                 num_res_blocks: int = 2,
                 attention_resolutions: Tuple[int, ...] = (16, 32),
                 channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 schedule_type: str = "linear"):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps
        
        # Initialize UNet
        self.unet = ImprovedUNet(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult
        )
        
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=schedule_type
        )

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
            beta_t = self.noise_scheduler.betas[t]
            
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
    def load_checkpoint(cls, filepath: str, device: torch.device = None) -> 'SeismicDiffusionModel':
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
    """Factory function to create a seismic diffusion model with sensible defaults."""
    return SeismicDiffusionModel(
        image_size=image_size,
        in_channels=1,  # Single channel for velocity models
        model_channels=128,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(16, 32),
        channel_mult=(1, 2, 4, 8),
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="linear",
        **kwargs
    )


if __name__ == "__main__":
    # Quick test of the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_seismic_diffusion_model(image_size=64)
    model.to(device)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 64, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    
    with torch.no_grad():
        output = model(x, timesteps)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    # Test training step
    loss = model.training_step(x)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling
    with torch.no_grad():
        samples = model.sample(batch_size=2, device=device, num_inference_steps=50)
        print(f"Samples shape: {samples.shape}")