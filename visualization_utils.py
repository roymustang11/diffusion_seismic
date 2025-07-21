"""
Visualization Utilities for Seismic Diffusion Model

This module provides comprehensive visualization capabilities for understanding
the diffusion process, including forward/reverse process visualization, training
monitoring, and comparative analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple, Union, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set up matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SeismicVisualizationUtils:
    """Comprehensive visualization utilities for seismic diffusion models."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.colormap = 'RdYlBu_r'  # Good for seismic velocity models
        
    def normalize_for_display(self, data: torch.Tensor) -> np.ndarray:
        """Normalize tensor data for display."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Handle different tensor shapes
        if data.ndim == 4:  # (batch, channels, height, width)
            data = data[0, 0]  # Take first sample, first channel
        elif data.ndim == 3:  # (channels, height, width)
            data = data[0]  # Take first channel
        
        # Normalize to [0, 1] for display
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        
        return data
    
    def plot_seismic_model(self, 
                          velocity_model: Union[torch.Tensor, np.ndarray],
                          title: str = "Seismic Velocity Model",
                          ax: Optional[plt.Axes] = None,
                          colorbar: bool = True,
                          vmin: Optional[float] = None,
                          vmax: Optional[float] = None) -> plt.Axes:
        """Plot a single seismic velocity model."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        data = self.normalize_for_display(velocity_model)
        
        im = ax.imshow(data, cmap=self.colormap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (samples)', fontsize=12)
        ax.set_ylabel('Depth (samples)', fontsize=12)
        
        if colorbar:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Normalized Velocity', fontsize=12)
        
        return ax
    
    def visualize_forward_process(self, 
                                 model,
                                 clean_data: torch.Tensor,
                                 timesteps: List[int] = [0, 200, 400, 600, 800, 1000],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Visualize the forward diffusion process (clean → noisy)."""
        device = clean_data.device
        batch_size = clean_data.shape[0]
        
        # Create figure
        n_steps = len(timesteps)
        fig, axes = plt.subplots(2, n_steps, figsize=(4*n_steps, 8), dpi=self.dpi)
        if n_steps == 1:
            axes = axes.reshape(2, 1)
        
        # Generate noisy versions
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                if t == 0:
                    noisy_data = clean_data
                else:
                    # Sample noise and add to clean data
                    noise = torch.randn_like(clean_data)
                    t_tensor = torch.full((batch_size,), t-1, device=device, dtype=torch.long)
                    noisy_data = model.noise_scheduler.add_noise(clean_data, noise, t_tensor)
                
                # Plot two different samples
                for row in range(2):
                    sample_idx = min(row, batch_size - 1)
                    ax = axes[row, i]
                    
                    self.plot_seismic_model(
                        noisy_data[sample_idx],
                        title=f't = {t}',
                        ax=ax,
                        colorbar=(i == n_steps - 1)  # Only show colorbar on last column
                    )
        
        plt.suptitle('Forward Diffusion Process (Clean → Noisy)', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Forward process visualization saved to {save_path}")
        
        return fig
    
    def visualize_reverse_process(self,
                                 model,
                                 num_samples: int = 2,
                                 num_inference_steps: int = 50,
                                 timesteps_to_show: List[int] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Visualize the reverse diffusion process (noisy → clean)."""
        device = next(model.parameters()).device
        
        if timesteps_to_show is None:
            # Show every 10th step
            timesteps_to_show = list(range(0, num_inference_steps, max(1, num_inference_steps // 6)))
            if num_inference_steps - 1 not in timesteps_to_show:
                timesteps_to_show.append(num_inference_steps - 1)
        
        n_steps = len(timesteps_to_show)
        fig, axes = plt.subplots(num_samples, n_steps, figsize=(4*n_steps, 4*num_samples), dpi=self.dpi)
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        if n_steps == 1:
            axes = axes.reshape(-1, 1)
        
        # Generate samples with intermediate steps
        shape = (num_samples, model.in_channels, model.image_size, model.image_size)
        x = torch.randn(shape, device=device)
        
        # Store intermediate results
        intermediate_results = {0: x.clone()}
        
        with torch.no_grad():
            timesteps = torch.linspace(model.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
            
            for step_idx, t in enumerate(timesteps):
                t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = model(x, t_tensor)
                
                # Compute previous sample using DDPM sampling
                alpha_t = model.noise_scheduler.alphas[t]
                alpha_cumprod_t = model.noise_scheduler.alphas_cumprod[t]
                
                # Mean of the reverse distribution
                pred_original_sample = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
                
                # Compute previous sample mean
                if t > 0:
                    alpha_cumprod_prev = model.noise_scheduler.alphas_cumprod[t - 1]
                    pred_sample_direction = torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise
                    pred_prev_sample = torch.sqrt(alpha_cumprod_prev) * pred_original_sample + pred_sample_direction
                    
                    # Add noise
                    variance = model.noise_scheduler.posterior_variance[t]
                    if variance > 0:
                        noise = torch.randn_like(x)
                        pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise
                else:
                    pred_prev_sample = pred_original_sample
                
                x = pred_prev_sample
                
                # Store if this step should be shown
                if step_idx in timesteps_to_show:
                    intermediate_results[step_idx] = x.clone()
        
        # Plot intermediate results
        for sample_idx in range(num_samples):
            for plot_idx, step_idx in enumerate(timesteps_to_show):
                ax = axes[sample_idx, plot_idx]
                
                step_t = int(timesteps[step_idx]) if step_idx < len(timesteps) else 0
                self.plot_seismic_model(
                    intermediate_results[step_idx][sample_idx],
                    title=f'Step {step_idx} (t={step_t})',
                    ax=ax,
                    colorbar=(plot_idx == n_steps - 1)  # Only show colorbar on last column
                )
        
        plt.suptitle('Reverse Diffusion Process (Noisy → Clean)', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Reverse process visualization saved to {save_path}")
        
        return fig
    
    def plot_training_progress(self,
                              losses: Dict[str, List[float]],
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot training progress with loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=self.dpi)
        
        # Plot loss curves
        ax = axes[0]
        for loss_name, loss_values in losses.items():
            epochs = range(1, len(loss_values) + 1)
            ax.plot(epochs, loss_values, label=loss_name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot loss in log scale
        ax = axes[1]
        for loss_name, loss_values in losses.items():
            epochs = range(1, len(loss_values) + 1)
            ax.semilogy(epochs, loss_values, label=loss_name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.set_title('Training Progress (Log Scale)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Training progress plot saved to {save_path}")
        
        return fig
    
    def compare_samples(self,
                       ground_truth: torch.Tensor,
                       generated: torch.Tensor,
                       num_comparisons: int = 4,
                       save_path: Optional[str] = None) -> plt.Figure:
        """Compare ground truth and generated samples side by side."""
        fig, axes = plt.subplots(2, num_comparisons, figsize=(4*num_comparisons, 8), dpi=self.dpi)
        if num_comparisons == 1:
            axes = axes.reshape(2, 1)
        
        batch_size = min(ground_truth.shape[0], generated.shape[0])
        
        for i in range(min(num_comparisons, batch_size)):
            # Ground truth
            self.plot_seismic_model(
                ground_truth[i],
                title=f'Ground Truth {i+1}',
                ax=axes[0, i],
                colorbar=(i == num_comparisons - 1)
            )
            
            # Generated
            self.plot_seismic_model(
                generated[i],
                title=f'Generated {i+1}',
                ax=axes[1, i],
                colorbar=(i == num_comparisons - 1)
            )
        
        plt.suptitle('Ground Truth vs Generated Samples', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Sample comparison saved to {save_path}")
        
        return fig
    
    def create_interactive_diffusion_plot(self,
                                        model,
                                        sample_data: torch.Tensor,
                                        num_inference_steps: int = 100) -> go.Figure:
        """Create an interactive plot showing the complete diffusion process."""
        device = sample_data.device
        
        # Generate forward process data
        forward_data = []
        timesteps_forward = torch.linspace(0, model.num_timesteps-1, 20).long()
        
        with torch.no_grad():
            for t in timesteps_forward:
                if t == 0:
                    noisy = sample_data
                else:
                    noise = torch.randn_like(sample_data)
                    t_tensor = torch.full((sample_data.shape[0],), t, device=device, dtype=torch.long)
                    noisy = model.noise_scheduler.add_noise(sample_data, noise, t_tensor)
                
                data_np = self.normalize_for_display(noisy)
                forward_data.append(data_np)
        
        # Generate reverse process data
        reverse_data = []
        shape = (1, model.in_channels, model.image_size, model.image_size)
        x = torch.randn(shape, device=device)
        
        timesteps_reverse = torch.linspace(model.num_timesteps - 1, 0, 20).long()
        
        with torch.no_grad():
            for t in timesteps_reverse:
                data_np = self.normalize_for_display(x)
                reverse_data.append(data_np)
                
                if t > 0:
                    t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                    predicted_noise = model(x, t_tensor)
                    
                    # Simplified sampling step
                    alpha_t = model.noise_scheduler.alphas[t]
                    alpha_cumprod_t = model.noise_scheduler.alphas_cumprod[t]
                    pred_original_sample = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                    
                    if t > 0:
                        alpha_cumprod_prev = model.noise_scheduler.alphas_cumprod[t - 1]
                        pred_sample_direction = torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise
                        x = torch.sqrt(alpha_cumprod_prev) * pred_original_sample + pred_sample_direction
        
        # Create interactive plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Forward Process (Clean → Noisy)', 'Reverse Process (Noisy → Clean)'),
            horizontal_spacing=0.1
        )
        
        # Add forward process heatmap
        fig.add_trace(
            go.Heatmap(
                z=forward_data[0],
                colorscale='RdYlBu_r',
                showscale=False,
                name='Forward'
            ),
            row=1, col=1
        )
        
        # Add reverse process heatmap
        fig.add_trace(
            go.Heatmap(
                z=reverse_data[0],
                colorscale='RdYlBu_r',
                showscale=True,
                name='Reverse'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Seismic Diffusion Process",
            title_x=0.5,
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_noise_levels(self,
                         model,
                         clean_data: torch.Tensor,
                         timesteps: List[int] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot different noise levels applied to the same clean data."""
        if timesteps is None:
            timesteps = [0, 100, 300, 500, 700, 900, 1000]
        
        device = clean_data.device
        
        # Create figure
        n_levels = len(timesteps)
        fig, axes = plt.subplots(1, n_levels, figsize=(4*n_levels, 4), dpi=self.dpi)
        if n_levels == 1:
            axes = [axes]
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                if t == 0:
                    noisy_data = clean_data
                else:
                    noise = torch.randn_like(clean_data)
                    t_tensor = torch.full((clean_data.shape[0],), t-1, device=device, dtype=torch.long)
                    noisy_data = model.noise_scheduler.add_noise(clean_data, noise, t_tensor)
                
                # Calculate noise level
                alpha_cumprod = model.noise_scheduler.alphas_cumprod[t-1] if t > 0 else 1.0
                noise_level = 1.0 - alpha_cumprod
                
                self.plot_seismic_model(
                    noisy_data[0],
                    title=f't = {t}\nNoise Level: {noise_level:.3f}',
                    ax=axes[i],
                    colorbar=(i == n_levels - 1)
                )
        
        plt.suptitle('Different Noise Levels Applied to Same Clean Data', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Noise levels plot saved to {save_path}")
        
        return fig
    
    def create_training_animation(self,
                                 model_checkpoints: List[str],
                                 test_data: torch.Tensor,
                                 save_path: Optional[str] = None) -> FuncAnimation:
        """Create an animation showing how generated samples improve during training."""
        device = test_data.device
        
        # Load models and generate samples
        generated_samples = []
        epochs = []
        
        for checkpoint_path in model_checkpoints:
            try:
                model, checkpoint = model.__class__.load_checkpoint(checkpoint_path, device)
                model.eval()
                
                with torch.no_grad():
                    samples = model.sample(batch_size=1, device=device, num_inference_steps=50)
                    generated_samples.append(self.normalize_for_display(samples))
                    epochs.append(checkpoint.get('epoch', 0))
                    
            except Exception as e:
                logger.warning(f"Could not load checkpoint {checkpoint_path}: {e}")
        
        if not generated_samples:
            logger.error("No valid checkpoints found for animation")
            return None
        
        # Create animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Ground truth (static)
        gt_data = self.normalize_for_display(test_data)
        im1 = ax1.imshow(gt_data, cmap=self.colormap, animated=True)
        ax1.set_title('Ground Truth')
        ax1.set_xlabel('X (samples)')
        ax1.set_ylabel('Depth (samples)')
        
        # Generated (animated)
        im2 = ax2.imshow(generated_samples[0], cmap=self.colormap, animated=True)
        ax2.set_title(f'Generated (Epoch {epochs[0]})')
        ax2.set_xlabel('X (samples)')
        ax2.set_ylabel('Depth (samples)')
        
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        def animate(frame):
            im2.set_array(generated_samples[frame])
            ax2.set_title(f'Generated (Epoch {epochs[frame]})')
            return [im2]
        
        anim = FuncAnimation(fig, animate, frames=len(generated_samples), 
                           interval=1000, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1)
            logger.info(f"Training animation saved to {save_path}")
        
        return anim


# Convenience functions for quick visualization
def quick_plot_seismic(data: Union[torch.Tensor, np.ndarray], 
                      title: str = "Seismic Data") -> plt.Figure:
    """Quick plot of seismic data."""
    vis = SeismicVisualizationUtils()
    fig, ax = plt.subplots(figsize=(10, 6))
    vis.plot_seismic_model(data, title=title, ax=ax)
    plt.tight_layout()
    return fig


def quick_compare(gt: Union[torch.Tensor, np.ndarray], 
                 gen: Union[torch.Tensor, np.ndarray]) -> plt.Figure:
    """Quick comparison of ground truth vs generated."""
    vis = SeismicVisualizationUtils()
    return vis.compare_samples(gt, gen, num_comparisons=1)


if __name__ == "__main__":
    # Test visualization utilities
    print("Testing visualization utilities...")
    
    # Create some dummy data
    dummy_data = torch.randn(4, 1, 64, 64)
    
    vis = SeismicVisualizationUtils()
    
    # Test basic plotting
    fig = quick_plot_seismic(dummy_data[0], "Test Seismic Data")
    plt.show()
    
    # Test comparison
    fig = quick_compare(dummy_data[:2], dummy_data[2:4])
    plt.show()
    
    print("Visualization utilities test completed!")