"""
Interactive Demo Script for Seismic Diffusion Process

This script provides an interactive demonstration of the complete diffusion process,
including step-by-step visualization of noise addition and denoising with parameter
adjustment capabilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from pathlib import Path
import argparse
import logging
from typing import Optional, List, Tuple
import time

# Import our modules
from simple_diffusion_seismic import SeismicDiffusionModel, create_seismic_diffusion_model
from visualization_utils import SeismicVisualizationUtils
from train_with_visualization import SeismicDataGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DiffusionDemo:
    """Interactive demonstration of seismic diffusion process."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 image_size: int = 64,
                 device: str = 'auto'):
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load or create model
        if model_path and Path(model_path).exists():
            logger.info(f"Loading model from {model_path}")
            self.model, checkpoint = SeismicDiffusionModel.load_checkpoint(model_path, self.device)
            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            logger.info("Creating new model (untrained)")
            self.model = create_seismic_diffusion_model(image_size=image_size)
            self.model.to(self.device)
        
        self.model.eval()
        
        # Initialize utilities
        self.visualizer = SeismicVisualizationUtils()
        self.data_generator = SeismicDataGenerator(image_size)
        
        # Create output directory
        self.output_dir = Path('./demo_outputs')
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Demo initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def generate_sample_data(self, num_samples: int = 4) -> torch.Tensor:
        """Generate sample seismic data for demonstration."""
        logger.info(f"Generating {num_samples} sample seismic velocity models...")
        return self.data_generator.create_dataset(num_samples, self.device)
    
    def demonstrate_forward_process(self, 
                                   clean_data: torch.Tensor,
                                   timesteps: List[int] = None,
                                   save_path: Optional[str] = None) -> None:
        """Demonstrate the forward diffusion process (clean → noisy)."""
        if timesteps is None:
            timesteps = [0, 100, 200, 400, 600, 800, 1000]
        
        logger.info(f"Demonstrating forward process with timesteps: {timesteps}")
        
        fig = self.visualizer.visualize_forward_process(
            self.model, 
            clean_data[:2], 
            timesteps=timesteps,
            save_path=save_path
        )
        
        if save_path is None:
            save_path = self.output_dir / "demo_forward_process.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Forward process demonstration saved to {save_path}")
    
    def demonstrate_reverse_process(self,
                                   num_samples: int = 2,
                                   num_inference_steps: int = 50,
                                   timesteps_to_show: List[int] = None,
                                   save_path: Optional[str] = None) -> torch.Tensor:
        """Demonstrate the reverse diffusion process (noisy → clean)."""
        if timesteps_to_show is None:
            # Show every 10th step plus the final result
            timesteps_to_show = list(range(0, num_inference_steps, max(1, num_inference_steps // 6)))
            if num_inference_steps - 1 not in timesteps_to_show:
                timesteps_to_show.append(num_inference_steps - 1)
        
        logger.info(f"Demonstrating reverse process with {num_inference_steps} inference steps")
        logger.info(f"Showing timesteps: {timesteps_to_show}")
        
        fig = self.visualizer.visualize_reverse_process(
            self.model,
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            timesteps_to_show=timesteps_to_show,
            save_path=save_path
        )
        
        if save_path is None:
            save_path = self.output_dir / "demo_reverse_process.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Also generate final samples for return
        with torch.no_grad():
            samples = self.model.sample(
                batch_size=num_samples,
                device=self.device,
                num_inference_steps=num_inference_steps
            )
        
        logger.info(f"Reverse process demonstration saved to {save_path}")
        return samples
    
    def demonstrate_noise_levels(self,
                                clean_data: torch.Tensor,
                                timesteps: List[int] = None,
                                save_path: Optional[str] = None) -> None:
        """Demonstrate different noise levels applied to the same clean data."""
        if timesteps is None:
            timesteps = [0, 100, 250, 500, 750, 900, 1000]
        
        logger.info(f"Demonstrating noise levels: {timesteps}")
        
        fig = self.visualizer.plot_noise_levels(
            self.model,
            clean_data[:1],
            timesteps=timesteps,
            save_path=save_path
        )
        
        if save_path is None:
            save_path = self.output_dir / "demo_noise_levels.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Noise levels demonstration saved to {save_path}")
    
    def compare_with_ground_truth(self,
                                 ground_truth: torch.Tensor,
                                 num_generated: int = 4,
                                 num_inference_steps: int = 100,
                                 save_path: Optional[str] = None) -> None:
        """Compare generated samples with ground truth data."""
        logger.info(f"Generating {num_generated} samples for comparison...")
        
        with torch.no_grad():
            generated_samples = self.model.sample(
                batch_size=num_generated,
                device=self.device,
                num_inference_steps=num_inference_steps
            )
        
        fig = self.visualizer.compare_samples(
            ground_truth[:num_generated],
            generated_samples,
            num_comparisons=num_generated,
            save_path=save_path
        )
        
        if save_path is None:
            save_path = self.output_dir / "demo_comparison.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Comparison demonstration saved to {save_path}")
    
    def interactive_parameter_demo(self,
                                  clean_data: torch.Tensor,
                                  inference_steps_list: List[int] = None,
                                  save_dir: Optional[Path] = None) -> None:
        """Demonstrate how different parameters affect generation quality."""
        if inference_steps_list is None:
            inference_steps_list = [10, 25, 50, 100]
        
        if save_dir is None:
            save_dir = self.output_dir / "parameter_comparison"
        save_dir.mkdir(exist_ok=True)
        
        logger.info(f"Demonstrating parameter effects with inference steps: {inference_steps_list}")
        
        # Generate samples with different inference steps
        for num_steps in inference_steps_list:
            logger.info(f"Generating samples with {num_steps} inference steps...")
            
            with torch.no_grad():
                samples = self.model.sample(
                    batch_size=4,
                    device=self.device,
                    num_inference_steps=num_steps
                )
            
            # Compare with ground truth
            fig = self.visualizer.compare_samples(
                clean_data[:4],
                samples,
                num_comparisons=4
            )
            
            save_path = save_dir / f"inference_steps_{num_steps}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved comparison for {num_steps} steps to {save_path}")
        
        logger.info(f"Parameter comparison demonstrations saved to {save_dir}")
    
    def create_animation_frames(self,
                               num_samples: int = 1,
                               num_inference_steps: int = 100,
                               save_dir: Optional[Path] = None) -> List[Path]:
        """Create frames for an animation of the reverse diffusion process."""
        if save_dir is None:
            save_dir = self.output_dir / "animation_frames"
        save_dir.mkdir(exist_ok=True)
        
        logger.info(f"Creating animation frames for {num_inference_steps} steps...")
        
        # Start from pure noise
        shape = (num_samples, self.model.in_channels, self.model.image_size, self.model.image_size)
        x = torch.randn(shape, device=self.device)
        
        frame_paths = []
        
        with torch.no_grad():
            timesteps = torch.linspace(self.model.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
            
            for step_idx, t in enumerate(timesteps):
                t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = self.model(x, t_tensor)
                
                # Compute previous sample using DDPM sampling
                alpha_t = self.model.noise_scheduler.alphas[t]
                alpha_cumprod_t = self.model.noise_scheduler.alphas_cumprod[t]
                
                # Mean of the reverse distribution
                pred_original_sample = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
                
                # Compute previous sample mean
                if t > 0:
                    alpha_cumprod_prev = self.model.noise_scheduler.alphas_cumprod[t - 1]
                    pred_sample_direction = torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise
                    pred_prev_sample = torch.sqrt(alpha_cumprod_prev) * pred_original_sample + pred_sample_direction
                    
                    # Add noise
                    variance = self.model.noise_scheduler.posterior_variance[t]
                    if variance > 0:
                        noise = torch.randn_like(x)
                        pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise
                else:
                    pred_prev_sample = pred_original_sample
                
                x = pred_prev_sample
                
                # Save frame every 5 steps
                if step_idx % 5 == 0 or step_idx == len(timesteps) - 1:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    
                    self.visualizer.plot_seismic_model(
                        x[0],
                        title=f'Denoising Step {step_idx}/{len(timesteps)-1} (t={int(t)})',
                        ax=ax
                    )
                    
                    frame_path = save_dir / f"frame_{step_idx:04d}.png"
                    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                    frame_paths.append(frame_path)
        
        logger.info(f"Created {len(frame_paths)} animation frames in {save_dir}")
        return frame_paths
    
    def run_complete_demo(self, num_samples: int = 4) -> None:
        """Run the complete demonstration suite."""
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE SEISMIC DIFFUSION DEMONSTRATION")
        logger.info("=" * 60)
        
        # Generate sample data
        sample_data = self.generate_sample_data(num_samples)
        
        # 1. Demonstrate forward process
        logger.info("\n1. Demonstrating Forward Diffusion Process...")
        self.demonstrate_forward_process(sample_data)
        
        # 2. Demonstrate reverse process
        logger.info("\n2. Demonstrating Reverse Diffusion Process...")
        generated_samples = self.demonstrate_reverse_process(num_samples=2)
        
        # 3. Demonstrate noise levels
        logger.info("\n3. Demonstrating Different Noise Levels...")
        self.demonstrate_noise_levels(sample_data)
        
        # 4. Compare with ground truth
        logger.info("\n4. Comparing Generated vs Ground Truth...")
        self.compare_with_ground_truth(sample_data, num_generated=4)
        
        # 5. Parameter effects demonstration
        logger.info("\n5. Demonstrating Parameter Effects...")
        self.interactive_parameter_demo(sample_data)
        
        # 6. Create animation frames
        logger.info("\n6. Creating Animation Frames...")
        frame_paths = self.create_animation_frames(num_samples=1, num_inference_steps=50)
        
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION COMPLETE!")
        logger.info(f"All outputs saved to: {self.output_dir}")
        logger.info("=" * 60)
        
        # Summary
        logger.info("\nGenerated visualizations:")
        for output_file in self.output_dir.glob("*.png"):
            logger.info(f"  - {output_file.name}")
        
        for subdir in self.output_dir.iterdir():
            if subdir.is_dir():
                file_count = len(list(subdir.glob("*.png")))
                logger.info(f"  - {subdir.name}/ ({file_count} files)")


def main():
    """Main function for the demo script."""
    parser = argparse.ArgumentParser(description="Seismic Diffusion Model Demo")
    parser.add_argument("--model_path", type=str, default=None, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--image_size", type=int, default=64, 
                       help="Image size for the model")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--num_samples", type=int, default=4, 
                       help="Number of samples to generate for demonstration")
    parser.add_argument("--demo_type", type=str, default="complete",
                       choices=["complete", "forward", "reverse", "noise", "comparison", "parameters"],
                       help="Type of demonstration to run")
    parser.add_argument("--output_dir", type=str, default="./demo_outputs",
                       help="Output directory for demonstrations")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = DiffusionDemo(
        model_path=args.model_path,
        image_size=args.image_size,
        device=args.device
    )
    
    # Set custom output directory if specified
    if args.output_dir != "./demo_outputs":
        demo.output_dir = Path(args.output_dir)
        demo.output_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    sample_data = demo.generate_sample_data(args.num_samples)
    
    # Run specified demonstration
    if args.demo_type == "complete":
        demo.run_complete_demo(args.num_samples)
    elif args.demo_type == "forward":
        demo.demonstrate_forward_process(sample_data)
    elif args.demo_type == "reverse":
        demo.demonstrate_reverse_process(num_samples=args.num_samples)
    elif args.demo_type == "noise":
        demo.demonstrate_noise_levels(sample_data)
    elif args.demo_type == "comparison":
        demo.compare_with_ground_truth(sample_data, num_generated=args.num_samples)
    elif args.demo_type == "parameters":
        demo.interactive_parameter_demo(sample_data)
    
    logger.info(f"Demo completed successfully! Check {demo.output_dir} for outputs.")


if __name__ == "__main__":
    main()