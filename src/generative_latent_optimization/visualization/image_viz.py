#!/usr/bin/env python3
"""
Image Visualization Module

Provides visualization utilities for comparing original images,
reconstructions, and optimization results.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


class ImageVisualizer:
    """Image visualization utilities for VAE optimization results"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 5)):
        self.figsize = figsize
        
    def create_comparison_grid(self, original: torch.Tensor,
                              initial_recon: torch.Tensor,
                              optimized_recon: torch.Tensor,
                              save_path: Union[str, Path],
                              title: Optional[str] = None,
                              metrics: Optional[dict] = None) -> Path:
        """
        Create side-by-side comparison grid of three images
        
        Args:
            original: Original image tensor [1, C, H, W] or [C, H, W]
            initial_recon: Initial reconstruction tensor
            optimized_recon: Optimized reconstruction tensor  
            save_path: Path to save the grid image
            title: Optional title for the grid
            metrics: Optional metrics to display
            
        Returns:
            Path to saved image
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        # Convert tensors to numpy arrays
        orig_np = self._tensor_to_numpy(original)
        init_np = self._tensor_to_numpy(initial_recon)
        opt_np = self._tensor_to_numpy(optimized_recon)
        
        # Display images
        axes[0].imshow(orig_np)
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(init_np)
        axes[1].set_title('Initial Reconstruction', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(opt_np)
        axes[2].set_title('Optimized Reconstruction', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Add metrics information if provided
        if metrics:
            info_text = []
            if 'initial_psnr' in metrics and 'final_psnr' in metrics:
                info_text.append(f"PSNR: {metrics['initial_psnr']:.2f} → {metrics['final_psnr']:.2f} dB")
            if 'psnr_improvement' in metrics:
                info_text.append(f"Improvement: +{metrics['psnr_improvement']:.2f} dB")
            if 'loss_reduction' in metrics:
                info_text.append(f"Loss reduction: {metrics['loss_reduction']:.1f}%")
            
            if info_text:
                fig.text(0.5, 0.02, " | ".join(info_text), 
                        ha='center', va='bottom', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        
        # Add title if provided
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Adjust layout to make room for metrics text
        if metrics:
            plt.subplots_adjust(bottom=0.15)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def create_detailed_comparison(self, original: torch.Tensor,
                                 initial_recon: torch.Tensor, 
                                 optimized_recon: torch.Tensor,
                                 save_path: Union[str, Path],
                                 metrics: Optional[dict] = None,
                                 losses: Optional[List[float]] = None) -> Path:
        """
        Create detailed comparison with metrics visualization
        
        Args:
            original: Original image tensor
            initial_recon: Initial reconstruction
            optimized_recon: Optimized reconstruction
            save_path: Save path
            metrics: Metrics dictionary
            losses: Loss history
            
        Returns:
            Path to saved image
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create subplots: 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Convert tensors
        orig_np = self._tensor_to_numpy(original)
        init_np = self._tensor_to_numpy(initial_recon)
        opt_np = self._tensor_to_numpy(optimized_recon)
        
        # Top row: Images
        axes[0, 0].imshow(orig_np)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(opt_np)
        axes[0, 1].set_title('Optimized Reconstruction', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Bottom left: Loss curve
        if losses:
            axes[1, 0].plot(losses, 'b-', linewidth=2, alpha=0.8)
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Optimization Progress')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        else:
            axes[1, 0].axis('off')
        
        # Bottom right: Metrics text
        axes[1, 1].axis('off')
        if metrics:
            metrics_text = []
            metrics_text.append("Optimization Results:")
            metrics_text.append("")
            
            if 'initial_psnr' in metrics and 'final_psnr' in metrics:
                metrics_text.append(f"Initial PSNR: {metrics['initial_psnr']:.2f} dB")
                metrics_text.append(f"Final PSNR: {metrics['final_psnr']:.2f} dB")
                metrics_text.append(f"PSNR Improvement: +{metrics['psnr_improvement']:.2f} dB")
                metrics_text.append("")
            
            if 'initial_ssim' in metrics and 'final_ssim' in metrics:
                metrics_text.append(f"Initial SSIM: {metrics['initial_ssim']:.3f}")
                metrics_text.append(f"Final SSIM: {metrics['final_ssim']:.3f}")
                metrics_text.append(f"SSIM Improvement: +{metrics['ssim_improvement']:.3f}")
                metrics_text.append("")
            
            if 'loss_reduction' in metrics:
                metrics_text.append(f"Loss Reduction: {metrics['loss_reduction']:.1f}%")
            
            if 'optimization_iterations' in metrics:
                metrics_text.append(f"Iterations: {metrics['optimization_iterations']}")
            
            axes[1, 1].text(0.05, 0.95, "\n".join(metrics_text), 
                           transform=axes[1, 1].transAxes, 
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def create_difference_map(self, image1: torch.Tensor, image2: torch.Tensor,
                            save_path: Union[str, Path], 
                            title: str = "Difference Map") -> Path:
        """
        Create difference map between two images
        
        Args:
            image1: First image tensor
            image2: Second image tensor
            save_path: Save path
            title: Title for the plot
            
        Returns:
            Path to saved image
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy
        img1_np = self._tensor_to_numpy(image1)
        img2_np = self._tensor_to_numpy(image2)
        
        # Calculate difference
        if img1_np.ndim == 3:  # Color image
            # Convert to grayscale for difference calculation
            img1_gray = np.mean(img1_np, axis=2)
            img2_gray = np.mean(img2_np, axis=2)
        else:
            img1_gray = img1_np
            img2_gray = img2_np
        
        diff = np.abs(img1_gray - img2_gray)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img1_np)
        axes[0].set_title('Image 1')
        axes[0].axis('off')
        
        axes[1].imshow(img2_np)
        axes[1].set_title('Image 2')
        axes[1].axis('off')
        
        im = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title(title)
        axes[2].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[2], shrink=0.6)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array for visualization"""
        # Handle batch dimension
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert from [C, H, W] to [H, W, C]
        if tensor.dim() == 3 and tensor.shape[0] <= 4:  # Assume channels first
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy and ensure correct range
        image_np = tensor.detach().cpu().numpy()
        image_np = np.clip(image_np, 0, 1)
        
        # Handle grayscale
        if image_np.shape[2] == 1:
            image_np = image_np.squeeze(2)
        
        return image_np
    
    def create_batch_overview(self, results: List[dict], save_path: Union[str, Path],
                            max_images: int = 16) -> Path:
        """
        Create overview grid of multiple optimization results
        
        Args:
            results: List of result dictionaries with images and metrics
            save_path: Save path
            max_images: Maximum images to show
            
        Returns:
            Path to saved image
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        n_images = min(len(results), max_images)
        cols = 4
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_images):
            row = i // cols
            col = i % cols
            
            result = results[i]
            
            # Show optimized reconstruction
            if 'optimized_reconstruction' in result:
                img_np = self._tensor_to_numpy(result['optimized_reconstruction'])
                axes[row, col].imshow(img_np)
                
                # Add metrics text
                if 'metrics' in result:
                    metrics = result['metrics']
                    text = f"{result.get('image_name', f'Image {i+1}')}\n"
                    if 'psnr_improvement' in metrics:
                        text += f"PSNR: +{metrics['psnr_improvement']:.2f}dB"
                    
                    axes[row, col].text(0.02, 0.98, text, transform=axes[row, col].transAxes,
                                      verticalalignment='top', fontsize=8,
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Optimization Results Overview ({n_images} images)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path