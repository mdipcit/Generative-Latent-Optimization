#!/usr/bin/env python3
"""
VAE Latent Optimization Module

Based on invertLDM.py methodology for optimizing latent representations
to minimize reconstruction loss.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class OptimizationConfig:
    """Configuration for latent optimization"""
    iterations: int = 150
    learning_rate: float = 0.4
    loss_function: str = 'mse'  # 'mse', 'l1', 'lpips'
    convergence_threshold: float = 1e-6
    checkpoint_interval: int = 20
    device: str = "cuda"


@dataclass
class OptimizationResult:
    """Result of latent optimization"""
    optimized_latents: torch.Tensor
    losses: List[float]
    metrics: Dict[str, float]
    convergence_iteration: Optional[int] = None
    initial_loss: float = 0.0
    final_loss: float = 0.0


class LatentOptimizer:
    """
    VAE Latent Representation Optimizer
    
    Optimizes latent representations to minimize reconstruction loss
    between target image and VAE-decoded image from latents.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def optimize(self, vae, initial_latents: torch.Tensor, 
                target_image: torch.Tensor) -> OptimizationResult:
        """
        Optimize latent representation to minimize reconstruction loss
        
        Args:
            vae: VAE model with encode/decode methods
            initial_latents: Initial latent representation [1, C, H, W]
            target_image: Target image tensor [1, C, H, W] in [0, 1] range
            
        Returns:
            OptimizationResult with optimized latents and metrics
        """
        # Freeze VAE decoder parameters
        for param in vae.decoder.parameters():
            param.requires_grad = False
        
        # Enable gradients for latents
        optimized_latents = initial_latents.clone().detach().requires_grad_(True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([optimized_latents], lr=self.config.learning_rate)
        
        # Optimization loop
        losses = []
        convergence_iteration = None
        
        with tqdm(range(self.config.iterations), desc="Optimizing latents") as pbar:
            for i in pbar:
                optimizer.zero_grad()
                
                # Decode latents to image
                outputs = vae.decode(optimized_latents)
                reconstructed = (outputs.sample / 2 + 0.5).clamp(0, 1)
                
                # Calculate loss
                if self.config.loss_function == 'mse':
                    loss = torch.nn.functional.mse_loss(target_image, reconstructed)
                elif self.config.loss_function == 'l1':
                    loss = torch.nn.functional.l1_loss(target_image, reconstructed)
                else:
                    raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                losses.append(current_loss)
                
                # Update progress bar
                if i % self.config.checkpoint_interval == 0:
                    psnr = self._calculate_psnr(target_image, reconstructed)
                    pbar.set_postfix({
                        'Loss': f'{current_loss:.6f}',
                        'PSNR': f'{psnr:.2f}dB'
                    })
                
                # Check convergence
                if (i > 0 and 
                    abs(losses[-1] - losses[-2]) < self.config.convergence_threshold):
                    convergence_iteration = i
                    break
        
        # Calculate final metrics
        with torch.no_grad():
            final_outputs = vae.decode(optimized_latents)
            final_reconstructed = (final_outputs.sample / 2 + 0.5).clamp(0, 1)
            final_psnr = self._calculate_psnr(target_image, final_reconstructed)
            final_ssim = self._calculate_ssim_basic(target_image, final_reconstructed)
        
        # Prepare result
        # Calculate loss reduction with zero-division protection
        if losses and losses[0] > 0:
            loss_reduction = ((losses[0] - losses[-1]) / losses[0]) * 100
        else:
            loss_reduction = 0.0  # No reduction if initial loss is zero
            
        metrics = {
            'final_psnr_db': final_psnr,
            'final_ssim': final_ssim,
            'loss_reduction_percent': loss_reduction,
            'total_iterations': len(losses)
        }
        
        return OptimizationResult(
            optimized_latents=optimized_latents.detach(),
            losses=losses,
            metrics=metrics,
            convergence_iteration=convergence_iteration,
            initial_loss=losses[0] if losses else 0.0,
            final_loss=losses[-1] if losses else 0.0
        )
    
    def optimize_batch(self, vae, latents_batch: torch.Tensor, 
                      targets_batch: torch.Tensor) -> List[OptimizationResult]:
        """
        Optimize a batch of latent representations
        
        Args:
            vae: VAE model
            latents_batch: Batch of initial latents [B, C, H, W]
            targets_batch: Batch of target images [B, C, H, W]
            
        Returns:
            List of OptimizationResult for each sample in batch
        """
        results = []
        batch_size = latents_batch.shape[0]
        
        for i in tqdm(range(batch_size), desc="Batch optimization"):
            result = self.optimize(
                vae, 
                latents_batch[i:i+1], 
                targets_batch[i:i+1]
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def _calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate PSNR between two images"""
        mse = torch.nn.functional.mse_loss(img1, img2)
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()
    
    @staticmethod
    def _calculate_ssim_basic(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Basic SSIM calculation (simplified version)"""
        # This is a simplified SSIM calculation
        # For full SSIM, consider using pytorch-ssim or torchmetrics
        
        # Convert to grayscale for simpler calculation
        if img1.shape[1] == 3:  # RGB
            img1_gray = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
            img2_gray = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        else:
            img1_gray = img1.squeeze(1)
            img2_gray = img2.squeeze(1)
        
        # Calculate means
        mu1 = img1_gray.mean()
        mu2 = img2_gray.mean()
        
        # Calculate variances and covariance
        var1 = ((img1_gray - mu1) ** 2).mean()
        var2 = ((img2_gray - mu2) ** 2).mean()
        cov = ((img1_gray - mu1) * (img2_gray - mu2)).mean()
        
        # SSIM constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # SSIM calculation
        numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
        
        ssim = numerator / denominator
        return ssim.item()