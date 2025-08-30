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
from contextlib import contextmanager
from ..metrics.image_metrics import ImageMetrics


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
        self.metrics = ImageMetrics(device=config.device)
    
    @contextmanager
    def _freeze_vae_decoder(self, vae):
        """Temporarily freeze VAE decoder parameters during optimization"""
        original_states = {}
        
        # Check if vae has proper decoder with named_parameters (not Mock)
        if hasattr(vae, 'decoder') and hasattr(vae.decoder, 'named_parameters'):
            try:
                # Store original requires_grad states
                for name, param in vae.decoder.named_parameters():
                    original_states[name] = param.requires_grad
                    param.requires_grad = False
            except (TypeError, AttributeError):
                # Handle Mock objects or other test doubles
                pass
        
        try:
            yield
        finally:
            # Restore original states
            if original_states and hasattr(vae, 'decoder') and hasattr(vae.decoder, 'named_parameters'):
                try:
                    for name, param in vae.decoder.named_parameters():
                        if name in original_states:
                            param.requires_grad = original_states[name]
                except (TypeError, AttributeError):
                    # Handle Mock objects or other test doubles
                    pass
    
    def _calculate_loss(self, target_image: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on configured loss function"""
        if self.config.loss_function == 'mse':
            return torch.nn.functional.mse_loss(target_image, reconstructed)
        elif self.config.loss_function == 'l1':
            return torch.nn.functional.l1_loss(target_image, reconstructed)
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
        
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
        with self._freeze_vae_decoder(vae):
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
                    loss = self._calculate_loss(target_image, reconstructed)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    current_loss = loss.item()
                    losses.append(current_loss)
                    
                    # Update progress bar
                    if i % self.config.checkpoint_interval == 0:
                        psnr = self.metrics.calculate_psnr(target_image, reconstructed)
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
            final_metrics = self.metrics.calculate_all_metrics(target_image, final_reconstructed)
            final_psnr = final_metrics.psnr_db
            final_ssim = final_metrics.ssim
        
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
        batch_size = latents_batch.shape[0]
        
        with self._freeze_vae_decoder(vae):
            # Enable gradients for all latents in batch
            optimized_latents = latents_batch.clone().detach().requires_grad_(True)
            
            # Setup optimizer for batch
            optimizer = torch.optim.Adam([optimized_latents], lr=self.config.learning_rate)
            
            # Track losses for each sample
            batch_losses = [[] for _ in range(batch_size)]
            convergence_iterations = [None] * batch_size
            
            with tqdm(range(self.config.iterations), desc="Batch optimization") as pbar:
                for i in pbar:
                    optimizer.zero_grad()
                    
                    # Decode all latents at once
                    outputs = vae.decode(optimized_latents)
                    reconstructed = (outputs.sample / 2 + 0.5).clamp(0, 1)
                    
                    # Calculate loss for entire batch
                    if self.config.loss_function == 'mse':
                        losses = torch.nn.functional.mse_loss(
                            targets_batch, reconstructed, reduction='none'
                        ).mean(dim=(1, 2, 3))
                    elif self.config.loss_function == 'l1':
                        losses = torch.nn.functional.l1_loss(
                            targets_batch, reconstructed, reduction='none'
                        ).mean(dim=(1, 2, 3))
                    else:
                        raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
                    
                    # Total loss for backpropagation
                    total_loss = losses.mean()
                    
                    # Backward pass
                    total_loss.backward()
                    optimizer.step()
                    
                    # Track individual losses
                    for j in range(batch_size):
                        loss_val = losses[j].item()
                        batch_losses[j].append(loss_val)
                    
                    # Update progress bar
                    if i % self.config.checkpoint_interval == 0:
                        avg_loss = total_loss.item()
                        pbar.set_postfix({'Avg Loss': f'{avg_loss:.6f}'})
                    
                    # Check convergence for each sample
                    if i > 0:
                        for j in range(batch_size):
                            if (convergence_iterations[j] is None and
                                abs(batch_losses[j][-1] - batch_losses[j][-2]) < self.config.convergence_threshold):
                                convergence_iterations[j] = i
            
            # Calculate final metrics for each sample
            results = []
            with torch.no_grad():
                final_outputs = vae.decode(optimized_latents)
                final_reconstructed = (final_outputs.sample / 2 + 0.5).clamp(0, 1)
                
                for j in range(batch_size):
                    single_target = targets_batch[j:j+1]
                    single_recon = final_reconstructed[j:j+1]
                    single_latent = optimized_latents[j:j+1]
                    
                    final_metrics = self.metrics.calculate_all_metrics(single_target, single_recon)
                    
                    # Calculate loss reduction
                    losses = batch_losses[j]
                    if losses and losses[0] > 0:
                        loss_reduction = ((losses[0] - losses[-1]) / losses[0]) * 100
                    else:
                        loss_reduction = 0.0
                    
                    metrics = {
                        'final_psnr_db': final_metrics.psnr_db,
                        'final_ssim': final_metrics.ssim,
                        'loss_reduction_percent': loss_reduction,
                        'total_iterations': len(losses)
                    }
                    
                    result = OptimizationResult(
                        optimized_latents=single_latent.detach(),
                        losses=losses,
                        metrics=metrics,
                        convergence_iteration=convergence_iterations[j],
                        initial_loss=losses[0] if losses else 0.0,
                        final_loss=losses[-1] if losses else 0.0
                    )
                    results.append(result)
        
        return results