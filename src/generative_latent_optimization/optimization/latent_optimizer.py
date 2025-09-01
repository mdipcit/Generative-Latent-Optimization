#!/usr/bin/env python3
"""
VAE Latent Optimization Module

Based on invertLDM.py methodology for optimizing latent representations
to minimize reconstruction loss.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
from contextlib import contextmanager
from ..metrics.unified_calculator import UnifiedMetricsCalculator
from ..core.device_manager import DeviceManager
from ..utils.io_utils import StatisticsCalculator


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
class BatchSetupResult:
    """Result from batch optimization setup"""
    optimized_latents: torch.Tensor
    optimizer: torch.optim.Optimizer
    batch_size: int
    batch_losses: List[List[float]]
    convergence_iterations: List[Optional[int]]


@dataclass
class RawOptimizationResult:
    """Raw results from optimization loop"""
    optimized_latents: torch.Tensor
    batch_losses: List[List[float]]
    convergence_iterations: List[Optional[int]]
    total_iterations: int


@dataclass
class ProcessedBatchResult:
    """Processed batch results with metrics"""
    individual_results: List[Any]
    batch_statistics: Dict[str, float]


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
        self.metrics = UnifiedMetricsCalculator(device=config.device, enable_lpips=False, enable_improved_ssim=False)
    
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
        """Calculate loss based on configured loss function (uses batch method internally)"""
        # Add batch dimension if needed
        if target_image.dim() == 3:
            target_image = target_image.unsqueeze(0)
        if reconstructed.dim() == 3:
            reconstructed = reconstructed.unsqueeze(0)
        
        # Use unified batch loss calculation
        losses = self._calculate_batch_loss(target_image, reconstructed)
        
        # Return scalar loss
        return losses.mean()
        
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
            final_metrics = self.metrics.calculate_legacy(target_image, final_reconstructed)
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
        Optimize a batch of latent representations (refactored)
        
        Args:
            vae: VAE model
            latents_batch: Batch of initial latents [B, C, H, W]
            targets_batch: Batch of target images [B, C, H, W]
            
        Returns:
            List of OptimizationResult for each sample in batch
        """
        with self._freeze_vae_decoder(vae):
            # Setup optimization
            setup = self._setup_batch_optimization(latents_batch)
            
            # Execute optimization loop
            raw_results = self._execute_batch_optimization_loop(
                vae, setup, targets_batch
            )
            
            # Calculate final results
            processed_results = self._calculate_batch_results(
                vae, raw_results, targets_batch
            )
            
            # Format output
            return self._format_batch_output(processed_results)
    
    def _setup_batch_optimization(self, latents_batch: torch.Tensor) -> BatchSetupResult:
        """
        Setup batch optimization environment
        
        Args:
            latents_batch: Batch of initial latents
            
        Returns:
            BatchSetupResult with initialized optimization components
        """
        batch_size = latents_batch.shape[0]
        
        # Enable gradients for all latents in batch
        optimized_latents = latents_batch.clone().detach().requires_grad_(True)
        
        # Setup optimizer for batch
        optimizer = torch.optim.Adam([optimized_latents], lr=self.config.learning_rate)
        
        # Initialize tracking structures
        batch_losses = [[] for _ in range(batch_size)]
        convergence_iterations = [None] * batch_size
        
        return BatchSetupResult(
            optimized_latents=optimized_latents,
            optimizer=optimizer,
            batch_size=batch_size,
            batch_losses=batch_losses,
            convergence_iterations=convergence_iterations
        )
    
    def _execute_batch_optimization_loop(
        self, vae, setup: BatchSetupResult, targets_batch: torch.Tensor
    ) -> RawOptimizationResult:
        """
        Execute the optimization loop
        
        Args:
            vae: VAE model
            setup: Batch setup result
            targets_batch: Target images
            
        Returns:
            RawOptimizationResult with optimization data
        """
        with tqdm(range(self.config.iterations), desc="Batch optimization") as pbar:
            for i in pbar:
                setup.optimizer.zero_grad()
                
                # Decode all latents at once
                outputs = vae.decode(setup.optimized_latents)
                reconstructed = (outputs.sample / 2 + 0.5).clamp(0, 1)
                
                # Calculate loss using unified method
                losses = self._calculate_batch_loss(targets_batch, reconstructed)
                
                # Total loss for backpropagation
                total_loss = losses.mean()
                
                # Backward pass
                total_loss.backward()
                setup.optimizer.step()
                
                # Track individual losses
                for j in range(setup.batch_size):
                    loss_val = losses[j].item()
                    setup.batch_losses[j].append(loss_val)
                
                # Update progress bar
                if i % self.config.checkpoint_interval == 0:
                    avg_loss = total_loss.item()
                    pbar.set_postfix({'Avg Loss': f'{avg_loss:.6f}'})
                
                # Check convergence
                self._check_batch_convergence(setup, i)
        
        return RawOptimizationResult(
            optimized_latents=setup.optimized_latents,
            batch_losses=setup.batch_losses,
            convergence_iterations=setup.convergence_iterations,
            total_iterations=self.config.iterations
        )
    
    def _calculate_batch_loss(
        self, targets: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate loss for batch (unified implementation)
        
        Args:
            targets: Target images
            reconstructed: Reconstructed images
            
        Returns:
            Per-sample losses
        """
        if self.config.loss_function == 'mse':
            losses = torch.nn.functional.mse_loss(
                targets, reconstructed, reduction='none'
            ).mean(dim=(1, 2, 3))
        elif self.config.loss_function == 'l1':
            losses = torch.nn.functional.l1_loss(
                targets, reconstructed, reduction='none'
            ).mean(dim=(1, 2, 3))
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
        
        return losses
    
    def _check_batch_convergence(self, setup: BatchSetupResult, iteration: int) -> None:
        """
        Check convergence for each sample in batch
        
        Args:
            setup: Batch setup with tracking data
            iteration: Current iteration
        """
        if iteration > 0:
            for j in range(setup.batch_size):
                if setup.convergence_iterations[j] is None:
                    losses = setup.batch_losses[j]
                    if len(losses) >= 2:
                        loss_change = abs(losses[-1] - losses[-2])
                        if loss_change < self.config.convergence_threshold:
                            setup.convergence_iterations[j] = iteration
    
    def _calculate_batch_results(
        self, vae, raw_results: RawOptimizationResult, targets_batch: torch.Tensor
    ) -> ProcessedBatchResult:
        """
        Calculate final metrics and statistics
        
        Args:
            vae: VAE model
            raw_results: Raw optimization results
            targets_batch: Target images
            
        Returns:
            ProcessedBatchResult with metrics
        """
        results = []
        batch_size = len(raw_results.batch_losses)
        
        with torch.no_grad():
            final_outputs = vae.decode(raw_results.optimized_latents)
            final_reconstructed = (final_outputs.sample / 2 + 0.5).clamp(0, 1)
            
            for j in range(batch_size):
                single_target = targets_batch[j:j+1]
                single_recon = final_reconstructed[j:j+1]
                single_latent = raw_results.optimized_latents[j:j+1]
                
                # Calculate metrics
                final_metrics = self.metrics.calculate_legacy(single_target, single_recon)
                
                # Calculate loss reduction
                losses = raw_results.batch_losses[j]
                loss_reduction = self._calculate_loss_reduction(losses)
                
                # Create result object
                result_data = {
                    'optimized_latents': single_latent.detach(),
                    'losses': losses,
                    'metrics': {
                        'final_psnr_db': final_metrics.psnr_db,
                        'final_ssim': final_metrics.ssim,
                        'loss_reduction_percent': loss_reduction,
                        'total_iterations': len(losses)
                    },
                    'convergence_iteration': raw_results.convergence_iterations[j],
                    'initial_loss': losses[0] if losses else 0.0,
                    'final_loss': losses[-1] if losses else 0.0
                }
                results.append(result_data)
        
        # Calculate batch statistics
        batch_stats = self._calculate_batch_statistics(results)
        
        return ProcessedBatchResult(
            individual_results=results,
            batch_statistics=batch_stats
        )
    
    def _calculate_loss_reduction(self, losses: List[float]) -> float:
        """
        Calculate percentage loss reduction
        
        Args:
            losses: List of loss values
            
        Returns:
            Loss reduction percentage
        """
        if losses and losses[0] > 0:
            return ((losses[0] - losses[-1]) / losses[0]) * 100
        return 0.0
    
    def _calculate_batch_statistics(self, results: List[Dict]) -> Dict[str, float]:
        """
        Calculate statistics across batch results
        
        Args:
            results: List of individual results
            
        Returns:
            Batch statistics dictionary
        """
        if not results:
            return {}
        
        # Extract metrics
        psnr_values = [r['metrics']['final_psnr_db'] for r in results]
        ssim_values = [r['metrics']['final_ssim'] for r in results]
        reduction_values = [r['metrics']['loss_reduction_percent'] for r in results]
        
        # Use StatisticsCalculator
        psnr_stats = StatisticsCalculator.calculate_basic_stats(psnr_values, 'psnr')
        ssim_stats = StatisticsCalculator.calculate_basic_stats(ssim_values, 'ssim')
        reduction_stats = StatisticsCalculator.calculate_basic_stats(reduction_values, 'reduction')
        
        return {
            'batch_psnr_mean': psnr_stats['mean'],
            'batch_psnr_std': psnr_stats['std'],
            'batch_ssim_mean': ssim_stats['mean'],
            'batch_ssim_std': ssim_stats['std'],
            'batch_reduction_mean': reduction_stats['mean'],
            'batch_reduction_std': reduction_stats['std']
        }
    
    def _format_batch_output(self, processed: ProcessedBatchResult) -> List[OptimizationResult]:
        """
        Format processed results into final output
        
        Args:
            processed: Processed batch results
            
        Returns:
            List of OptimizationResult objects
        """
        output_results = []
        
        for result_data in processed.individual_results:
            result = OptimizationResult(
                optimized_latents=result_data['optimized_latents'],
                losses=result_data['losses'],
                metrics=result_data['metrics'],
                convergence_iteration=result_data['convergence_iteration'],
                initial_loss=result_data['initial_loss'],
                final_loss=result_data['final_loss']
            )
            output_results.append(result)
        
        return output_results