#!/usr/bin/env python3
"""
Metrics Integration Module

Provides unified interface for calculating all individual image metrics.
FID is excluded as it requires dataset-level evaluation.
"""

import torch
from typing import List, Dict, Any, Optional
import logging

try:
    from .image_metrics import ImageMetrics, IndividualImageMetrics
    from .individual_metrics import LPIPSMetric, ImprovedSSIM
except ImportError:
    # For direct execution as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from image_metrics import ImageMetrics, IndividualImageMetrics
    from individual_metrics import LPIPSMetric, ImprovedSSIM

logger = logging.getLogger(__name__)


class IndividualMetricsCalculator:
    """
    Unified calculator for individual image metrics (FID excluded)
    
    Provides a single interface to calculate all individual image quality metrics
    including basic metrics (PSNR, SSIM, MSE, MAE) and advanced metrics (LPIPS, Improved SSIM).
    """
    
    def __init__(self, device='cuda', use_lpips=True, use_improved_ssim=True):
        """
        Initialize metrics calculator
        
        Args:
            device: Computation device
            use_lpips: Whether to use LPIPS calculation
            use_improved_ssim: Whether to use improved SSIM calculation
        """
        self.device = device
        self.use_lpips = use_lpips
        self.use_improved_ssim = use_improved_ssim
        
        # Initialize basic metrics
        self.basic_metrics = ImageMetrics(device=device)
        
        # Initialize advanced metrics
        self.lpips = None
        self.ssim_improved = None
        
        if use_lpips:
            try:
                self.lpips = LPIPSMetric(device=device)
                logger.info("LPIPS metric enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LPIPS: {e}")
                self.use_lpips = False
        
        if use_improved_ssim:
            try:
                self.ssim_improved = ImprovedSSIM(device=device)
                logger.info("Improved SSIM metric enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize improved SSIM: {e}")
                self.use_improved_ssim = False
        
        logger.info(f"Individual metrics calculator initialized on {device}")
        logger.info(f"  LPIPS: {'enabled' if self.use_lpips else 'disabled'}")
        logger.info(f"  Improved SSIM: {'enabled' if self.use_improved_ssim else 'disabled'}")
    
    def calculate_all_individual_metrics(self, original: torch.Tensor, 
                                       reconstructed: torch.Tensor) -> IndividualImageMetrics:
        """
        Calculate all individual image metrics for a single image pair (FID excluded)
        
        Args:
            original: Original image tensor [B, C, H, W]
            reconstructed: Reconstructed image tensor [B, C, H, W]
            
        Returns:
            IndividualImageMetrics: Complete metrics results
        """
        try:
            # Validate inputs
            if original.shape != reconstructed.shape:
                raise ValueError(f"Image shapes must match: {original.shape} vs {reconstructed.shape}")
            
            # Calculate basic metrics
            basic_results = self.basic_metrics.calculate_all_metrics(original, reconstructed)
            
            # Calculate advanced metrics
            lpips_value = None
            if self.use_lpips and self.lpips is not None:
                try:
                    lpips_value = self.lpips.calculate(original, reconstructed)
                except Exception as e:
                    logger.warning(f"LPIPS calculation failed: {e}")
            
            ssim_improved_value = None
            if self.use_improved_ssim and self.ssim_improved is not None:
                try:
                    ssim_improved_value = self.ssim_improved.calculate(original, reconstructed)
                except Exception as e:
                    logger.warning(f"Improved SSIM calculation failed: {e}")
            
            return IndividualImageMetrics(
                psnr_db=basic_results.psnr_db,
                ssim=basic_results.ssim,  # Original SSIM
                mse=basic_results.mse,
                mae=basic_results.mae,
                lpips=lpips_value,
                ssim_improved=ssim_improved_value
            )
            
        except Exception as e:
            logger.error(f"Individual metrics calculation failed: {e}")
            raise
    
    def calculate_batch_individual_metrics(self, original_batch: torch.Tensor,
                                         reconstructed_batch: torch.Tensor) -> List[IndividualImageMetrics]:
        """
        Calculate individual metrics for each image in a batch
        
        Args:
            original_batch: Batch of original images [B, C, H, W]
            reconstructed_batch: Batch of reconstructed images [B, C, H, W]
            
        Returns:
            List of IndividualImageMetrics for each image pair
        """
        batch_size = original_batch.shape[0]
        batch_results = []
        
        logger.debug(f"Calculating individual metrics for batch of {batch_size} images")
        
        for i in range(batch_size):
            try:
                metrics = self.calculate_all_individual_metrics(
                    original_batch[i:i+1],
                    reconstructed_batch[i:i+1]
                )
                batch_results.append(metrics)
            except Exception as e:
                logger.error(f"Failed to calculate metrics for batch item {i}: {e}")
                # Add None or default metrics to maintain batch consistency
                batch_results.append(None)
        
        return batch_results
    
    def get_batch_statistics(self, batch_results: List[IndividualImageMetrics]) -> Dict[str, float]:
        """
        Calculate statistics across a batch of individual metrics
        
        Args:
            batch_results: List of IndividualImageMetrics
            
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        from ..utils.io_utils import StatisticsCalculator
        
        # Filter out None results
        valid_results = [r for r in batch_results if r is not None]
        
        if not valid_results:
            return {}
        
        # Extract values for each metric
        psnr_values = [r.psnr_db for r in valid_results]
        ssim_values = [r.ssim for r in valid_results]
        mse_values = [r.mse for r in valid_results]
        mae_values = [r.mae for r in valid_results]
        
        # Use StatisticsCalculator for all metrics
        psnr_stats = StatisticsCalculator.calculate_basic_stats(psnr_values, 'psnr')
        ssim_stats = StatisticsCalculator.calculate_basic_stats(ssim_values, 'ssim')
        mse_stats = StatisticsCalculator.calculate_basic_stats(mse_values, 'mse')
        mae_stats = StatisticsCalculator.calculate_basic_stats(mae_values, 'mae')
        
        statistics = {
            'psnr_mean': psnr_stats['mean'],
            'psnr_std': psnr_stats['std'],
            'psnr_min': psnr_stats['min'],
            'psnr_max': psnr_stats['max'],
            'ssim_mean': ssim_stats['mean'],
            'ssim_std': ssim_stats['std'],
            'ssim_min': ssim_stats['min'],
            'ssim_max': ssim_stats['max'],
            'mse_mean': mse_stats['mean'],
            'mse_std': mse_stats['std'],
            'mae_mean': mae_stats['mean'],
            'mae_std': mae_stats['std'],
            'valid_samples': len(valid_results),
            'total_samples': len(batch_results)
        }
        
        # Add LPIPS statistics if available
        lpips_values = [r.lpips for r in valid_results if r.lpips is not None]
        if lpips_values:
            lpips_stats = StatisticsCalculator.calculate_basic_stats(lpips_values, 'lpips')
            statistics.update({
                'lpips_mean': lpips_stats['mean'],
                'lpips_std': lpips_stats['std'],
                'lpips_min': lpips_stats['min'],
                'lpips_max': lpips_stats['max']
            })
        
        # Add improved SSIM statistics if available
        ssim_improved_values = [r.ssim_improved for r in valid_results if r.ssim_improved is not None]
        if ssim_improved_values:
            ssim_imp_stats = StatisticsCalculator.calculate_basic_stats(ssim_improved_values, 'ssim_improved')
            statistics.update({
                'ssim_improved_mean': ssim_imp_stats['mean'],
                'ssim_improved_std': ssim_imp_stats['std'],
                'ssim_improved_min': ssim_imp_stats['min'],
                'ssim_improved_max': ssim_imp_stats['max']
            })
        
        return statistics


# Utility functions for testing
