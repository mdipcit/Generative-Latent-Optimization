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
    
    def __init__(self, device='cuda', enable_lpips=True, enable_improved_ssim=True):
        """
        Initialize metrics calculator
        
        Args:
            device: Computation device
            enable_lpips: Whether to enable LPIPS calculation
            enable_improved_ssim: Whether to enable improved SSIM calculation
        """
        self.device = device
        self.enable_lpips = enable_lpips
        self.enable_improved_ssim = enable_improved_ssim
        
        # Initialize basic metrics
        self.basic_metrics = ImageMetrics(device=device)
        
        # Initialize advanced metrics
        self.lpips = None
        self.ssim_improved = None
        
        if enable_lpips:
            try:
                self.lpips = LPIPSMetric(device=device)
                logger.info("LPIPS metric enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize LPIPS: {e}")
                self.enable_lpips = False
        
        if enable_improved_ssim:
            try:
                self.ssim_improved = ImprovedSSIM(device=device)
                logger.info("Improved SSIM metric enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize improved SSIM: {e}")
                self.enable_improved_ssim = False
        
        logger.info(f"Individual metrics calculator initialized on {device}")
        logger.info(f"  LPIPS: {'enabled' if self.enable_lpips else 'disabled'}")
        logger.info(f"  Improved SSIM: {'enabled' if self.enable_improved_ssim else 'disabled'}")
    
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
            if self.enable_lpips and self.lpips is not None:
                try:
                    lpips_value = self.lpips.calculate(original, reconstructed)
                except Exception as e:
                    logger.warning(f"LPIPS calculation failed: {e}")
            
            ssim_improved_value = None
            if self.enable_improved_ssim and self.ssim_improved is not None:
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
        # Filter out None results
        valid_results = [r for r in batch_results if r is not None]
        
        if not valid_results:
            return {}
        
        # Extract values for each metric
        psnr_values = [r.psnr_db for r in valid_results]
        ssim_values = [r.ssim for r in valid_results]
        mse_values = [r.mse for r in valid_results]
        mae_values = [r.mae for r in valid_results]
        
        statistics = {
            'psnr_mean': sum(psnr_values) / len(psnr_values),
            'psnr_std': self._calculate_std(psnr_values),
            'psnr_min': min(psnr_values),
            'psnr_max': max(psnr_values),
            'ssim_mean': sum(ssim_values) / len(ssim_values),
            'ssim_std': self._calculate_std(ssim_values),
            'ssim_min': min(ssim_values),
            'ssim_max': max(ssim_values),
            'mse_mean': sum(mse_values) / len(mse_values),
            'mse_std': self._calculate_std(mse_values),
            'mae_mean': sum(mae_values) / len(mae_values),
            'mae_std': self._calculate_std(mae_values),
            'valid_samples': len(valid_results),
            'total_samples': len(batch_results)
        }
        
        # Add LPIPS statistics if available
        lpips_values = [r.lpips for r in valid_results if r.lpips is not None]
        if lpips_values:
            statistics.update({
                'lpips_mean': sum(lpips_values) / len(lpips_values),
                'lpips_std': self._calculate_std(lpips_values),
                'lpips_min': min(lpips_values),
                'lpips_max': max(lpips_values)
            })
        
        # Add improved SSIM statistics if available
        ssim_improved_values = [r.ssim_improved for r in valid_results if r.ssim_improved is not None]
        if ssim_improved_values:
            statistics.update({
                'ssim_improved_mean': sum(ssim_improved_values) / len(ssim_improved_values),
                'ssim_improved_std': self._calculate_std(ssim_improved_values),
                'ssim_improved_min': min(ssim_improved_values),
                'ssim_improved_max': max(ssim_improved_values)
            })
        
        return statistics
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


# Utility functions for testing
def test_individual_metrics_calculator(device='cuda'):
    """Test individual metrics calculator functionality"""
    print("Testing individual metrics calculator...")
    
    try:
        calculator = IndividualMetricsCalculator(
            device=device,
            enable_lpips=True,
            enable_improved_ssim=True
        )
        
        # Create test images
        original = torch.rand(1, 3, 256, 256).to(device)
        reconstructed = original + torch.randn_like(original) * 0.1
        
        # Test single image calculation
        print("  Testing single image metrics...")
        metrics = calculator.calculate_all_individual_metrics(original, reconstructed)
        
        print(f"    PSNR: {metrics.psnr_db:.2f} dB")
        print(f"    SSIM: {metrics.ssim:.4f}")
        print(f"    MSE: {metrics.mse:.6f}")
        print(f"    MAE: {metrics.mae:.6f}")
        if metrics.lpips is not None:
            print(f"    LPIPS: {metrics.lpips:.4f}")
        if metrics.ssim_improved is not None:
            print(f"    SSIM (improved): {metrics.ssim_improved:.4f}")
        
        # Test batch calculation
        print("  Testing batch metrics...")
        batch_original = torch.rand(4, 3, 256, 256).to(device)
        batch_reconstructed = batch_original + torch.randn_like(batch_original) * 0.1
        
        batch_results = calculator.calculate_batch_individual_metrics(
            batch_original, batch_reconstructed
        )
        
        print(f"    Batch size: {len(batch_results)}")
        
        # Test statistics calculation
        stats = calculator.get_batch_statistics(batch_results)
        print(f"    Mean PSNR: {stats['psnr_mean']:.2f} dB")
        print(f"    Mean SSIM: {stats['ssim_mean']:.4f}")
        if 'lpips_mean' in stats:
            print(f"    Mean LPIPS: {stats['lpips_mean']:.4f}")
        
        print("  Individual metrics calculator test passed!")
        
    except Exception as e:
        print(f"  Individual metrics calculator test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run functionality tests
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running individual metrics calculator tests on device: {device}")
    
    test_individual_metrics_calculator(device)
    
    print("Metrics integration module tests completed.")