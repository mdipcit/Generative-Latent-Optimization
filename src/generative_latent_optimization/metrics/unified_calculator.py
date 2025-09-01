"""Unified metrics calculator for consolidated metric computation."""

from typing import Dict, List, Any, Optional, Union
import torch
from ..core.base_classes import BaseMetric
from .image_metrics import ImageMetrics, MetricResults, IndividualImageMetrics
from .individual_metrics import LPIPSMetric, ImprovedSSIM
from ..utils.io_utils import StatisticsCalculator


class UnifiedMetricsCalculator(BaseMetric):
    """Unified metrics calculation interface.
    
    Consolidates duplicate metric calculation implementations across:
    - ImageMetrics.calculate_all_metrics()
    - IndividualImageMetricsCalculator.calculate_all_individual_metrics()
    - LatentOptimizer metric calculations
    """
    
    def __init__(self, device: str = 'cuda', use_lpips: bool = True, use_improved_ssim: bool = True):
        """Initialize unified metrics calculator.
        
        Args:
            device: Device for computations ('cuda' or 'cpu')
            use_lpips: Whether to use LPIPS calculations
            use_improved_ssim: Whether to use improved SSIM calculations
        """
        # Set configuration before calling super().__init__
        self.use_lpips = use_lpips
        self.use_improved_ssim = use_improved_ssim
        super().__init__(device)
        
    def _setup_metric_specific_resources(self) -> None:
        """Initialize required metric calculators."""
        # Basic metrics calculator
        self.basic_metrics = ImageMetrics(device=self.device)
        
        # Advanced metrics (conditional initialization)
        self.lpips_metric = None
        if self.use_lpips:
            try:
                self.lpips_metric = LPIPSMetric(device=self.device)
            except Exception as e:
                print(f"Warning: LPIPS initialization failed: {e}")
                self.use_lpips = False
                
        self.improved_ssim = None
        if self.use_improved_ssim:
            try:
                self.improved_ssim = ImprovedSSIM(device=self.device)
            except Exception as e:
                print(f"Warning: Improved SSIM initialization failed: {e}")
                self.use_improved_ssim = False
    
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> IndividualImageMetrics:
        """Calculate comprehensive metrics for a single image pair.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            
        Returns:
            IndividualImageMetrics with all calculated metrics
        """
        # Validate inputs
        if img1.shape != img2.shape:
            raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
        
        # Calculate basic metrics
        basic_results = self.basic_metrics.calculate_all_metrics(img1, img2)
        
        # Initialize advanced metrics
        lpips_value = None
        improved_ssim_value = None
        
        # Calculate LPIPS if enabled
        if self.use_lpips and self.lpips_metric is not None:
            try:
                lpips_value = self.lpips_metric.calculate(img1, img2)
            except Exception as e:
                print(f"Warning: LPIPS calculation failed: {e}")
                
        # Calculate improved SSIM if enabled
        if self.use_improved_ssim and self.improved_ssim is not None:
            try:
                improved_ssim_value = self.improved_ssim.calculate(img1, img2)
            except Exception as e:
                print(f"Warning: Improved SSIM calculation failed: {e}")
        
        # Return unified results
        return IndividualImageMetrics(
            psnr_db=basic_results.psnr_db,
            ssim=basic_results.ssim,
            mse=basic_results.mse,
            mae=basic_results.mae,
            lpips=lpips_value,
            ssim_improved=improved_ssim_value
        )
    
    def calculate_legacy(self, img1: torch.Tensor, img2: torch.Tensor) -> MetricResults:
        """Calculate basic metrics in legacy format for backward compatibility.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            
        Returns:
            MetricResults with basic metrics
        """
        return self.basic_metrics.calculate_all_metrics(img1, img2)
    
    def calculate_batch_metrics(self, imgs1: List[torch.Tensor], imgs2: List[torch.Tensor]) -> Dict[str, Any]:
        """Calculate metrics for batch of images with statistics.
        
        Args:
            imgs1: List of first image tensors
            imgs2: List of second image tensors
            
        Returns:
            Dictionary with individual results and comprehensive statistics
        """
        if len(imgs1) != len(imgs2):
            raise ValueError(f"Batch sizes must match: {len(imgs1)} vs {len(imgs2)}")
        
        # Calculate individual metrics for each pair
        individual_results = []
        for img1, img2 in zip(imgs1, imgs2):
            result = self.calculate(img1, img2)
            individual_results.append(result)
        
        # Calculate comprehensive statistics using StatisticsCalculator
        stats_calc = StatisticsCalculator()
        comprehensive_stats = stats_calc.calculate_comprehensive_stats(individual_results)
        
        return {
            'individual_results': individual_results,
            'statistics': comprehensive_stats,
            'batch_size': len(individual_results)
        }
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metric names.
        
        Returns:
            List of metric names that can be calculated
        """
        metrics = ['psnr_db', 'ssim', 'mse', 'mae']
        
        if self.use_lpips and self.lpips_metric is not None:
            metrics.append('lpips')
            
        if self.use_improved_ssim and self.improved_ssim is not None:
            metrics.append('ssim_improved')
            
        return metrics
    
    def get_batch_statistics(self, batch_results: List[IndividualImageMetrics]) -> Dict[str, float]:
        """Calculate statistics across a batch of individual metrics (compatibility method).
        
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
        
        # Use StatisticsCalculator for all metrics
        psnr_stats = StatisticsCalculator.calculate_basic_stats(psnr_values, 'psnr')
        ssim_stats = StatisticsCalculator.calculate_basic_stats(ssim_values, 'ssim')
        mse_stats = StatisticsCalculator.calculate_basic_stats(mse_values, 'mse')
        mae_stats = StatisticsCalculator.calculate_basic_stats(mae_values, 'mae')
        
        # Create properly prefixed statistics dictionary
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
            'mse_min': mse_stats['min'],
            'mse_max': mse_stats['max'],
            'mae_mean': mae_stats['mean'],
            'mae_std': mae_stats['std'],
            'mae_min': mae_stats['min'],
            'mae_max': mae_stats['max'],
        }
        
        # Add advanced metrics if available
        lpips_values = [r.lpips for r in valid_results if r.lpips is not None]
        if lpips_values:
            lpips_stats = StatisticsCalculator.calculate_basic_stats(lpips_values, 'lpips')
            statistics.update({
                'lpips_mean': lpips_stats['mean'],
                'lpips_std': lpips_stats['std'],
                'lpips_min': lpips_stats['min'],
                'lpips_max': lpips_stats['max'],
            })
        
        improved_ssim_values = [r.ssim_improved for r in valid_results if r.ssim_improved is not None]
        if improved_ssim_values:
            improved_ssim_stats = StatisticsCalculator.calculate_basic_stats(improved_ssim_values, 'ssim_improved')
            statistics.update({
                'ssim_improved_mean': improved_ssim_stats['mean'],
                'ssim_improved_std': improved_ssim_stats['std'],
                'ssim_improved_min': improved_ssim_stats['min'],
                'ssim_improved_max': improved_ssim_stats['max'],
            })
        
        return statistics
    
    # Compatibility methods for backward compatibility with individual calculations
    def calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate PSNR (compatibility method)."""
        return self.basic_metrics.calculate_psnr(img1, img2)
    
    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM (compatibility method).""" 
        return self.basic_metrics.calculate_ssim(img1, img2)
    
    def calculate_mse(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate MSE (compatibility method)."""
        return self.basic_metrics.calculate_mse(img1, img2)
    
    def calculate_mae(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate MAE (compatibility method)."""
        return self.basic_metrics.calculate_mae(img1, img2)