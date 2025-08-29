#!/usr/bin/env python3
"""
Image Quality Metrics Module

Provides various image quality assessment metrics for evaluating
VAE reconstruction and optimization results.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MetricResults:
    """Container for image quality metrics results"""
    psnr_db: float
    ssim: float
    mse: float
    mae: float
    lpips: Optional[float] = None


class ImageMetrics:
    """
    Collection of image quality assessment metrics
    
    Provides PSNR, SSIM, MSE, MAE and other metrics for comparing
    original and reconstructed images.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    def calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        
        Args:
            img1: Reference image tensor [B, C, H, W] in [0, 1] range
            img2: Comparison image tensor [B, C, H, W] in [0, 1] range
            
        Returns:
            PSNR value in dB
        """
        mse = torch.nn.functional.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()
    
    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor, 
                      window_size: int = 11) -> float:
        """
        Calculate Structural Similarity Index (SSIM)
        
        Args:
            img1: Reference image tensor [B, C, H, W]
            img2: Comparison image tensor [B, C, H, W]
            window_size: Size of the sliding window
            
        Returns:
            SSIM value between 0 and 1
        """
        # Simplified SSIM implementation
        # For production use, consider torchmetrics.SSIM
        
        # Convert to grayscale if RGB
        if img1.shape[1] == 3:
            img1_gray = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
            img2_gray = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        else:
            img1_gray = img1.squeeze(1)
            img2_gray = img2.squeeze(1)
        
        # Calculate local statistics
        mu1 = self._gaussian_filter(img1_gray, window_size)
        mu2 = self._gaussian_filter(img2_gray, window_size)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = self._gaussian_filter(img1_gray * img1_gray, window_size) - mu1_sq
        sigma2_sq = self._gaussian_filter(img2_gray * img2_gray, window_size) - mu2_sq
        sigma12 = self._gaussian_filter(img1_gray * img2_gray, window_size) - mu1_mu2
        
        # SSIM constants
        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2
        
        # SSIM calculation
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim_map = numerator / denominator
        return ssim_map.mean().item()
    
    def calculate_mse(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate Mean Squared Error"""
        mse = torch.nn.functional.mse_loss(img1, img2)
        return mse.item()
    
    def calculate_mae(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate Mean Absolute Error"""
        mae = torch.nn.functional.l1_loss(img1, img2)
        return mae.item()
    
    def calculate_all_metrics(self, img1: torch.Tensor, img2: torch.Tensor) -> MetricResults:
        """
        Calculate all available metrics
        
        Args:
            img1: Reference image tensor
            img2: Comparison image tensor
            
        Returns:
            MetricResults containing all calculated metrics
        """
        return MetricResults(
            psnr_db=self.calculate_psnr(img1, img2),
            ssim=self.calculate_ssim(img1, img2),
            mse=self.calculate_mse(img1, img2),
            mae=self.calculate_mae(img1, img2)
        )
    
    def calculate_batch_metrics(self, original_batch: torch.Tensor, 
                              reconstructed_batch: torch.Tensor) -> List[MetricResults]:
        """
        Calculate metrics for a batch of images
        
        Args:
            original_batch: Batch of original images [B, C, H, W]
            reconstructed_batch: Batch of reconstructed images [B, C, H, W]
            
        Returns:
            List of MetricResults for each sample in the batch
        """
        batch_size = original_batch.shape[0]
        results = []
        
        for i in range(batch_size):
            result = self.calculate_all_metrics(
                original_batch[i:i+1],
                reconstructed_batch[i:i+1]
            )
            results.append(result)
        
        return results
    
    def get_batch_statistics(self, results: List[MetricResults]) -> Dict[str, float]:
        """
        Calculate statistics across a batch of metric results
        
        Args:
            results: List of MetricResults
            
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not results:
            return {}
        
        psnr_values = [r.psnr_db for r in results]
        ssim_values = [r.ssim for r in results]
        mse_values = [r.mse for r in results]
        mae_values = [r.mae for r in results]
        
        return {
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'psnr_min': np.min(psnr_values),
            'psnr_max': np.max(psnr_values),
            'ssim_mean': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values),
            'ssim_min': np.min(ssim_values),
            'ssim_max': np.max(ssim_values),
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values),
            'mae_mean': np.mean(mae_values),
            'mae_std': np.std(mae_values),
        }
    
    def _gaussian_filter(self, input_tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Apply Gaussian filter for SSIM calculation
        
        This is a simplified implementation. For production use,
        consider using torchvision.transforms.GaussianBlur
        """
        # Create Gaussian kernel
        sigma = window_size / 6.0
        kernel = self._create_gaussian_kernel(window_size, sigma)
        kernel = kernel.to(input_tensor.device)
        
        # Apply convolution with padding
        padding = window_size // 2
        filtered = torch.nn.functional.conv2d(
            input_tensor.unsqueeze(1), 
            kernel.unsqueeze(0).unsqueeze(0), 
            padding=padding
        )
        
        return filtered.squeeze(1)
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # Create 2D kernel
        kernel = g[:, None] * g[None, :]
        return kernel


class MetricsTracker:
    """Track metrics during optimization process"""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, float]] = []
        self.image_metrics = ImageMetrics()
    
    def add_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                   iteration: int, loss: float):
        """Add metrics for current iteration"""
        metrics = self.image_metrics.calculate_all_metrics(original, reconstructed)
        
        record = {
            'iteration': iteration,
            'loss': loss,
            'psnr_db': metrics.psnr_db,
            'ssim': metrics.ssim,
            'mse': metrics.mse,
            'mae': metrics.mae
        }
        
        self.metrics_history.append(record)
    
    def get_history(self) -> List[Dict[str, float]]:
        """Get complete metrics history"""
        return self.metrics_history
    
    def get_final_summary(self) -> Dict[str, float]:
        """Get summary of final metrics"""
        if not self.metrics_history:
            return {}
        
        final = self.metrics_history[-1]
        initial = self.metrics_history[0]
        
        return {
            'initial_psnr': initial['psnr_db'],
            'final_psnr': final['psnr_db'],
            'psnr_improvement': final['psnr_db'] - initial['psnr_db'],
            'initial_ssim': initial['ssim'],
            'final_ssim': final['ssim'],
            'ssim_improvement': final['ssim'] - initial['ssim'],
            'initial_loss': initial['loss'],
            'final_loss': final['loss'],
            'loss_reduction_percent': ((initial['loss'] - final['loss']) / initial['loss']) * 100
        }


# Utility functions for backward compatibility
def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Standalone PSNR calculation function"""
    metrics = ImageMetrics()
    return metrics.calculate_psnr(img1, img2)