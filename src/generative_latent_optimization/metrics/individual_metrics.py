#!/usr/bin/env python3
"""
Individual Image Metrics Module

Provides advanced individual image quality metrics including LPIPS and improved SSIM.
These metrics are calculated per image during the optimization process.
"""

import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LPIPSMetric:
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric
    
    Measures perceptual similarity between images based on human visual perception.
    Lower values indicate more perceptually similar images.
    """
    
    def __init__(self, net='alex', device='cuda', use_gpu=True):
        """
        Initialize LPIPS metric
        
        Args:
            net: Network backbone ('alex', 'vgg', 'squeeze')
            device: Computation device
            use_gpu: Whether to use GPU acceleration
        """
        self.device = device
        self.net = net
        
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net=net, verbose=False)
            if use_gpu and torch.cuda.is_available():
                self.loss_fn = self.loss_fn.to(device)
            logger.info(f"LPIPS metric initialized with {net} network on {device}")
        except ImportError:
            raise ImportError("lpips package is required but not installed. Install with: pip install lpips")
        except Exception as e:
            logger.error(f"Failed to initialize LPIPS: {e}")
            raise
    
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate LPIPS distance between two images
        
        Args:
            img1: Reference image tensor [B, C, H, W] in [-1,1] range
            img2: Comparison image tensor [B, C, H, W] in [-1,1] range
            
        Returns:
            LPIPS distance (lower = more perceptually similar)
        """
        try:
            # Validate inputs
            if img1.shape != img2.shape:
                raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
            
            # Ensure images are in [-1, 1] range
            if img1.min() >= 0 and img1.max() <= 1:
                # Convert [0,1] to [-1,1]
                img1 = img1 * 2.0 - 1.0
            if img2.min() >= 0 and img2.max() <= 1:
                # Convert [0,1] to [-1,1]  
                img2 = img2 * 2.0 - 1.0
            
            # Ensure tensors are on correct device
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            # Calculate LPIPS
            with torch.no_grad():
                lpips_value = self.loss_fn(img1, img2)
                return lpips_value.item()
                
        except Exception as e:
            logger.error(f"LPIPS calculation failed: {e}")
            return None
    
    def calculate_batch(self, img1_batch: torch.Tensor, img2_batch: torch.Tensor) -> list:
        """
        Calculate LPIPS for a batch of image pairs
        
        Args:
            img1_batch: Batch of reference images [B, C, H, W]
            img2_batch: Batch of comparison images [B, C, H, W]
            
        Returns:
            List of LPIPS values for each pair
        """
        batch_size = img1_batch.shape[0]
        lpips_values = []
        
        for i in range(batch_size):
            lpips_val = self.calculate(img1_batch[i:i+1], img2_batch[i:i+1])
            lpips_values.append(lpips_val)
            
        return lpips_values


class ImprovedSSIM:
    """
    Improved SSIM using TorchMetrics implementation
    
    Provides more accurate and efficient SSIM computation compared to custom implementation.
    """
    
    def __init__(self, data_range=1.0, device='cuda'):
        """
        Initialize improved SSIM metric
        
        Args:
            data_range: Range of input images (1.0 for [0,1], 2.0 for [-1,1])
            device: Computation device
        """
        self.device = device
        self.data_range = data_range
        
        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            self.ssim = StructuralSimilarityIndexMeasure(
                data_range=data_range,
                gaussian_kernel=True,
                kernel_size=11,
                sigma=1.5,
                reduction='elementwise_mean',
                k1=0.01,
                k2=0.03
            )
            
            if torch.cuda.is_available():
                self.ssim = self.ssim.to(device)
            logger.info(f"Improved SSIM initialized with data_range={data_range} on {device}")
            
        except ImportError:
            raise ImportError("torchmetrics package is required but not installed.")
        except Exception as e:
            logger.error(f"Failed to initialize improved SSIM: {e}")
            raise
    
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate improved SSIM between two images
        
        Args:
            img1: Reference image tensor [B, C, H, W]
            img2: Comparison image tensor [B, C, H, W]
            
        Returns:
            SSIM value between 0 and 1 (higher = more similar)
        """
        try:
            # Validate inputs
            if img1.shape != img2.shape:
                raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
            
            # Ensure correct value range
            if self.data_range == 1.0:
                # Expect [0, 1] range
                if img1.min() < 0 or img1.max() > 1:
                    img1 = torch.clamp((img1 + 1) / 2, 0, 1)  # Convert [-1,1] to [0,1]
                if img2.min() < 0 or img2.max() > 1:
                    img2 = torch.clamp((img2 + 1) / 2, 0, 1)
            
            # Ensure tensors are on correct device
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            # Calculate SSIM
            ssim_value = self.ssim(img1, img2)
            return ssim_value.item()
            
        except Exception as e:
            logger.error(f"Improved SSIM calculation failed: {e}")
            return None
    
    def calculate_batch(self, img1_batch: torch.Tensor, img2_batch: torch.Tensor) -> list:
        """
        Calculate improved SSIM for a batch of image pairs
        
        Args:
            img1_batch: Batch of reference images [B, C, H, W]
            img2_batch: Batch of comparison images [B, C, H, W]
            
        Returns:
            List of SSIM values for each pair
        """
        batch_size = img1_batch.shape[0]
        ssim_values = []
        
        for i in range(batch_size):
            ssim_val = self.calculate(img1_batch[i:i+1], img2_batch[i:i+1])
            ssim_values.append(ssim_val)
            
        return ssim_values


# Utility functions for backward compatibility and testing
def test_lpips_functionality(device='cuda'):
    """Test LPIPS metric functionality"""
    print("Testing LPIPS functionality...")
    
    try:
        lpips_metric = LPIPSMetric(device=device)
        
        # Create test images
        img1 = torch.randn(1, 3, 256, 256).to(device)
        img2 = img1 + torch.randn_like(img1) * 0.1  # Similar image with noise
        img3 = torch.randn(1, 3, 256, 256).to(device)  # Different image
        
        # Test calculations
        lpips_similar = lpips_metric.calculate(img1, img2)
        lpips_different = lpips_metric.calculate(img1, img3)
        
        print(f"  LPIPS (similar images): {lpips_similar:.4f}")
        print(f"  LPIPS (different images): {lpips_different:.4f}")
        print(f"  Test passed: {lpips_similar < lpips_different}")
        
    except Exception as e:
        print(f"  LPIPS test failed: {e}")


def test_improved_ssim_functionality(device='cuda'):
    """Test improved SSIM metric functionality"""
    print("Testing improved SSIM functionality...")
    
    try:
        ssim_metric = ImprovedSSIM(device=device)
        
        # Create test images
        img1 = torch.rand(1, 3, 256, 256).to(device)  # [0,1] range
        img2 = img1 + torch.randn_like(img1) * 0.05   # Similar image with small noise
        img3 = torch.rand(1, 3, 256, 256).to(device)  # Different image
        
        # Test calculations
        ssim_similar = ssim_metric.calculate(img1, img2)
        ssim_different = ssim_metric.calculate(img1, img3)
        
        print(f"  SSIM (similar images): {ssim_similar:.4f}")
        print(f"  SSIM (different images): {ssim_different:.4f}")
        print(f"  Test passed: {ssim_similar > ssim_different}")
        
    except Exception as e:
        print(f"  Improved SSIM test failed: {e}")


if __name__ == "__main__":
    # Run functionality tests
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running tests on device: {device}")
    
    test_lpips_functionality(device)
    test_improved_ssim_functionality(device)
    
    print("Individual metrics module tests completed.")