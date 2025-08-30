#!/usr/bin/env python3
"""
Test Helper Functions and Fixtures

Common functionality used across tests and experiments.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Re-export commonly used functions from external modules
try:
    from vae_toolkit import VAELoader, load_and_preprocess_image, tensor_to_pil
except ImportError:
    print("Warning: vae_toolkit not available")


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate PSNR between two images
    
    Args:
        img1: First image tensor [B, C, H, W] in [0, 1] range
        img2: Second image tensor [B, C, H, W] in [0, 1] range
        
    Returns:
        PSNR value in dB
    """
    mse = torch.nn.functional.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


def calculate_metrics(img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, float]:
    """
    Calculate common image quality metrics
    
    Args:
        img1: First image tensor [B, C, H, W] in [0, 1] range  
        img2: Second image tensor [B, C, H, W] in [0, 1] range
        
    Returns:
        Dictionary with computed metrics
    """
    mse = torch.nn.functional.mse_loss(img1, img2).item()
    mae = torch.nn.functional.l1_loss(img1, img2).item()
    psnr = calculate_psnr(img1, img2)
    
    return {
        'mse': mse,
        'mae': mae,
        'psnr_db': psnr
    }


def save_image_tensor(tensor: torch.Tensor, path: Path, format: str = "PNG") -> None:
    """
    Convert tensor to PIL image and save
    
    Args:
        tensor: Image tensor [C, H, W] or [1, C, H, W] in [0, 1] range
        path: Output path
        format: Image format (PNG, JPEG, etc.)
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert from [C, H, W] to [H, W, C]
    image_np = tensor.detach().cpu().permute(1, 2, 0).numpy()
    # Convert to uint8
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    # Save as PIL image
    Image.fromarray(image_np).save(path, format=format)


def create_test_image(size: Tuple[int, int] = (512, 512), 
                     channels: int = 3,
                     device: str = "cpu") -> torch.Tensor:
    """
    Create a random test image tensor
    
    Args:
        size: Image dimensions (H, W)
        channels: Number of channels 
        device: Device to create tensor on
        
    Returns:
        Random image tensor [1, C, H, W] in [0, 1] range
    """
    return torch.rand(1, channels, *size, device=device)


def setup_test_device() -> str:
    """
    Setup device for testing (prefer GPU if available)
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_vae_for_testing(model_name: str = "sd14", 
                        device: Optional[str] = None) -> Tuple[Any, str]:
    """
    Load VAE model for testing
    
    Args:
        model_name: Model identifier (sd14, sd15, etc.)
        device: Device to load on (auto-detected if None)
        
    Returns:
        Tuple of (vae_model, device_used)
    """
    if device is None:
        device = setup_test_device()
    
    try:
        vae, actual_device = VAELoader.load_sd_vae_simple(model_name, device)
        return vae, actual_device
    except Exception as e:
        raise RuntimeError(f"Failed to load VAE model: {e}")


def basic_optimization_loop(vae, initial_latents: torch.Tensor, 
                          target_image: torch.Tensor,
                          iterations: int = 50,
                          lr: float = 0.1) -> Tuple[torch.Tensor, list]:
    """
    Simple optimization loop for testing
    
    Args:
        vae: VAE model
        initial_latents: Initial latent representation
        target_image: Target image to reconstruct
        iterations: Number of optimization steps
        lr: Learning rate
        
    Returns:
        Tuple of (optimized_latents, loss_history)
    """
    # Freeze VAE decoder parameters
    for param in vae.decoder.parameters():
        param.requires_grad = False
    
    # Enable gradients for latents
    optimized_latents = initial_latents.clone().detach().requires_grad_(True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam([optimized_latents], lr=lr)
    
    # Optimization loop
    losses = []
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Decode latents to image
        outputs = vae.decode(optimized_latents)
        reconstructed = (outputs.sample / 2 + 0.5).clamp(0, 1)
        
        # Calculate MSE loss
        loss = torch.nn.functional.mse_loss(target_image, reconstructed)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return optimized_latents.detach(), losses


def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestImageGenerator:
    """
    Helper class for generating test images with known properties
    """
    
    @staticmethod
    def solid_color(color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                   size: Tuple[int, int] = (512, 512),
                   device: str = "cpu") -> torch.Tensor:
        """Generate solid color image"""
        image = torch.zeros(1, 3, *size, device=device)
        for i, c in enumerate(color):
            image[0, i, :, :] = c
        return image
    
    @staticmethod
    def gradient_horizontal(size: Tuple[int, int] = (512, 512),
                          device: str = "cpu") -> torch.Tensor:
        """Generate horizontal gradient image"""
        image = torch.zeros(1, 3, *size, device=device)
        gradient = torch.linspace(0, 1, size[1], device=device)
        for c in range(3):
            image[0, c, :, :] = gradient.unsqueeze(0)
        return image
    
    @staticmethod
    def checkerboard(square_size: int = 32,
                    size: Tuple[int, int] = (512, 512),
                    device: str = "cpu") -> torch.Tensor:
        """Generate checkerboard pattern"""
        image = torch.zeros(1, 3, *size, device=device)
        h, w = size
        
        for i in range(0, h, square_size):
            for j in range(0, w, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 1:
                    image[0, :, i:i+square_size, j:j+square_size] = 1.0
        
        return image


def print_test_header(test_name: str, width: int = 50) -> None:
    """Print formatted test section header"""
    print("=" * width)
    print(f"{test_name:^{width}}")
    print("=" * width)


def print_test_result(test_name: str, success: bool, details: str = "") -> None:
    """Print formatted test result"""
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"{test_name}: {status}")
    if details:
        print(f"  {details}")


def print_metrics_summary(metrics: Dict[str, float]) -> None:
    """Print formatted metrics summary"""
    print("\n📊 Metrics Summary:")
    for name, value in metrics.items():
        if name.endswith('_db'):
            print(f"  {name}: {value:.2f} dB")
        elif name.startswith('psnr'):
            print(f"  {name}: {value:.2f} dB")
        else:
            print(f"  {name}: {value:.6f}")