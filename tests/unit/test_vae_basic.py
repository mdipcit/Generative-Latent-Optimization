#!/usr/bin/env python3
"""
Unit tests for VAE basic functionality

Tests basic VAE operations including model loading, encode/decode,
device compatibility, and scaling factor validation.
"""

import os
import torch
from diffusers import AutoencoderKL
import numpy as np
import time
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def vae():
    """Load SD 1.4 VAE model for testing"""
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN environment variable not set")
    
    try:
        vae_model = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            token=token,
        )
        return vae_model
    except Exception as e:
        pytest.skip(f"Failed to load VAE model: {e}")

@pytest.fixture
def sample_images():
    """Generate sample images for testing"""
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 512, 512) * 2 - 1
    return test_images.clamp(-1, 1)

def test_model_loading():
    """Test SD 1.4 VAE model loading"""
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN environment variable not set")
    
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        token=token,
    )
    
    # Assertions instead of prints
    assert vae is not None
    assert hasattr(vae.config, 'scaling_factor')
    assert hasattr(vae, 'encode')
    assert hasattr(vae, 'decode')
    assert abs(vae.config.scaling_factor - 0.18215) < 1e-5

def test_encode_decode_shapes(vae, sample_images):
    """Test encode/decode shape validation"""
    batch_size = sample_images.shape[0]
    
    # Ensure model and input are on same device
    device = next(vae.parameters()).device
    test_images = sample_images.to(device)
    
    # Test encoding
    with torch.no_grad():
        posterior = vae.encode(test_images)
        latents = posterior.latent_dist.mode()
        scaled_latents = latents * vae.config.scaling_factor
    
    # Assert latent shape
    expected_shape = (batch_size, 4, 64, 64)
    assert latents.shape == expected_shape, f"Expected {expected_shape}, got {latents.shape}"
    
    # Assert latent ranges are reasonable
    assert torch.isfinite(latents).all(), "Latents contain infinite or NaN values"
    assert torch.isfinite(scaled_latents).all(), "Scaled latents contain infinite or NaN values"
    
    # Test decoding
    with torch.no_grad():
        decoded_images = vae.decode(latents).sample
    
    # Assert decoded shape
    assert decoded_images.shape == test_images.shape, f"Decoded shape mismatch"
    assert torch.isfinite(decoded_images).all(), "Decoded images contain infinite or NaN values"

def test_reconstruction_quality(vae, sample_images):
    """Test reconstruction quality"""
    device = next(vae.parameters()).device
    original = sample_images.to(device)
    
    # Encode and decode
    with torch.no_grad():
        posterior = vae.encode(original)
        latents = posterior.latent_dist.mode()
        reconstructed = vae.decode(latents).sample
    
    # Calculate losses
    mse_loss = torch.nn.functional.mse_loss(original, reconstructed)
    mae_loss = torch.nn.functional.l1_loss(original, reconstructed)
    
    # Assert reconstruction quality
    assert mse_loss.item() < 2.0, f"MSE loss too high: {mse_loss.item():.6f}"
    assert mae_loss.item() < 1.0, f"MAE loss too high: {mae_loss.item():.6f}"
    
    # Assert output range is reasonable (allow some tolerance beyond [-1,1])
    assert torch.all(reconstructed >= -1.5), f"Output too negative: {reconstructed.min():.3f}"
    assert torch.all(reconstructed <= 1.5), f"Output too positive: {reconstructed.max():.3f}"
    
    # Assert finite values
    assert torch.isfinite(reconstructed).all(), "Reconstructed images contain infinite or NaN values"
    
    # PSNR should be reasonable
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse_loss))
    assert psnr.item() > 10.0, f"PSNR too low: {psnr.item():.2f} dB"

def test_different_input_sizes(vae):
    """Test VAE with different input sizes"""
    device = next(vae.parameters()).device
    
    test_sizes = [
        (1, 3, 512, 512),   # Standard size
        (1, 3, 256, 256),   # Smaller size
        (4, 3, 512, 512),   # Batch size 4
    ]
    
    for size in test_sizes:
        test_input = torch.randn(*size).clamp(-1, 1).to(device)
        
        with torch.no_grad():
            latents = vae.encode(test_input).latent_dist.mode()
            decoded = vae.decode(latents).sample
        
        # VAE downsamples by factor of 8
        expected_latent_h = size[2] // 8
        expected_latent_w = size[3] // 8
        expected_latent_shape = (size[0], 4, expected_latent_h, expected_latent_w)
        
        assert latents.shape == expected_latent_shape, f"Size {size}: Expected {expected_latent_shape}, got {latents.shape}"
        assert decoded.shape == size, f"Size {size}: Decoded shape mismatch"
        
        # Assert finite values
        assert torch.isfinite(latents).all(), f"Size {size}: Latents contain infinite or NaN"
        assert torch.isfinite(decoded).all(), f"Size {size}: Decoded contains infinite or NaN"

def test_device_compatibility(vae):
    """Test CPU/CUDA compatibility"""
    test_input = torch.randn(1, 3, 512, 512).clamp(-1, 1)
    
    # CPU Test
    vae_cpu = vae.to('cpu')
    test_input_cpu = test_input.to('cpu')
    
    with torch.no_grad():
        latents_cpu = vae_cpu.encode(test_input_cpu).latent_dist.mode()
        decoded_cpu = vae_cpu.decode(latents_cpu).sample
    
    # Assert device consistency
    assert latents_cpu.device.type == 'cpu', "Latents not on CPU"
    assert decoded_cpu.device.type == 'cpu', "Decoded not on CPU"
    
    # Assert shapes and finite values
    assert latents_cpu.shape == (1, 4, 64, 64)
    assert decoded_cpu.shape == test_input_cpu.shape
    assert torch.isfinite(latents_cpu).all()
    assert torch.isfinite(decoded_cpu).all()
    
    # CUDA test if available
    if torch.cuda.is_available():
        vae_cuda = vae.to('cuda')
        test_input_cuda = test_input.to('cuda')
        
        with torch.no_grad():
            latents_cuda = vae_cuda.encode(test_input_cuda).latent_dist.mode()
            decoded_cuda = vae_cuda.decode(latents_cuda).sample
        
        # Assert device consistency
        assert latents_cuda.device.type == 'cuda', "Latents not on CUDA"
        assert decoded_cuda.device.type == 'cuda', "Decoded not on CUDA"
        
        # Assert shapes and finite values
        assert latents_cuda.shape == (1, 4, 64, 64)
        assert decoded_cuda.shape == test_input_cuda.shape
        assert torch.isfinite(latents_cuda).all()
        assert torch.isfinite(decoded_cuda).all()

def test_scaling_factor_validation(vae):
    """Test SD 1.4 scaling factor accuracy"""
    device = next(vae.parameters()).device
    test_input = torch.randn(1, 3, 512, 512).clamp(-1, 1).to(device)
    
    # Test scaling factor value
    expected_scaling = 0.18215
    actual_scaling = vae.config.scaling_factor
    
    assert abs(actual_scaling - expected_scaling) < 1e-5, f"Scaling factor mismatch: {actual_scaling} vs {expected_scaling}"
    
    # Test scaling application
    with torch.no_grad():
        posterior = vae.encode(test_input)
        latents_mode = posterior.latent_dist.mode()
        scaled_latents = latents_mode * actual_scaling
    
    # Assert finite values
    assert torch.isfinite(latents_mode).all(), "Raw latents contain infinite or NaN"
    assert torch.isfinite(scaled_latents).all(), "Scaled latents contain infinite or NaN"
    
    # Assert reasonable ranges
    assert latents_mode.abs().max() < 100, f"Raw latent values too large: {latents_mode.abs().max()}"
    assert scaled_latents.abs().max() < 20, f"Scaled latent values too large: {scaled_latents.abs().max()}"

# Remove the main() function - not needed for pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])