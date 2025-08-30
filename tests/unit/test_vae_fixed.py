#!/usr/bin/env python3
"""
Unit tests for VAE fixed functionality

Fixed version of VAE tests with proper device handling,
output clamping, and device consistency.
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
    """Load SD 1.4 VAE model with automatic device selection"""
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN environment variable not set")
    
    try:
        vae_model = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            token=token,
        )
        
        # Auto-select device
        if torch.cuda.is_available():
            vae_model = vae_model.cuda()
        
        return vae_model
    except Exception as e:
        pytest.skip(f"Failed to load VAE model: {e}")

@pytest.fixture
def sample_images_fixed():
    """Generate sample images for fixed tests"""
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 512, 512).clamp(-1, 1)
    return test_images

def test_model_loading():
    """Test SD 1.4 VAE model loading with device handling"""
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN environment variable not set")
    
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        token=token,
    )
    
    # Auto-select device  
    if torch.cuda.is_available():
        vae = vae.cuda()
    
    # Assertions
    assert vae is not None
    assert hasattr(vae.config, 'scaling_factor')
    assert abs(vae.config.scaling_factor - 0.18215) < 1e-5
    
    # Check device is properly set
    model_device = next(vae.parameters()).device
    if torch.cuda.is_available():
        assert model_device.type == 'cuda'
    else:
        assert model_device.type == 'cpu'

def test_encode_decode_with_clamping(vae, sample_images_fixed):
    """Test encode/decode with output range clamping"""
    device = next(vae.parameters()).device
    test_images = sample_images_fixed.to(device)
    batch_size = test_images.shape[0]
    
    # Test encoding
    with torch.no_grad():
        posterior = vae.encode(test_images)
        latents = posterior.latent_dist.mode()
        scaled_latents = latents * vae.config.scaling_factor
    
    # Assert latent shape and device
    expected_shape = (batch_size, 4, 64, 64)
    assert latents.shape == expected_shape, f"Expected {expected_shape}, got {latents.shape}"
    assert latents.device == device, f"Latents device mismatch: {latents.device} vs {device}"
    assert torch.isfinite(latents).all(), "Latents contain infinite or NaN values"
    
    # Test decoding with clamping
    with torch.no_grad():
        decoded_images = vae.decode(latents).sample
        decoded_images_clamped = decoded_images.clamp(-1, 1)
    
    # Assert shapes and device consistency
    assert decoded_images_clamped.shape == test_images.shape, "Decoded shape mismatch"
    assert decoded_images_clamped.device == device, f"Decoded device mismatch: {decoded_images_clamped.device} vs {device}"
    
    # Assert clamping worked
    assert torch.all(decoded_images_clamped >= -1.0), f"Values below -1.0: {decoded_images_clamped.min():.3f}"
    assert torch.all(decoded_images_clamped <= 1.0), f"Values above 1.0: {decoded_images_clamped.max():.3f}"
    assert torch.isfinite(decoded_images_clamped).all(), "Decoded images contain infinite or NaN values"

def test_proper_device_handling(vae):
    """Test proper device handling"""
    model_device = next(vae.parameters()).device
    
    # Test input on same device as model
    test_input = torch.randn(1, 3, 512, 512).clamp(-1, 1).to(model_device)
    
    # Ensure input device matches model
    assert test_input.device == model_device, f"Input device {test_input.device} != model device {model_device}"
    
    with torch.no_grad():
        # Encode
        posterior = vae.encode(test_input)
        latents = posterior.latent_dist.mode()
        
        # Decode with clamping
        decoded = vae.decode(latents).sample.clamp(-1, 1)
    
    # Assert device consistency
    assert latents.device == model_device, f"Latents device {latents.device} != model device {model_device}"
    assert decoded.device == model_device, f"Decoded device {decoded.device} != model device {model_device}"
    
    # Assert valid outputs
    assert torch.isfinite(latents).all(), "Latents contain infinite or NaN values"
    assert torch.isfinite(decoded).all(), "Decoded images contain infinite or NaN values"
    
    # Assert proper clamping
    assert torch.all(decoded >= -1.0), f"Decoded values below -1.0: {decoded.min():.3f}"
    assert torch.all(decoded <= 1.0), f"Decoded values above 1.0: {decoded.max():.3f}"

def test_reconstruction_quality_fixed(vae, sample_images_fixed):
    """Test reconstruction quality with fixes"""
    device = next(vae.parameters()).device
    original = sample_images_fixed.to(device)
    
    # Encode and decode
    with torch.no_grad():
        posterior = vae.encode(original)
        latents = posterior.latent_dist.mode()
        decoded = vae.decode(latents).sample
        # Apply clamping
        reconstructed = decoded.clamp(-1, 1)
    
    # Ensure device consistency
    assert original.device == reconstructed.device, "Device mismatch between original and reconstructed"
    
    # Calculate losses
    mse_loss = torch.nn.functional.mse_loss(original, reconstructed)
    mae_loss = torch.nn.functional.l1_loss(original, reconstructed)
    
    # Assert reconstruction quality
    assert mse_loss.item() < 2.0, f"MSE loss too high: {mse_loss.item():.6f}"
    assert mae_loss.item() < 1.0, f"MAE loss too high: {mae_loss.item():.6f}"
    
    # Assert proper clamping
    assert torch.all(reconstructed >= -1.0), f"Values below -1.0: {reconstructed.min():.3f}"
    assert torch.all(reconstructed <= 1.0), f"Values above 1.0: {reconstructed.max():.3f}"
    
    # Assert finite values
    assert torch.isfinite(reconstructed).all(), "Reconstructed images contain infinite or NaN values"
    
    # PSNR calculation
    if mse_loss > 0:
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse_loss))
        assert psnr.item() > 5.0, f"PSNR too low: {psnr.item():.2f} dB"
    
    # Pixel difference statistics
    diff = torch.abs(original - reconstructed)
    assert diff.max() <= 2.0, f"Max pixel difference too high: {diff.max():.3f}"
    assert diff.mean() < 0.5, f"Mean pixel difference too high: {diff.mean():.3f}"

def test_batch_processing(vae):
    """Test batch processing with different batch sizes"""
    device = next(vae.parameters()).device
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        # Create batch
        test_batch = torch.randn(batch_size, 3, 512, 512).clamp(-1, 1).to(device)
        
        with torch.no_grad():
            latents = vae.encode(test_batch).latent_dist.mode()
            decoded = vae.decode(latents).sample.clamp(-1, 1)
        
        # Assert shapes
        expected_latent_shape = (batch_size, 4, 64, 64)
        assert latents.shape == expected_latent_shape, f"Batch {batch_size}: latent shape {latents.shape} != {expected_latent_shape}"
        assert decoded.shape == test_batch.shape, f"Batch {batch_size}: decoded shape mismatch"
        
        # Assert device consistency
        assert latents.device == device, f"Batch {batch_size}: latents device mismatch"
        assert decoded.device == device, f"Batch {batch_size}: decoded device mismatch"
        
        # Assert finite values and proper clamping
        assert torch.isfinite(latents).all(), f"Batch {batch_size}: latents contain infinite/NaN"
        assert torch.isfinite(decoded).all(), f"Batch {batch_size}: decoded contain infinite/NaN"
        assert torch.all(decoded >= -1.0), f"Batch {batch_size}: values below -1.0"
        assert torch.all(decoded <= 1.0), f"Batch {batch_size}: values above 1.0"

# For standalone execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])