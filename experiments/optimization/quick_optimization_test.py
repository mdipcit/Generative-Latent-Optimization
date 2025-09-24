#!/usr/bin/env python3
"""
Quick test of the integrated loss visualization functionality
"""

import sys
import torch
from pathlib import Path

# Add project root to path to access test helpers
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tests.fixtures.test_helpers import (
    load_vae_for_testing, setup_test_device, calculate_psnr,
    basic_optimization_loop, save_image_tensor, ensure_directory
)
from vae_toolkit import load_and_preprocess_image
from single_image_optimization import create_comparison_grid, create_loss_graphs

# Override the main function for quick testing
def quick_test():
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    experiments_dir = Path(__file__).parent.parent
    image_path = project_root / "document.png"
    results_dir = experiments_dir / "results" / "quick_test"
    ensure_directory(results_dir)
    
    # Device setup
    device = setup_test_device()
    print(f"Using device: {device}")
    
    # Load VAE model
    print("Loading VAE model...")
    vae, device = load_vae_for_testing("sd14", device)
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    original_tensor, original_pil = load_and_preprocess_image(str(image_path), target_size=512)
    original_tensor = original_tensor.to(device)
    
    # Initial VAE encoding
    print("Performing initial VAE encoding...")
    with torch.no_grad():
        initial_latents = vae.encode(original_tensor).latent_dist.mode()
        initial_reconstruction = vae.decode(initial_latents)
        initial_recon_tensor = (initial_reconstruction.sample / 2 + 0.5).clamp(0, 1)
    
    # Calculate initial PSNR
    initial_psnr = calculate_psnr(original_tensor, initial_recon_tensor)
    print(f"Initial PSNR: {initial_psnr:.2f} dB")
    
    # Quick optimization (only 20 iterations for testing)
    print("Starting quick latent optimization (20 iterations)...")
    optimized_latents, losses = basic_optimization_loop(
        vae, initial_latents, original_tensor, 
        iterations=20, lr=0.1  # Reduced iterations
    )
    
    # Final reconstruction
    with torch.no_grad():
        final_reconstruction = vae.decode(optimized_latents)
        final_recon_tensor = (final_reconstruction.sample / 2 + 0.5).clamp(0, 1)
    
    # Calculate final PSNR
    final_psnr = calculate_psnr(original_tensor, final_recon_tensor)
    print(f"Final PSNR: {final_psnr:.2f} dB")
    print(f"PSNR improvement: {final_psnr - initial_psnr:.2f} dB")
    
    # Save results with loss graphs
    print("Saving results...")
    
    # Save individual images
    save_image_tensor(original_tensor, results_dir / "original.png")
    save_image_tensor(initial_recon_tensor, results_dir / "initial_reconstruction.png")
    save_image_tensor(final_recon_tensor, results_dir / "optimized_reconstruction.png")
    
    # Save comparison grid
    create_comparison_grid(
        original_tensor, initial_recon_tensor, final_recon_tensor,
        results_dir / "comparison_grid.png"
    )
    
    # Create loss visualization graphs - THIS IS THE KEY TEST!
    print("Creating loss visualization graphs...")
    create_loss_graphs(losses, results_dir)
    
    print(f"Quick test completed! Check {results_dir} for all outputs including loss graphs.")


if __name__ == "__main__":
    quick_test()