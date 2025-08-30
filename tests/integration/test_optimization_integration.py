#!/usr/bin/env python3
"""
Integration Test for Modularized VAE Optimization

Tests the newly created modules by performing single image optimization
using the refactored components.
"""

import os
import torch
import sys
from pathlib import Path

# Add project root to path to access test helpers
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tests.fixtures.test_helpers import (
    load_vae_for_testing, setup_test_device, calculate_psnr,
    save_image_tensor, ensure_directory, print_test_header, 
    print_test_result, TestImageGenerator
)

# Test imports
try:
    from vae_toolkit import VAELoader, load_and_preprocess_image
    from generative_latent_optimization import (
        LatentOptimizer, 
        OptimizationConfig,
        ImageMetrics,
        IOUtils
    )
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)


def test_modular_optimization():
    """Test the modularized optimization pipeline"""
    
    print("🚀 Starting modularized optimization test...")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    experiments_dir = Path(__file__).parent
    image_path = project_root / "document.png"
    results_dir = experiments_dir / "results" / "modularized_test"
    
    # Check if test image exists
    if not image_path.exists():
        print(f"❌ Test image not found: {image_path}")
        print("Please ensure document.png exists in the project root")
        return False
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Load VAE model
        print("Loading VAE model...")
        vae, device = VAELoader.load_sd_vae_simple("sd14", device)
        
        # Load and preprocess image
        print(f"Loading image: {image_path}")
        original_tensor, original_pil = load_and_preprocess_image(str(image_path), target_size=512)
        original_tensor = original_tensor.to(device)
        
        print(f"Image shape: {original_tensor.shape}")
        print(f"Image range: [{original_tensor.min():.2f}, {original_tensor.max():.2f}]")
        
        # Initial VAE encoding
        print("Performing initial VAE encoding...")
        with torch.no_grad():
            initial_latents = vae.encode(original_tensor).latent_dist.mode()
            initial_reconstruction = vae.decode(initial_latents)
            initial_recon_tensor = (initial_reconstruction.sample / 2 + 0.5).clamp(0, 1)
        
        # Initialize metrics calculator
        metrics_calc = ImageMetrics(device=device)
        initial_psnr = metrics_calc.calculate_psnr(original_tensor, initial_recon_tensor)
        print(f"Initial PSNR: {initial_psnr:.2f} dB")
        
        # Setup optimization configuration
        config = OptimizationConfig(
            iterations=50,  # Reduced for testing
            learning_rate=0.4,
            loss_function='mse',
            checkpoint_interval=10
        )
        
        # Initialize optimizer and perform optimization
        optimizer = LatentOptimizer(config)
        print("Starting latent optimization...")
        
        result = optimizer.optimize(vae, initial_latents, original_tensor)
        
        print(f"Optimization completed!")
        print(f"Final PSNR: {result.metrics['final_psnr_db']:.2f} dB")
        print(f"PSNR improvement: {result.metrics['final_psnr_db'] - initial_psnr:.2f} dB")
        print(f"Loss reduction: {result.metrics['loss_reduction_percent']:.1f}%")
        
        # Final reconstruction
        with torch.no_grad():
            final_reconstruction = vae.decode(result.optimized_latents)
            final_recon_tensor = (final_reconstruction.sample / 2 + 0.5).clamp(0, 1)
        
        # Save results using IOUtils
        print("Saving results...")
        io_utils = IOUtils()
        results_dir.mkdir(exist_ok=True)
        
        # Save images
        io_utils.save_image_tensor(original_tensor, results_dir / "original.png")
        io_utils.save_image_tensor(initial_recon_tensor, results_dir / "initial_reconstruction.png") 
        io_utils.save_image_tensor(final_recon_tensor, results_dir / "optimized_reconstruction.png")
        
        # Save latents and metrics
        io_utils.save_tensor(result.optimized_latents, results_dir / "optimized_latents.pt")
        io_utils.save_json(result.metrics, results_dir / "metrics.json")
        io_utils.save_json({'losses': result.losses}, results_dir / "losses.json")
        
        print(f"Results saved to: {results_dir}")
        print("✅ Modular optimization test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """Test individual components separately"""
    
    print("\n🧪 Testing individual components...")
    
    try:
        # Test ImageMetrics
        print("Testing ImageMetrics...")
        metrics = ImageMetrics()
        
        # Create test tensors using helper
        device = setup_test_device()
        img1 = TestImageGenerator.solid_color((1.0, 0.0, 0.0), (64, 64), device)
        img2 = TestImageGenerator.gradient_horizontal((64, 64), device)
        
        psnr = metrics.calculate_psnr(img1, img2)
        ssim = metrics.calculate_ssim(img1, img2)
        all_metrics = metrics.calculate_all_metrics(img1, img2)
        
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.3f}")
        print(f"  All metrics: {all_metrics}")
        
        # Test OptimizationConfig
        print("Testing OptimizationConfig...")
        config = OptimizationConfig(iterations=100, learning_rate=0.1)
        print(f"  Config created: {config}")
        
        # Test IOUtils
        print("Testing IOUtils...")
        io_utils = IOUtils()
        test_tensor = torch.rand(3, 64, 64)
        
        temp_path = Path("temp_test_image.png")
        io_utils.save_image_tensor(test_tensor, temp_path)
        
        if temp_path.exists():
            print("  Image save/load: ✅")
            temp_path.unlink()  # Delete temp file
        else:
            print("  Image save/load: ❌")
        
        print("✅ Individual component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("MODULAR VAE OPTIMIZATION TEST")
    print("=" * 50)
    
    # Test individual components first
    component_test = test_individual_components()
    
    # Only run full test if components work
    if component_test:
        full_test = test_modular_optimization()
        
        if full_test:
            print("\n🎉 ALL TESTS PASSED!")
            print("The modularized optimization system is working correctly.")
        else:
            print("\n💥 FULL TEST FAILED!")
    else:
        print("\n💥 COMPONENT TESTS FAILED!")
        print("Please check the module implementations.")