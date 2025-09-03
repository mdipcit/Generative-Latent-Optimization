#!/usr/bin/env python3
"""
Test MSE optimization with actual VAE model on single image
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

from generative_latent_optimization.optimization.latent_optimizer import LatentOptimizer, OptimizationConfig

def test_mse_optimization_with_vae():
    """Test MSE optimization using actual VAE model"""
    print("🔍 Testing MSE optimization with VAE model...")
    
    try:
        # Load VAE model
        from vae_toolkit import VAELoader, load_and_preprocess_image
        
        print("Loading VAE model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vae, device = VAELoader.load_sd_vae_simple('sd15', device)
        print(f"VAE loaded on device: {device}")
        
        # Create a simple test image (solid color for predictable results)
        test_image = torch.ones(1, 3, 512, 512) * 0.5  # Gray image
        test_image = test_image.to(device)
        
        print(f"Test image shape: {test_image.shape}")
        print(f"Test image range: [{test_image.min():.3f}, {test_image.max():.3f}]")
        
        # Test MSE optimization
        config = OptimizationConfig(
            iterations=10,
            learning_rate=0.05,
            loss_function='mse',
            device=device
        )
        
        # Encode image to get initial latents
        with torch.no_grad():
            initial_latents = vae.encode(test_image * 2 - 1).latent_dist.sample()
            initial_latents = initial_latents * vae.config.scaling_factor
        
        print(f"Initial latents shape: {initial_latents.shape}")
        
        optimizer = LatentOptimizer(config)
        print("\n🔄 Running MSE optimization...")
        
        result = optimizer.optimize(vae, initial_latents, test_image)
        
        print(f"\n📊 MSE Optimization Results:")
        print(f"  Initial loss: {result.initial_loss:.8f}")
        print(f"  Final loss: {result.final_loss:.8f}")
        print(f"  Loss reduction: {result.metrics.get('loss_reduction_percent', 0):.2f}%")
        print(f"  Final PSNR: {result.metrics.get('final_psnr_db', 0):.6f} dB")
        print(f"  Iterations: {result.metrics.get('total_iterations', 0)}")
        
        # Calculate actual PSNR improvement manually
        with torch.no_grad():
            # Encode original image
            initial_latents = vae.encode(test_image * 2 - 1).latent_dist.sample()
            initial_latents = initial_latents * vae.config.scaling_factor
            
            # Decode with initial latents
            initial_outputs = vae.decode(initial_latents)
            initial_recon = (initial_outputs.sample / 2 + 0.5).clamp(0, 1)
            
            # Decode with optimized latents
            final_outputs = vae.decode(result.optimized_latents)
            final_recon = (final_outputs.sample / 2 + 0.5).clamp(0, 1)
            
            # Calculate PSNR
            initial_mse = torch.nn.functional.mse_loss(test_image, initial_recon)
            final_mse = torch.nn.functional.mse_loss(test_image, final_recon)
            
            initial_psnr = 10 * torch.log10(1.0 / initial_mse)
            final_psnr = 10 * torch.log10(1.0 / final_mse)
            
            print(f"\n🔬 Manual PSNR Calculation:")
            print(f"  Initial MSE: {initial_mse.item():.8f}")
            print(f"  Final MSE: {final_mse.item():.8f}")
            print(f"  Initial PSNR: {initial_psnr.item():.6f} dB")
            print(f"  Final PSNR: {final_psnr.item():.6f} dB")
            print(f"  PSNR improvement: {final_psnr.item() - initial_psnr.item():.6f} dB")
            
            # Check if loss matches MSE
            expected_loss = final_mse.item()
            actual_loss = result.final_loss
            print(f"\n🧪 Loss Consistency Check:")
            print(f"  Expected final loss (MSE): {expected_loss:.8f}")
            print(f"  Actual final loss: {actual_loss:.8f}")
            print(f"  Difference: {abs(expected_loss - actual_loss):.8f}")
            
            if abs(expected_loss - actual_loss) < 1e-6:
                print("✅ Loss calculation is consistent")
            else:
                print("❌ Loss calculation inconsistency detected")
                
            return final_psnr.item() - initial_psnr.item(), result.metrics.get('loss_reduction_percent', 0)
        
    except ImportError as e:
        print(f"❌ VAE toolkit not available: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error during VAE optimization: {e}")
        return None, None

def test_psnr_optimization_with_vae():
    """Test PSNR optimization using actual VAE model for comparison"""
    print("\n🔍 Testing PSNR optimization with VAE model...")
    
    try:
        # Load VAE model
        from vae_toolkit import VAELoader
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vae, device = VAELoader.load_sd_vae_simple('sd15', device)
        
        # Create the same test image
        test_image = torch.ones(1, 3, 512, 512) * 0.5
        test_image = test_image.to(device)
        
        # Encode image to get initial latents
        with torch.no_grad():
            initial_latents = vae.encode(test_image * 2 - 1).latent_dist.sample()
            initial_latents = initial_latents * vae.config.scaling_factor
        
        # Test PSNR optimization
        config = OptimizationConfig(
            iterations=10,
            learning_rate=0.05,
            loss_function='psnr',
            device=device
        )
        
        optimizer = LatentOptimizer(config)
        print("🔄 Running PSNR optimization...")
        
        result = optimizer.optimize(vae, initial_latents, test_image)
        
        print(f"\n📊 PSNR Optimization Results:")
        print(f"  Initial loss: {result.initial_loss:.8f}")
        print(f"  Final loss: {result.final_loss:.8f}")
        print(f"  Loss reduction: {result.metrics.get('loss_reduction_percent', 0):.2f}%")
        print(f"  Final PSNR: {result.metrics.get('final_psnr_db', 0):.6f} dB")
        
        return result.metrics.get('final_psnr_db', 0), result.metrics.get('loss_reduction_percent', 0)
        
    except Exception as e:
        print(f"❌ Error during PSNR optimization: {e}")
        return None, None

def main():
    """Compare MSE vs PSNR optimization with actual VAE"""
    print("🎯 MSE vs PSNR Optimization with VAE Comparison")
    print("=" * 60)
    
    # Test MSE optimization
    mse_psnr_improvement, mse_loss_reduction = test_mse_optimization_with_vae()
    
    # Test PSNR optimization for comparison
    psnr_final_value, psnr_loss_reduction = test_psnr_optimization_with_vae()
    
    print("\n" + "=" * 60)
    print("🔍 VAE OPTIMIZATION COMPARISON:")
    
    if mse_psnr_improvement is not None and psnr_final_value is not None:
        print(f"  MSE Loss  → PSNR improvement: {mse_psnr_improvement:.6f} dB")
        print(f"  MSE Loss  → Loss reduction: {mse_loss_reduction:.2f}%")
        print(f"  PSNR Loss → Final PSNR: {psnr_final_value:.6f} dB")
        print(f"  PSNR Loss → Loss reduction: {psnr_loss_reduction:.2f}%")
        
        print("\n🎯 RECOMMENDATION:")
        if mse_loss_reduction > psnr_loss_reduction:
            print("✅ Use MSE optimization - better loss tracking and equivalent PSNR improvement")
        else:
            print("⚡ Both methods work, MSE provides cleaner implementation")
    else:
        print("❌ Tests failed - VAE model issues")

if __name__ == "__main__":
    main()