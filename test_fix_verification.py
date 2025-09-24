#!/usr/bin/env python3
"""
Test the image range consistency fix
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

from generative_latent_optimization.optimization.latent_optimizer import LatentOptimizer, OptimizationConfig

def test_single_image_with_fix():
    """Test single image optimization with the range consistency fix"""
    print("🔍 Testing single image optimization with range fix...")
    
    try:
        # Load VAE model
        from vae_toolkit import VAELoader
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vae, device = VAELoader.load_sd_vae_simple('sd15', device)
        print(f"VAE loaded on device: {device}")
        
        # Create test image
        test_image = torch.ones(1, 3, 512, 512) * 0.5
        test_image = test_image.to(device)
        
        # Test MSE optimization
        config = OptimizationConfig(
            iterations=20,
            learning_rate=0.05,
            loss_function='mse',
            device=device
        )
        
        optimizer = LatentOptimizer(config)
        
        # Manual implementation to verify fix
        print("🔄 Manual optimization with range consistency...")
        
        # Convert original image to [0,1] range (simulating the fix)
        original_tensor_raw = test_image * 2 - 1  # [0,1] → [-1,1] (simulating vae_toolkit format)
        target_tensor = (original_tensor_raw + 1.0) / 2.0  # [-1,1] → [0,1] (our fix)
        
        print(f"Original tensor range: [{original_tensor_raw.min():.3f}, {original_tensor_raw.max():.3f}]")
        print(f"Target tensor range: [{target_tensor.min():.3f}, {target_tensor.max():.3f}]")
        
        # Initial encoding
        with torch.no_grad():
            initial_latents = vae.encode(original_tensor_raw).latent_dist.mode()
            initial_reconstruction = vae.decode(initial_latents)
            initial_recon_tensor = (initial_reconstruction.sample / 2 + 0.5).clamp(0, 1)
        
        print(f"Initial recon range: [{initial_recon_tensor.min():.3f}, {initial_recon_tensor.max():.3f}]")
        
        # Calculate initial PSNR with consistent ranges
        initial_mse = torch.nn.functional.mse_loss(target_tensor, initial_recon_tensor)
        initial_psnr = 10 * torch.log10(1.0 / initial_mse)
        print(f"Initial PSNR (consistent): {initial_psnr.item():.6f} dB")
        
        # Optimization
        result = optimizer.optimize(vae, initial_latents, target_tensor)
        
        # Final reconstruction 
        with torch.no_grad():
            final_reconstruction = vae.decode(result.optimized_latents)
            final_recon_tensor = (final_reconstruction.sample / 2 + 0.5).clamp(0, 1)
        
        # Calculate final PSNR with consistent ranges
        final_mse = torch.nn.functional.mse_loss(target_tensor, final_recon_tensor)
        final_psnr = 10 * torch.log10(1.0 / final_mse)
        print(f"Final PSNR (consistent): {final_psnr.item():.6f} dB")
        
        # PSNR improvement
        psnr_improvement = final_psnr.item() - initial_psnr.item()
        print(f"PSNR improvement: {psnr_improvement:.6f} dB")
        
        # Compare with old method (inconsistent ranges)
        print(f"\n🔬 Comparing with old inconsistent method:")
        initial_mse_old = torch.nn.functional.mse_loss(original_tensor_raw, initial_recon_tensor)
        final_mse_old = torch.nn.functional.mse_loss(original_tensor_raw, final_recon_tensor)
        
        initial_psnr_old = 10 * torch.log10(1.0 / initial_mse_old)
        final_psnr_old = 10 * torch.log10(1.0 / final_mse_old)
        psnr_improvement_old = final_psnr_old.item() - initial_psnr_old.item()
        
        print(f"Old method PSNR improvement: {psnr_improvement_old:.6f} dB")
        print(f"Difference: {psnr_improvement - psnr_improvement_old:.6f} dB")
        
        print(f"\n📊 Results Summary:")
        print(f"  MSE Loss reduction: {result.metrics.get('loss_reduction_percent', 0):.2f}%")
        print(f"  PSNR improvement (fixed): {psnr_improvement:.6f} dB")
        print(f"  PSNR improvement (old): {psnr_improvement_old:.6f} dB")
        
        if psnr_improvement > 0:
            print("✅ Fixed method shows positive PSNR improvement!")
        if psnr_improvement > psnr_improvement_old:
            print("✅ Fixed method shows better results than old method!")
            
        return psnr_improvement, psnr_improvement_old
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Test the range consistency fix"""
    print("🎯 TESTING IMAGE RANGE CONSISTENCY FIX")
    print("=" * 60)
    
    fixed_improvement, old_improvement = test_single_image_with_fix()
    
    print("\n" + "=" * 60)
    print("🔍 FIX VERIFICATION SUMMARY:")
    
    if fixed_improvement is not None and old_improvement is not None:
        print(f"  Fixed method PSNR improvement: {fixed_improvement:.6f} dB")
        print(f"  Old method PSNR improvement: {old_improvement:.6f} dB")
        print(f"  Improvement difference: {fixed_improvement - old_improvement:.6f} dB")
        
        if fixed_improvement > 0 and old_improvement < 0:
            print("🎉 SUCCESS: Fix resolves negative PSNR improvement issue!")
        elif fixed_improvement > old_improvement:
            print("✅ SUCCESS: Fix improves PSNR calculation accuracy!")
        else:
            print("⚠️  Fix doesn't resolve the issue completely")
    else:
        print("❌ Test failed")

if __name__ == "__main__":
    main()