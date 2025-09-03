#!/usr/bin/env python3
"""
Debug PSNR optimization to find why it fails to improve PSNR values
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

from generative_latent_optimization.optimization.latent_optimizer import LatentOptimizer, OptimizationConfig

def test_psnr_calculation_consistency():
    """Test if optimization and evaluation PSNR calculations are consistent"""
    print("🔍 Testing PSNR calculation consistency...")
    
    # Create test images
    torch.manual_seed(42)
    img1 = torch.rand(1, 3, 64, 64)  # Small size for debugging
    img2 = torch.rand(1, 3, 64, 64)
    
    # Optimization PSNR calculation (with epsilon)
    mse_opt = torch.nn.functional.mse_loss(img1, img2, reduction='none').mean(dim=[1, 2, 3])
    epsilon = 1e-8
    psnr_opt = -10 * torch.log10(mse_opt + epsilon)
    
    # Evaluation PSNR calculation (without epsilon)
    mse_eval = torch.nn.functional.mse_loss(img1, img2)
    psnr_eval = 10 * torch.log10(1.0 / mse_eval)
    
    print(f"MSE (optimization): {mse_opt.item():.8f}")
    print(f"MSE (evaluation): {mse_eval.item():.8f}")
    print(f"PSNR (optimization): {psnr_opt.item():.6f}")
    print(f"PSNR (evaluation): {psnr_eval.item():.6f}")
    print(f"Difference: {abs(psnr_opt.item() - psnr_eval.item()):.6f}")
    print()
    
    return psnr_opt.item(), psnr_eval.item()

def test_gradient_flow():
    """Test if gradients flow properly through PSNR loss"""
    print("🔍 Testing gradient flow through PSNR loss...")
    
    # Create test setup
    torch.manual_seed(42)
    target = torch.rand(1, 3, 64, 64)
    
    # Create latents that need gradients
    latents = torch.rand(1, 4, 8, 8, requires_grad=True)
    
    # Mock a simple "decoder" (just reshape and interpolate)
    def mock_decode(latents):
        # Simple upsampling as mock decoder
        decoded = torch.nn.functional.interpolate(latents, size=(64, 64), mode='bilinear')
        # Convert from 4 channels to 3 channels to match target
        if decoded.shape[1] != target.shape[1]:
            # Use linear layer to convert channels
            channel_converter = torch.nn.Linear(decoded.shape[1], target.shape[1])
            decoded = channel_converter(decoded.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return decoded
    
    # Test PSNR loss calculation
    reconstructed = mock_decode(latents)
    
    # Calculate PSNR loss
    mse = torch.nn.functional.mse_loss(target, reconstructed, reduction='none').mean(dim=[1, 2, 3])
    epsilon = 1e-8
    psnr = -10 * torch.log10(mse + epsilon)
    loss = -psnr  # Minimize negative PSNR = maximize PSNR
    
    print(f"Initial MSE: {mse.item():.8f}")
    print(f"Initial PSNR: {psnr.item():.6f}")
    print(f"Initial Loss: {loss.item():.6f}")
    
    # Check if gradients exist
    loss.backward()
    grad_norm = latents.grad.norm().item() if latents.grad is not None else 0
    print(f"Gradient norm: {grad_norm:.8f}")
    
    if grad_norm > 0:
        print("✅ Gradients are flowing")
    else:
        print("❌ No gradients detected!")
    
    return grad_norm > 1e-10

def test_epsilon_impact():
    """Test impact of epsilon on PSNR calculation"""
    print("🔍 Testing epsilon impact on PSNR...")
    
    # Test with various MSE values
    mse_values = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    epsilon = 1e-8
    
    print("MSE\t\tPSNR (no ε)\tPSNR (with ε)\tDifference")
    print("-" * 60)
    
    for mse in mse_values:
        mse_tensor = torch.tensor(mse)
        
        # Without epsilon
        psnr_no_eps = -10 * torch.log10(mse_tensor)
        
        # With epsilon
        psnr_with_eps = -10 * torch.log10(mse_tensor + epsilon)
        
        diff = abs(psnr_no_eps.item() - psnr_with_eps.item())
        
        print(f"{mse:.2e}\t{psnr_no_eps.item():.4f}\t\t{psnr_with_eps.item():.4f}\t\t{diff:.6f}")
    
    print()

def test_simple_optimization():
    """Test simple PSNR optimization without VAE"""
    print("🔍 Testing simple PSNR optimization...")
    
    # Create target and initial reconstruction
    torch.manual_seed(42)
    target = torch.rand(1, 3, 64, 64)
    initial_recon = torch.rand(1, 3, 64, 64, requires_grad=True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam([initial_recon], lr=0.01)
    
    # Calculate initial PSNR
    with torch.no_grad():
        initial_mse = torch.nn.functional.mse_loss(target, initial_recon)
        initial_psnr = 10 * torch.log10(1.0 / initial_mse)
    
    print(f"Initial PSNR: {initial_psnr.item():.6f}")
    
    # Optimization loop
    losses = []
    for i in range(10):
        optimizer.zero_grad()
        
        # Calculate PSNR loss (same as in LatentOptimizer)
        mse = torch.nn.functional.mse_loss(target, initial_recon, reduction='none').mean(dim=[1, 2, 3])
        epsilon = 1e-8
        psnr = -10 * torch.log10(mse + epsilon)
        loss = -psnr  # Minimize negative PSNR
        
        losses.append(loss.item())
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        if i % 2 == 0:
            print(f"Iteration {i}: Loss = {loss.item():.8f}, PSNR = {psnr.item():.6f}")
    
    # Calculate final PSNR
    with torch.no_grad():
        final_mse = torch.nn.functional.mse_loss(target, initial_recon)
        final_psnr = 10 * torch.log10(1.0 / final_mse)
    
    print(f"Final PSNR: {final_psnr.item():.6f}")
    print(f"PSNR improvement: {final_psnr.item() - initial_psnr.item():.6f}")
    
    # Check loss reduction (corrected for negative losses)
    if losses[0] < 0:  # PSNR losses are negative
        loss_reduction = ((losses[0] - losses[-1]) / abs(losses[0])) * 100
    else:
        loss_reduction = 0.0
    
    print(f"Loss reduction: {loss_reduction:.2f}%")
    
    return final_psnr.item() - initial_psnr.item(), loss_reduction

def test_mse_optimization():
    """Test MSE optimization for PSNR improvement"""
    print("🔍 Testing MSE optimization for PSNR improvement...")
    
    # Create target and initial reconstruction
    torch.manual_seed(42)
    target = torch.rand(1, 3, 64, 64)
    initial_recon = torch.rand(1, 3, 64, 64, requires_grad=True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam([initial_recon], lr=0.01)
    
    # Calculate initial metrics
    with torch.no_grad():
        initial_mse = torch.nn.functional.mse_loss(target, initial_recon)
        initial_psnr = 10 * torch.log10(1.0 / initial_mse)
    
    print(f"Initial MSE: {initial_mse.item():.8f}")
    print(f"Initial PSNR: {initial_psnr.item():.6f}")
    
    # Optimization loop
    losses = []
    for i in range(10):
        optimizer.zero_grad()
        
        # Calculate MSE loss (directly minimize MSE)
        mse_loss = torch.nn.functional.mse_loss(target, initial_recon)
        
        losses.append(mse_loss.item())
        
        # Backprop
        mse_loss.backward()
        optimizer.step()
        
        if i % 2 == 0:
            with torch.no_grad():
                current_psnr = 10 * torch.log10(1.0 / mse_loss)
            print(f"Iteration {i}: MSE = {mse_loss.item():.8f}, PSNR = {current_psnr.item():.6f}")
    
    # Calculate final metrics
    with torch.no_grad():
        final_mse = torch.nn.functional.mse_loss(target, initial_recon)
        final_psnr = 10 * torch.log10(1.0 / final_mse)
    
    print(f"Final MSE: {final_mse.item():.8f}")
    print(f"Final PSNR: {final_psnr.item():.6f}")
    print(f"PSNR improvement: {final_psnr.item() - initial_psnr.item():.6f}")
    
    # Check loss reduction (positive losses for MSE)
    if losses[0] > 0:
        loss_reduction = ((losses[0] - losses[-1]) / losses[0]) * 100
    else:
        loss_reduction = 0.0
    
    print(f"MSE reduction: {loss_reduction:.2f}%")
    
    return final_psnr.item() - initial_psnr.item(), loss_reduction

def main():
    """Main debugging function"""
    print("🐛 PSNR vs MSE Optimization Comparison")
    print("=" * 60)
    
    # Test 1: PSNR calculation consistency
    opt_psnr, eval_psnr = test_psnr_calculation_consistency()
    
    # Test 2: Gradient flow
    has_gradients = test_gradient_flow()
    
    # Test 3: Epsilon impact
    test_epsilon_impact()
    
    # Test 4: PSNR optimization
    psnr_improvement_psnr, loss_reduction_psnr = test_simple_optimization()
    
    print()
    # Test 5: MSE optimization  
    psnr_improvement_mse, loss_reduction_mse = test_mse_optimization()
    
    print("=" * 60)
    print("🔍 COMPARISON SUMMARY:")
    print(f"  PSNR calculation difference: {abs(opt_psnr - eval_psnr):.6f}")
    print(f"  Gradients flowing: {'✅' if has_gradients else '❌'}")
    print()
    print("📊 OPTIMIZATION COMPARISON:")
    print(f"  PSNR Loss → PSNR improvement: {psnr_improvement_psnr:.6f} dB")
    print(f"  PSNR Loss → Loss reduction: {loss_reduction_psnr:.2f}%")
    print(f"  MSE Loss  → PSNR improvement: {psnr_improvement_mse:.6f} dB")
    print(f"  MSE Loss  → Loss reduction: {loss_reduction_mse:.2f}%")
    print()
    
    # Recommendation
    if psnr_improvement_mse > psnr_improvement_psnr:
        print("✅ RECOMMENDATION: Use MSE optimization for better PSNR improvement")
    elif abs(psnr_improvement_mse - psnr_improvement_psnr) < 0.01:
        print("⚡ RECOMMENDATION: MSE optimization provides cleaner implementation")
    else:
        print("❓ PSNR optimization performs better, but has implementation issues")
        
    if loss_reduction_mse > 0 and loss_reduction_psnr == 0:
        print("🔧 CONFIRMED: MSE optimization fixes loss reduction calculation bug")

if __name__ == "__main__":
    main()