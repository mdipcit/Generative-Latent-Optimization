#!/usr/bin/env python3
"""
Single Image Latent Optimization Experiment

Based on invertLDM.py methodology:
- Encode image to latent space using VAE
- Optimize latents to minimize reconstruction loss
- Compare original, initial reconstruction, and optimized reconstruction
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

from vae_toolkit import VAELoader, load_and_preprocess_image


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.nn.functional.mse_loss(img1, img2)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


def optimize_latents(vae, initial_latents, target_image, iterations=100, lr=0.1, device="cuda"):
    """
    Optimize latent representation to minimize reconstruction loss
    Based on invertLDM.py methodology
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
    for i in tqdm(range(iterations), desc="Optimizing latents"):
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
        
        if i % 20 == 0:
            psnr = calculate_psnr(target_image, reconstructed)
            print(f"Iteration {i}: Loss={loss.item():.6f}, PSNR={psnr:.2f}dB")
    
    return optimized_latents.detach(), losses


def save_image_tensor(tensor, path):
    """Convert tensor to PIL image and save"""
    # Convert from [1, C, H, W] to [H, W, C]
    image_np = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    # Convert to uint8
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    # Save as PIL image
    Image.fromarray(image_np).save(path)


def create_comparison_grid(original, initial_recon, optimized_recon, save_path):
    """Create side-by-side comparison grid"""
    from matplotlib import pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy for display
    def tensor_to_numpy(tensor):
        return tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    
    axes[0].imshow(tensor_to_numpy(original))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(tensor_to_numpy(initial_recon))
    axes[1].set_title('Initial Reconstruction')
    axes[1].axis('off')
    
    axes[2].imshow(tensor_to_numpy(optimized_recon))
    axes[2].set_title('Optimized Reconstruction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_loss_graphs(losses, save_dir):
    """Create loss and PSNR visualization graphs"""
    from matplotlib import pyplot as plt
    import numpy as np
    
    iterations = range(len(losses))
    psnr_values = [10 * np.log10(1.0 / loss) for loss in losses]
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('VAE Latent Optimization Results', fontsize=16, fontweight='bold')
    
    # 1. Loss over iterations
    ax1.plot(iterations, losses, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Reconstruction Loss over Iterations')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add loss reduction percentage
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
    ax1.text(0.05, 0.95, f'Loss Reduction: {loss_reduction:.1f}%', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
             verticalalignment='top')
    
    # 2. PSNR over iterations
    ax2.plot(iterations, psnr_values, 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('PSNR Improvement over Iterations')
    ax2.grid(True, alpha=0.3)
    
    # Add PSNR improvement
    initial_psnr = psnr_values[0]
    final_psnr = psnr_values[-1]
    psnr_improvement = final_psnr - initial_psnr
    ax2.text(0.05, 0.95, f'PSNR Improvement: +{psnr_improvement:.2f} dB', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8),
             verticalalignment='top')
    
    # 3. Loss convergence (last 50 iterations if available)
    if len(losses) >= 50:
        last_50_iterations = iterations[-50:]
        last_50_losses = losses[-50:]
        ax3.plot(last_50_iterations, last_50_losses, 'r-', linewidth=2, alpha=0.8)
        ax3.set_title('Loss Convergence (Last 50 Iterations)')
    else:
        ax3.plot(iterations, losses, 'r-', linewidth=2, alpha=0.8)
        ax3.set_title('Loss Convergence (All Iterations)')
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('MSE Loss')
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss gradient (rate of change)
    loss_gradient = np.gradient(losses)
    ax4.plot(iterations[1:], loss_gradient[1:], 'purple', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Loss Gradient')
    ax4.set_title('Rate of Loss Change')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_dir / "loss_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create simple loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, 'b-', linewidth=2, alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.title('VAE Latent Optimization: Reconstruction Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add annotations
    plt.text(0.02, 0.98, f'Initial Loss: {initial_loss:.6f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    plt.text(0.02, 0.90, f'Final Loss: {final_loss:.6f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.8))
    plt.text(0.02, 0.82, f'Reduction: {loss_reduction:.1f}%', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8))
    
    plt.savefig(save_dir / "loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss visualization saved to: {save_dir / 'loss_visualization.png'}")
    print(f"Simple loss curve saved to: {save_dir / 'loss_curve.png'}")


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    experiments_dir = Path(__file__).parent
    image_path = project_root / "document.png"
    results_dir = experiments_dir / "results" / "single_image_optimization"
    results_dir.mkdir(exist_ok=True)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
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
    
    # Calculate initial PSNR
    initial_psnr = calculate_psnr(original_tensor, initial_recon_tensor)
    print(f"Initial PSNR: {initial_psnr:.2f} dB")
    
    # Optimize latents
    print("Starting latent optimization...")
    optimized_latents, losses = optimize_latents(
        vae, initial_latents, original_tensor, 
        iterations=150, lr=0.4, device=device
    )
    
    # Final reconstruction
    with torch.no_grad():
        final_reconstruction = vae.decode(optimized_latents)
        final_recon_tensor = (final_reconstruction.sample / 2 + 0.5).clamp(0, 1)
    
    # Calculate final PSNR
    final_psnr = calculate_psnr(original_tensor, final_recon_tensor)
    print(f"Final PSNR: {final_psnr:.2f} dB")
    print(f"PSNR improvement: {final_psnr - initial_psnr:.2f} dB")
    
    # Save results
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
    
    # Create loss visualization graphs
    print("Creating loss visualization graphs...")
    create_loss_graphs(losses, results_dir)
    
    # Save latents
    torch.save(initial_latents, results_dir / "initial_latents.pt")
    torch.save(optimized_latents, results_dir / "optimized_latents.pt")
    
    # Save metrics
    metrics = {
        "initial_psnr_db": initial_psnr,
        "final_psnr_db": final_psnr,
        "psnr_improvement_db": final_psnr - initial_psnr,
        "optimization_iterations": 150,
        "learning_rate": 0.4,
        "final_loss": losses[-1],
        "loss_history": losses
    }
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to: {results_dir}")
    print(f"PSNR: {initial_psnr:.2f} → {final_psnr:.2f} dB (+{final_psnr - initial_psnr:.2f} dB)")


if __name__ == "__main__":
    main()