#!/usr/bin/env python3
"""
Loss History Visualization Script

Reads loss history from metrics.json and creates visualization graphs
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def calculate_psnr_from_loss(loss_values):
    """Calculate PSNR values from MSE loss values"""
    return [10 * np.log10(1.0 / loss) for loss in loss_values]


def create_loss_visualization(metrics_path, save_dir):
    """Create comprehensive loss and PSNR visualization"""
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    loss_history = metrics['loss_history']
    iterations = range(len(loss_history))
    psnr_history = calculate_psnr_from_loss(loss_history)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('VAE Latent Optimization Results', fontsize=16, fontweight='bold')
    
    # 1. Loss over iterations
    ax1.plot(iterations, loss_history, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Reconstruction Loss over Iterations')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Add loss reduction percentage
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
    ax1.text(0.05, 0.95, f'Loss Reduction: {loss_reduction:.1f}%', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
             verticalalignment='top')
    
    # 2. PSNR over iterations
    ax2.plot(iterations, psnr_history, 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('PSNR Improvement over Iterations')
    ax2.grid(True, alpha=0.3)
    
    # Add PSNR improvement
    initial_psnr = psnr_history[0]
    final_psnr = psnr_history[-1]
    psnr_improvement = final_psnr - initial_psnr
    ax2.text(0.05, 0.95, f'PSNR Improvement: +{psnr_improvement:.2f} dB', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8),
             verticalalignment='top')
    
    # 3. Loss convergence (last 50 iterations)
    last_50_iterations = iterations[-50:]
    last_50_losses = loss_history[-50:]
    ax3.plot(last_50_iterations, last_50_losses, 'r-', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('MSE Loss')
    ax3.set_title('Loss Convergence (Last 50 Iterations)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss gradient (rate of change)
    loss_gradient = np.gradient(loss_history)
    ax4.plot(iterations[1:], loss_gradient[1:], 'purple', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Loss Gradient')
    ax4.set_title('Rate of Loss Change')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add convergence indicator
    convergence_threshold = 1e-6
    converged_at = None
    for i in range(1, len(loss_gradient)):
        if abs(loss_gradient[i]) < convergence_threshold:
            converged_at = i
            break
    
    if converged_at:
        ax4.axvline(x=converged_at, color='red', linestyle='--', alpha=0.7, 
                   label=f'Convergence at iter {converged_at}')
        ax4.legend()
    
    plt.tight_layout()
    
    # Save the visualization
    save_path = Path(save_dir) / "loss_visualization.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss visualization saved to: {save_path}")
    
    # Save a simple loss-only graph as well
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, loss_history, 'b-', linewidth=2, alpha=0.8)
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
    
    simple_save_path = Path(save_dir) / "loss_curve.png"
    plt.savefig(simple_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Simple loss curve saved to: {simple_save_path}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Total Iterations: {len(loss_history)}")
    print(f"Initial Loss: {initial_loss:.6f}")
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Loss Reduction: {loss_reduction:.1f}%")
    print(f"Initial PSNR: {initial_psnr:.2f} dB")
    print(f"Final PSNR: {final_psnr:.2f} dB")
    print(f"PSNR Improvement: +{psnr_improvement:.2f} dB")
    
    if converged_at:
        print(f"Convergence achieved at iteration: {converged_at}")
    else:
        print("Convergence not achieved within threshold")
    
    print("="*50)


def main():
    project_root = Path(__file__).parent.parent
    experiments_dir = Path(__file__).parent
    # Look for recent results in experiments/results/
    metrics_path = experiments_dir / "results" / "metrics.json"
    results_dir = experiments_dir / "results" / "visualization"
    results_dir.mkdir(exist_ok=True)
    
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found. Please run the optimization experiment first.")
        return
    
    print("Creating loss visualization...")
    create_loss_visualization(metrics_path, results_dir)
    print("Visualization complete!")


if __name__ == "__main__":
    main()