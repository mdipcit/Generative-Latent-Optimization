#!/usr/bin/env python3
"""
Calculate baseline VAE metrics (LPIPS, FID, MAE) from initial reconstructions
"""

import json
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Import our metrics
from generative_latent_optimization.metrics import UnifiedMetricsCalculator
from generative_latent_optimization.metrics import DatasetFIDEvaluator


def load_image_as_tensor(image_path: Path) -> torch.Tensor:
    """Load image and convert to tensor"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(img)


def calculate_baseline_metrics():
    """Calculate baseline metrics from initial VAE reconstructions"""
    
    print("=" * 70)
    print("🎯 BASELINE VAE METRICS CALCULATION")
    print("=" * 70)
    
    # Setup paths
    base_dir = Path("experiments/full_mse_comparison/mse_experiment_20250903_101919_png")
    
    # Check if directory exists
    if not base_dir.exists():
        print(f"❌ Error: Directory not found: {base_dir}")
        return
    
    # Initialize metrics calculators
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📊 Device: {device}")
    
    metrics_calc = UnifiedMetricsCalculator(
        device=device,
        use_lpips=True,
        use_improved_ssim=False  # Not needed for baseline
    )
    
    fid_evaluator = DatasetFIDEvaluator(device=device)
    
    # Collect all image pairs
    print("\n📂 Collecting image pairs...")
    
    image_pairs = []
    lpips_scores = []
    mae_scores = []
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
            
        print(f"\n📁 Processing {split} split...")
        
        # Get all subdirectories (image IDs)
        image_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for img_dir in tqdm(image_dirs, desc=f"  {split}"):
            original_path = img_dir / "original.png"
            initial_recon_path = img_dir / "initial_recon.png"
            
            if not (original_path.exists() and initial_recon_path.exists()):
                continue
            
            # Load images as tensors
            original_tensor = load_image_as_tensor(original_path).unsqueeze(0).to(device)
            initial_tensor = load_image_as_tensor(initial_recon_path).unsqueeze(0).to(device)
            
            # Calculate LPIPS and MAE
            with torch.no_grad():
                metrics = metrics_calc.calculate(original_tensor, initial_tensor)
                lpips_scores.append(metrics.lpips)
                mae_scores.append(metrics.mae)
            
            # Store paths for FID calculation
            image_pairs.append((original_path, initial_recon_path))
    
    print(f"\n✅ Collected {len(image_pairs)} image pairs")
    
    # Calculate LPIPS statistics
    if lpips_scores:
        lpips_array = np.array(lpips_scores)
        lpips_stats = {
            'mean': float(np.mean(lpips_array)),
            'std': float(np.std(lpips_array)),
            'min': float(np.min(lpips_array)),
            'max': float(np.max(lpips_array)),
            'median': float(np.median(lpips_array))
        }
        print(f"\n📊 Baseline LPIPS: {lpips_stats['mean']:.4f} ± {lpips_stats['std']:.4f}")
    else:
        lpips_stats = None
        print("\n❌ No LPIPS scores calculated")
    
    # Calculate MAE statistics
    if mae_scores:
        mae_array = np.array(mae_scores)
        mae_stats = {
            'mean': float(np.mean(mae_array)),
            'std': float(np.std(mae_array)),
            'min': float(np.min(mae_array)),
            'max': float(np.max(mae_array)),
            'median': float(np.median(mae_array))
        }
        print(f"📊 Baseline MAE: {mae_stats['mean']:.4f} ± {mae_stats['std']:.4f}")
    else:
        mae_stats = None
    
    # Calculate FID
    print("\n🔄 Calculating FID score...")
    
    # Create temporary directories for FID calculation
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        original_dir = temp_path / "original"
        recon_dir = temp_path / "reconstructed"
        original_dir.mkdir()
        recon_dir.mkdir()
        
        # Copy images to temporary directories
        print("  Preparing images for FID calculation...")
        for i, (orig_path, recon_path) in enumerate(tqdm(image_pairs, desc="  Copying")):
            shutil.copy(orig_path, original_dir / f"{i:06d}.png")
            shutil.copy(recon_path, recon_dir / f"{i:06d}.png")
        
        # Calculate FID using correct method
        try:
            fid_results = fid_evaluator.evaluate_created_dataset_vs_original(
                str(recon_dir),
                str(original_dir)
            )
            fid_score = fid_results.fid_score
            print(f"\n📊 Baseline FID: {fid_score:.2f}")
        except Exception as e:
            print(f"\n❌ FID calculation failed: {e}")
            fid_score = None
    
    # Prepare results
    results = {
        'total_samples': len(image_pairs),
        'lpips': lpips_stats,
        'mae': mae_stats,
        'fid': fid_score
    }
    
    # Save results
    output_file = Path("baseline_vae_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("📋 BASELINE VAE METRICS SUMMARY")
    print("=" * 70)
    print(f"  Samples: {len(image_pairs)}")
    if lpips_stats:
        print(f"  LPIPS: {lpips_stats['mean']:.4f} ± {lpips_stats['std']:.4f}")
    if mae_stats:
        print(f"  MAE: {mae_stats['mean']:.4f} ± {mae_stats['std']:.4f}")
    if fid_score is not None:
        print(f"  FID: {fid_score:.2f}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Set environment for better performance
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Run calculation
    results = calculate_baseline_metrics()
    
    # Print recommendation for table update
    if results and results.get('lpips') and results.get('fid'):
        print("\n📝 For comparison table update:")
        print(f"  LPIPS: {results['lpips']['mean']:.3f}")
        if results['mae']:
            print(f"  MAE: {results['mae']['mean']:.3f}")
        print(f"  FID: {results['fid']:.1f}")