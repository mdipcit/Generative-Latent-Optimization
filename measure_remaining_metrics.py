#!/usr/bin/env python3
"""
Measure remaining metrics (LPIPS, MAE) for MSE optimization results
"""

import torch
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

from generative_latent_optimization.metrics.unified_calculator import UnifiedMetricsCalculator

def load_image_as_tensor(image_path, device='cuda'):
    """Load PNG image and convert to tensor"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

def collect_image_pairs(dataset_path):
    """Collect all image pairs for evaluation"""
    image_pairs = []
    
    # Iterate through all splits (train, val, test)
    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
            
        # Get all image directories in this split
        for image_dir in split_dir.iterdir():
            if image_dir.is_dir():
                original_file = image_dir / "original.png"
                optimized_file = image_dir / "optimized_recon.png"
                
                if original_file.exists() and optimized_file.exists():
                    image_pairs.append({
                        'original': str(original_file),
                        'optimized': str(optimized_file),
                        'image_id': image_dir.name,
                        'split': split
                    })
    
    return image_pairs

def main():
    print("🔬 Measuring remaining metrics for MSE optimization results...")
    print("-" * 60)
    
    # Setup paths
    dataset_path = Path("./experiments/full_mse_comparison/mse_experiment_20250903_101919_png")
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return None
    
    print(f"📁 Dataset path: {dataset_path}")
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Setup unified metrics calculator (with LPIPS enabled)
    print("🔧 Initializing metrics calculator with LPIPS support...")
    calculator = UnifiedMetricsCalculator(
        device=device,
        use_lpips=True,
        use_improved_ssim=True
    )
    
    try:
        # Collect all image pairs
        print("📊 Collecting image pairs...")
        image_pairs = collect_image_pairs(dataset_path)
        print(f"✅ Found {len(image_pairs)} image pairs")
        
        # Initialize result storage
        results = {
            'lpips_scores': [],
            'mae_scores': [],
            'individual_metrics': [],
            'split_breakdown': {'train': [], 'val': [], 'test': []}
        }
        
        # Process each image pair
        print("🔄 Processing image pairs...")
        
        for i, pair in enumerate(tqdm(image_pairs, desc="Measuring metrics")):
            try:
                # Load images as tensors
                original_tensor = load_image_as_tensor(pair['original'], device)
                optimized_tensor = load_image_as_tensor(pair['optimized'], device)
                
                # Calculate all metrics
                metrics = calculator.calculate(original_tensor, optimized_tensor)
                
                # Extract LPIPS and MAE
                lpips_score = metrics.lpips if hasattr(metrics, 'lpips') else None
                mae_score = metrics.mae if hasattr(metrics, 'mae') else None
                
                if lpips_score is not None:
                    results['lpips_scores'].append(float(lpips_score))
                if mae_score is not None:
                    results['mae_scores'].append(float(mae_score))
                
                # Store individual results
                individual_result = {
                    'image_id': pair['image_id'],
                    'split': pair['split'],
                    'lpips': float(lpips_score) if lpips_score is not None else None,
                    'mae': float(mae_score) if mae_score is not None else None,
                    'psnr': float(metrics.psnr_db),
                    'ssim': float(metrics.ssim)
                }
                results['individual_metrics'].append(individual_result)
                results['split_breakdown'][pair['split']].append(individual_result)
                
            except Exception as e:
                print(f"⚠️  Warning: Failed to process {pair['image_id']}: {e}")
                continue
        
        # Calculate statistics
        print("📊 Calculating statistics...")
        
        lpips_stats = {}
        if results['lpips_scores']:
            lpips_scores = np.array(results['lpips_scores'])
            lpips_stats = {
                'mean': float(np.mean(lpips_scores)),
                'std': float(np.std(lpips_scores)),
                'min': float(np.min(lpips_scores)),
                'max': float(np.max(lpips_scores)),
                'median': float(np.median(lpips_scores)),
                'percentile_25': float(np.percentile(lpips_scores, 25)),
                'percentile_75': float(np.percentile(lpips_scores, 75))
            }
        
        mae_stats = {}
        if results['mae_scores']:
            mae_scores = np.array(results['mae_scores'])
            mae_stats = {
                'mean': float(np.mean(mae_scores)),
                'std': float(np.std(mae_scores)),
                'min': float(np.min(mae_scores)),
                'max': float(np.max(mae_scores)),
                'median': float(np.median(mae_scores)),
                'percentile_25': float(np.percentile(mae_scores, 25)),
                'percentile_75': float(np.percentile(mae_scores, 75))
            }
        
        # Split-wise statistics
        split_stats = {}
        for split in ['train', 'val', 'test']:
            split_data = results['split_breakdown'][split]
            if split_data:
                split_lpips = [item['lpips'] for item in split_data if item['lpips'] is not None]
                split_mae = [item['mae'] for item in split_data if item['mae'] is not None]
                
                split_stats[split] = {
                    'count': len(split_data),
                    'lpips_mean': float(np.mean(split_lpips)) if split_lpips else None,
                    'mae_mean': float(np.mean(split_mae)) if split_mae else None
                }
        
        # Prepare comprehensive results
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(dataset_path),
            'total_images': len(image_pairs),
            'device': device,
            'optimization_method': 'MSE',
            'lpips_statistics': lpips_stats,
            'mae_statistics': mae_stats,
            'split_statistics': split_stats,
            'measurement_success_rate': len(results['individual_metrics']) / len(image_pairs) * 100
        }
        
        # Save results
        results_file = dataset_path / "comprehensive_metrics.json"
        with open(results_file, "w") as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Display results
        print("\n" + "="*60)
        print("🏆 **COMPREHENSIVE METRICS RESULTS**")
        print("="*60)
        
        if lpips_stats:
            print(f"\n📊 **LPIPS Statistics:**")
            print(f"   Mean: {lpips_stats['mean']:.4f} ± {lpips_stats['std']:.4f}")
            print(f"   Range: {lpips_stats['min']:.4f} - {lpips_stats['max']:.4f}")
            print(f"   Median: {lpips_stats['median']:.4f}")
        
        if mae_stats:
            print(f"\n📊 **MAE Statistics:**")
            print(f"   Mean: {mae_stats['mean']:.4f} ± {mae_stats['std']:.4f}")
            print(f"   Range: {mae_stats['min']:.4f} - {mae_stats['max']:.4f}")
            print(f"   Median: {mae_stats['median']:.4f}")
        
        print(f"\n📈 **Split Breakdown:**")
        for split, stats in split_stats.items():
            if stats['lpips_mean'] is not None:
                print(f"   {split}: LPIPS={stats['lpips_mean']:.4f}, MAE={stats['mae_mean']:.4f}")
        
        print(f"\n✅ Processing completed!")
        print(f"📊 Success rate: {comprehensive_results['measurement_success_rate']:.1f}%")
        print(f"💾 Results saved to: {results_file}")
        
        return comprehensive_results
        
    except Exception as e:
        print(f"❌ Metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()