#!/usr/bin/env python3
"""
Re-evaluate existing optimization datasets with corrected evaluation system
"""

import torch
import json
import sys
import tempfile
import shutil
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
from generative_latent_optimization.metrics.dataset_metrics import DatasetFIDEvaluator

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

def collect_images_by_type(dataset_path, image_type):
    """Collect all images of a specific type from the dataset"""
    image_paths = []
    
    # Iterate through all splits (train, val, test)
    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
            
        # Get all image directories in this split
        for image_dir in split_dir.iterdir():
            if image_dir.is_dir():
                image_file = image_dir / f"{image_type}.png"
                if image_file.exists():
                    image_paths.append(str(image_file))
    
    return image_paths

def re_evaluate_dataset(dataset_path, method_name, device='cuda'):
    """Re-evaluate a dataset with corrected evaluation system"""
    
    print(f"\n🔄 **Re-evaluating {method_name} Dataset**")
    print(f"📁 Path: {dataset_path}")
    print("-" * 50)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return None
    
    # Setup calculators
    calculator = UnifiedMetricsCalculator(
        device=device,
        use_lpips=True,
        use_improved_ssim=True
    )
    fid_evaluator = DatasetFIDEvaluator(device=device)
    
    try:
        # Collect image pairs
        image_pairs = collect_image_pairs(dataset_path)
        print(f"📊 Found {len(image_pairs)} image pairs")
        
        # Initialize result storage
        results = {
            'psnr_improvements': [],
            'ssim_improvements': [],
            'lpips_scores': [],
            'mae_scores': [],
            'individual_metrics': [],
            'split_breakdown': {'train': [], 'val': [], 'test': []}
        }
        
        # Process each image pair
        print("🔄 Processing image pairs with corrected evaluation...")
        
        for i, pair in enumerate(tqdm(image_pairs, desc=f"Re-evaluating {method_name}")):
            try:
                # Load images as tensors
                original_tensor = load_image_as_tensor(pair['original'], device)
                optimized_tensor = load_image_as_tensor(pair['optimized'], device)
                
                # Calculate all metrics with corrected system
                metrics = calculator.calculate(original_tensor, optimized_tensor)
                
                # Load old metrics for comparison
                old_metrics_file = Path(pair['original']).parent / "metrics.json"
                old_metrics = {}
                if old_metrics_file.exists():
                    with open(old_metrics_file, 'r') as f:
                        old_metrics = json.load(f)
                
                # Calculate improvements (corrected)
                initial_metrics_file = Path(pair['original']).parent / "metrics.json"
                if initial_metrics_file.exists():
                    with open(initial_metrics_file, 'r') as f:
                        old_data = json.load(f)
                    
                    # For PSNR improvement, we need initial reconstruction
                    initial_recon_file = Path(pair['original']).parent / "initial_recon.png"
                    if initial_recon_file.exists():
                        initial_recon_tensor = load_image_as_tensor(str(initial_recon_file), device)
                        initial_metrics = calculator.calculate(original_tensor, initial_recon_tensor)
                        
                        psnr_improvement = metrics.psnr_db - initial_metrics.psnr_db
                        ssim_improvement = metrics.ssim - initial_metrics.ssim
                    else:
                        # Use old initial values if available
                        psnr_improvement = None
                        ssim_improvement = None
                else:
                    psnr_improvement = None
                    ssim_improvement = None
                
                # Store results
                if psnr_improvement is not None:
                    results['psnr_improvements'].append(float(psnr_improvement))
                if ssim_improvement is not None:
                    results['ssim_improvements'].append(float(ssim_improvement))
                
                if metrics.lpips is not None:
                    results['lpips_scores'].append(float(metrics.lpips))
                if metrics.mae is not None:
                    results['mae_scores'].append(float(metrics.mae))
                
                # Store individual results
                individual_result = {
                    'image_id': pair['image_id'],
                    'split': pair['split'],
                    'psnr_improvement': float(psnr_improvement) if psnr_improvement is not None else None,
                    'ssim_improvement': float(ssim_improvement) if ssim_improvement is not None else None,
                    'lpips': float(metrics.lpips) if metrics.lpips is not None else None,
                    'mae': float(metrics.mae) if metrics.mae is not None else None,
                    'final_psnr': float(metrics.psnr_db),
                    'final_ssim': float(metrics.ssim),
                    'old_psnr_improvement': old_metrics.get('psnr_improvement', None),
                    'old_ssim_improvement': old_metrics.get('ssim_improvement', None)
                }
                results['individual_metrics'].append(individual_result)
                results['split_breakdown'][pair['split']].append(individual_result)
                
            except Exception as e:
                print(f"⚠️  Warning: Failed to process {pair['image_id']}: {e}")
                continue
        
        # Calculate statistics
        print("📊 Calculating corrected statistics...")
        
        # PSNR improvement statistics
        psnr_stats = {}
        if results['psnr_improvements']:
            psnr_improvements = np.array(results['psnr_improvements'])
            psnr_stats = {
                'mean': float(np.mean(psnr_improvements)),
                'std': float(np.std(psnr_improvements)),
                'min': float(np.min(psnr_improvements)),
                'max': float(np.max(psnr_improvements)),
                'median': float(np.median(psnr_improvements))
            }
        
        # SSIM improvement statistics
        ssim_stats = {}
        if results['ssim_improvements']:
            ssim_improvements = np.array(results['ssim_improvements'])
            ssim_stats = {
                'mean': float(np.mean(ssim_improvements)),
                'std': float(np.std(ssim_improvements)),
                'min': float(np.min(ssim_improvements)),
                'max': float(np.max(ssim_improvements)),
                'median': float(np.median(ssim_improvements))
            }
        
        # LPIPS statistics
        lpips_stats = {}
        if results['lpips_scores']:
            lpips_scores = np.array(results['lpips_scores'])
            lpips_stats = {
                'mean': float(np.mean(lpips_scores)),
                'std': float(np.std(lpips_scores)),
                'min': float(np.min(lpips_scores)),
                'max': float(np.max(lpips_scores)),
                'median': float(np.median(lpips_scores))
            }
        
        # MAE statistics
        mae_stats = {}
        if results['mae_scores']:
            mae_scores = np.array(results['mae_scores'])
            mae_stats = {
                'mean': float(np.mean(mae_scores)),
                'std': float(np.std(mae_scores)),
                'min': float(np.min(mae_scores)),
                'max': float(np.max(mae_scores)),
                'median': float(np.median(mae_scores))
            }
        
        # Split-wise statistics
        split_stats = {}
        for split in ['train', 'val', 'test']:
            split_data = results['split_breakdown'][split]
            if split_data:
                split_psnr = [item['psnr_improvement'] for item in split_data if item['psnr_improvement'] is not None]
                split_ssim = [item['ssim_improvement'] for item in split_data if item['ssim_improvement'] is not None]
                split_lpips = [item['lpips'] for item in split_data if item['lpips'] is not None]
                split_mae = [item['mae'] for item in split_data if item['mae'] is not None]
                
                split_stats[split] = {
                    'count': len(split_data),
                    'psnr_improvement_mean': float(np.mean(split_psnr)) if split_psnr else None,
                    'ssim_improvement_mean': float(np.mean(split_ssim)) if split_ssim else None,
                    'lpips_mean': float(np.mean(split_lpips)) if split_lpips else None,
                    'mae_mean': float(np.mean(split_mae)) if split_mae else None
                }
        
        # Calculate FID score
        print("🔄 Calculating FID score...")
        original_images = collect_images_by_type(dataset_path, "original")
        optimized_images = collect_images_by_type(dataset_path, "optimized_recon")
        
        fid_score = None
        if len(original_images) > 0 and len(optimized_images) > 0:
            # Create temporary directories for FID calculation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                original_temp_dir = temp_path / "original"
                optimized_temp_dir = temp_path / "optimized"
                
                original_temp_dir.mkdir()
                optimized_temp_dir.mkdir()
                
                # Copy images
                for i, img_path in enumerate(original_images):
                    shutil.copy2(img_path, original_temp_dir / f"{i:05d}.png")
                
                for i, img_path in enumerate(optimized_images):
                    shutil.copy2(img_path, optimized_temp_dir / f"{i:05d}.png")
                
                # Calculate FID
                fid_results = fid_evaluator.evaluate_created_dataset_vs_original(
                    optimized_temp_dir, 
                    original_temp_dir
                )
                fid_score = fid_results.fid_score
        
        # Prepare comprehensive results
        comprehensive_results = {
            'method': method_name,
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(dataset_path),
            'total_images': len(image_pairs),
            'device': device,
            'corrected_evaluation': True,
            'psnr_improvement_statistics': psnr_stats,
            'ssim_improvement_statistics': ssim_stats,
            'lpips_statistics': lpips_stats,
            'mae_statistics': mae_stats,
            'fid_score': float(fid_score) if fid_score is not None else None,
            'split_statistics': split_stats,
            'measurement_success_rate': len(results['individual_metrics']) / len(image_pairs) * 100
        }
        
        # Save results
        results_file = dataset_path / "corrected_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Display results
        print("\n" + "="*60)
        print(f"🏆 **{method_name.upper()} CORRECTED EVALUATION RESULTS**")
        print("="*60)
        
        if psnr_stats:
            print(f"\n📊 **PSNR Improvement (Corrected):**")
            print(f"   Mean: {psnr_stats['mean']:+.3f} ± {psnr_stats['std']:.3f} dB")
            print(f"   Range: {psnr_stats['min']:+.3f} to {psnr_stats['max']:+.3f} dB")
            print(f"   Median: {psnr_stats['median']:+.3f} dB")
        
        if ssim_stats:
            print(f"\n📊 **SSIM Improvement (Corrected):**")
            print(f"   Mean: {ssim_stats['mean']:+.4f} ± {ssim_stats['std']:.4f}")
            print(f"   Range: {ssim_stats['min']:+.4f} to {ssim_stats['max']:+.4f}")
            print(f"   Median: {ssim_stats['median']:+.4f}")
        
        if lpips_stats:
            print(f"\n📊 **LPIPS Score:**")
            print(f"   Mean: {lpips_stats['mean']:.4f} ± {lpips_stats['std']:.4f}")
            print(f"   Range: {lpips_stats['min']:.4f} - {lpips_stats['max']:.4f}")
        
        if mae_stats:
            print(f"\n📊 **MAE Score:**")
            print(f"   Mean: {mae_stats['mean']:.4f} ± {mae_stats['std']:.4f}")
            print(f"   Range: {mae_stats['min']:.4f} - {mae_stats['max']:.4f}")
        
        if fid_score is not None:
            print(f"\n📊 **FID Score:** {fid_score:.2f}")
            
            # FID assessment
            if fid_score < 20:
                print("🏆 EXCELLENT: Very low FID, high distribution preservation")
            elif fid_score < 50:
                print("✅ GOOD: Reasonable FID, acceptable distribution preservation") 
            elif fid_score < 100:
                print("⚠️  MODERATE: Higher FID, some distribution degradation")
            else:
                print("❌ POOR: Very high FID, significant distribution loss")
        
        print(f"\n📈 **Split Breakdown:**")
        for split, stats in split_stats.items():
            if stats['psnr_improvement_mean'] is not None:
                print(f"   {split} ({stats['count']}): PSNR={stats['psnr_improvement_mean']:+.3f}dB, "
                      f"SSIM={stats['ssim_improvement_mean']:+.4f}, "
                      f"LPIPS={stats['lpips_mean']:.4f}, MAE={stats['mae_mean']:.4f}")
        
        print(f"\n✅ Re-evaluation completed!")
        print(f"📊 Success rate: {comprehensive_results['measurement_success_rate']:.1f}%")
        print(f"💾 Results saved to: {results_file}")
        
        return comprehensive_results
        
    except Exception as e:
        print(f"❌ Re-evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🔬 **Existing Data Re-evaluation with Corrected System**")
    print("="*70)
    print("🎯 Purpose: Re-evaluate existing optimization results with fixed evaluation")
    print("⚡ Advantage: 90% time saving vs new experiments")
    print("🔧 Fix: Corrected image range normalization in evaluation")
    print("-"*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    print(f"⏰ Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Define datasets to re-evaluate
    base_path = Path("./experiments/fixed_comparison/experiments/full_comparison")
    
    datasets_to_evaluate = [
        {
            'path': base_path / "improved_ssim_dataset_png",
            'method': 'Improved SSIM',
            'expected': '+2-3dB PSNR (from 45.8% loss reduction)'
        },
        {
            'path': base_path / "lpips_dataset_png",
            'method': 'LPIPS',
            'expected': '+3-4dB PSNR (from 90.5% loss reduction)'
        },
        {
            'path': base_path / "psnr_dataset_png", 
            'method': 'PSNR',
            'expected': '+2-3dB PSNR (similar to MSE performance)'
        }
    ]
    
    results = {}
    
    # Process each dataset
    for dataset_info in datasets_to_evaluate:
        dataset_path = dataset_info['path']
        method_name = dataset_info['method']
        
        print(f"\n🎯 **{method_name} Re-evaluation**")
        print(f"📈 Expected improvement: {dataset_info['expected']}")
        
        if dataset_path.exists():
            result = re_evaluate_dataset(dataset_path, method_name, device)
            if result:
                results[method_name] = result
        else:
            print(f"❌ Dataset not found: {dataset_path}")
    
    print(f"\n🏆 **RE-EVALUATION SUMMARY**")
    print("="*50)
    
    for method, result in results.items():
        if result:
            psnr_mean = result['psnr_improvement_statistics'].get('mean', 'N/A')
            fid_score = result.get('fid_score', 'N/A')
            print(f"{method}: PSNR={psnr_mean:+.2f}dB, FID={fid_score}")
    
    return results

if __name__ == "__main__":
    main()