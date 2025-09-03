#!/usr/bin/env python3
"""
Calculate FID score for MSE optimization results
"""

import torch
import json
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

from generative_latent_optimization.metrics.dataset_metrics import DatasetFIDEvaluator

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

def main():
    print("🔍 Calculating FID for MSE optimization results...")
    print("-" * 50)
    
    # Setup paths
    dataset_path = Path("./experiments/full_mse_comparison/mse_experiment_20250903_101919_png")
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return None
    
    print(f"📁 Dataset path: {dataset_path}")
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Setup FID evaluator
    fid_evaluator = DatasetFIDEvaluator(device=device)
    
    try:
        print("🚀 Starting FID calculation...")
        
        # Collect original and optimized images
        original_images = collect_images_by_type(dataset_path, "original")
        optimized_images = collect_images_by_type(dataset_path, "optimized_recon")
        
        print(f"📊 Found {len(original_images)} original images")
        print(f"📊 Found {len(optimized_images)} optimized images")
        
        if len(original_images) != len(optimized_images):
            print("⚠️  Warning: Mismatch in image counts")
        
        if len(original_images) == 0:
            print("❌ No images found")
            return None
        
        # Create temporary directories for FID calculation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            original_temp_dir = temp_path / "original"
            optimized_temp_dir = temp_path / "optimized"
            
            original_temp_dir.mkdir()
            optimized_temp_dir.mkdir()
            
            print("📁 Creating temporary directories for FID calculation...")
            
            # Copy original images
            for i, img_path in enumerate(original_images):
                shutil.copy2(img_path, original_temp_dir / f"{i:05d}.png")
            
            # Copy optimized images  
            for i, img_path in enumerate(optimized_images):
                shutil.copy2(img_path, optimized_temp_dir / f"{i:05d}.png")
            
            print(f"📁 Copied {len(original_images)} image pairs to temporary directories")
            
            # Calculate FID score using the dataset evaluator
            print("🔄 Computing FID score...")
            results = fid_evaluator.evaluate_created_dataset_vs_original(
                optimized_temp_dir, 
                original_temp_dir
            )
            
            fid_score = results.fid_score
            print(f"📊 FID Score: {fid_score:.2f}")
        
        # Prepare results
        fid_result = {
            "fid_score": fid_score,
            "timestamp": datetime.now().isoformat(),
            "dataset_path": str(dataset_path),
            "device": device,
            "optimization_method": "MSE",
            "experiment_date": "2025-09-03",
            "total_images": len(original_images),
            "original_images_count": len(original_images),
            "optimized_images_count": len(optimized_images)
        }
        
        # Save results
        results_file = dataset_path / "fid_results.json"
        with open(results_file, "w") as f:
            json.dump(fid_result, f, indent=2)
        
        print(f"✅ FID calculation completed!")
        print(f"📊 Final FID Score: {fid_score:.2f}")
        print(f"💾 Results saved to: {results_file}")
        
        # Performance assessment
        print(f"\n🎯 FID PERFORMANCE ASSESSMENT:")
        if fid_score < 20:
            print("🏆 EXCELLENT: Very low FID, high distribution preservation")
        elif fid_score < 50:
            print("✅ GOOD: Reasonable FID, acceptable distribution preservation") 
        elif fid_score < 100:
            print("⚠️  MODERATE: Higher FID, some distribution degradation")
        else:
            print("❌ POOR: Very high FID, significant distribution loss")
        
        return fid_score
        
    except Exception as e:
        print(f"❌ FID calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()