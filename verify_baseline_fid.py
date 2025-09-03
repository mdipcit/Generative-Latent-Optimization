#!/usr/bin/env python3
"""
Create initial_recon collection and verify FID measurement
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import FID evaluator
from generative_latent_optimization.metrics import DatasetFIDEvaluator


def create_initial_recon_collection():
    """Create directory containing only initial_recon.png files for FID verification"""
    
    print("=" * 70)
    print("🔍 INITIAL_RECON FID VERIFICATION")
    print("=" * 70)
    
    # Source directory (MSE experiment)
    source_dir = Path("experiments/full_mse_comparison/mse_experiment_20250903_101919_png")
    
    # Create verification directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    verification_dir = Path("experiments/baseline_verification") / f"initial_recon_fid_check_{timestamp}"
    verification_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    original_dir = verification_dir / "original_images"
    initial_recon_dir = verification_dir / "initial_reconstructions"
    original_dir.mkdir()
    initial_recon_dir.mkdir()
    
    print(f"📂 Source: {source_dir}")
    print(f"📁 Verification: {verification_dir}")
    print(f"   └── original_images/")
    print(f"   └── initial_reconstructions/")
    
    # Statistics
    total_copied = 0
    splits_stats = {'train': 0, 'val': 0, 'test': 0}
    
    # Collect all initial_recon.png files
    for split in ['train', 'val', 'test']:
        source_split = source_dir / split
        
        if not source_split.exists():
            continue
            
        print(f"\n📁 Processing {split} split...")
        
        # Get all image directories
        image_dirs = sorted([d for d in source_split.iterdir() if d.is_dir()])
        
        for img_dir in tqdm(image_dirs, desc=f"  {split}"):
            image_id = img_dir.name
            
            original_src = img_dir / "original.png"
            initial_recon_src = img_dir / "initial_recon.png"
            
            if original_src.exists() and initial_recon_src.exists():
                # Copy with sequential naming for FID calculation
                original_dst = original_dir / f"{total_copied:06d}_{split}_{image_id}.png"
                initial_dst = initial_recon_dir / f"{total_copied:06d}_{split}_{image_id}.png"
                
                shutil.copy2(original_src, original_dst)
                shutil.copy2(initial_recon_src, initial_dst)
                
                total_copied += 1
                splits_stats[split] += 1
        
        print(f"    ✅ {split}: {splits_stats[split]} images copied")
    
    print(f"\n✅ Total images collected: {total_copied}")
    
    # Calculate FID
    print("\n🔄 Calculating FID for verification...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fid_evaluator = DatasetFIDEvaluator(device=device, batch_size=50)
    
    try:
        fid_results = fid_evaluator.evaluate_created_dataset_vs_original(
            str(initial_recon_dir),
            str(original_dir)
        )
        fid_score = fid_results.fid_score
        
        print(f"📊 Verification FID Score: {fid_score:.2f}")
        
        # Compare with previous result
        previous_fid = 13.79
        difference = abs(fid_score - previous_fid)
        
        print(f"📊 Previous FID Score: {previous_fid:.2f}")
        print(f"📊 Difference: {difference:.2f}")
        
        if difference < 1.0:
            print("✅ FID測定結果が一致 - ベースライン性能確認済み")
            verification_status = "CONSISTENT"
        else:
            print("⚠️ FID測定結果に差異 - 再確認が必要")
            verification_status = "INCONSISTENT"
            
    except Exception as e:
        print(f"❌ FID calculation failed: {e}")
        fid_score = None
        verification_status = "FAILED"
    
    # Save verification results
    verification_results = {
        'timestamp': datetime.now().isoformat(),
        'total_images': total_copied,
        'splits_breakdown': splits_stats,
        'source_experiment': str(source_dir),
        'verification_fid': fid_score,
        'previous_fid': previous_fid,
        'difference': difference if fid_score else None,
        'verification_status': verification_status,
        'original_dir': str(original_dir),
        'initial_recon_dir': str(initial_recon_dir)
    }
    
    with open(verification_dir / "fid_verification_results.json", 'w') as f:
        json.dump(verification_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("📋 FID VERIFICATION RESULTS")
    print("=" * 70)
    print(f"  Original FID: {previous_fid:.2f}")
    if fid_score:
        print(f"  Verification FID: {fid_score:.2f}")
        print(f"  Difference: ±{difference:.2f}")
        print(f"  Status: {verification_status}")
    print(f"  Images: {total_copied}")
    print(f"  Directory: {verification_dir}")
    print("=" * 70)
    
    return verification_dir, fid_score, verification_status


if __name__ == "__main__":
    import torch
    
    verification_dir, fid_score, status = create_initial_recon_collection()
    
    if status == "CONSISTENT":
        print("\n🎯 結論: ベースラインVAEの高性能は確実")
        print("   → SD1.5 VAEの再構成能力が予想以上に優秀")
        print("   → 最適化手法の改善余地は限定的")
    elif status == "INCONSISTENT":
        print("\n⚠️ 結論: 測定に不整合あり - 要調査")
    else:
        print("\n❌ 結論: 測定失敗 - 要再実行")