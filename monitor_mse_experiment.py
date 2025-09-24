#!/usr/bin/env python3
"""
Monitor MSE optimization experiment progress
"""

import time
import json
from pathlib import Path
from datetime import datetime

def check_experiment_progress():
    """Check current experiment progress"""
    print(f"🔍 Experiment Progress Check - {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 50)
    
    # Expected experiment directory
    experiment_dir = Path("./experiments/full_mse_comparison/mse_experiment_20250903_101919")
    
    if not experiment_dir.exists():
        print("❌ Experiment directory not found")
        return
    
    # Check for temporary processing directory
    temp_dirs = list(experiment_dir.parent.glob("temp_processing_*"))
    if temp_dirs:
        temp_dir = temp_dirs[0]
        print(f"📁 Processing directory: {temp_dir}")
        
        # Check processed results
        processed_dir = temp_dir / "processed"
        if processed_dir.exists():
            splits = ["train", "val", "test"]
            total_processed = 0
            
            for split in splits:
                split_dir = processed_dir / split
                if split_dir.exists():
                    processed_images = len(list(split_dir.glob("*")))
                    total_processed += processed_images
                    print(f"  {split}: {processed_images} images processed")
            
            print(f"  Total: {total_processed}/500 images ({total_processed/500*100:.1f}%)")
        
        # Check for summary files
        summary_files = list(temp_dir.glob("**/processing_summary.json"))
        if summary_files:
            with open(summary_files[0], 'r') as f:
                summary = json.load(f)
            
            print(f"\n📊 Current Results:")
            print(f"  Avg PSNR improvement: {summary.get('average_psnr_improvement', 0):.4f} dB")
            print(f"  Avg loss reduction: {summary.get('average_loss_reduction', 0):.2f}%")
            print(f"  Processing time: {summary.get('processing_time_hours', 0):.2f} hours")
    
    # Check final results
    final_stats = experiment_dir / "mse_dataset_png" / "statistics.json"
    if final_stats.exists():
        print(f"\n🎉 EXPERIMENT COMPLETED!")
        with open(final_stats, 'r') as f:
            stats = json.load(f)
        
        psnr_stats = stats.get('psnr_improvement', {})
        loss_stats = stats.get('loss_reduction', {})
        
        print(f"📊 Final Results:")
        print(f"  PSNR improvement: {psnr_stats.get('mean', 0):.4f} ± {psnr_stats.get('std', 0):.4f} dB")
        print(f"  Loss reduction: {loss_stats.get('mean', 0):.2f} ± {loss_stats.get('std', 0):.2f}%")
        print(f"  Total samples: {stats.get('total_samples', 0)}")
        
        return True  # Experiment completed
    
    return False  # Still running

def main():
    """Monitor experiment with periodic checks"""
    print("📊 MSE Experiment Monitoring System")
    print("=" * 60)
    
    completed = False
    check_count = 0
    
    while not completed and check_count < 20:  # Max 20 checks
        completed = check_experiment_progress()
        
        if not completed:
            print(f"\n⏳ Experiment still running... (check {check_count + 1}/20)")
            print("   Next check in 5 minutes")
            time.sleep(300)  # Wait 5 minutes
        else:
            print(f"\n✅ Experiment completed successfully!")
            break
            
        check_count += 1
    
    if not completed:
        print(f"\n⚠️  Monitoring timeout after {check_count} checks")
        print("   Experiment may still be running")

if __name__ == "__main__":
    main()