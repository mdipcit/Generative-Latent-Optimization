#!/usr/bin/env python3
"""
Small-scale MSE optimization test with 10 BSDS500 images
"""

import torch
import json
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

from generative_latent_optimization.workflows.batch_processing import process_bsds500_dataset
from generative_latent_optimization.optimization.latent_optimizer import OptimizationConfig

def run_small_scale_mse_test():
    """Run MSE optimization on 10 BSDS500 images"""
    print("🔬 Running small-scale MSE optimization test...")
    print("Target: 10 BSDS500 images with MSE loss")
    
    try:
        # Get BSDS500 path
        import os
        bsds500_path = os.environ.get("BSDS500_PATH")
        if not bsds500_path:
            raise ValueError("BSDS500_PATH environment variable not set")
        
        # Setup output directory
        output_dir = Path(__file__).parent / "experiments" / "small_mse_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"BSDS500 path: {bsds500_path}")
        print(f"Output directory: {output_dir}")
        
        # Create MSE optimization config (reduced iterations for faster testing)
        mse_config = OptimizationConfig(
            iterations=20,
            learning_rate=0.05,
            loss_function='mse',  # Use MSE instead of PSNR
            device='cuda' if torch.cuda.is_available() else 'cpu',
            checkpoint_interval=10
        )
        
        print(f"Using MSE optimization config: {mse_config.loss_function}")
        
        # Run optimization with MSE loss
        result_summary = process_bsds500_dataset(
            bsds500_path=bsds500_path,
            output_path=str(output_dir),
            config=mse_config,
            max_images_per_split=10,
            save_visualizations=True,
            create_pytorch_dataset=True,
            create_png_dataset=True
        )
        
        print(f"\n✅ Small-scale MSE test completed!")
        print(f"Results saved to: {output_dir}")
        
        # Analyze results
        analyze_mse_test_results(output_dir, result_summary)
        
        return result_summary
        
    except Exception as e:
        print(f"❌ Error during small-scale test: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_mse_test_results(output_dir, result_summary):
    """Analyze the results of MSE optimization test"""
    print("\n" + "="*60)
    print("📊 SMALL-SCALE MSE TEST ANALYSIS")
    print("="*60)
    
    if not result_summary:
        print("❌ No results to analyze")
        return
    
    # Extract performance metrics
    png_info = result_summary.get('png_dataset_info', {})
    
    if png_info:
        print(f"📁 PNG Dataset Created: {png_info.get('total_images', 0)} images")
        print(f"📂 Output Location: {png_info.get('base_path', 'N/A')}")
        
        # Look for statistics file
        stats_file = Path(png_info.get('base_path', '')) / "statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print(f"\n📈 OPTIMIZATION STATISTICS:")
            print(f"  Total samples: {stats.get('total_samples', 0)}")
            
            # PSNR improvement stats
            psnr_stats = stats.get('psnr_improvement', {})
            if psnr_stats:
                print(f"  PSNR improvement mean: {psnr_stats.get('mean', 0):.6f} dB")
                print(f"  PSNR improvement std:  {psnr_stats.get('std', 0):.6f} dB")
                print(f"  PSNR improvement min:  {psnr_stats.get('min', 0):.6f} dB")
                print(f"  PSNR improvement max:  {psnr_stats.get('max', 0):.6f} dB")
            
            # Loss reduction stats
            loss_stats = stats.get('loss_reduction', {})
            if loss_stats:
                print(f"  Loss reduction mean: {loss_stats.get('mean', 0):.2f}%")
                print(f"  Loss reduction std:  {loss_stats.get('std', 0):.2f}%")
                print(f"  Loss reduction min:  {loss_stats.get('min', 0):.2f}%")
                print(f"  Loss reduction max:  {loss_stats.get('max', 0):.2f}%")
        
        # Sample individual results
        sample_results_dir = Path(png_info.get('base_path', '')) / "test"
        if sample_results_dir.exists():
            print(f"\n🔍 SAMPLE RESULTS:")
            
            sample_dirs = list(sample_results_dir.glob("*/"))[:3]  # First 3 samples
            for sample_dir in sample_dirs:
                metrics_file = sample_dir / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    image_name = metrics.get('image_name', sample_dir.name)
                    psnr_improvement = metrics.get('psnr_improvement', 0)
                    loss_reduction = metrics.get('loss_reduction', 0)
                    
                    print(f"  {image_name}: PSNR +{psnr_improvement:.4f}dB, Loss -{loss_reduction:.1f}%")
    
    print("\n🎯 MSE OPTIMIZATION ASSESSMENT:")
    
    # Read a sample metrics to check if MSE optimization worked
    sample_results_dir = Path(output_dir) / "mse_dataset_png" / "test"
    if sample_results_dir.exists():
        sample_dirs = list(sample_results_dir.glob("*/"))
        if sample_dirs:
            sample_metrics = sample_dirs[0] / "metrics.json"
            if sample_metrics.exists():
                with open(sample_metrics, 'r') as f:
                    sample_data = json.load(f)
                
                psnr_improvement = sample_data.get('psnr_improvement', 0)
                loss_reduction = sample_data.get('loss_reduction', 0)
                
                if psnr_improvement > 0 and loss_reduction > 0:
                    print("✅ MSE optimization is working correctly")
                    print("✅ Both PSNR improvement and loss reduction are positive")
                elif psnr_improvement > 0 and loss_reduction == 0:
                    print("⚠️  MSE optimization improving PSNR but loss reduction calculation issue")
                else:
                    print("❌ MSE optimization not working as expected")
    
    print("="*60)

def compare_with_original_experiment():
    """Compare MSE results with original PSNR experiment results"""
    print("\n🔍 COMPARING WITH ORIGINAL PSNR EXPERIMENT...")
    
    # Path to original PSNR results
    original_psnr_stats = Path(__file__).parent / "experiments" / "fixed_comparison" / "experiments" / "full_comparison" / "psnr_dataset_png" / "statistics.json"
    
    if original_psnr_stats.exists():
        with open(original_psnr_stats, 'r') as f:
            psnr_stats = json.load(f)
        
        print("📊 ORIGINAL PSNR OPTIMIZATION (500 images):")
        psnr_improvement = psnr_stats.get('psnr_improvement', {})
        loss_reduction = psnr_stats.get('loss_reduction', {})
        
        print(f"  PSNR improvement mean: {psnr_improvement.get('mean', 0):.6f} dB")
        print(f"  Loss reduction mean: {loss_reduction.get('mean', 0):.2f}%")
        
        if psnr_improvement.get('mean', 0) < 0:
            print("❌ Original PSNR optimization shows negative improvement")
        if loss_reduction.get('mean', 0) == 0:
            print("❌ Original PSNR optimization shows 0% loss reduction")
            
        print("\n💡 EXPECTED MSE IMPROVEMENT:")
        print("  MSE optimization should show:")
        print("  - Positive PSNR improvement (>0 dB)")  
        print("  - Positive loss reduction (>0%)")
        print("  - Similar or better performance than PSNR optimization")
    else:
        print("❌ Original PSNR experiment results not found")

def main():
    """Main function for small-scale MSE test"""
    print("🎯 SMALL-SCALE MSE OPTIMIZATION TEST")
    print("="*60)
    
    # Compare with original results first
    compare_with_original_experiment()
    
    # Run MSE test
    result_summary = run_small_scale_mse_test()
    
    if result_summary:
        print("\n✅ Small-scale MSE test completed successfully")
        print("🔄 Ready to proceed with larger-scale experiments")
    else:
        print("\n❌ Small-scale test failed")
        print("🔧 Need to debug before proceeding")

if __name__ == "__main__":
    main()