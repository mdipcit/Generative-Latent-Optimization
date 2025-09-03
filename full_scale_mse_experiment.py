#!/usr/bin/env python3
"""
Full-scale MSE optimization experiment - Step by step execution
Based on successful 30-image test: +2.13 dB PSNR improvement
"""

import torch
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

from generative_latent_optimization.workflows.batch_processing import process_bsds500_dataset
from generative_latent_optimization.optimization.latent_optimizer import OptimizationConfig

def setup_experiment_configuration():
    """Step 1: Setup experiment configuration"""
    print("📋 STEP 1: Setting up MSE experiment configuration")
    print("-" * 50)
    
    # Verify environment
    import os
    bsds500_path = os.environ.get("BSDS500_PATH")
    if not bsds500_path:
        raise ValueError("BSDS500_PATH environment variable not set")
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  Using CPU (will be slower)")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(__file__).parent / "experiments" / "full_mse_comparison"
    output_dir = output_base / f"mse_experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 BSDS500 path: {bsds500_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create optimized MSE configuration
    config = OptimizationConfig(
        iterations=50,  # Based on CLAUDE.md recommendations
        learning_rate=0.05,  # Proven effective in small-scale test
        loss_function='mse',  # Direct MSE minimization
        device=device,
        checkpoint_interval=10,
        convergence_threshold=1e-6,
        patience=15  # Allow early stopping for efficiency
    )
    
    print(f"\n⚙️  MSE OPTIMIZATION CONFIGURATION:")
    print(f"   Loss function: {config.loss_function}")
    print(f"   Iterations: {config.iterations}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Device: {config.device}")
    print(f"   Convergence threshold: {config.convergence_threshold}")
    print(f"   Patience: {config.patience}")
    
    # Expected results based on small-scale test
    print(f"\n📊 EXPECTED RESULTS (based on 30-image test):")
    print(f"   PSNR improvement: +2.13 ± 0.22 dB")
    print(f"   Loss reduction: ~38%")
    print(f"   Processing time: ~1.5-2 hours")
    print(f"   Success rate: 100%")
    
    return config, output_dir, bsds500_path

def execute_mse_optimization(config, output_dir, bsds500_path):
    """Step 2: Execute MSE optimization on 500 images"""
    print(f"\n🔄 STEP 2: Executing MSE optimization")
    print("-" * 50)
    print(f"Target: 500 BSDS500 images")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Record experiment start
    experiment_start = time.time()
    
    try:
        # Launch optimization process
        print(f"🚀 Starting MSE optimization process...")
        
        result_summary = process_bsds500_dataset(
            bsds500_path=bsds500_path,
            output_path=str(output_dir),
            config=config,
            max_images_per_split=None,  # Process all images
            save_visualizations=True,
            create_pytorch_dataset=True,
            create_png_dataset=True
        )
        
        experiment_time = time.time() - experiment_start
        
        print(f"\n✅ MSE optimization completed!")
        print(f"   Total time: {experiment_time/3600:.2f} hours")
        print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return result_summary, experiment_time
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def monitor_and_analyze_results(output_dir, result_summary, experiment_time):
    """Step 3: Monitor and analyze experiment results"""
    print(f"\n📊 STEP 3: Analyzing experiment results")
    print("-" * 50)
    
    if not result_summary:
        print("❌ No results to analyze")
        return None
    
    # Get PNG dataset info
    png_info = result_summary.get('png_dataset_info', {})
    if not png_info:
        print("❌ PNG dataset info not available")
        return None
    
    # Load statistics
    stats_file = Path(png_info.get('base_path', '')) / "statistics.json"
    if not stats_file.exists():
        print(f"❌ Statistics file not found: {stats_file}")
        return None
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print(f"📈 MSE OPTIMIZATION RESULTS (500 images):")
    print(f"   Total samples: {stats.get('total_samples', 0)}")
    print(f"   Processing time: {experiment_time/3600:.2f} hours")
    
    # PSNR analysis
    psnr_stats = stats.get('psnr_improvement', {})
    if psnr_stats:
        print(f"\n🎯 PSNR IMPROVEMENT ANALYSIS:")
        print(f"   Mean: {psnr_stats.get('mean', 0):.4f} dB")
        print(f"   Std:  {psnr_stats.get('std', 0):.4f} dB")
        print(f"   Min:  {psnr_stats.get('min', 0):.4f} dB") 
        print(f"   Max:  {psnr_stats.get('max', 0):.4f} dB")
        print(f"   Median: {psnr_stats.get('median', 0):.4f} dB")
    
    # Loss reduction analysis
    loss_stats = stats.get('loss_reduction', {})
    if loss_stats:
        print(f"\n⚡ LOSS REDUCTION ANALYSIS:")
        print(f"   Mean: {loss_stats.get('mean', 0):.2f}%")
        print(f"   Std:  {loss_stats.get('std', 0):.2f}%")
        print(f"   Min:  {loss_stats.get('min', 0):.2f}%")
        print(f"   Max:  {loss_stats.get('max', 0):.2f}%")
    
    # Performance assessment
    mean_psnr = psnr_stats.get('mean', 0)
    mean_loss = loss_stats.get('mean', 0)
    
    print(f"\n✨ PERFORMANCE ASSESSMENT:")
    if mean_psnr > 2.0:
        print("🏆 EXCELLENT: >2dB PSNR improvement achieved")
    elif mean_psnr > 1.0:
        print("✅ GOOD: >1dB PSNR improvement achieved")
    elif mean_psnr > 0:
        print("⚠️  MODEST: Positive but small improvement")
    else:
        print("❌ FAILED: Negative PSNR improvement")
    
    if mean_loss > 30:
        print("✅ OPTIMIZATION EFFECTIVE: >30% loss reduction")
    else:
        print("⚠️  OPTIMIZATION CONCERN: <30% loss reduction")
    
    return stats

def calculate_fid_for_mse(output_dir):
    """Step 4: Calculate FID score for MSE optimized images"""
    print(f"\n📊 STEP 4: Calculating FID score")
    print("-" * 50)
    
    # Find PNG dataset directory
    png_dataset_path = None
    for path in output_dir.iterdir():
        if path.is_dir() and "png" in path.name.lower():
            png_dataset_path = path
            break
    
    if not png_dataset_path:
        print("❌ PNG dataset directory not found")
        return None
    
    print(f"📁 PNG dataset: {png_dataset_path}")
    
    # Create FID evaluation script
    fid_script_content = f'''#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generative_latent_optimization.evaluation.simple_evaluator import SimpleAllMetricsEvaluator

def main():
    print("🔍 Calculating FID for MSE optimization...")
    
    # Setup evaluator
    evaluator = SimpleAllMetricsEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate dataset
    dataset_path = "{png_dataset_path}"
    results = evaluator.evaluate_dataset_all_metrics(dataset_path, None)
    
    print(f"📊 FID Score: {{results.fid_score:.2f}}")
    
    # Save FID result
    fid_result = {{
        "fid_score": results.fid_score,
        "timestamp": "{datetime.now().isoformat()}",
        "dataset_path": dataset_path
    }}
    
    import json
    with open("{output_dir}/fid_results.json", "w") as f:
        json.dump(fid_result, f, indent=2)
    
    return results.fid_score

if __name__ == "__main__":
    import torch
    from datetime import datetime
    main()
'''
    
    fid_script_path = output_dir / "calculate_mse_fid.py"
    with open(fid_script_path, 'w') as f:
        f.write(fid_script_content)
    
    print(f"📝 FID calculation script created: {fid_script_path}")
    return fid_script_path

def main():
    """Execute full MSE optimization experiment step by step"""
    print("🎯 FULL-SCALE MSE OPTIMIZATION EXPERIMENT")
    print("=" * 60)
    print("Target: 500 BSDS500 images with corrected evaluation")
    print("Expected: +2.1 dB based on 30-image verification")
    print()
    
    try:
        # Step 1: Configuration
        config, output_dir, bsds500_path = setup_experiment_configuration()
        
        # Step 2: Execute optimization
        result_summary, experiment_time = execute_mse_optimization(config, output_dir, bsds500_path)
        
        if result_summary and experiment_time:
            # Step 3: Analyze results
            stats = monitor_and_analyze_results(output_dir, result_summary, experiment_time)
            
            if stats:
                # Step 4: FID calculation
                fid_script_path = calculate_fid_for_mse(output_dir)
                
                print(f"\n🎉 MSE EXPERIMENT COMPLETED SUCCESSFULLY!")
                print(f"📁 Results: {output_dir}")
                print(f"📊 Ready for method comparison")
                
                return stats, output_dir
            else:
                print("❌ Failed to analyze results")
                return None, None
        else:
            print("❌ Optimization failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()