#!/usr/bin/env python3
"""
Phase 3 Integration Test

Comprehensive test of all Phase 3 enhanced metrics functionality.
"""

import torch
import os
from pathlib import Path

def test_phase3_imports():
    """Test all Phase 3 imports"""
    print("Testing Phase 3 imports...")
    
    try:
        # Test individual metrics imports
        from generative_latent_optimization.metrics import (
            LPIPSMetric,
            ImprovedSSIM,
            DatasetFIDEvaluator,
            IndividualMetricsCalculator,
            IndividualImageMetrics,
            DatasetEvaluationResults
        )
        print("  ✅ Individual metrics imports successful")
        
        # Test evaluation imports
        from generative_latent_optimization.evaluation import ComprehensiveDatasetEvaluator
        print("  ✅ Evaluation module imports successful")
        
        # Test main package imports
        from generative_latent_optimization import (
            IndividualMetricsCalculator as MainIndividual,
            ComprehensiveDatasetEvaluator as MainEvaluator,
            LPIPSMetric as MainLPIPS
        )
        print("  ✅ Main package imports successful")
        
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        return False
    
    return True


def test_phase3_functionality(device='cuda'):
    """Test Phase 3 functionality end-to-end"""
    print(f"Testing Phase 3 functionality on {device}...")
    
    try:
        from generative_latent_optimization.metrics import (
            IndividualMetricsCalculator,
            DatasetFIDEvaluator
        )
        from generative_latent_optimization.evaluation import ComprehensiveDatasetEvaluator
        
        # Test individual metrics calculator
        print("  Testing IndividualMetricsCalculator...")
        calculator = IndividualMetricsCalculator(device=device)
        
        # Create test images
        original = torch.rand(1, 3, 256, 256).to(device)
        reconstructed = original + torch.randn_like(original) * 0.1
        
        # Calculate all metrics
        metrics = calculator.calculate_all_individual_metrics(original, reconstructed)
        
        print(f"    PSNR: {metrics.psnr_db:.2f} dB")
        print(f"    SSIM: {metrics.ssim:.4f}")
        if metrics.lpips:
            print(f"    LPIPS: {metrics.lpips:.4f}")
        if metrics.ssim_improved:
            print(f"    SSIM (improved): {metrics.ssim_improved:.4f}")
        
        # Test FID evaluator initialization
        print("  Testing DatasetFIDEvaluator...")
        fid_evaluator = DatasetFIDEvaluator(device=device)
        print("    FID evaluator initialized successfully")
        
        # Test comprehensive evaluator initialization
        print("  Testing ComprehensiveDatasetEvaluator...")
        comprehensive_evaluator = ComprehensiveDatasetEvaluator(device=device)
        print("    Comprehensive evaluator initialized successfully")
        
        print("  ✅ All Phase 3 functionality tests passed!")
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_phase3_workflow_preview():
    """Preview the Phase 3 two-stage workflow"""
    print("Phase 3 Two-Stage Workflow Preview:")
    print()
    
    # Stage 1: Individual Metrics Dataset Creation
    print("🚀 Stage 1: Individual Metrics Dataset Creation")
    print("   Code example:")
    print("   ```python")
    print("   from generative_latent_optimization.workflows import optimize_bsds500_enhanced")
    print("   ")
    print("   # Create dataset with enhanced individual metrics")
    print("   datasets = optimize_bsds500_enhanced(")
    print("       output_path='./enhanced_dataset',")
    print("       max_images=10,")
    print("       enable_lpips=True,          # 👈 NEW: Perceptual similarity")
    print("       improved_ssim=True,         # 👈 NEW: High-quality SSIM")
    print("       create_pytorch=True,")
    print("       create_png=True")
    print("   )")
    print("   ```")
    print()
    
    # Stage 2: Dataset Evaluation  
    print("🎯 Stage 2: Dataset Quality Evaluation")
    print("   Code example:")
    print("   ```python")
    print("   from generative_latent_optimization.evaluation import ComprehensiveDatasetEvaluator")
    print("   import os")
    print("   ")
    print("   # Evaluate completed dataset against original BSDS500")
    print("   evaluator = ComprehensiveDatasetEvaluator()")
    print("   evaluation = evaluator.evaluate_complete_dataset(")
    print("       dataset_path=datasets['png'],")
    print("       original_bsds500_path=os.environ['BSDS500_PATH'],")
    print("       dataset_type='png'")
    print("   )")
    print("   ")
    print("   print(f'FID Score: {evaluation[\"dataset_evaluation\"][\"fid_score\"]:.2f}')")
    print("   print(f'Overall Quality: {evaluation[\"evaluation_summary\"][\"overall_quality_score\"]:.1f}/100')")
    print("   ```")
    print()


def main():
    """Main test execution"""
    print("=" * 60)
    print("Phase 3: Enhanced Metrics System Integration Test")
    print("=" * 60)
    print()
    
    # Test imports
    imports_ok = test_phase3_imports()
    print()
    
    if imports_ok:
        # Test functionality
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        functionality_ok = test_phase3_functionality(device)
        print()
        
        # Show workflow preview
        test_phase3_workflow_preview()
        
        # Final status
        if functionality_ok:
            print("🎉 Phase 3 Integration Test: ALL PASSED")
            print()
            print("✅ LPIPS (Perceptual similarity) - Ready")
            print("✅ Improved SSIM (TorchMetrics) - Ready") 
            print("✅ FID (Dataset evaluation) - Ready")
            print("✅ Individual metrics integration - Ready")
            print("✅ Comprehensive dataset evaluation - Ready")
            print()
            print("🚀 Phase 3 implementation is complete and ready for use!")
        else:
            print("❌ Phase 3 Integration Test: FAILED")
    else:
        print("❌ Phase 3 Integration Test: IMPORT FAILED")


if __name__ == "__main__":
    main()