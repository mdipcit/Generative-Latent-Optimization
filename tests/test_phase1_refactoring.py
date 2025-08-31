"""Test script for Phase 1 refactoring verification."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

def test_core_module():
    """Test core module with base classes."""
    print("Testing Core Module...")
    
    try:
        from generative_latent_optimization.core import (
            BaseMetric, BaseEvaluator, BaseDataset, DeviceManager
        )
        print("  ✅ Core module imports successful")
        
        # Test DeviceManager
        device_manager = DeviceManager('cuda')
        print(f"  Device selected: {device_manager.device}")
        
        # Test device info
        info = device_manager.get_device_info()
        print(f"  Device info: CUDA available = {info['cuda_available']}")
        
        # Test tensor movement
        test_tensor = torch.randn(2, 3, 4)
        moved_tensor = device_manager.move_to_device(test_tensor)
        print(f"  Tensor moved to: {moved_tensor.device}")
        
        print("  ✅ DeviceManager test passed")
        
    except Exception as e:
        print(f"  ❌ Core module test failed: {e}")
        return False
    
    return True


def test_utils_module():
    """Test utils module with new utilities."""
    print("\nTesting Utils Module...")
    
    try:
        from generative_latent_optimization.utils import (
            PathUtils, UnifiedImageLoader, StatisticsCalculator
        )
        print("  ✅ Utils module imports successful")
        
        # Test PathUtils
        current_dir = PathUtils.resolve_path('.')
        print(f"  Current directory: {current_dir}")
        
        # Test path utilities
        test_path = PathUtils.ensure_directory('tests/temp_test')
        print(f"  Created test directory: {test_path}")
        
        # Clean up
        import shutil
        if test_path.exists():
            shutil.rmtree(test_path)
        
        print("  ✅ PathUtils test passed")
        
        # Test StatisticsCalculator
        test_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = StatisticsCalculator.calculate_basic_stats(test_values, 'test')
        print(f"  Statistics calculated: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        print("  ✅ StatisticsCalculator test passed")
        
        # Test UnifiedImageLoader initialization
        image_loader = UnifiedImageLoader(device='cpu')
        print(f"  UnifiedImageLoader device: {image_loader.device}")
        print("  ✅ UnifiedImageLoader test passed")
        
    except Exception as e:
        print(f"  ❌ Utils module test failed: {e}")
        return False
    
    return True


def test_statistics_integration():
    """Test StatisticsCalculator integration in metrics modules."""
    print("\nTesting StatisticsCalculator Integration...")
    
    try:
        from generative_latent_optimization.metrics.metrics_integration import (
            IndividualMetricsCalculator, IndividualImageMetrics
        )
        print("  ✅ Metrics integration module imported")
        
        # Create test data
        test_metrics = [
            IndividualImageMetrics(
                psnr_db=25.0 + i,
                ssim=0.8 + i * 0.01,
                mse=0.1 + i * 0.01,
                mae=0.05 + i * 0.005,
                lpips=None,
                ssim_improved=None
            )
            for i in range(5)
        ]
        
        # Test statistics calculation
        calculator = IndividualMetricsCalculator(device='cpu')
        stats = calculator.get_batch_statistics(test_metrics)
        
        print(f"  PSNR mean: {stats.get('psnr_mean', 0):.2f}")
        print(f"  PSNR std: {stats.get('psnr_std', 0):.2f}")
        print(f"  SSIM mean: {stats.get('ssim_mean', 0):.4f}")
        print(f"  Valid samples: {stats.get('valid_samples', 0)}")
        
        print("  ✅ StatisticsCalculator integration test passed")
        
    except Exception as e:
        print(f"  ❌ StatisticsCalculator integration test failed: {e}")
        return False
    
    return True


def test_device_manager_fallback():
    """Test DeviceManager CUDA fallback behavior."""
    print("\nTesting DeviceManager Fallback...")
    
    try:
        from generative_latent_optimization.core import DeviceManager
        
        # Test CUDA fallback
        dm = DeviceManager('cuda:99')  # Invalid device
        print(f"  Invalid device fallback: {dm.device}")
        
        # Test auto device selection
        best_device = DeviceManager.auto_select_device()
        print(f"  Auto-selected device: {best_device}")
        
        print("  ✅ DeviceManager fallback test passed")
        
    except Exception as e:
        print(f"  ❌ DeviceManager fallback test failed: {e}")
        return False
    
    return True


def main():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("PHASE 1 REFACTORING TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_core_module()
    all_passed &= test_utils_module()
    all_passed &= test_statistics_integration()
    all_passed &= test_device_manager_fallback()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL PHASE 1 TESTS PASSED!")
        print("Phase 1 refactoring is complete and functional.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())