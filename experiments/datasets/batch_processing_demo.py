#!/usr/bin/env python3
"""
Test Script for Batch Processing

Tests the batch processing functionality with a small subset
of BSDS500 images to verify the implementation works correctly.
"""

import os
import torch
from pathlib import Path

# Test imports
try:
    from generative_latent_optimization import OptimizationConfig
    from generative_latent_optimization.workflows import optimize_bsds500_test
    from generative_latent_optimization.dataset import BatchProcessor, BatchProcessingConfig
    print("✅ All batch processing imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)


def test_batch_processing():
    """Test batch processing with a small subset of BSDS500"""
    
    print("🚀 Starting batch processing test...")
    
    # Check BSDS500_PATH
    bsds500_path = os.environ.get("BSDS500_PATH")
    if not bsds500_path:
        print("❌ BSDS500_PATH environment variable not set")
        return False
    
    print(f"BSDS500 path: {bsds500_path}")
    
    # Check if path exists
    if not Path(bsds500_path).exists():
        print(f"❌ BSDS500 path not found: {bsds500_path}")
        return False
    
    try:
        # Test with small subset
        print("🧪 Testing with 3 images from BSDS500...")
        output_path = "./test_batch_dataset.pt"
        
        dataset_path = optimize_bsds500_test(
            output_path=output_path,
            max_images=3  # Very small test
        )
        
        print(f"✅ Test batch processing completed!")
        print(f"Dataset created: {dataset_path}")
        
        # Verify dataset
        if Path(dataset_path).exists():
            dataset = torch.load(dataset_path)
            print(f"📊 Dataset verification:")
            print(f"  Total samples: {dataset['metadata']['total_samples']}")
            print(f"  Splits: {dataset['metadata']['splits_count']}")
            
            if dataset['metadata']['total_samples'] > 0:
                avg_metrics = dataset['metadata']['average_metrics']
                print(f"  Avg PSNR improvement: {avg_metrics['avg_psnr_improvement']:.2f} dB")
                print(f"  Avg loss reduction: {avg_metrics['avg_loss_reduction']:.1f}%")
                
                print("✅ Batch processing test successful!")
                return True
            else:
                print("❌ No samples in dataset")
                return False
        else:
            print(f"❌ Dataset file not created: {dataset_path}")
            return False
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_batch_components():
    """Test individual batch processing components"""
    
    print("\n🧪 Testing individual batch components...")
    
    try:
        # Test BatchProcessingConfig
        print("Testing BatchProcessingConfig...")
        batch_config = BatchProcessingConfig(
            batch_size=4,
            max_images=10,
            save_visualizations=True
        )
        print(f"  Config created: batch_size={batch_config.batch_size}")
        
        # Test OptimizationConfig 
        print("Testing OptimizationConfig for batch...")
        opt_config = OptimizationConfig(
            iterations=25,
            learning_rate=0.2,
            checkpoint_interval=5
        )
        print(f"  Config created: iterations={opt_config.iterations}")
        
        print("✅ Individual batch component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("BATCH PROCESSING TEST")
    print("=" * 50)
    
    # Test components first
    component_test = test_individual_batch_components()
    
    # Run batch test if components work  
    if component_test:
        batch_test = test_batch_processing()
        
        if batch_test:
            print("\n🎉 ALL BATCH TESTS PASSED!")
            print("The batch processing system is working correctly.")
            print("\nTo run full BSDS500 optimization:")
            print("  from generative_latent_optimization.workflows import optimize_bsds500_full")
            print("  dataset_path = optimize_bsds500_full('./full_bsds500_dataset.pt')")
        else:
            print("\n💥 BATCH TEST FAILED!")
    else:
        print("\n💥 COMPONENT TESTS FAILED!")
        print("Please check the batch processing implementations.")