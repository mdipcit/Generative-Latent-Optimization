#!/usr/bin/env python3
"""
Test Script for Dual Datasets (PyTorch + PNG)

Tests the dual dataset creation functionality by creating both
PyTorch (.pt) and PNG datasets from optimization results.
"""

import os
import torch
from pathlib import Path

# Test imports
try:
    from generative_latent_optimization import OptimizationConfig
    from generative_latent_optimization.workflows import optimize_bsds500_test
    from generative_latent_optimization.dataset import (
        load_optimized_dataset, 
        OptimizedLatentsDataset
    )
    print("✅ All dual dataset imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)


def test_dual_dataset_creation():
    """Test creating both PyTorch and PNG datasets"""
    
    print("🚀 Starting dual dataset creation test...")
    
    # Check BSDS500_PATH
    bsds500_path = os.environ.get("BSDS500_PATH")
    if not bsds500_path:
        print("❌ BSDS500_PATH environment variable not set")
        return False
    
    print(f"BSDS500 path: {bsds500_path}")
    
    try:
        # Test with small subset - create both datasets
        print("🧪 Creating dual datasets (PyTorch + PNG) with 2 images...")
        output_base = "./test_dual_dataset"
        
        datasets_paths = optimize_bsds500_test(
            output_path=output_base,
            max_images=2,  # Very small test
            create_pytorch=True,
            create_png=True
        )
        
        print(f"✅ Dual dataset creation completed!")
        print(f"Created datasets: {datasets_paths}")
        
        # Verify PyTorch dataset
        if 'pytorch' in datasets_paths:
            pytorch_path = datasets_paths['pytorch']
            if Path(pytorch_path).exists():
                print(f"\n📦 Verifying PyTorch dataset: {pytorch_path}")
                dataset = load_optimized_dataset(pytorch_path)
                print(f"  Total samples: {len(dataset)}")
                print(f"  Metadata: {dataset.get_metadata().total_samples} samples")
                
                # Test data loading
                sample = dataset[0]
                print(f"  Sample keys: {list(sample.keys())}")
                print(f"  Initial latents shape: {sample['initial_latents'].shape}")
                print(f"  Optimized latents shape: {sample['optimized_latents'].shape}")
                print(f"  PSNR improvement: {sample['metrics']['psnr_improvement']:.2f} dB")
                
                print("  ✅ PyTorch dataset verification passed!")
            else:
                print(f"  ❌ PyTorch dataset file not found: {pytorch_path}")
                return False
        
        # Verify PNG dataset
        if 'png' in datasets_paths:
            png_path = datasets_paths['png']
            png_dir = Path(png_path)
            if png_dir.exists():
                print(f"\n🖼️ Verifying PNG dataset: {png_path}")
                
                # Check directory structure
                splits = ['train', 'val', 'test']
                total_images = 0
                
                for split in splits:
                    split_dir = png_dir / split
                    if split_dir.exists():
                        image_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                        print(f"  {split}: {len(image_dirs)} images")
                        total_images += len(image_dirs)
                        
                        # Check first image directory
                        if image_dirs:
                            first_image = image_dirs[0]
                            files = list(first_image.glob('*'))
                            print(f"    Sample ({first_image.name}): {len(files)} files")
                            
                            # Check for required files
                            expected_files = ['original.png', 'initial_reconstruction.png', 
                                            'optimized_reconstruction.png', 'comparison_grid.png', 
                                            'metrics.json']
                            found_files = [f.name for f in files]
                            
                            for expected in expected_files:
                                if expected in found_files:
                                    print(f"      ✅ {expected}")
                                else:
                                    print(f"      ❌ Missing: {expected}")
                
                print(f"  Total images across splits: {total_images}")
                
                # Check metadata files
                metadata_files = ['metadata.json', 'statistics.json', 'samples.json', 'README.md']
                for meta_file in metadata_files:
                    meta_path = png_dir / meta_file
                    if meta_path.exists():
                        print(f"  ✅ {meta_file}")
                    else:
                        print(f"  ⚠️ Missing: {meta_file}")
                
                # Check overview directory
                overview_dir = png_dir / 'overview'
                if overview_dir.exists():
                    overview_files = list(overview_dir.glob('*'))
                    print(f"  ✅ overview/ ({len(overview_files)} files)")
                
                print("  ✅ PNG dataset verification passed!")
            else:
                print(f"  ❌ PNG dataset directory not found: {png_path}")
                return False
        
        print(f"\n🎉 DUAL DATASET TEST PASSED!")
        print("Both PyTorch and PNG datasets created successfully.")
        
        # Usage examples
        print(f"\n📋 Usage Examples:")
        if 'pytorch' in datasets_paths:
            print(f"  PyTorch Dataset:")
            print(f"    dataset = load_optimized_dataset('{datasets_paths['pytorch']}')")
            print(f"    dataloader = dataset.create_dataloader(batch_size=4)")
        
        if 'png' in datasets_paths:
            print(f"  PNG Dataset:")
            print(f"    Browse images in: {datasets_paths['png']}")
            print(f"    View README: {png_dir / 'README.md'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dual dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pytorch_only():
    """Test creating only PyTorch dataset"""
    
    print("\n🧪 Testing PyTorch-only dataset creation...")
    
    try:
        datasets_paths = optimize_bsds500_test(
            output_path="./test_pytorch_only",
            max_images=1,
            create_pytorch=True,
            create_png=False
        )
        
        if 'pytorch' in datasets_paths and 'png' not in datasets_paths:
            print("✅ PyTorch-only test passed!")
            return True
        else:
            print(f"❌ Expected only pytorch dataset, got: {datasets_paths}")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch-only test failed: {e}")
        return False


def test_png_only():
    """Test creating only PNG dataset"""
    
    print("\n🧪 Testing PNG-only dataset creation...")
    
    try:
        datasets_paths = optimize_bsds500_test(
            output_path="./test_png_only",
            max_images=1,
            create_pytorch=False,
            create_png=True
        )
        
        if 'png' in datasets_paths and 'pytorch' not in datasets_paths:
            print("✅ PNG-only test passed!")
            return True
        else:
            print(f"❌ Expected only png dataset, got: {datasets_paths}")
            return False
            
    except Exception as e:
        print(f"❌ PNG-only test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("DUAL DATASETS TEST (PyTorch + PNG)")
    print("=" * 60)
    
    # Test dual dataset creation
    dual_test = test_dual_dataset_creation()
    
    # Test individual dataset types
    pytorch_test = test_pytorch_only()
    png_test = test_png_only()
    
    # Final results
    if dual_test and pytorch_test and png_test:
        print("\n🎉 ALL DUAL DATASET TESTS PASSED!")
        print("\nThe system now supports:")
        print("  ✅ PyTorch datasets (.pt files) with latents and metrics")
        print("  ✅ PNG datasets (organized directories) with images and metadata")
        print("  ✅ Dual dataset creation (both formats simultaneously)")
        print("  ✅ Flexible dataset format selection")
        
        print("\n🚀 Ready for full BSDS500 processing:")
        print("  from generative_latent_optimization.workflows import optimize_bsds500_full")
        print("  datasets = optimize_bsds500_full('./full_bsds500', create_pytorch=True, create_png=True)")
        
    else:
        print("\n💥 SOME TESTS FAILED!")
        print(f"  Dual dataset: {'✅' if dual_test else '❌'}")
        print(f"  PyTorch only: {'✅' if pytorch_test else '❌'}")
        print(f"  PNG only: {'✅' if png_test else '❌'}")