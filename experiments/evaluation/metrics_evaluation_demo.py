#!/usr/bin/env python3
"""
Simple All Metrics Evaluator Test

Test the simplified metrics evaluation system implementation.
"""

import torch
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def create_test_images(output_dir: Path, num_images=5):
    """Create test images for evaluation"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} test images in {output_dir}")
    
    # Create some test images
    for i in range(num_images):
        # Generate random 512x512 RGB images
        image_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, 'RGB')
        
        image_path = output_dir / f"test_{i:04d}.png"
        image.save(image_path)
    
    print(f"Created {len(list(output_dir.glob('*.png')))} test images")
    return list(output_dir.glob('*.png'))

def create_modified_images(original_dir: Path, modified_dir: Path, noise_level=0.1):
    """Create modified versions of original images"""
    modified_dir.mkdir(parents=True, exist_ok=True)
    
    original_images = list(original_dir.glob('*.png'))
    print(f"Creating modified versions of {len(original_images)} images")
    
    for orig_path in original_images:
        # Load original image
        orig_image = Image.open(orig_path)
        orig_array = np.array(orig_image).astype(np.float32) / 255.0
        
        # Add some noise to simulate optimization result
        noise = np.random.randn(*orig_array.shape) * noise_level
        modified_array = np.clip(orig_array + noise, 0, 1)
        
        # Convert back to PIL and save
        modified_array = (modified_array * 255).astype(np.uint8)
        modified_image = Image.fromarray(modified_array, 'RGB')
        
        modified_path = modified_dir / orig_path.name
        modified_image.save(modified_path)
    
    print(f"Created {len(list(modified_dir.glob('*.png')))} modified images")

def test_simple_evaluator_import():
    """Test importing the SimpleAllMetricsEvaluator"""
    print("\n" + "="*50)
    print("Testing SimpleAllMetricsEvaluator Import")
    print("="*50)
    
    try:
        from generative_latent_optimization.evaluation import SimpleAllMetricsEvaluator
        print("✅ Successfully imported SimpleAllMetricsEvaluator from evaluation module")
        
        from generative_latent_optimization import SimpleAllMetricsEvaluator as MainEvaluator
        print("✅ Successfully imported SimpleAllMetricsEvaluator from main package")
        
        return SimpleAllMetricsEvaluator
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return None

def test_evaluator_initialization(evaluator_class):
    """Test evaluator initialization"""
    print("\n" + "="*50)
    print("Testing Evaluator Initialization")
    print("="*50)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Test basic initialization
        evaluator = evaluator_class(device=device)
        print("✅ Basic initialization successful")
        
        # Test with LPIPS disabled (faster for testing)
        evaluator_no_lpips = evaluator_class(
            device=device, 
            enable_lpips=False, 
            enable_improved_ssim=False
        )
        print("✅ Initialization with disabled features successful")
        
        return evaluator_no_lpips
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_full_evaluation(evaluator):
    """Test full evaluation workflow"""
    print("\n" + "="*50)
    print("Testing Full Evaluation Workflow")
    print("="*50)
    
    # Create temporary directories for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        original_dir = temp_path / "original"
        modified_dir = temp_path / "modified"
        
        try:
            # Create test datasets
            create_test_images(original_dir, num_images=3)
            create_modified_images(original_dir, modified_dir, noise_level=0.05)
            
            print(f"\nRunning evaluation...")
            print(f"Original: {original_dir}")
            print(f"Modified: {modified_dir}")
            
            # Run evaluation
            results = evaluator.evaluate_dataset_all_metrics(
                created_dataset_path=modified_dir,
                original_dataset_path=original_dir
            )
            
            print("✅ Evaluation completed successfully!")
            
            # Verify results structure
            if hasattr(results, 'individual_metrics') and results.individual_metrics:
                print(f"✅ Got {len(results.individual_metrics)} individual metric results")
            
            if hasattr(results, 'fid_score'):
                print(f"✅ FID Score: {results.fid_score:.2f}")
            
            if hasattr(results, 'statistics') and results.statistics:
                print(f"✅ Statistics computed for {len(results.statistics)} metric types")
                for metric_name, stats in results.statistics.items():
                    print(f"   {metric_name}: mean={stats.get('mean', 'N/A'):.4f}")
            
            # Test print summary
            print("\nTesting print_summary method...")
            evaluator.print_summary(results)
            
            return True
            
        except Exception as e:
            print(f"❌ Full evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_error_handling(evaluator):
    """Test error handling for non-existent directories"""
    print("\n" + "="*50)
    print("Testing Error Handling")
    print("="*50)
    
    try:
        # Test with non-existent directories
        results = evaluator.evaluate_dataset_all_metrics(
            created_dataset_path="/nonexistent/path",
            original_dataset_path="/also/nonexistent"
        )
        
        # Should handle gracefully and return results with empty/default values
        print("✅ Handled non-existent directories gracefully")
        return True
        
    except Exception as e:
        print(f"⚠️  Error handling test: {e}")
        return False

def main():
    """Main test execution"""
    print("🧪 Simple All Metrics Evaluator - Comprehensive Test")
    print("=" * 60)
    
    # Test 1: Import
    evaluator_class = test_simple_evaluator_import()
    if not evaluator_class:
        print("❌ Test failed at import stage")
        return
    
    # Test 2: Initialization  
    evaluator = test_evaluator_initialization(evaluator_class)
    if not evaluator:
        print("❌ Test failed at initialization stage")
        return
    
    # Test 3: Full evaluation
    full_eval_success = test_full_evaluation(evaluator)
    
    # Test 4: Error handling
    error_handling_success = test_error_handling(evaluator)
    
    # Summary
    print("\n" + "="*60)
    print("🏁 Test Results Summary")
    print("="*60)
    print("✅ Import: PASSED")
    print("✅ Initialization: PASSED")
    print(f"{'✅' if full_eval_success else '❌'} Full Evaluation: {'PASSED' if full_eval_success else 'FAILED'}")
    print(f"{'✅' if error_handling_success else '❌'} Error Handling: {'PASSED' if error_handling_success else 'FAILED'}")
    
    if full_eval_success and error_handling_success:
        print("\n🎉 All tests PASSED! SimpleAllMetricsEvaluator is ready for use!")
        
        print("\n📋 Usage Example:")
        print("```python")
        print("from generative_latent_optimization import SimpleAllMetricsEvaluator")
        print("")
        print("# Initialize evaluator")
        print("evaluator = SimpleAllMetricsEvaluator(device='cuda')")
        print("")
        print("# Evaluate dataset")
        print("results = evaluator.evaluate_dataset_all_metrics(")
        print("    created_dataset_path='./my_created_dataset',")
        print("    original_dataset_path='./original_bsds500'")
        print(")")
        print("")
        print("# Display summary")
        print("evaluator.print_summary(results)")
        print("```")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()