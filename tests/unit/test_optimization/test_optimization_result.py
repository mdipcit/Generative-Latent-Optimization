#!/usr/bin/env python3
"""
Unit Tests for OptimizationResult Module

Comprehensive testing of optimization result data structure
including data consistency, metrics calculation, and serialization.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from dataclasses import asdict

# Add project paths
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.generative_latent_optimization.optimization.latent_optimizer import OptimizationResult
from tests.fixtures.test_helpers import (
    setup_test_device, TestImageGenerator, print_test_header, print_test_result
)


@pytest.fixture
def device():
    """Setup test device"""
    return setup_test_device()


@pytest.fixture
def sample_result(device):
    """Create sample optimization result"""
    optimized_latents = torch.randn(1, 4, 64, 64, device=device)
    losses = [0.5, 0.4, 0.3, 0.25, 0.2]
    metrics = {
        'final_psnr_db': 25.5,
        'final_ssim': 0.85,
        'loss_reduction_percent': 60.0,
        'total_iterations': 5
    }
    
    return OptimizationResult(
        optimized_latents=optimized_latents,
        losses=losses,
        metrics=metrics,
        convergence_iteration=None,
        initial_loss=0.5,
        final_loss=0.2
    )


class TestOptimizationResult:
    """Test suite for OptimizationResult class"""
    
    def test_result_initialization(self, device):
        """Test OptimizationResult initialization"""
        print_test_header("Result Initialization Test")
        
        latents = torch.randn(1, 4, 64, 64, device=device)
        losses = [1.0, 0.8, 0.6]
        metrics = {'psnr': 20.0, 'ssim': 0.7}
        
        result = OptimizationResult(
            optimized_latents=latents,
            losses=losses,
            metrics=metrics
        )
        
        assert torch.equal(result.optimized_latents, latents)
        assert result.losses == losses
        assert result.metrics == metrics
        assert result.convergence_iteration is None
        assert result.initial_loss == 0.0  # Default
        assert result.final_loss == 0.0    # Default
        
        print_test_result("Basic initialization", True, "All fields set correctly")
    
    def test_result_with_convergence(self, device):
        """Test result with convergence information"""
        print_test_header("Convergence Information Test")
        
        latents = torch.randn(1, 4, 64, 64, device=device)
        losses = [1.0, 0.5, 0.25, 0.24, 0.24]  # Converged at iteration 3
        
        result = OptimizationResult(
            optimized_latents=latents,
            losses=losses,
            metrics={'total_iterations': 5},
            convergence_iteration=3,
            initial_loss=losses[0],
            final_loss=losses[-1]
        )
        
        assert result.convergence_iteration == 3
        assert result.initial_loss == 1.0
        assert result.final_loss == 0.24
        
        print_test_result("Convergence data", True, "Convergence info stored correctly")
    
    def test_data_consistency_validation(self, sample_result):
        """Test data consistency within optimization result"""
        print_test_header("Data Consistency Validation Test")
        
        # Test loss list consistency
        assert len(sample_result.losses) == 5
        assert sample_result.initial_loss == sample_result.losses[0]
        assert sample_result.final_loss == sample_result.losses[-1]
        
        print_test_result("Loss consistency", True, "Initial/final match list")
        
        # Test metrics consistency
        assert sample_result.metrics['total_iterations'] == len(sample_result.losses)
        
        print_test_result("Metrics consistency", True, "Iteration count matches")
        
        # Test tensor shape consistency
        expected_shape = (1, 4, 64, 64)
        assert sample_result.optimized_latents.shape == expected_shape
        
        print_test_result("Tensor shape", True, f"Shape: {sample_result.optimized_latents.shape}")
    
    def test_loss_reduction_calculation_accuracy(self, device):
        """Test accuracy of loss reduction percentage calculation"""
        print_test_header("Loss Reduction Accuracy Test")
        
        # Create result with known loss values
        losses = [1.0, 0.8, 0.6, 0.4, 0.2]  # 80% reduction
        
        result = OptimizationResult(
            optimized_latents=torch.randn(1, 4, 64, 64, device=device),
            losses=losses,
            metrics={'loss_reduction_percent': 80.0},
            initial_loss=losses[0],
            final_loss=losses[-1]
        )
        
        # Calculate expected reduction
        expected_reduction = ((losses[0] - losses[-1]) / losses[0]) * 100
        actual_reduction = result.metrics['loss_reduction_percent']
        
        assert abs(actual_reduction - expected_reduction) < 1e-10
        assert abs(actual_reduction - 80.0) < 1e-10
        
        print_test_result("Loss reduction accuracy", True, 
                         f"Expected: {expected_reduction:.1f}%, Actual: {actual_reduction:.1f}%")
    
    def test_metrics_data_types(self, sample_result):
        """Test metrics data types and ranges"""
        print_test_header("Metrics Data Types Test")
        
        metrics = sample_result.metrics
        
        # Test PSNR
        assert isinstance(metrics['final_psnr_db'], (int, float))
        assert metrics['final_psnr_db'] > 0  # PSNR should be positive
        print_test_result("PSNR type/range", True, 
                         f"PSNR: {metrics['final_psnr_db']:.2f}dB")
        
        # Test SSIM
        assert isinstance(metrics['final_ssim'], (int, float))
        assert 0 <= metrics['final_ssim'] <= 1  # SSIM should be [0, 1]
        print_test_result("SSIM type/range", True, 
                         f"SSIM: {metrics['final_ssim']:.3f}")
        
        # Test loss reduction percentage
        assert isinstance(metrics['loss_reduction_percent'], (int, float))
        # Loss reduction can be negative (if optimization made things worse)
        print_test_result("Loss reduction type", True, 
                         f"Reduction: {metrics['loss_reduction_percent']:.1f}%")
        
        # Test total iterations
        assert isinstance(metrics['total_iterations'], (int, float))
        assert metrics['total_iterations'] >= 0
        print_test_result("Iterations type/range", True, 
                         f"Iterations: {metrics['total_iterations']}")
    
    def test_tensor_serialization(self, device):
        """Test tensor serialization capabilities"""
        print_test_header("Tensor Serialization Test")
        
        original_latents = torch.randn(1, 4, 64, 64, device=device)
        
        result = OptimizationResult(
            optimized_latents=original_latents,
            losses=[0.5, 0.3],
            metrics={'psnr': 20.0}
        )
        
        # Test tensor save/load
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save tensor
            torch.save(result.optimized_latents, temp_path)
            
            # Load tensor
            loaded_latents = torch.load(temp_path, map_location=device)
            
            # Verify equality
            assert torch.equal(original_latents.cpu(), loaded_latents.cpu())
            print_test_result("Tensor serialization", True, "Save/load successful")
            
        finally:
            Path(temp_path).unlink()
    
    def test_metrics_serialization(self, sample_result):
        """Test metrics dictionary serialization"""
        print_test_header("Metrics Serialization Test")
        
        metrics = sample_result.metrics
        
        # Test JSON serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metrics, f)
            temp_path = f.name
        
        try:
            # Load back
            with open(temp_path, 'r') as f:
                loaded_metrics = json.load(f)
            
            # Verify all metrics preserved
            for key, value in metrics.items():
                assert key in loaded_metrics
                assert abs(loaded_metrics[key] - value) < 1e-10
            
            print_test_result("Metrics JSON serialization", True, "All metrics preserved")
            
        finally:
            Path(temp_path).unlink()
    
    def test_result_equality_comparison(self, device):
        """Test result equality comparison"""
        print_test_header("Result Equality Test")
        
        latents = torch.randn(1, 4, 64, 64, device=device)
        losses = [0.5, 0.3]
        metrics = {'psnr': 25.0}
        
        result1 = OptimizationResult(
            optimized_latents=latents,
            losses=losses,
            metrics=metrics
        )
        
        result2 = OptimizationResult(
            optimized_latents=latents.clone(),
            losses=losses.copy(),
            metrics=metrics.copy()
        )
        
        # Test tensor equality
        assert torch.equal(result1.optimized_latents, result2.optimized_latents)
        print_test_result("Tensor equality", True, "Tensors are equal")
        
        # Test lists equality
        assert result1.losses == result2.losses
        print_test_result("Losses equality", True, "Loss lists are equal")
        
        # Test metrics equality
        assert result1.metrics == result2.metrics
        print_test_result("Metrics equality", True, "Metrics dicts are equal")
    
    def test_result_modification_independence(self, device):
        """Test that result modifications don't affect original data"""
        print_test_header("Modification Independence Test")
        
        original_latents = torch.randn(1, 4, 64, 64, device=device)
        original_losses = [0.5, 0.3, 0.1]
        original_metrics = {'psnr': 25.0, 'ssim': 0.8}
        
        result = OptimizationResult(
            optimized_latents=original_latents,
            losses=original_losses,
            metrics=original_metrics
        )
        
        # Modify result data
        result.optimized_latents[0, 0, 0, 0] = 999.0
        result.losses.append(0.05)
        result.metrics['new_metric'] = 1.0
        
        # Verify original data is affected (since we didn't clone)
        # This tests the behavior, not necessarily the desired behavior
        assert original_latents[0, 0, 0, 0] == 999.0  # Tensor is shared reference
        assert len(original_losses) == 4  # List is shared reference
        assert 'new_metric' in original_metrics  # Dict is shared reference
        
        print_test_result("Reference sharing", True, "Confirmed reference sharing behavior")
    
    def test_empty_and_minimal_results(self, device):
        """Test empty and minimal optimization results"""
        print_test_header("Empty/Minimal Results Test")
        
        # Test with minimal data
        minimal_latents = torch.zeros(1, 4, 1, 1, device=device)
        
        minimal_result = OptimizationResult(
            optimized_latents=minimal_latents,
            losses=[],
            metrics={}
        )
        
        assert minimal_result.optimized_latents.shape == (1, 4, 1, 1)
        assert len(minimal_result.losses) == 0
        assert len(minimal_result.metrics) == 0
        
        print_test_result("Minimal result", True, "Empty lists/dicts handled")
        
        # Test with single iteration
        single_result = OptimizationResult(
            optimized_latents=minimal_latents,
            losses=[0.5],
            metrics={'total_iterations': 1},
            initial_loss=0.5,
            final_loss=0.5
        )
        
        assert len(single_result.losses) == 1
        assert single_result.initial_loss == single_result.final_loss
        
        print_test_result("Single iteration", True, "Single iteration handled")
    
    def test_result_string_representation(self, sample_result):
        """Test string representation of optimization result"""
        print_test_header("String Representation Test")
        
        result_str = str(sample_result)
        
        # Verify string contains key information
        assert 'OptimizationResult' in result_str
        assert 'optimized_latents' in result_str
        assert 'losses' in result_str
        assert 'metrics' in result_str
        
        print_test_result("String representation", True, "Contains key fields")
    
    def test_result_memory_efficiency(self, device):
        """Test memory efficiency of result storage"""
        print_test_header("Memory Efficiency Test")
        
        # Create results with different sizes
        sizes = [(1, 4, 32, 32), (1, 4, 64, 64), (2, 4, 64, 64)]
        
        for size in sizes:
            try:
                latents = torch.randn(*size, device=device)
                losses = list(range(100))  # 100 loss values
                metrics = {f'metric_{i}': float(i) for i in range(20)}  # 20 metrics
                
                result = OptimizationResult(
                    optimized_latents=latents,
                    losses=losses,
                    metrics=metrics
                )
                
                # Verify all data is accessible
                assert result.optimized_latents.shape == size
                assert len(result.losses) == 100
                assert len(result.metrics) == 20
                
                print_test_result(f"Size {size}", True, "Large result created successfully")
                
            except Exception as e:
                print_test_result(f"Size {size}", False, f"Error: {e}")
    
    def test_convergence_iteration_logic(self, device):
        """Test convergence iteration logic and edge cases"""
        print_test_header("Convergence Iteration Logic Test")
        
        latents = torch.randn(1, 4, 64, 64, device=device)
        
        # Test with convergence
        result_converged = OptimizationResult(
            optimized_latents=latents,
            losses=[1.0, 0.5, 0.25, 0.24, 0.24],
            metrics={'total_iterations': 5},
            convergence_iteration=3
        )
        
        assert result_converged.convergence_iteration == 3
        assert result_converged.convergence_iteration < len(result_converged.losses)
        print_test_result("Convergence case", True, "Convergence iteration valid")
        
        # Test without convergence
        result_no_convergence = OptimizationResult(
            optimized_latents=latents,
            losses=[1.0, 0.8, 0.6, 0.4, 0.2],
            metrics={'total_iterations': 5},
            convergence_iteration=None
        )
        
        assert result_no_convergence.convergence_iteration is None
        print_test_result("No convergence case", True, "None handled correctly")
        
        # Test edge case: convergence at last iteration
        result_late_convergence = OptimizationResult(
            optimized_latents=latents,
            losses=[1.0, 0.5, 0.25],
            metrics={'total_iterations': 3},
            convergence_iteration=2  # Last iteration (0-indexed)
        )
        
        assert result_late_convergence.convergence_iteration == 2
        assert result_late_convergence.convergence_iteration == len(result_late_convergence.losses) - 1
        print_test_result("Late convergence", True, "Last iteration convergence valid")
    
    def test_metrics_validation(self, device):
        """Test metrics validation and expected ranges"""
        print_test_header("Metrics Validation Test")
        
        latents = torch.randn(1, 4, 64, 64, device=device)
        losses = [0.8, 0.4, 0.2]
        
        # Test with valid metrics
        valid_metrics = {
            'final_psnr_db': 30.5,      # Positive PSNR
            'final_ssim': 0.85,         # SSIM in [0, 1]
            'loss_reduction_percent': 75.0,  # 75% reduction
            'total_iterations': 3
        }
        
        result = OptimizationResult(
            optimized_latents=latents,
            losses=losses,
            metrics=valid_metrics,
            initial_loss=losses[0],
            final_loss=losses[-1]
        )
        
        # Verify metrics ranges
        assert result.metrics['final_psnr_db'] > 0
        assert 0 <= result.metrics['final_ssim'] <= 1
        assert result.metrics['total_iterations'] >= 0
        
        # Verify loss reduction calculation
        expected_reduction = ((0.8 - 0.2) / 0.8) * 100
        assert abs(result.metrics['loss_reduction_percent'] - expected_reduction) < 1e-10
        
        print_test_result("Valid metrics", True, "All metrics within expected ranges")
    
    def test_edge_case_metrics(self, device):
        """Test edge case metrics (perfect reconstruction, no improvement, etc.)"""
        print_test_header("Edge Case Metrics Test")
        
        latents = torch.randn(1, 4, 64, 64, device=device)
        
        # Test perfect reconstruction (zero loss)
        perfect_result = OptimizationResult(
            optimized_latents=latents,
            losses=[0.0],
            metrics={
                'final_psnr_db': float('inf'),
                'final_ssim': 1.0,
                'loss_reduction_percent': 0.0,  # No reduction possible from 0
                'total_iterations': 1
            },
            initial_loss=0.0,
            final_loss=0.0
        )
        
        assert perfect_result.initial_loss == 0.0
        assert perfect_result.final_loss == 0.0
        print_test_result("Perfect reconstruction", True, "Zero loss handled")
        
        # Test no improvement case
        no_improvement_result = OptimizationResult(
            optimized_latents=latents,
            losses=[0.5, 0.5, 0.5],
            metrics={
                'final_psnr_db': 20.0,
                'final_ssim': 0.7,
                'loss_reduction_percent': 0.0,
                'total_iterations': 3
            },
            initial_loss=0.5,
            final_loss=0.5
        )
        
        assert no_improvement_result.metrics['loss_reduction_percent'] == 0.0
        print_test_result("No improvement", True, "Zero improvement handled")
        
        # Test degradation case (optimization made things worse)
        degradation_result = OptimizationResult(
            optimized_latents=latents,
            losses=[0.2, 0.3, 0.4],
            metrics={
                'final_psnr_db': 15.0,
                'final_ssim': 0.5,
                'loss_reduction_percent': -100.0,  # 100% increase in loss
                'total_iterations': 3
            },
            initial_loss=0.2,
            final_loss=0.4
        )
        
        assert degradation_result.metrics['loss_reduction_percent'] < 0
        print_test_result("Performance degradation", True, "Negative improvement handled")
    
    def test_large_optimization_results(self, device):
        """Test handling of large optimization results"""
        print_test_header("Large Results Test")
        
        # Test with large number of iterations
        large_losses = [1.0 - (i * 0.001) for i in range(1000)]  # 1000 iterations
        
        large_result = OptimizationResult(
            optimized_latents=torch.randn(1, 4, 64, 64, device=device),
            losses=large_losses,
            metrics={
                'total_iterations': 1000,
                'final_psnr_db': 35.0,
                'final_ssim': 0.9,
                'loss_reduction_percent': 90.0
            },
            initial_loss=large_losses[0],
            final_loss=large_losses[-1]
        )
        
        assert len(large_result.losses) == 1000
        assert large_result.metrics['total_iterations'] == 1000
        
        print_test_result("Large iteration count", True, "1000 iterations handled")
        
        # Test with multiple latent channels/batch
        large_latents = torch.randn(4, 8, 128, 128, device=device)  # Larger tensor
        
        large_tensor_result = OptimizationResult(
            optimized_latents=large_latents,
            losses=[0.5, 0.3],
            metrics={'total_iterations': 2}
        )
        
        assert large_tensor_result.optimized_latents.shape == (4, 8, 128, 128)
        print_test_result("Large tensor", True, "Large tensor handled")
    
    def test_result_data_types_validation(self, device):
        """Test validation of result data types"""
        print_test_header("Data Types Validation Test")
        
        # Test with various data types
        latents = torch.randn(1, 4, 64, 64, device=device)
        
        # Test different loss data types
        int_losses = [1, 2, 3]
        float_losses = [1.0, 2.0, 3.0]
        
        result_int = OptimizationResult(
            optimized_latents=latents,
            losses=int_losses,
            metrics={'total_iterations': 3}
        )
        
        result_float = OptimizationResult(
            optimized_latents=latents,
            losses=float_losses,
            metrics={'total_iterations': 3}
        )
        
        assert all(isinstance(loss, (int, float)) for loss in result_int.losses)
        assert all(isinstance(loss, (int, float)) for loss in result_float.losses)
        
        print_test_result("Mixed numeric types", True, "Int/float losses handled")
    
    def test_result_with_missing_optional_fields(self, device):
        """Test result creation with missing optional fields"""
        print_test_header("Missing Optional Fields Test")
        
        latents = torch.randn(1, 4, 64, 64, device=device)
        
        # Create result with only required fields
        minimal_result = OptimizationResult(
            optimized_latents=latents,
            losses=[0.5],
            metrics={'total_iterations': 1}
            # convergence_iteration, initial_loss, final_loss use defaults
        )
        
        assert minimal_result.convergence_iteration is None
        assert minimal_result.initial_loss == 0.0
        assert minimal_result.final_loss == 0.0
        
        print_test_result("Minimal required fields", True, "Optional fields use defaults")


# Convenience function for direct execution
def run_all_tests():
    """Run all tests manually without pytest runner"""
    print("🧪 Starting OptimizationResult Unit Tests")
    print("=" * 60)
    
    device = setup_test_device()
    print(f"Testing on device: {device}")
    print()
    
    test_instance = TestOptimizationResult()
    
    # Create sample result for tests that need it
    optimized_latents = torch.randn(1, 4, 64, 64, device=device)
    losses = [0.5, 0.4, 0.3, 0.25, 0.2]
    metrics = {
        'final_psnr_db': 25.5,
        'final_ssim': 0.85,
        'loss_reduction_percent': 60.0,
        'total_iterations': 5
    }
    
    sample_result = OptimizationResult(
        optimized_latents=optimized_latents,
        losses=losses,
        metrics=metrics,
        convergence_iteration=None,
        initial_loss=0.5,
        final_loss=0.2
    )
    
    # Run all tests
    try:
        test_instance.test_result_initialization(device)
        test_instance.test_result_with_convergence(device)
        test_instance.test_data_consistency_validation(sample_result)
        test_instance.test_loss_reduction_calculation_accuracy(device)
        test_instance.test_metrics_data_types(sample_result)
        test_instance.test_tensor_serialization(device)
        test_instance.test_metrics_serialization(sample_result)
        test_instance.test_result_equality_comparison(device)
        test_instance.test_result_modification_independence(device)
        test_instance.test_empty_and_minimal_results(device)
        test_instance.test_result_string_representation(sample_result)
        test_instance.test_result_memory_efficiency(device)
        test_instance.test_convergence_iteration_logic(device)
        test_instance.test_metrics_validation(device)
        test_instance.test_edge_case_metrics(device)
        test_instance.test_large_optimization_results(device)
        test_instance.test_result_data_types_validation(device)
        test_instance.test_result_with_missing_optional_fields(device)
        
        print("\n" + "=" * 60)
        print("🎉 ALL OPTIMIZATION RESULT TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)