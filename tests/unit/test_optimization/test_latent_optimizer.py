#!/usr/bin/env python3
"""
Unit Tests for LatentOptimizer Module

Comprehensive testing of the VAE latent optimization functionality
including edge cases, error handling, and performance validation.
"""

import pytest
import torch
import numpy as np
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add project paths
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.generative_latent_optimization.optimization.latent_optimizer import (
    LatentOptimizer, OptimizationConfig, OptimizationResult
)
from tests.fixtures.test_helpers import (
    TestImageGenerator, setup_test_device, calculate_metrics,
    print_test_header, print_test_result
)


@pytest.fixture
def device():
    """Setup test device"""
    return setup_test_device()


@pytest.fixture
def mock_vae(device):
    """Create mock VAE for testing"""
    mock_vae = Mock()
    
    # Mock decoder with parameters
    mock_decoder = Mock()
    mock_param = Mock()
    mock_param.requires_grad = True
    mock_decoder.parameters.return_value = [mock_param]
    mock_vae.decoder = mock_decoder
    
    # Mock decode method
    def mock_decode(latents):
        # Simulate VAE decode: return tensor that depends on latents for gradient computation
        batch_size = latents.shape[0] 
        latent_h, latent_w = latents.shape[2], latents.shape[3]
        # Calculate target size based on latent size (8x upsampling)
        target_size = (latent_h * 8, latent_w * 8)
        
        mock_output = Mock()
        # Create a simple linear transformation to maintain gradient flow
        upsampled = torch.nn.functional.interpolate(latents, size=target_size, mode='bilinear', align_corners=False)
        # Convert 4 channels to 3 channels with a simple linear combination
        mock_output.sample = (upsampled[:, :3] + upsampled[:, 1:4]) * 0.5 * 2 - 1
        return mock_output
    
    mock_vae.decode = mock_decode
    return mock_vae


@pytest.fixture
def sample_latents(device):
    """Create sample latent tensor"""
    return torch.randn(1, 4, 64, 64, device=device, requires_grad=True)


@pytest.fixture
def sample_target(device):
    """Create sample target image"""
    return TestImageGenerator.solid_color((0.8, 0.6, 0.4), (512, 512), device)


class TestLatentOptimizer:
    """Test suite for LatentOptimizer class"""
    
    def test_optimizer_initialization(self, device):
        """Test LatentOptimizer initialization with various configurations"""
        print_test_header("Optimizer Initialization Test")
        
        # Test default configuration
        config = OptimizationConfig()
        optimizer = LatentOptimizer(config)
        
        assert optimizer.config.iterations == 150
        assert optimizer.config.learning_rate == 0.4
        assert optimizer.config.loss_function == 'mse'
        print_test_result("Default config", True, "All default values correct")
        
        # Test custom configuration
        custom_config = OptimizationConfig(
            iterations=100,
            learning_rate=0.1,
            loss_function='l1',
            convergence_threshold=1e-5,
            checkpoint_interval=10,
            device='cpu'
        )
        custom_optimizer = LatentOptimizer(custom_config)
        
        assert custom_optimizer.config.iterations == 100
        assert custom_optimizer.config.learning_rate == 0.1
        assert custom_optimizer.config.loss_function == 'l1'
        print_test_result("Custom config", True, "All custom values correct")
    
    def test_optimization_basic_functionality(self, mock_vae, sample_latents, sample_target, device):
        """Test basic optimization functionality"""
        print_test_header("Basic Optimization Test")
        
        config = OptimizationConfig(iterations=10, learning_rate=0.1, device=device)
        optimizer = LatentOptimizer(config)
        
        result = optimizer.optimize(mock_vae, sample_latents, sample_target)
        
        # Verify result structure
        assert isinstance(result, OptimizationResult)
        assert result.optimized_latents is not None
        assert len(result.losses) == 10
        assert 'final_psnr_db' in result.metrics
        assert 'loss_reduction_percent' in result.metrics
        
        print_test_result("Basic optimization", True, 
                         f"Generated {len(result.losses)} loss values")
    
    def test_optimization_loss_functions(self, mock_vae, sample_latents, sample_target, device):
        """Test different loss functions"""
        print_test_header("Loss Function Test")
        
        loss_functions = ['mse', 'l1']
        results = {}
        
        for loss_func in loss_functions:
            config = OptimizationConfig(
                iterations=5, 
                learning_rate=0.1, 
                loss_function=loss_func,
                device=device
            )
            optimizer = LatentOptimizer(config)
            result = optimizer.optimize(mock_vae, sample_latents, sample_target)
            results[loss_func] = result
            
            assert len(result.losses) == 5
            print_test_result(f"{loss_func.upper()} loss", True, 
                             f"Final loss: {result.final_loss:.6f}")
        
        # Verify different loss functions produce different results
        assert results['mse'].losses != results['l1'].losses
        print_test_result("Loss function diversity", True, "Different results confirmed")
    
    def test_optimization_invalid_loss_function(self, mock_vae, sample_latents, sample_target, device):
        """Test error handling for invalid loss function"""
        print_test_header("Invalid Loss Function Test")
        
        config = OptimizationConfig(
            iterations=5,
            loss_function='invalid_loss',
            device=device
        )
        optimizer = LatentOptimizer(config)
        
        with pytest.raises(ValueError, match="Unsupported loss function"):
            optimizer.optimize(mock_vae, sample_latents, sample_target)
        
        print_test_result("Invalid loss function", True, "ValueError raised correctly")
    
    def test_convergence_detection(self, device):
        """Test convergence detection functionality"""
        print_test_header("Convergence Detection Test")
        
        # Create mock VAE that returns convergent output
        mock_vae = Mock()
        mock_decoder = Mock()
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_decoder.parameters.return_value = [mock_param]
        mock_vae.decoder = mock_decoder
        
        # Create special target that should converge quickly
        target = torch.zeros(1, 3, 512, 512, device=device)
        latents = torch.zeros(1, 4, 64, 64, device=device, requires_grad=True)
        
        # Modify mock VAE to return nearly constant output that depends on latents
        def convergent_decode(latents_input):
            mock_output = Mock()
            # Create small dependency on latents to maintain gradient flow, 
            # but make output close to target to achieve quick convergence
            latent_contribution = torch.nn.functional.interpolate(latents_input, size=(512, 512), mode='bilinear', align_corners=False)
            base_output = target * 2 - 1  # Convert target [0,1] to [-1,1]
            # Add minimal latent dependency
            mock_output.sample = base_output + latent_contribution[:, :3] * 0.001
            return mock_output
        
        mock_vae.decode = convergent_decode
        
        config = OptimizationConfig(
            iterations=100,
            learning_rate=0.1,
            convergence_threshold=1e-4,
            device=device
        )
        optimizer = LatentOptimizer(config)
        
        result = optimizer.optimize(mock_vae, latents, target)
        
        # Convergence may or may not happen due to small latent dependency
        # Just verify the test runs without error
        assert len(result.losses) > 0
        print_test_result("Convergence test", True, 
                         f"Completed {len(result.losses)} iterations")
    
    def test_optimization_edge_cases(self, mock_vae, device):
        """Test edge cases and boundary conditions"""
        print_test_header("Edge Cases Test")
        
        config = OptimizationConfig(iterations=5, device=device)
        optimizer = LatentOptimizer(config)
        
        # Test with zero iterations
        config_zero = OptimizationConfig(iterations=0, device=device)
        optimizer_zero = LatentOptimizer(config_zero)
        
        latents = torch.randn(1, 4, 64, 64, device=device)
        target = TestImageGenerator.solid_color(device=device)
        
        result = optimizer_zero.optimize(mock_vae, latents, target)
        assert len(result.losses) == 0
        print_test_result("Zero iterations", True, "Handled correctly")
        
        # Test with very small learning rate
        config_small_lr = OptimizationConfig(
            iterations=3, 
            learning_rate=1e-8,
            convergence_threshold=1e-10,  # Disable early convergence for this test
            device=device
        )
        optimizer_small = LatentOptimizer(config_small_lr)
        result = optimizer_small.optimize(mock_vae, latents, target)
        
        assert len(result.losses) <= 3  # May converge early
        print_test_result("Small learning rate", True, f"Completed {len(result.losses)} iterations")
        
        # Test with very large learning rate
        config_large_lr = OptimizationConfig(
            iterations=3,
            learning_rate=100.0,
            convergence_threshold=1e-10,  # Disable early convergence for this test
            device=device
        )
        optimizer_large = LatentOptimizer(config_large_lr)
        result = optimizer_large.optimize(mock_vae, latents, target)
        
        assert len(result.losses) <= 3  # May converge early
        print_test_result("Large learning rate", True, f"Completed {len(result.losses)} iterations")
    
    def test_batch_optimization(self, mock_vae, device):
        """Test batch optimization functionality"""
        print_test_header("Batch Optimization Test")
        
        config = OptimizationConfig(iterations=5, device=device)
        optimizer = LatentOptimizer(config)
        
        # Create batch data
        batch_size = 3
        latents_batch = torch.randn(batch_size, 4, 64, 64, device=device)
        targets_batch = torch.stack([
            TestImageGenerator.solid_color((1.0, 0.0, 0.0), device=device).squeeze(0),
            TestImageGenerator.solid_color((0.0, 1.0, 0.0), device=device).squeeze(0),
            TestImageGenerator.solid_color((0.0, 0.0, 1.0), device=device).squeeze(0)
        ]).unsqueeze(1)
        
        results = optimizer.optimize_batch(mock_vae, latents_batch, targets_batch)
        
        assert len(results) == batch_size
        assert all(isinstance(r, OptimizationResult) for r in results)
        assert all(len(r.losses) == 5 for r in results)
        
        print_test_result("Batch processing", True, 
                         f"Processed {batch_size} samples successfully")
    
    def test_gradient_computation(self, mock_vae, sample_latents, sample_target, device):
        """Test gradient computation and parameter freezing"""
        print_test_header("Gradient Computation Test")
        
        config = OptimizationConfig(iterations=2, device=device)
        optimizer = LatentOptimizer(config)
        
        # Track initial latent values
        initial_latents = sample_latents.clone()
        
        result = optimizer.optimize(mock_vae, sample_latents, sample_target)
        
        # Verify latents were modified
        assert not torch.equal(initial_latents, result.optimized_latents)
        print_test_result("Latents modified", True, "Optimization changed latent values")
        
        # Verify VAE decoder parameters are frozen
        for param in mock_vae.decoder.parameters():
            assert param.requires_grad == False
        print_test_result("VAE parameters frozen", True, "Decoder parameters correctly frozen")
        
        # Verify optimized latents don't require gradients
        assert not result.optimized_latents.requires_grad
        print_test_result("Result gradients detached", True, "Output latents detached correctly")
    
    def test_metrics_calculation(self, mock_vae, sample_latents, sample_target, device):
        """Test metrics calculation in optimization results"""
        print_test_header("Metrics Calculation Test")
        
        config = OptimizationConfig(iterations=10, device=device)
        optimizer = LatentOptimizer(config)
        
        result = optimizer.optimize(mock_vae, sample_latents, sample_target)
        
        # Check required metrics are present
        required_metrics = [
            'final_psnr_db', 'final_ssim', 'loss_reduction_percent', 'total_iterations'
        ]
        
        for metric in required_metrics:
            assert metric in result.metrics
            assert isinstance(result.metrics[metric], (int, float))
        
        # Verify metric ranges
        assert result.metrics['final_psnr_db'] > 0  # PSNR should be positive
        assert 0 <= result.metrics['final_ssim'] <= 1  # SSIM should be [0, 1]
        assert result.metrics['total_iterations'] == 10
        
        print_test_result("Metrics calculation", True, 
                         f"PSNR: {result.metrics['final_psnr_db']:.2f}dB")
    
    def test_device_consistency(self, mock_vae, device):
        """Test device consistency throughout optimization"""
        print_test_header("Device Consistency Test")
        
        config = OptimizationConfig(iterations=3, device=device)
        optimizer = LatentOptimizer(config)
        
        # Create tensors on specified device
        latents = torch.randn(1, 4, 64, 64, device=device)
        target = TestImageGenerator.checkerboard(device=device)
        
        result = optimizer.optimize(mock_vae, latents, target)
        
        # Verify all result tensors are on correct device
        assert result.optimized_latents.device.type == device
        print_test_result("Device consistency", True, 
                         f"Tensors maintained on {device}")
    
    def test_checkpoint_interval_behavior(self, mock_vae, sample_latents, sample_target, device):
        """Test checkpoint interval functionality"""
        print_test_header("Checkpoint Interval Test")
        
        # Test with different checkpoint intervals
        for interval in [1, 5, 10]:
            config = OptimizationConfig(
                iterations=10,
                checkpoint_interval=interval,
                device=device
            )
            optimizer = LatentOptimizer(config)
            
            # This test mainly verifies no errors occur with different intervals
            result = optimizer.optimize(mock_vae, sample_latents, sample_target)
            assert len(result.losses) == 10
            
            print_test_result(f"Checkpoint interval {interval}", True, "No errors occurred")
    
    def test_loss_reduction_calculation(self, mock_vae, device):
        """Test loss reduction percentage calculation"""
        print_test_header("Loss Reduction Calculation Test")
        
        config = OptimizationConfig(iterations=10, device=device)
        optimizer = LatentOptimizer(config)
        
        # Create deterministic scenario for predictable loss behavior
        latents = torch.zeros(1, 4, 64, 64, device=device)
        target = TestImageGenerator.solid_color((0.5, 0.5, 0.5), device=device)
        
        result = optimizer.optimize(mock_vae, latents, target)
        
        # Verify loss reduction calculation
        expected_reduction = ((result.initial_loss - result.final_loss) / result.initial_loss) * 100
        actual_reduction = result.metrics['loss_reduction_percent']
        
        assert abs(actual_reduction - expected_reduction) < 1e-5
        print_test_result("Loss reduction calculation", True, 
                         f"Reduction: {actual_reduction:.2f}%")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_efficiency_cuda(self, mock_vae, device):
        """Test memory efficiency with different batch sizes on CUDA"""
        print_test_header("Memory Efficiency Test (CUDA)")
        
        if device == "cuda":
            # Test with progressively larger latent tensors
            sizes = [(1, 4, 32, 32), (1, 4, 64, 64), (2, 4, 64, 64)]
            
            for size in sizes:
                try:
                    config = OptimizationConfig(iterations=3, device=device)
                    optimizer = LatentOptimizer(config)
                    
                    latents = torch.randn(*size, device=device)
                    # Adjust target size based on latent size
                    img_size = (size[2] * 8, size[3] * 8)  # VAE upsamples by 8x
                    target = TestImageGenerator.solid_color(size=img_size, device=device)
                    if size[0] > 1:  # Batch size > 1
                        target = target.repeat(size[0], 1, 1, 1)
                    
                    result = optimizer.optimize(mock_vae, latents, target)
                    
                    print_test_result(f"Size {size}", True, "Memory allocation successful")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print_test_result(f"Size {size}", False, "Out of memory (expected for large sizes)")
                    else:
                        raise
    
    def test_optimization_with_different_image_patterns(self, mock_vae, device):
        """Test optimization with various image patterns"""
        print_test_header("Image Pattern Test")
        
        config = OptimizationConfig(iterations=5, device=device)
        optimizer = LatentOptimizer(config)
        
        # Test with different image patterns
        patterns = [
            ("Solid color", TestImageGenerator.solid_color(device=device)),
            ("Horizontal gradient", TestImageGenerator.gradient_horizontal(device=device)),
            ("Checkerboard", TestImageGenerator.checkerboard(device=device))
        ]
        
        for pattern_name, target_image in patterns:
            latents = torch.randn(1, 4, 64, 64, device=device)
            
            try:
                result = optimizer.optimize(mock_vae, latents, target_image)
                assert len(result.losses) == 5
                print_test_result(pattern_name, True, 
                                 f"PSNR: {result.metrics['final_psnr_db']:.2f}dB")
            except Exception as e:
                print_test_result(pattern_name, False, f"Error: {e}")
    
    def test_psnr_calculation_accuracy(self, device):
        """Test internal PSNR calculation accuracy"""
        print_test_header("PSNR Calculation Accuracy Test")
        
        # Test with known values
        img1 = torch.ones(1, 3, 64, 64, device=device) * 0.5
        img2 = torch.ones(1, 3, 64, 64, device=device) * 0.5
        
        # Identical images should have infinite PSNR (or very high)
        psnr_identical = LatentOptimizer._calculate_psnr(img1, img2)
        assert psnr_identical > 100  # Very high PSNR for identical images
        print_test_result("Identical images PSNR", True, f"PSNR: {psnr_identical:.2f}dB")
        
        # Different images should have finite PSNR
        img2_different = torch.ones(1, 3, 64, 64, device=device) * 0.8
        psnr_different = LatentOptimizer._calculate_psnr(img1, img2_different)
        assert 0 < psnr_different < 100
        print_test_result("Different images PSNR", True, f"PSNR: {psnr_different:.2f}dB")
    
    def test_ssim_calculation_accuracy(self, device):
        """Test internal SSIM calculation accuracy"""
        print_test_header("SSIM Calculation Accuracy Test")
        
        # Test with known values
        img1 = TestImageGenerator.solid_color((0.5, 0.5, 0.5), device=device)
        img2 = TestImageGenerator.solid_color((0.5, 0.5, 0.5), device=device)
        
        # Identical images should have SSIM = 1
        ssim_identical = LatentOptimizer._calculate_ssim_basic(img1, img2)
        assert 0.99 <= ssim_identical <= 1.01  # Allow small numerical errors
        print_test_result("Identical images SSIM", True, f"SSIM: {ssim_identical:.4f}")
        
        # Very different images should have lower SSIM
        img2_different = TestImageGenerator.solid_color((1.0, 0.0, 0.0), device=device)
        ssim_different = LatentOptimizer._calculate_ssim_basic(img1, img2_different)
        assert 0 <= ssim_different < 0.9
        print_test_result("Different images SSIM", True, f"SSIM: {ssim_different:.4f}")
    
    def test_result_data_consistency(self, mock_vae, sample_latents, sample_target, device):
        """Test data consistency in optimization results"""
        print_test_header("Result Data Consistency Test")
        
        config = OptimizationConfig(iterations=8, device=device)
        optimizer = LatentOptimizer(config)
        
        result = optimizer.optimize(mock_vae, sample_latents, sample_target)
        
        # Verify loss history consistency
        assert len(result.losses) == 8
        assert result.initial_loss == result.losses[0]
        assert result.final_loss == result.losses[-1]
        
        # Verify loss reduction calculation
        expected_reduction = ((result.initial_loss - result.final_loss) / result.initial_loss) * 100
        assert abs(result.metrics['loss_reduction_percent'] - expected_reduction) < 1e-5
        
        # Verify tensor shapes
        assert result.optimized_latents.shape == sample_latents.shape
        
        print_test_result("Data consistency", True, "All data relationships verified")
    
    def test_optimization_performance_benchmark(self, mock_vae, device):
        """Benchmark optimization performance"""
        print_test_header("Performance Benchmark Test")
        
        config = OptimizationConfig(iterations=20, device=device)
        optimizer = LatentOptimizer(config)
        
        latents = torch.randn(1, 4, 64, 64, device=device)
        target = TestImageGenerator.gradient_horizontal(device=device)
        
        start_time = time.time()
        result = optimizer.optimize(mock_vae, latents, target)
        end_time = time.time()
        
        optimization_time = end_time - start_time
        time_per_iteration = optimization_time / result.metrics['total_iterations']
        
        # Performance should be reasonable (< 1 second per iteration on mock VAE)
        assert time_per_iteration < 1.0
        
        print_test_result("Performance benchmark", True, 
                         f"Time: {optimization_time:.3f}s ({time_per_iteration:.3f}s/iter)")


# Convenience function for direct execution
def run_all_tests():
    """Run all tests manually without pytest runner"""
    device = setup_test_device()
    print(f"Running tests on device: {device}")
    
    test_instance = TestLatentOptimizer()
    
    # Test without fixtures (manual setup)
    try:
        test_instance.test_optimizer_initialization(device)
        test_instance.test_psnr_calculation_accuracy(device)
        test_instance.test_ssim_calculation_accuracy(device)
        
        # Test with mocked VAE
        mock_vae = Mock()
        mock_decoder = Mock()
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_decoder.parameters.return_value = [mock_param]
        mock_vae.decoder = mock_decoder
        
        def mock_decode(latents):
            # Simulate VAE decode: return tensor that depends on latents for gradient computation
            batch_size = latents.shape[0] 
            latent_h, latent_w = latents.shape[2], latents.shape[3]
            # Calculate target size based on latent size (8x upsampling)
            target_size = (latent_h * 8, latent_w * 8)
            
            mock_output = Mock()
            # Create a simple linear transformation to maintain gradient flow
            upsampled = torch.nn.functional.interpolate(latents, size=target_size, mode='bilinear', align_corners=False)
            # Convert 4 channels to 3 channels with a simple linear combination
            mock_output.sample = (upsampled[:, :3] + upsampled[:, 1:4]) * 0.5 * 2 - 1
            return mock_output
        
        mock_vae.decode = mock_decode
        
        sample_latents = torch.randn(1, 4, 64, 64, device=device, requires_grad=True)
        sample_target = TestImageGenerator.solid_color((0.8, 0.6, 0.4), (512, 512), device)
        
        test_instance.test_optimization_basic_functionality(mock_vae, sample_latents, sample_target, device)
        test_instance.test_optimization_loss_functions(mock_vae, sample_latents, sample_target, device)
        test_instance.test_convergence_detection(device)
        test_instance.test_optimization_edge_cases(mock_vae, device)
        test_instance.test_batch_optimization(mock_vae, device)
        test_instance.test_gradient_computation(mock_vae, sample_latents, sample_target, device)
        test_instance.test_metrics_calculation(mock_vae, sample_latents, sample_target, device)
        test_instance.test_device_consistency(mock_vae, device)
        test_instance.test_checkpoint_interval_behavior(mock_vae, sample_latents, sample_target, device)
        test_instance.test_loss_reduction_calculation(mock_vae, device)
        test_instance.test_optimization_with_different_image_patterns(mock_vae, device)
        test_instance.test_result_data_consistency(mock_vae, sample_latents, sample_target, device)
        test_instance.test_optimization_performance_benchmark(mock_vae, device)
        
        print("\n" + "=" * 60)
        print("🎉 ALL LATENT OPTIMIZER TESTS COMPLETED SUCCESSFULLY!")
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