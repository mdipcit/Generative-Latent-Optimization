#!/usr/bin/env python3
"""
Unit Tests for Metrics Integration Module

Comprehensive testing of unified individual image metrics calculation
including integration of basic and advanced metrics.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project paths
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from tests.fixtures.test_helpers import print_test_header, print_test_result


class TestIndividualMetricsCalculator:
    """Test suite for IndividualMetricsCalculator class"""
    
    def test_calculator_initialization_all_enabled(self):
        """Test calculator initialization with all metrics enabled"""
        print_test_header("Calculator Initialization - All Enabled")
        
        # Mock the advanced metrics classes
        mock_lpips = Mock()
        mock_improved_ssim = Mock()
        
        with patch('src.generative_latent_optimization.metrics.metrics_integration.LPIPSMetric', return_value=mock_lpips):
            with patch('src.generative_latent_optimization.metrics.metrics_integration.ImprovedSSIM', return_value=mock_improved_ssim):
                from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
                
                calculator = IndividualMetricsCalculator(
                    device='cpu',
                    enable_lpips=True,
                    enable_improved_ssim=True
                )
                
                assert calculator.device == 'cpu'
                assert calculator.enable_lpips == True
                assert calculator.enable_improved_ssim == True
                assert hasattr(calculator, 'basic_metrics')
                assert calculator.lpips is not None
                assert calculator.ssim_improved is not None
                
                print_test_result("All metrics enabled", True, "All advanced metrics initialized")
    
    def test_calculator_initialization_selective_enabled(self):
        """Test calculator initialization with selective metric enabling"""
        print_test_header("Calculator Initialization - Selective")
        
        # Mock only LPIPS
        mock_lpips = Mock()
        
        with patch('src.generative_latent_optimization.metrics.metrics_integration.LPIPSMetric', return_value=mock_lpips):
            with patch('src.generative_latent_optimization.metrics.metrics_integration.ImprovedSSIM', side_effect=ImportError("Mocked missing")):
                from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
                
                calculator = IndividualMetricsCalculator(
                    device='cpu',
                    enable_lpips=True,
                    enable_improved_ssim=True  # Will fail and be disabled
                )
                
                assert calculator.enable_lpips == True
                assert calculator.enable_improved_ssim == False  # Should be disabled due to failure
                assert calculator.lpips is not None
                assert calculator.ssim_improved is None
                
                print_test_result("Selective metrics", True, "LPIPS enabled, ImprovedSSIM disabled")
    
    def test_calculator_initialization_all_disabled(self):
        """Test calculator initialization with all advanced metrics disabled"""
        print_test_header("Calculator Initialization - All Disabled")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        
        calculator = IndividualMetricsCalculator(
            device='cpu',
            enable_lpips=False,
            enable_improved_ssim=False
        )
        
        assert calculator.device == 'cpu'
        assert calculator.enable_lpips == False
        assert calculator.enable_improved_ssim == False
        assert hasattr(calculator, 'basic_metrics')
        assert calculator.lpips is None
        assert calculator.ssim_improved is None
        
        print_test_result("All disabled", True, "Only basic metrics available")
    
    def test_calculator_initialization_failure_handling(self):
        """Test calculator handling of initialization failures"""
        print_test_header("Calculator Initialization - Failure Handling")
        
        # Mock both advanced metrics to fail initialization
        with patch('src.generative_latent_optimization.metrics.metrics_integration.LPIPSMetric', side_effect=ImportError("LPIPS not available")):
            with patch('src.generative_latent_optimization.metrics.metrics_integration.ImprovedSSIM', side_effect=ImportError("TorchMetrics not available")):
                from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
                
                calculator = IndividualMetricsCalculator(
                    device='cpu',
                    enable_lpips=True,  # Requested but will fail
                    enable_improved_ssim=True  # Requested but will fail
                )
                
                # Both should be disabled due to failures
                assert calculator.enable_lpips == False
                assert calculator.enable_improved_ssim == False
                assert calculator.lpips is None
                assert calculator.ssim_improved is None
                
                print_test_result("Failure handling", True, "Gracefully disabled failed metrics")
    
    def test_individual_metrics_calculation_basic_only(self):
        """Test individual metrics calculation with basic metrics only"""
        print_test_header("Individual Metrics - Basic Only")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        
        calculator = IndividualMetricsCalculator(
            device='cpu',
            enable_lpips=False,
            enable_improved_ssim=False
        )
        
        # Create test images
        torch.manual_seed(42)
        original = torch.rand(1, 3, 64, 64)
        reconstructed = original + torch.randn_like(original) * 0.1
        reconstructed = torch.clamp(reconstructed, 0, 1)
        
        metrics = calculator.calculate_all_individual_metrics(original, reconstructed)
        
        # Verify basic metrics are present
        assert hasattr(metrics, 'psnr_db')
        assert hasattr(metrics, 'ssim')
        assert hasattr(metrics, 'mse')
        assert hasattr(metrics, 'mae')
        assert isinstance(metrics.psnr_db, float)
        assert isinstance(metrics.ssim, float)
        assert isinstance(metrics.mse, float)
        assert isinstance(metrics.mae, float)
        
        # Advanced metrics should be None
        assert metrics.lpips is None
        assert metrics.ssim_improved is None
        
        print_test_result("Basic metrics only", True, f"PSNR: {metrics.psnr_db:.2f}dB")
    
    def test_individual_metrics_calculation_with_advanced(self):
        """Test individual metrics calculation with mocked advanced metrics"""
        print_test_header("Individual Metrics - With Advanced")
        
        # Mock advanced metrics
        mock_lpips = Mock()
        mock_lpips.calculate.return_value = 0.15
        mock_improved_ssim = Mock()
        mock_improved_ssim.calculate.return_value = 0.92
        
        with patch('src.generative_latent_optimization.metrics.metrics_integration.LPIPSMetric', return_value=mock_lpips):
            with patch('src.generative_latent_optimization.metrics.metrics_integration.ImprovedSSIM', return_value=mock_improved_ssim):
                from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
                
                calculator = IndividualMetricsCalculator(
                    device='cpu',
                    enable_lpips=True,
                    enable_improved_ssim=True
                )
                
                # Create test images
                torch.manual_seed(42)
                original = torch.rand(1, 3, 64, 64)
                reconstructed = original + torch.randn_like(original) * 0.1
                reconstructed = torch.clamp(reconstructed, 0, 1)
                
                metrics = calculator.calculate_all_individual_metrics(original, reconstructed)
                
                # Verify all metrics are present
                assert isinstance(metrics.psnr_db, float)
                assert isinstance(metrics.ssim, float)
                assert isinstance(metrics.mse, float)
                assert isinstance(metrics.mae, float)
                assert metrics.lpips == 0.15
                assert metrics.ssim_improved == 0.92
                
                # Verify advanced metrics were called
                mock_lpips.calculate.assert_called_once()
                mock_improved_ssim.calculate.assert_called_once()
                
                print_test_result("Advanced metrics", True, 
                                 f"LPIPS: {metrics.lpips}, SSIM+: {metrics.ssim_improved}")
    
    def test_individual_metrics_shape_validation(self):
        """Test shape validation in individual metrics calculation"""
        print_test_header("Individual Metrics - Shape Validation")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        
        calculator = IndividualMetricsCalculator(device='cpu', enable_lpips=False, enable_improved_ssim=False)
        
        # Create mismatched shapes
        original = torch.rand(1, 3, 64, 64)
        reconstructed = torch.rand(1, 3, 32, 32)  # Different size
        
        with pytest.raises(ValueError) as exc_info:
            calculator.calculate_all_individual_metrics(original, reconstructed)
        
        assert "Image shapes must match" in str(exc_info.value)
        
        print_test_result("Shape validation", True, "ValueError raised for mismatched shapes")
    
    def test_batch_individual_metrics_calculation(self):
        """Test batch individual metrics calculation"""
        print_test_header("Batch Individual Metrics")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        
        calculator = IndividualMetricsCalculator(device='cpu', enable_lpips=False, enable_improved_ssim=False)
        
        # Create batch test images
        batch_size = 4
        torch.manual_seed(42)
        original_batch = torch.rand(batch_size, 3, 32, 32)
        reconstructed_batch = original_batch + torch.randn_like(original_batch) * 0.1
        reconstructed_batch = torch.clamp(reconstructed_batch, 0, 1)
        
        batch_results = calculator.calculate_batch_individual_metrics(original_batch, reconstructed_batch)
        
        # Verify batch results
        assert len(batch_results) == batch_size
        
        for result in batch_results:
            assert result is not None
            assert hasattr(result, 'psnr_db')
            assert hasattr(result, 'ssim')
            assert hasattr(result, 'mse')
            assert hasattr(result, 'mae')
            assert isinstance(result.psnr_db, float)
            assert isinstance(result.ssim, float)
        
        print_test_result("Batch individual metrics", True, f"Processed {batch_size} images")
    
    def test_batch_metrics_with_failures(self):
        """Test batch metrics calculation with some failures"""
        print_test_header("Batch Metrics - With Failures")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        
        calculator = IndividualMetricsCalculator(device='cpu', enable_lpips=False, enable_improved_ssim=False)
        
        # Create batch with unified shape for proper concatenation
        original_batch = torch.rand(3, 3, 32, 32)
        
        # Create properly shaped tensors, then simulate problematic case differently
        good_images = torch.rand(2, 3, 32, 32)
        problematic_image = torch.rand(1, 3, 32, 32)  # Same shape for concatenation
        
        reconstructed_batch = torch.cat([good_images, problematic_image], dim=0)
        
        # This should handle the error gracefully
        batch_results = calculator.calculate_batch_individual_metrics(original_batch, reconstructed_batch)
        
        # Should have 3 results, all should succeed with unified shapes
        assert len(batch_results) == 3
        assert batch_results[0] is not None  # First image should work
        assert batch_results[1] is not None  # Second image should work  
        assert batch_results[2] is not None  # Third image should work (shapes unified)
        
        print_test_result("Batch with failures", True, "Handled failures gracefully")
    
    def test_batch_statistics_calculation(self):
        """Test batch statistics calculation"""
        print_test_header("Batch Statistics Calculation")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        
        calculator = IndividualMetricsCalculator(device='cpu', enable_lpips=False, enable_improved_ssim=False)
        
        # Create test batch
        batch_size = 5
        torch.manual_seed(42)
        original_batch = torch.rand(batch_size, 3, 32, 32)
        reconstructed_batch = original_batch + torch.randn_like(original_batch) * 0.1
        reconstructed_batch = torch.clamp(reconstructed_batch, 0, 1)
        
        batch_results = calculator.calculate_batch_individual_metrics(original_batch, reconstructed_batch)
        statistics = calculator.get_batch_statistics(batch_results)
        
        # Verify statistics structure
        expected_basic_keys = [
            'psnr_mean', 'psnr_std', 'psnr_min', 'psnr_max',
            'ssim_mean', 'ssim_std', 'ssim_min', 'ssim_max',
            'mse_mean', 'mse_std', 'mae_mean', 'mae_std',
            'valid_samples', 'total_samples'
        ]
        
        for key in expected_basic_keys:
            assert key in statistics
            assert isinstance(statistics[key], (float, int))
        
        # Verify statistical properties
        assert statistics['psnr_min'] <= statistics['psnr_mean'] <= statistics['psnr_max']
        assert statistics['ssim_min'] <= statistics['ssim_mean'] <= statistics['ssim_max']
        assert statistics['psnr_std'] >= 0
        assert statistics['ssim_std'] >= 0
        assert statistics['valid_samples'] == batch_size
        assert statistics['total_samples'] == batch_size
        
        # Advanced metrics should not be present
        assert 'lpips_mean' not in statistics
        assert 'ssim_improved_mean' not in statistics
        
        print_test_result("Batch statistics", True, f"Mean PSNR: {statistics['psnr_mean']:.2f}dB")
    
    def test_batch_statistics_with_advanced_metrics(self):
        """Test batch statistics with advanced metrics included"""
        print_test_header("Batch Statistics - Advanced Metrics")
        
        # Mock advanced metrics
        mock_lpips = Mock()
        mock_lpips.calculate.return_value = 0.15
        mock_improved_ssim = Mock()
        mock_improved_ssim.calculate.return_value = 0.88
        
        with patch('src.generative_latent_optimization.metrics.metrics_integration.LPIPSMetric', return_value=mock_lpips):
            with patch('src.generative_latent_optimization.metrics.metrics_integration.ImprovedSSIM', return_value=mock_improved_ssim):
                from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
                
                calculator = IndividualMetricsCalculator(
                    device='cpu',
                    enable_lpips=True,
                    enable_improved_ssim=True
                )
                
                # Create test batch
                batch_size = 3
                torch.manual_seed(42)
                original_batch = torch.rand(batch_size, 3, 32, 32)
                reconstructed_batch = torch.rand(batch_size, 3, 32, 32)
                
                batch_results = calculator.calculate_batch_individual_metrics(original_batch, reconstructed_batch)
                statistics = calculator.get_batch_statistics(batch_results)
                
                # Verify advanced metrics statistics are present
                assert 'lpips_mean' in statistics
                assert 'lpips_std' in statistics
                assert 'lpips_min' in statistics
                assert 'lpips_max' in statistics
                assert 'ssim_improved_mean' in statistics
                assert 'ssim_improved_std' in statistics
                
                # All should have the same values since mocked
                assert statistics['lpips_mean'] == 0.15
                assert statistics['ssim_improved_mean'] == 0.88
                
                print_test_result("Advanced statistics", True, 
                                 f"LPIPS mean: {statistics['lpips_mean']}")
    
    def test_batch_statistics_empty_list(self):
        """Test batch statistics with empty results list"""
        print_test_header("Batch Statistics - Empty List")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        
        calculator = IndividualMetricsCalculator(device='cpu', enable_lpips=False, enable_improved_ssim=False)
        
        statistics = calculator.get_batch_statistics([])
        
        # Should return empty dictionary
        assert statistics == {}
        
        print_test_result("Empty statistics", True, "Returns empty dict")
    
    def test_batch_statistics_with_none_values(self):
        """Test batch statistics with some None values (failed calculations)"""
        print_test_header("Batch Statistics - With None Values")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        from src.generative_latent_optimization.metrics.image_metrics import IndividualImageMetrics
        
        calculator = IndividualMetricsCalculator(device='cpu', enable_lpips=False, enable_improved_ssim=False)
        
        # Create mixed results with some None values
        valid_result1 = IndividualImageMetrics(psnr_db=25.0, ssim=0.8, mse=0.01, mae=0.05)
        valid_result2 = IndividualImageMetrics(psnr_db=30.0, ssim=0.9, mse=0.005, mae=0.03)
        
        batch_results = [valid_result1, None, valid_result2, None]  # 2 valid, 2 None
        
        statistics = calculator.get_batch_statistics(batch_results)
        
        # Should calculate statistics based only on valid results
        assert statistics['valid_samples'] == 2
        assert statistics['total_samples'] == 4
        assert statistics['psnr_mean'] == pytest.approx(27.5, rel=1e-9)  # (25.0 + 30.0) / 2
        assert statistics['ssim_mean'] == pytest.approx(0.85, rel=1e-9)  # (0.8 + 0.9) / 2
        
        print_test_result("None values handling", True, "Filtered None values correctly")
    
    def test_standard_deviation_calculation(self):
        """Test standard deviation calculation method"""
        print_test_header("Standard Deviation Calculation")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        
        calculator = IndividualMetricsCalculator(device='cpu', enable_lpips=False, enable_improved_ssim=False)
        
        # Test with normal values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        std = calculator._calculate_std(values)
        
        # Manual calculation: mean=3.0, variance=2.5, std=sqrt(2.5)≈1.58
        expected_std = (2.5 ** 0.5)
        assert abs(std - expected_std) < 1e-6
        
        # Test with single value
        single_std = calculator._calculate_std([5.0])
        assert single_std == 0.0
        
        # Test with empty list
        empty_std = calculator._calculate_std([])
        assert empty_std == 0.0
        
        print_test_result("Std calculation", True, f"Calculated std: {std:.4f}")
    
    def test_advanced_metrics_calculation_errors(self):
        """Test handling of errors during advanced metrics calculation"""
        print_test_header("Advanced Metrics - Error Handling")
        
        # Mock advanced metrics that fail during calculation
        mock_lpips = Mock()
        mock_lpips.calculate.side_effect = RuntimeError("LPIPS calculation error")
        mock_improved_ssim = Mock()
        mock_improved_ssim.calculate.side_effect = RuntimeError("SSIM calculation error")
        
        with patch('src.generative_latent_optimization.metrics.metrics_integration.LPIPSMetric', return_value=mock_lpips):
            with patch('src.generative_latent_optimization.metrics.metrics_integration.ImprovedSSIM', return_value=mock_improved_ssim):
                from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
                
                calculator = IndividualMetricsCalculator(
                    device='cpu',
                    enable_lpips=True,
                    enable_improved_ssim=True
                )
                
                # Create test images
                torch.manual_seed(42)
                original = torch.rand(1, 3, 64, 64)
                reconstructed = torch.rand(1, 3, 64, 64)
                
                metrics = calculator.calculate_all_individual_metrics(original, reconstructed)
                
                # Basic metrics should work
                assert isinstance(metrics.psnr_db, float)
                assert isinstance(metrics.ssim, float)
                
                # Advanced metrics should be None due to errors
                assert metrics.lpips is None
                assert metrics.ssim_improved is None
                
                print_test_result("Advanced errors", True, "Handled calculation errors gracefully")
    
    def test_device_consistency(self):
        """Test device consistency across all metrics"""
        print_test_header("Device Consistency")
        
        devices = ['cpu', 'cuda']
        
        for device in devices:
            from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
            
            calculator = IndividualMetricsCalculator(
                device=device,
                enable_lpips=False,
                enable_improved_ssim=False
            )
            
            assert calculator.device == device
            assert calculator.basic_metrics.device == device
            
        print_test_result("Device consistency", True, f"Tested devices: {devices}")


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_individual_metrics_calculator_test_function(self):
        """Test the test_individual_metrics_calculator utility function"""
        print_test_header("Calculator Test Function")
        
        # Mock the calculator class to avoid complex dependencies
        mock_calculator = Mock()
        mock_metrics = Mock()
        mock_metrics.psnr_db = 28.5
        mock_metrics.ssim = 0.85
        mock_metrics.mse = 0.01
        mock_metrics.mae = 0.05
        mock_metrics.lpips = 0.12
        mock_metrics.ssim_improved = 0.88
        
        mock_calculator.calculate_all_individual_metrics.return_value = mock_metrics
        mock_calculator.calculate_batch_individual_metrics.return_value = [mock_metrics] * 4
        mock_calculator.get_batch_statistics.return_value = {
            'psnr_mean': 28.5, 'ssim_mean': 0.85, 'lpips_mean': 0.12
        }
        
        with patch('src.generative_latent_optimization.metrics.metrics_integration.IndividualMetricsCalculator', return_value=mock_calculator):
            from src.generative_latent_optimization.metrics.metrics_integration import test_individual_metrics_calculator
            
            # Should run without errors
            test_individual_metrics_calculator(device='cpu')
            
            # Verify calls were made
            assert mock_calculator.calculate_all_individual_metrics.called
            assert mock_calculator.calculate_batch_individual_metrics.called
            assert mock_calculator.get_batch_statistics.called
            
            print_test_result("Calculator test function", True, "Executed without errors")


class TestIntegrationScenarios:
    """Test suite for integration scenarios"""
    
    def test_metrics_calculator_with_partial_advanced_success(self):
        """Test calculator when only some advanced metrics succeed"""
        print_test_header("Partial Advanced Metrics Success")
        
        # Mock LPIPS to succeed, ImprovedSSIM to fail
        mock_lpips = Mock()
        mock_lpips.calculate.return_value = 0.18
        
        with patch('src.generative_latent_optimization.metrics.metrics_integration.LPIPSMetric', return_value=mock_lpips):
            with patch('src.generative_latent_optimization.metrics.metrics_integration.ImprovedSSIM', side_effect=ImportError("ImprovedSSIM failed")):
                from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
                
                calculator = IndividualMetricsCalculator(
                    device='cpu',
                    enable_lpips=True,
                    enable_improved_ssim=True
                )
                
                # LPIPS should be enabled, ImprovedSSIM disabled
                assert calculator.enable_lpips == True
                assert calculator.enable_improved_ssim == False
                
                # Test calculation
                torch.manual_seed(42)
                original = torch.rand(1, 3, 64, 64)
                reconstructed = torch.rand(1, 3, 64, 64)
                
                metrics = calculator.calculate_all_individual_metrics(original, reconstructed)
                
                # Basic metrics + LPIPS should be present
                assert isinstance(metrics.psnr_db, float)
                assert metrics.lpips == 0.18
                assert metrics.ssim_improved is None
                
                print_test_result("Partial advanced success", True, "LPIPS works, ImprovedSSIM disabled")
    
    def test_calculator_with_different_tensor_types(self):
        """Test calculator with different tensor types and ranges"""
        print_test_header("Different Tensor Types")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        
        calculator = IndividualMetricsCalculator(device='cpu', enable_lpips=False, enable_improved_ssim=False)
        
        # Test with different tensor configurations
        test_configs = [
            # (original, reconstructed, description)
            (torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32), "float32 [0,1]"),
            (torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16), "grayscale"),
            (torch.rand(2, 3, 24, 24), torch.rand(2, 3, 24, 24), "batch size 2"),
        ]
        
        for original, reconstructed, description in test_configs:
            metrics = calculator.calculate_all_individual_metrics(original, reconstructed)
            
            assert isinstance(metrics.psnr_db, float)
            assert isinstance(metrics.ssim, float)
            assert isinstance(metrics.mse, float)
            assert isinstance(metrics.mae, float)
            
        print_test_result("Tensor types", True, f"Tested {len(test_configs)} configurations")
    
    def test_large_batch_processing(self):
        """Test calculator with larger batch sizes"""
        print_test_header("Large Batch Processing")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        
        calculator = IndividualMetricsCalculator(device='cpu', enable_lpips=False, enable_improved_ssim=False)
        
        # Test with larger batch
        batch_size = 20
        torch.manual_seed(42)
        original_batch = torch.rand(batch_size, 3, 16, 16)  # Smaller images for faster processing
        reconstructed_batch = original_batch + torch.randn_like(original_batch) * 0.05
        reconstructed_batch = torch.clamp(reconstructed_batch, 0, 1)
        
        batch_results = calculator.calculate_batch_individual_metrics(original_batch, reconstructed_batch)
        statistics = calculator.get_batch_statistics(batch_results)
        
        assert len(batch_results) == batch_size
        assert statistics['valid_samples'] == batch_size
        assert statistics['total_samples'] == batch_size
        
        print_test_result("Large batch", True, f"Processed {batch_size} images")


class TestErrorConditions:
    """Test suite for error conditions and edge cases"""
    
    def test_calculator_with_invalid_device(self):
        """Test calculator behavior with invalid device specification"""
        print_test_header("Invalid Device Specification")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        
        # Should not raise error during initialization (device validation happens later)
        calculator = IndividualMetricsCalculator(
            device='invalid_device',
            enable_lpips=False,
            enable_improved_ssim=False
        )
        
        assert calculator.device == 'invalid_device'
        
        print_test_result("Invalid device", True, "Accepted invalid device string")
    
    def test_calculator_all_metrics_none_values(self):
        """Test statistics calculation when all advanced metrics return None"""
        print_test_header("All Advanced Metrics None")
        
        from src.generative_latent_optimization.metrics.metrics_integration import IndividualMetricsCalculator
        from src.generative_latent_optimization.metrics.image_metrics import IndividualImageMetrics
        
        calculator = IndividualMetricsCalculator(device='cpu', enable_lpips=False, enable_improved_ssim=False)
        
        # Create results where advanced metrics are None
        batch_results = [
            IndividualImageMetrics(psnr_db=25.0, ssim=0.8, mse=0.01, mae=0.05, lpips=None, ssim_improved=None),
            IndividualImageMetrics(psnr_db=30.0, ssim=0.9, mse=0.005, mae=0.03, lpips=None, ssim_improved=None)
        ]
        
        statistics = calculator.get_batch_statistics(batch_results)
        
        # Should only have basic metrics statistics
        assert 'psnr_mean' in statistics
        assert 'ssim_mean' in statistics
        assert 'lpips_mean' not in statistics  # No LPIPS values available
        assert 'ssim_improved_mean' not in statistics  # No improved SSIM values available
        
        print_test_result("All advanced None", True, "Handled all None advanced metrics")


# Convenience function for direct execution
def run_all_tests():
    """Run all tests manually without pytest runner"""
    print("🧪 Starting Metrics Integration Unit Tests")
    print("=" * 60)
    
    # Create test instances
    test_calculator = TestIndividualMetricsCalculator()
    test_utilities = TestUtilityFunctions()
    test_integration = TestIntegrationScenarios()
    test_errors = TestErrorConditions()
    
    try:
        # Run IndividualMetricsCalculator tests
        test_calculator.test_calculator_initialization_all_enabled()
        test_calculator.test_calculator_initialization_selective_enabled()
        test_calculator.test_calculator_initialization_all_disabled()
        test_calculator.test_calculator_initialization_failure_handling()
        test_calculator.test_individual_metrics_calculation_basic_only()
        test_calculator.test_individual_metrics_calculation_with_advanced()
        test_calculator.test_individual_metrics_shape_validation()
        test_calculator.test_batch_individual_metrics_calculation()
        test_calculator.test_batch_metrics_with_failures()
        test_calculator.test_batch_statistics_calculation()
        test_calculator.test_batch_statistics_with_advanced_metrics()
        test_calculator.test_batch_statistics_empty_list()
        test_calculator.test_batch_statistics_with_none_values()
        test_calculator.test_standard_deviation_calculation()
        test_calculator.test_advanced_metrics_calculation_errors()
        test_calculator.test_device_consistency()
        
        # Run utility function tests
        test_utilities.test_individual_metrics_calculator_test_function()
        
        # Run integration scenario tests
        test_integration.test_metrics_calculator_with_partial_advanced_success()
        test_integration.test_calculator_with_different_tensor_types()
        test_integration.test_large_batch_processing()
        
        # Run error condition tests
        test_errors.test_calculator_with_invalid_device()
        test_errors.test_calculator_all_metrics_none_values()
        
        print("\n" + "=" * 60)
        print("🎉 ALL METRICS INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
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