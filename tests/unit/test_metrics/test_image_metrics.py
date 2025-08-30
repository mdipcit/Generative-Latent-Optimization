#!/usr/bin/env python3
"""
Unit Tests for Image Metrics Module

Comprehensive testing of image quality assessment metrics including
PSNR, SSIM, MSE, MAE calculations and batch processing functionality.
"""

import pytest
import torch
import numpy as np
import json
import tempfile
from pathlib import Path
from dataclasses import asdict
from unittest.mock import Mock, patch

# Add project paths
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.generative_latent_optimization.metrics.image_metrics import (
    ImageMetrics, MetricsTracker, MetricResults, IndividualImageMetrics,
    DatasetEvaluationResults, AllMetricsResults, calculate_psnr
)
from tests.fixtures.test_helpers import print_test_header, print_test_result


class TestImageMetrics:
    """Test suite for ImageMetrics class"""
    
    @pytest.fixture
    def image_metrics(self):
        """Create ImageMetrics instance for testing"""
        return ImageMetrics(device='cpu')
    
    @pytest.fixture
    def sample_images(self):
        """Create sample image tensors for testing"""
        # Create reproducible test images
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 32, 32)  # Original image
        img2 = img1 + torch.randn_like(img1) * 0.1  # Slightly noisy version
        img2 = torch.clamp(img2, 0, 1)  # Ensure valid range
        
        return img1, img2
    
    @pytest.fixture
    def identical_images(self):
        """Create identical image tensors for edge case testing"""
        torch.manual_seed(123)
        img = torch.rand(1, 3, 16, 16)
        return img, img.clone()
    
    def test_psnr_calculation_normal(self, image_metrics, sample_images):
        """Test PSNR calculation with normal images"""
        print_test_header("PSNR Calculation - Normal Images")
        
        img1, img2 = sample_images
        psnr = image_metrics.calculate_psnr(img1, img2)
        
        # PSNR should be positive and reasonable
        assert isinstance(psnr, float)
        assert psnr > 0
        assert psnr < 100  # Reasonable upper bound
        
        print_test_result("PSNR calculation", True, f"PSNR: {psnr:.2f} dB")
    
    def test_psnr_identical_images(self, image_metrics, identical_images):
        """Test PSNR calculation with identical images"""
        print_test_header("PSNR Calculation - Identical Images")
        
        img1, img2 = identical_images
        psnr = image_metrics.calculate_psnr(img1, img2)
        
        # PSNR should be infinite for identical images
        assert psnr == float('inf')
        
        print_test_result("PSNR identical", True, "PSNR: inf dB (correct)")
    
    def test_psnr_different_devices(self, sample_images):
        """Test PSNR calculation with different device settings"""
        print_test_header("PSNR Calculation - Device Compatibility")
        
        img1, img2 = sample_images
        
        # Test CPU
        metrics_cpu = ImageMetrics(device='cpu')
        psnr_cpu = metrics_cpu.calculate_psnr(img1, img2)
        
        # Test with CUDA string (device parameter doesn't affect calculation)
        metrics_cuda_str = ImageMetrics(device='cuda')
        psnr_cuda_str = metrics_cuda_str.calculate_psnr(img1, img2)
        
        # Results should be identical regardless of device string
        assert abs(psnr_cpu - psnr_cuda_str) < 1e-6
        
        print_test_result("Device compatibility", True, "Results consistent across devices")
    
    def test_ssim_calculation_normal(self, image_metrics, sample_images):
        """Test SSIM calculation with normal images"""
        print_test_header("SSIM Calculation - Normal Images")
        
        img1, img2 = sample_images
        ssim = image_metrics.calculate_ssim(img1, img2)
        
        # SSIM should be between 0 and 1
        assert isinstance(ssim, float)
        assert 0 <= ssim <= 1
        
        print_test_result("SSIM calculation", True, f"SSIM: {ssim:.4f}")
    
    def test_ssim_identical_images(self, image_metrics, identical_images):
        """Test SSIM calculation with identical images"""
        print_test_header("SSIM Calculation - Identical Images")
        
        img1, img2 = identical_images
        ssim = image_metrics.calculate_ssim(img1, img2)
        
        # SSIM should be 1.0 for identical images (with small tolerance)
        assert abs(ssim - 1.0) < 1e-4
        
        print_test_result("SSIM identical", True, f"SSIM: {ssim:.6f} (≈1.0)")
    
    def test_ssim_window_size_parameter(self, image_metrics, sample_images):
        """Test SSIM calculation with different window sizes"""
        print_test_header("SSIM Calculation - Window Size Variation")
        
        img1, img2 = sample_images
        
        # Test different window sizes
        window_sizes = [3, 7, 11, 15]
        ssim_values = []
        
        for window_size in window_sizes:
            ssim = image_metrics.calculate_ssim(img1, img2, window_size=window_size)
            ssim_values.append(ssim)
            assert 0 <= ssim <= 1
        
        # All values should be reasonable
        assert all(0 <= s <= 1 for s in ssim_values)
        
        print_test_result("SSIM window sizes", True, f"All windows valid: {window_sizes}")
    
    def test_mse_calculation(self, image_metrics, sample_images):
        """Test MSE calculation"""
        print_test_header("MSE Calculation")
        
        img1, img2 = sample_images
        mse = image_metrics.calculate_mse(img1, img2)
        
        # MSE should be non-negative
        assert isinstance(mse, float)
        assert mse >= 0
        
        print_test_result("MSE calculation", True, f"MSE: {mse:.6f}")
    
    def test_mse_identical_images(self, image_metrics, identical_images):
        """Test MSE with identical images"""
        print_test_header("MSE Calculation - Identical Images")
        
        img1, img2 = identical_images
        mse = image_metrics.calculate_mse(img1, img2)
        
        # MSE should be 0 for identical images
        assert abs(mse) < 1e-6
        
        print_test_result("MSE identical", True, f"MSE: {mse:.8f} (≈0.0)")
    
    def test_mae_calculation(self, image_metrics, sample_images):
        """Test MAE calculation"""
        print_test_header("MAE Calculation")
        
        img1, img2 = sample_images
        mae = image_metrics.calculate_mae(img1, img2)
        
        # MAE should be non-negative
        assert isinstance(mae, float)
        assert mae >= 0
        
        print_test_result("MAE calculation", True, f"MAE: {mae:.6f}")
    
    def test_mae_identical_images(self, image_metrics, identical_images):
        """Test MAE with identical images"""
        print_test_header("MAE Calculation - Identical Images")
        
        img1, img2 = identical_images
        mae = image_metrics.calculate_mae(img1, img2)
        
        # MAE should be 0 for identical images
        assert abs(mae) < 1e-6
        
        print_test_result("MAE identical", True, f"MAE: {mae:.8f} (≈0.0)")
    
    def test_all_metrics_calculation(self, image_metrics, sample_images):
        """Test calculate_all_metrics method"""
        print_test_header("All Metrics Calculation")
        
        img1, img2 = sample_images
        results = image_metrics.calculate_all_metrics(img1, img2)
        
        # Verify result type and structure
        assert isinstance(results, MetricResults)
        assert hasattr(results, 'psnr_db')
        assert hasattr(results, 'ssim')
        assert hasattr(results, 'mse')
        assert hasattr(results, 'mae')
        assert hasattr(results, 'lpips')
        
        # Verify value ranges
        assert results.psnr_db > 0
        assert 0 <= results.ssim <= 1
        assert results.mse >= 0
        assert results.mae >= 0
        assert results.lpips is None  # Not implemented in basic metrics
        
        print_test_result("All metrics", True, 
                         f"PSNR: {results.psnr_db:.2f}, SSIM: {results.ssim:.4f}")
    
    def test_batch_metrics_calculation(self, image_metrics):
        """Test batch metrics calculation"""
        print_test_header("Batch Metrics Calculation")
        
        # Create batch of test images
        batch_size = 4
        torch.manual_seed(42)
        original_batch = torch.rand(batch_size, 3, 16, 16)
        reconstructed_batch = original_batch + torch.randn_like(original_batch) * 0.05
        reconstructed_batch = torch.clamp(reconstructed_batch, 0, 1)
        
        results = image_metrics.calculate_batch_metrics(original_batch, reconstructed_batch)
        
        # Verify batch results structure
        assert isinstance(results, list)
        assert len(results) == batch_size
        
        for result in results:
            assert isinstance(result, MetricResults)
            assert result.psnr_db > 0
            assert 0 <= result.ssim <= 1
            assert result.mse >= 0
            assert result.mae >= 0
        
        print_test_result("Batch metrics", True, f"Processed {batch_size} images")
    
    def test_batch_statistics_calculation(self, image_metrics):
        """Test batch statistics calculation"""
        print_test_header("Batch Statistics Calculation")
        
        # Create sample results
        torch.manual_seed(42)
        sample_results = []
        for i in range(5):
            result = MetricResults(
                psnr_db=20.0 + np.random.normal(0, 2),
                ssim=0.8 + np.random.normal(0, 0.1),
                mse=0.01 + np.random.normal(0, 0.002),
                mae=0.05 + np.random.normal(0, 0.01)
            )
            sample_results.append(result)
        
        stats = image_metrics.get_batch_statistics(sample_results)
        
        # Verify statistics structure
        expected_keys = [
            'psnr_mean', 'psnr_std', 'psnr_min', 'psnr_max',
            'ssim_mean', 'ssim_std', 'ssim_min', 'ssim_max',
            'mse_mean', 'mse_std', 'mae_mean', 'mae_std'
        ]
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (float, np.floating))
        
        # Verify statistical properties
        assert stats['psnr_min'] <= stats['psnr_mean'] <= stats['psnr_max']
        assert stats['ssim_min'] <= stats['ssim_mean'] <= stats['ssim_max']
        assert stats['psnr_std'] >= 0
        assert stats['ssim_std'] >= 0
        
        print_test_result("Batch statistics", True, f"Mean PSNR: {stats['psnr_mean']:.2f}")
    
    def test_batch_statistics_empty_list(self, image_metrics):
        """Test batch statistics with empty list"""
        print_test_header("Batch Statistics - Empty List")
        
        stats = image_metrics.get_batch_statistics([])
        
        # Should return empty dictionary
        assert stats == {}
        
        print_test_result("Empty batch stats", True, "Returns empty dict")
    
    def test_gaussian_kernel_creation(self, image_metrics):
        """Test Gaussian kernel creation"""
        print_test_header("Gaussian Kernel Creation")
        
        # Test different kernel sizes
        sizes = [3, 5, 7, 11]
        
        for size in sizes:
            kernel = image_metrics._create_gaussian_kernel(size, sigma=1.0)
            
            # Verify kernel properties
            assert kernel.shape == (size, size)
            assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-6)
            assert kernel.min() >= 0  # All values should be non-negative
            
            # Verify symmetry
            assert torch.allclose(kernel, kernel.t())
            
        print_test_result("Gaussian kernels", True, f"Tested sizes: {sizes}")
    
    def test_gaussian_filter_application(self, image_metrics):
        """Test Gaussian filter application"""
        print_test_header("Gaussian Filter Application")
        
        # Create test image
        torch.manual_seed(42)
        test_image = torch.rand(1, 16, 16)
        
        # Apply filter
        filtered = image_metrics._gaussian_filter(test_image, window_size=5)
        
        # Verify output properties
        assert filtered.shape == test_image.shape
        assert filtered.dtype == test_image.dtype
        
        # Filtered image should be smoother (less variance)
        original_var = test_image.var()
        filtered_var = filtered.var()
        assert filtered_var <= original_var
        
        print_test_result("Gaussian filter", True, "Applied successfully")
    
    def test_tensor_shapes_validation(self, image_metrics):
        """Test various tensor shape configurations"""
        print_test_header("Tensor Shape Validation")
        
        # Test different valid shapes
        test_shapes = [
            (1, 1, 8, 8),    # Grayscale
            (1, 3, 16, 16),  # RGB small
            (2, 3, 32, 32),  # RGB batch
            (1, 3, 64, 64),  # RGB larger
        ]
        
        for shape in test_shapes:
            torch.manual_seed(42)
            img1 = torch.rand(shape)
            img2 = torch.rand(shape)
            
            # All metrics should work with different shapes
            psnr = image_metrics.calculate_psnr(img1, img2)
            ssim = image_metrics.calculate_ssim(img1, img2)
            mse = image_metrics.calculate_mse(img1, img2)
            mae = image_metrics.calculate_mae(img1, img2)
            
            # Verify all results are valid
            assert isinstance(psnr, float)
            assert isinstance(ssim, float)
            assert isinstance(mse, float)
            assert isinstance(mae, float)
            
            assert psnr > 0
            assert 0 <= ssim <= 1
            assert mse >= 0
            assert mae >= 0
        
        print_test_result("Shape validation", True, f"Tested {len(test_shapes)} shapes")
    
    def test_extreme_image_values(self, image_metrics):
        """Test metrics with extreme image values"""
        print_test_header("Extreme Image Values Test")
        
        # Test with all zeros
        zero_img = torch.zeros(1, 3, 16, 16)
        ones_img = torch.ones(1, 3, 16, 16)
        
        # Zero vs Ones (maximum difference)
        psnr_extreme = image_metrics.calculate_psnr(zero_img, ones_img)
        ssim_extreme = image_metrics.calculate_ssim(zero_img, ones_img)
        mse_extreme = image_metrics.calculate_mse(zero_img, ones_img)
        mae_extreme = image_metrics.calculate_mae(zero_img, ones_img)
        
        # Verify extreme case handling
        assert isinstance(psnr_extreme, float)
        assert psnr_extreme == 0.0  # PSNR is 0 for maximum difference (MSE=1.0)
        assert isinstance(ssim_extreme, float)
        assert 0 <= ssim_extreme <= 1
        assert mse_extreme == 1.0  # Perfect MSE for 0 vs 1
        assert mae_extreme == 1.0  # Perfect MAE for 0 vs 1
        
        print_test_result("Extreme values", True, 
                         f"PSNR: {psnr_extreme:.2f}, SSIM: {ssim_extreme:.4f}")
    
    def test_grayscale_vs_rgb_consistency(self, image_metrics):
        """Test consistency between grayscale and RGB processing"""
        print_test_header("Grayscale vs RGB Consistency")
        
        # Create grayscale images
        torch.manual_seed(42)
        gray1 = torch.rand(1, 1, 16, 16)
        gray2 = gray1 + torch.randn_like(gray1) * 0.1
        gray2 = torch.clamp(gray2, 0, 1)
        
        # Calculate metrics for grayscale
        psnr_gray = image_metrics.calculate_psnr(gray1, gray2)
        ssim_gray = image_metrics.calculate_ssim(gray1, gray2)
        
        # Convert to RGB (repeat channels)
        rgb1 = gray1.repeat(1, 3, 1, 1)
        rgb2 = gray2.repeat(1, 3, 1, 1)
        
        # Calculate metrics for RGB version
        psnr_rgb = image_metrics.calculate_psnr(rgb1, rgb2)
        ssim_rgb = image_metrics.calculate_ssim(rgb1, rgb2)
        
        # PSNR should be identical (pixel-wise calculation)
        assert abs(psnr_gray - psnr_rgb) < 1e-6
        
        # SSIM should be very similar (uses grayscale conversion anyway)
        assert abs(ssim_gray - ssim_rgb) < 1e-4
        
        print_test_result("Gray vs RGB", True, "Consistent results")


class TestMetricResults:
    """Test suite for MetricResults dataclass"""
    
    def test_metric_results_creation(self):
        """Test MetricResults creation and field access"""
        print_test_header("MetricResults Creation")
        
        results = MetricResults(
            psnr_db=25.5,
            ssim=0.85,
            mse=0.01,
            mae=0.05,
            lpips=0.12
        )
        
        assert results.psnr_db == 25.5
        assert results.ssim == 0.85
        assert results.mse == 0.01
        assert results.mae == 0.05
        assert results.lpips == 0.12
        
        print_test_result("MetricResults creation", True, "All fields accessible")
    
    def test_metric_results_optional_fields(self):
        """Test MetricResults with optional fields"""
        print_test_header("MetricResults Optional Fields")
        
        # Create without optional lpips
        results = MetricResults(
            psnr_db=30.0,
            ssim=0.9,
            mse=0.005,
            mae=0.02
        )
        
        assert results.lpips is None
        
        print_test_result("Optional fields", True, "LPIPS defaults to None")
    
    def test_metric_results_serialization(self):
        """Test MetricResults serialization"""
        print_test_header("MetricResults Serialization")
        
        results = MetricResults(
            psnr_db=28.3,
            ssim=0.78,
            mse=0.015,
            mae=0.07,
            lpips=0.08
        )
        
        # Convert to dictionary
        results_dict = asdict(results)
        
        # Verify all fields present
        expected_fields = ['psnr_db', 'ssim', 'mse', 'mae', 'lpips']
        for field in expected_fields:
            assert field in results_dict
        
        # Test JSON serialization
        json_str = json.dumps(results_dict)
        restored_dict = json.loads(json_str)
        
        # Verify values preserved
        assert restored_dict['psnr_db'] == 28.3
        assert restored_dict['ssim'] == 0.78
        
        print_test_result("Serialization", True, "JSON round-trip successful")


class TestIndividualImageMetrics:
    """Test suite for IndividualImageMetrics dataclass"""
    
    def test_individual_metrics_creation(self):
        """Test IndividualImageMetrics creation"""
        print_test_header("IndividualImageMetrics Creation")
        
        metrics = IndividualImageMetrics(
            psnr_db=32.1,
            ssim=0.92,
            mse=0.008,
            mae=0.03,
            lpips=0.05,
            ssim_improved=0.94
        )
        
        # Verify all fields
        assert metrics.psnr_db == 32.1
        assert metrics.ssim == 0.92
        assert metrics.mse == 0.008
        assert metrics.mae == 0.03
        assert metrics.lpips == 0.05
        assert metrics.ssim_improved == 0.94
        
        print_test_result("IndividualImageMetrics", True, "All fields set correctly")
    
    def test_individual_metrics_optional_advanced(self):
        """Test IndividualImageMetrics with optional advanced metrics"""
        print_test_header("IndividualImageMetrics Optional Advanced")
        
        # Create with only basic metrics
        metrics = IndividualImageMetrics(
            psnr_db=25.0,
            ssim=0.80,
            mse=0.02,
            mae=0.08
        )
        
        assert metrics.lpips is None
        assert metrics.ssim_improved is None
        
        print_test_result("Optional advanced", True, "Defaults to None")


class TestDatasetEvaluationResults:
    """Test suite for DatasetEvaluationResults dataclass"""
    
    def test_dataset_results_creation(self):
        """Test DatasetEvaluationResults creation"""
        print_test_header("DatasetEvaluationResults Creation")
        
        results = DatasetEvaluationResults(
            fid_score=15.2,
            total_images=100,
            original_dataset_path="/path/to/original",
            generated_dataset_path="/path/to/generated",
            evaluation_timestamp="2024-01-01T12:00:00",
            individual_metrics_summary={'psnr_mean': 28.5, 'ssim_mean': 0.85}
        )
        
        # Verify all fields
        assert results.fid_score == 15.2
        assert results.total_images == 100
        assert results.original_dataset_path == "/path/to/original"
        assert results.generated_dataset_path == "/path/to/generated"
        assert results.evaluation_timestamp == "2024-01-01T12:00:00"
        assert isinstance(results.individual_metrics_summary, dict)
        
        print_test_result("DatasetEvaluationResults", True, "All fields set correctly")


class TestAllMetricsResults:
    """Test suite for AllMetricsResults dataclass"""
    
    def test_all_metrics_results_creation(self):
        """Test AllMetricsResults creation"""
        print_test_header("AllMetricsResults Creation")
        
        # Create sample individual metrics
        individual_metrics = [
            IndividualImageMetrics(psnr_db=30.0, ssim=0.9, mse=0.01, mae=0.04),
            IndividualImageMetrics(psnr_db=28.5, ssim=0.85, mse=0.015, mae=0.06)
        ]
        
        statistics = {
            'psnr': {'mean': 29.25, 'std': 1.06},
            'ssim': {'mean': 0.875, 'std': 0.035}
        }
        
        results = AllMetricsResults(
            individual_metrics=individual_metrics,
            fid_score=12.8,
            statistics=statistics,
            total_images=2,
            evaluation_timestamp="2024-01-01T12:00:00",
            created_dataset_path="/path/to/created",
            original_dataset_path="/path/to/original"
        )
        
        # Verify structure
        assert len(results.individual_metrics) == 2
        assert results.fid_score == 12.8
        assert results.total_images == 2
        assert isinstance(results.statistics, dict)
        
        print_test_result("AllMetricsResults", True, "Created with 2 metrics")
    
    def test_metric_summary_generation(self):
        """Test get_metric_summary method"""
        print_test_header("Metric Summary Generation")
        
        statistics = {
            'psnr': {'mean': 30.5, 'std': 2.1},
            'ssim': {'mean': 0.88, 'std': 0.05},
            'lpips': {'mean': 0.12, 'std': 0.03}
        }
        
        results = AllMetricsResults(
            individual_metrics=[],
            fid_score=15.7,
            statistics=statistics,
            total_images=10,
            evaluation_timestamp="2024-01-01T12:00:00",
            created_dataset_path="/path/to/created",
            original_dataset_path="/path/to/original"
        )
        
        summary = results.get_metric_summary()
        
        # Verify summary contains key information
        assert "30.50dB" in summary  # PSNR
        assert "0.8800" in summary  # SSIM  
        assert "0.12" in summary     # LPIPS
        assert "15.70" in summary    # FID
        
        print_test_result("Metric summary", True, "Summary generated correctly")
    
    def test_metric_summary_missing_metrics(self):
        """Test get_metric_summary with missing metrics"""
        print_test_header("Metric Summary - Missing Metrics")
        
        # Create results with minimal statistics
        statistics = {}
        
        results = AllMetricsResults(
            individual_metrics=[],
            fid_score=20.0,
            statistics=statistics,
            total_images=5,
            evaluation_timestamp="2024-01-01T12:00:00",
            created_dataset_path="/path/to/created",
            original_dataset_path="/path/to/original"
        )
        
        summary = results.get_metric_summary()
        
        # Should handle missing metrics gracefully
        assert "0dB" in summary or "0.00dB" in summary  # Default PSNR
        assert "0.0000" in summary                       # Default SSIM
        assert "N/A" in summary                          # Default LPIPS
        assert "20.00" in summary                        # FID present
        
        print_test_result("Missing metrics", True, "Handled gracefully")


class TestMetricsTracker:
    """Test suite for MetricsTracker class"""
    
    @pytest.fixture
    def metrics_tracker(self):
        """Create MetricsTracker instance for testing"""
        return MetricsTracker()
    
    @pytest.fixture
    def sample_optimization_data(self):
        """Create sample optimization data for tracking"""
        torch.manual_seed(42)
        original = torch.rand(1, 3, 16, 16)
        
        # Simulate optimization progression (improving reconstruction)
        iterations_data = []
        for i in range(5):
            # Gradually improving reconstruction
            noise_level = 0.2 * (1 - i/10)  # Decreasing noise
            reconstructed = original + torch.randn_like(original) * noise_level
            reconstructed = torch.clamp(reconstructed, 0, 1)
            loss = 0.1 * (1 - i/10)  # Decreasing loss
            
            iterations_data.append((original, reconstructed, i, loss))
        
        return iterations_data
    
    def test_add_metrics_single(self, metrics_tracker, sample_optimization_data):
        """Test adding single metrics entry"""
        print_test_header("MetricsTracker - Single Entry")
        
        original, reconstructed, iteration, loss = sample_optimization_data[0]
        
        metrics_tracker.add_metrics(original, reconstructed, iteration, loss)
        
        history = metrics_tracker.get_history()
        assert len(history) == 1
        
        record = history[0]
        assert record['iteration'] == iteration
        assert record['loss'] == loss
        assert 'psnr_db' in record
        assert 'ssim' in record
        assert 'mse' in record
        assert 'mae' in record
        
        print_test_result("Single entry", True, f"Iteration {iteration} tracked")
    
    def test_add_metrics_multiple(self, metrics_tracker, sample_optimization_data):
        """Test adding multiple metrics entries"""
        print_test_header("MetricsTracker - Multiple Entries")
        
        # Add all optimization steps
        for original, reconstructed, iteration, loss in sample_optimization_data:
            metrics_tracker.add_metrics(original, reconstructed, iteration, loss)
        
        history = metrics_tracker.get_history()
        assert len(history) == len(sample_optimization_data)
        
        # Verify progression (metrics should generally improve)
        first_record = history[0]
        last_record = history[-1]
        
        assert first_record['iteration'] == 0
        assert last_record['iteration'] == 4
        assert last_record['loss'] <= first_record['loss']  # Loss should decrease
        
        print_test_result("Multiple entries", True, f"Tracked {len(history)} iterations")
    
    def test_get_final_summary(self, metrics_tracker, sample_optimization_data):
        """Test final summary generation"""
        print_test_header("MetricsTracker - Final Summary")
        
        # Add optimization progression
        for original, reconstructed, iteration, loss in sample_optimization_data:
            metrics_tracker.add_metrics(original, reconstructed, iteration, loss)
        
        summary = metrics_tracker.get_final_summary()
        
        # Verify summary structure
        expected_keys = [
            'initial_psnr', 'final_psnr', 'psnr_improvement',
            'initial_ssim', 'final_ssim', 'ssim_improvement',
            'initial_loss', 'final_loss', 'loss_reduction_percent'
        ]
        
        for key in expected_keys:
            assert key in summary
            assert isinstance(summary[key], (float, int))
        
        # Verify improvements (should be positive or zero)
        assert summary['psnr_improvement'] >= 0  # PSNR should improve or stay same
        assert summary['ssim_improvement'] >= 0  # SSIM should improve or stay same
        assert summary['loss_reduction_percent'] >= 0  # Loss should reduce
        
        print_test_result("Final summary", True, 
                         f"PSNR improvement: {summary['psnr_improvement']:.2f}dB")
    
    def test_empty_tracker_summary(self, metrics_tracker):
        """Test summary generation with empty tracker"""
        print_test_header("MetricsTracker - Empty Summary")
        
        summary = metrics_tracker.get_final_summary()
        
        # Should return empty dictionary
        assert summary == {}
        
        print_test_result("Empty summary", True, "Returns empty dict")
    
    def test_single_entry_summary(self, metrics_tracker, sample_optimization_data):
        """Test summary with single entry"""
        print_test_header("MetricsTracker - Single Entry Summary")
        
        original, reconstructed, iteration, loss = sample_optimization_data[0]
        metrics_tracker.add_metrics(original, reconstructed, iteration, loss)
        
        summary = metrics_tracker.get_final_summary()
        
        # With single entry, improvements should be zero
        assert summary['psnr_improvement'] == 0
        assert summary['ssim_improvement'] == 0
        assert summary['loss_reduction_percent'] == 0
        
        print_test_result("Single entry summary", True, "Zero improvements (expected)")


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_standalone_psnr_function(self):
        """Test standalone calculate_psnr function"""
        print_test_header("Standalone PSNR Function")
        
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 16, 16)
        img2 = img1 + torch.randn_like(img1) * 0.1
        img2 = torch.clamp(img2, 0, 1)
        
        psnr = calculate_psnr(img1, img2)
        
        # Should match ImageMetrics result
        metrics = ImageMetrics()
        psnr_class = metrics.calculate_psnr(img1, img2)
        
        assert abs(psnr - psnr_class) < 1e-6
        
        print_test_result("Standalone PSNR", True, f"PSNR: {psnr:.2f}dB")


class TestErrorHandling:
    """Test suite for error handling scenarios"""
    
    def test_mismatched_tensor_shapes(self):
        """Test handling of mismatched tensor shapes"""
        print_test_header("Error Handling - Shape Mismatch")
        
        metrics = ImageMetrics()
        
        img1 = torch.rand(1, 3, 16, 16)
        img2 = torch.rand(1, 3, 32, 32)  # Different size
        
        # Should raise an error for mismatched shapes
        with pytest.raises(RuntimeError):
            metrics.calculate_psnr(img1, img2)
        
        print_test_result("Shape mismatch", True, "RuntimeError raised correctly")
    
    def test_invalid_tensor_ranges(self):
        """Test handling of invalid tensor value ranges"""
        print_test_header("Error Handling - Invalid Ranges")
        
        metrics = ImageMetrics()
        
        # Create tensors outside [0, 1] range
        img1 = torch.rand(1, 3, 16, 16)
        img2 = torch.rand(1, 3, 16, 16) * 2.0  # Values in [0, 2]
        
        # Metrics should still calculate (no range validation)
        psnr = metrics.calculate_psnr(img1, img2)
        ssim = metrics.calculate_ssim(img1, img2)
        
        assert isinstance(psnr, float)
        assert isinstance(ssim, float)
        
        print_test_result("Invalid ranges", True, "Calculated without range validation")
    
    def test_zero_variance_images(self):
        """Test metrics with zero variance (constant) images"""
        print_test_header("Error Handling - Zero Variance")
        
        metrics = ImageMetrics()
        
        # Create constant images
        constant_img1 = torch.full((1, 3, 16, 16), 0.5)
        constant_img2 = torch.full((1, 3, 16, 16), 0.7)
        
        # Should handle constant images gracefully
        psnr = metrics.calculate_psnr(constant_img1, constant_img2)
        ssim = metrics.calculate_ssim(constant_img1, constant_img2)
        mse = metrics.calculate_mse(constant_img1, constant_img2)
        mae = metrics.calculate_mae(constant_img1, constant_img2)
        
        assert isinstance(psnr, float)
        assert isinstance(ssim, float)
        assert isinstance(mse, float)
        assert isinstance(mae, float)
        
        # For constant images with different values
        assert psnr > 0
        assert abs(mse - 0.04) < 1e-6  # (0.7 - 0.5)^2 = 0.04
        assert abs(mae - 0.2) < 1e-6   # |0.7 - 0.5| = 0.2
        
        print_test_result("Zero variance", True, "Handled gracefully")


# Convenience function for direct execution
def run_all_tests():
    """Run all tests manually without pytest runner"""
    print("🧪 Starting Image Metrics Unit Tests")
    print("=" * 60)
    
    # Create test instances
    test_image_metrics = TestImageMetrics()
    test_metric_results = TestMetricResults()
    test_individual_metrics = TestIndividualImageMetrics()
    test_dataset_results = TestDatasetEvaluationResults()
    test_all_metrics_results = TestAllMetricsResults()
    test_metrics_tracker = TestMetricsTracker()
    test_utility_functions = TestUtilityFunctions()
    test_error_handling = TestErrorHandling()
    
    try:
        # Create fixtures manually
        image_metrics = ImageMetrics(device='cpu')
        torch.manual_seed(42)
        sample_images = (torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
        torch.manual_seed(123)
        identical_images = (torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
        identical_images = (identical_images[0], identical_images[0].clone())
        
        metrics_tracker = MetricsTracker()
        
        # Sample optimization data
        torch.manual_seed(42)
        original = torch.rand(1, 3, 16, 16)
        sample_optimization_data = []
        for i in range(5):
            noise_level = 0.2 * (1 - i/10)
            reconstructed = original + torch.randn_like(original) * noise_level
            reconstructed = torch.clamp(reconstructed, 0, 1)
            loss = 0.1 * (1 - i/10)
            sample_optimization_data.append((original, reconstructed, i, loss))
        
        # Run ImageMetrics tests
        test_image_metrics.test_psnr_calculation_normal(image_metrics, sample_images)
        test_image_metrics.test_psnr_identical_images(image_metrics, identical_images)
        test_image_metrics.test_psnr_different_devices(sample_images)
        test_image_metrics.test_ssim_calculation_normal(image_metrics, sample_images)
        test_image_metrics.test_ssim_identical_images(image_metrics, identical_images)
        test_image_metrics.test_ssim_window_size_parameter(image_metrics, sample_images)
        test_image_metrics.test_mse_calculation(image_metrics, sample_images)
        test_image_metrics.test_mse_identical_images(image_metrics, identical_images)
        test_image_metrics.test_mae_calculation(image_metrics, sample_images)
        test_image_metrics.test_mae_identical_images(image_metrics, identical_images)
        test_image_metrics.test_all_metrics_calculation(image_metrics, sample_images)
        test_image_metrics.test_batch_metrics_calculation(image_metrics)
        test_image_metrics.test_batch_statistics_calculation(image_metrics)
        test_image_metrics.test_batch_statistics_empty_list(image_metrics)
        test_image_metrics.test_gaussian_kernel_creation(image_metrics)
        test_image_metrics.test_gaussian_filter_application(image_metrics)
        test_image_metrics.test_tensor_shapes_validation(image_metrics)
        test_image_metrics.test_extreme_image_values(image_metrics)
        test_image_metrics.test_grayscale_vs_rgb_consistency(image_metrics)
        
        # Run MetricResults tests
        test_metric_results.test_metric_results_creation()
        test_metric_results.test_metric_results_optional_fields()
        test_metric_results.test_metric_results_serialization()
        
        # Run IndividualImageMetrics tests
        test_individual_metrics.test_individual_metrics_creation()
        test_individual_metrics.test_individual_metrics_optional_advanced()
        
        # Run DatasetEvaluationResults tests
        test_dataset_results.test_dataset_results_creation()
        
        # Run AllMetricsResults tests
        test_all_metrics_results.test_all_metrics_results_creation()
        test_all_metrics_results.test_metric_summary_generation()
        test_all_metrics_results.test_metric_summary_missing_metrics()
        
        # Run MetricsTracker tests
        test_metrics_tracker.test_add_metrics_single(metrics_tracker, sample_optimization_data)
        
        # Create new tracker for multiple entries test
        metrics_tracker_multi = MetricsTracker()
        test_metrics_tracker.test_add_metrics_multiple(metrics_tracker_multi, sample_optimization_data)
        test_metrics_tracker.test_get_final_summary(metrics_tracker_multi, sample_optimization_data)
        
        # Create new tracker for empty test
        metrics_tracker_empty = MetricsTracker()
        test_metrics_tracker.test_empty_tracker_summary(metrics_tracker_empty)
        
        # Create new tracker for single entry test
        metrics_tracker_single = MetricsTracker()
        test_metrics_tracker.test_single_entry_summary(metrics_tracker_single, sample_optimization_data)
        
        # Run utility function tests
        test_utility_functions.test_standalone_psnr_function()
        
        # Run error handling tests
        test_error_handling.test_mismatched_tensor_shapes()
        test_error_handling.test_invalid_tensor_ranges()
        test_error_handling.test_zero_variance_images(image_metrics)
        
        print("\n" + "=" * 60)
        print("🎉 ALL IMAGE METRICS TESTS COMPLETED SUCCESSFULLY!")
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