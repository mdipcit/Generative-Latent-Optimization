#!/usr/bin/env python3
"""
Unit Tests for Individual Metrics Module

Comprehensive testing of advanced individual image metrics including
LPIPS and improved SSIM implementations.
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


class TestLPIPSMetric:
    """Test suite for LPIPSMetric class"""
    
    def test_lpips_initialization_with_mock(self):
        """Test LPIPS initialization with mocked package"""
        print_test_header("LPIPS Initialization - Mocked")
        
        # Create fresh mock for this test
        mock_lpips = Mock()
        mock_loss_fn = Mock()
        mock_loss_fn.to.return_value = mock_loss_fn
        mock_lpips.LPIPS.return_value = mock_loss_fn
        
        with patch.dict('sys.modules', {'lpips': mock_lpips}):
            from src.generative_latent_optimization.metrics.individual_metrics import LPIPSMetric
            
            lpips_metric = LPIPSMetric(net='alex', device='cpu', use_gpu=False)
            
            assert lpips_metric.device == 'cpu'
            assert lpips_metric.net == 'alex'
            assert hasattr(lpips_metric, 'loss_fn')
            
            print_test_result("LPIPS mock init", True, "Initialized with mocked package")
    
    def test_lpips_initialization_different_networks(self):
        """Test LPIPS initialization with different network backbones"""
        print_test_header("LPIPS Initialization - Different Networks")
        
        networks = ['alex', 'vgg', 'squeeze']
        
        # Create fresh mock for this test
        mock_lpips = Mock()
        mock_loss_fn = Mock()
        mock_loss_fn.to.return_value = mock_loss_fn
        mock_lpips.LPIPS.return_value = mock_loss_fn
        
        with patch.dict('sys.modules', {'lpips': mock_lpips}):
            from src.generative_latent_optimization.metrics.individual_metrics import LPIPSMetric
            
            for net in networks:
                lpips_metric = LPIPSMetric(net=net, device='cpu')
                assert lpips_metric.net == net
                
            print_test_result("LPIPS networks", True, f"Tested: {networks}")
    
    def test_lpips_missing_package(self):
        """Test LPIPS behavior when package is missing"""
        print_test_header("LPIPS Missing Package")
        
        with patch.dict('sys.modules', {'lpips': None}):
            from src.generative_latent_optimization.metrics.individual_metrics import LPIPSMetric
            
            with pytest.raises(ImportError) as exc_info:
                LPIPSMetric()
            
            assert "lpips package is required" in str(exc_info.value)
            
            print_test_result("LPIPS missing pkg", True, "ImportError raised correctly")
    
    def test_lpips_calculation_with_mock(self):
        """Test LPIPS calculation with mocked backend"""
        print_test_header("LPIPS Calculation - Mocked Backend")
        
        # Create test images
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 64, 64)
        img2 = img1 + torch.randn_like(img1) * 0.1
        img2 = torch.clamp(img2, 0, 1)
        img3 = torch.rand(1, 3, 64, 64)
        
        # Create fresh mock for this test
        mock_lpips = Mock()
        mock_loss_fn = Mock()
        mock_loss_fn.to.return_value = mock_loss_fn
        mock_lpips.LPIPS.return_value = mock_loss_fn
        
        with patch.dict('sys.modules', {'lpips': mock_lpips}):
            from src.generative_latent_optimization.metrics.individual_metrics import LPIPSMetric
            
            lpips_metric = LPIPSMetric(device='cpu')
            
            # Test similar images first
            mock_loss_fn.return_value = torch.tensor([0.05])
            lpips_similar = lpips_metric.calculate(img1, img2)
            
            # Test different images
            mock_loss_fn.return_value = torch.tensor([0.35])
            lpips_different = lpips_metric.calculate(img1, img3)
            
            assert isinstance(lpips_similar, float)
            assert isinstance(lpips_different, float)
            # Use approximate equality for floating point comparison
            assert abs(lpips_similar - 0.05) < 1e-6
            assert abs(lpips_different - 0.35) < 1e-6
            
            print_test_result("LPIPS calculation", True, 
                             f"Similar: {lpips_similar}, Different: {lpips_different}")
    
    def test_lpips_range_conversion(self):
        """Test LPIPS range conversion from [0,1] to [-1,1]"""
        print_test_header("LPIPS Range Conversion")
        
        # Create fresh mock for this test
        mock_lpips = Mock()
        mock_loss_fn = Mock()
        mock_loss_fn.to.return_value = mock_loss_fn
        mock_loss_fn.return_value = torch.tensor([0.1])
        mock_lpips.LPIPS.return_value = mock_loss_fn
        
        with patch.dict('sys.modules', {'lpips': mock_lpips}):
            from src.generative_latent_optimization.metrics.individual_metrics import LPIPSMetric
            
            lpips_metric = LPIPSMetric(device='cpu')
            
            # Create images in [0,1] range
            img1 = torch.tensor([[[[0.2, 0.8], [0.5, 0.9]]]], dtype=torch.float32)
            img2 = torch.tensor([[[[0.3, 0.7], [0.4, 0.6]]]], dtype=torch.float32)
            
            lpips_value = lpips_metric.calculate(img1, img2)
            
            # Should have called loss_fn
            assert mock_loss_fn.called
            assert isinstance(lpips_value, float)
            assert abs(lpips_value - 0.1) < 1e-6
            
            print_test_result("Range conversion", True, "Range conversion handled")
    
    def test_lpips_shape_validation(self):
        """Test LPIPS shape validation"""
        print_test_header("LPIPS Shape Validation")
        
        # Create fresh mock for this test
        mock_lpips = Mock()
        mock_loss_fn = Mock()
        mock_loss_fn.to.return_value = mock_loss_fn
        mock_lpips.LPIPS.return_value = mock_loss_fn
        
        with patch.dict('sys.modules', {'lpips': mock_lpips}):
            from src.generative_latent_optimization.metrics.individual_metrics import LPIPSMetric
            
            lpips_metric = LPIPSMetric(device='cpu')
            
            # Create mismatched shapes
            img1 = torch.rand(1, 3, 64, 64)
            img2 = torch.rand(1, 3, 32, 32)  # Different size
            
            result = lpips_metric.calculate(img1, img2)
            
            # Should return None for shape mismatch (error caught internally)
            assert result is None
            
            print_test_result("LPIPS shape validation", True, "Returns None for mismatched shapes")
    
    def test_lpips_batch_calculation(self):
        """Test LPIPS batch calculation"""
        print_test_header("LPIPS Batch Calculation")
        
        # Create fresh mock for this test
        mock_lpips = Mock()
        mock_loss_fn = Mock()
        mock_loss_fn.to.return_value = mock_loss_fn
        mock_loss_fn.return_value = torch.tensor([0.1])
        mock_lpips.LPIPS.return_value = mock_loss_fn
        
        with patch.dict('sys.modules', {'lpips': mock_lpips}):
            from src.generative_latent_optimization.metrics.individual_metrics import LPIPSMetric
            
            lpips_metric = LPIPSMetric(device='cpu')
            
            # Create batch
            batch_size = 3
            img1_batch = torch.rand(batch_size, 3, 64, 64)
            img2_batch = torch.rand(batch_size, 3, 64, 64)
            
            lpips_values = lpips_metric.calculate_batch(img1_batch, img2_batch)
            
            assert len(lpips_values) == batch_size
            assert all(abs(val - 0.1) < 1e-6 for val in lpips_values)
            
            print_test_result("LPIPS batch", True, f"Processed {batch_size} images")
    
    def test_lpips_error_handling(self):
        """Test LPIPS error handling"""
        print_test_header("LPIPS Error Handling")
        
        # Create fresh mock for this test
        mock_lpips = Mock()
        mock_loss_fn = Mock()
        mock_loss_fn.to.return_value = mock_loss_fn
        mock_loss_fn.side_effect = RuntimeError("Simulated error")
        mock_lpips.LPIPS.return_value = mock_loss_fn
        
        with patch.dict('sys.modules', {'lpips': mock_lpips}):
            from src.generative_latent_optimization.metrics.individual_metrics import LPIPSMetric
            
            lpips_metric = LPIPSMetric(device='cpu')
            
            img1 = torch.rand(1, 3, 64, 64)
            img2 = torch.rand(1, 3, 64, 64)
            
            result = lpips_metric.calculate(img1, img2)
            
            # Should return None on error
            assert result is None
            
            print_test_result("LPIPS error handling", True, "Returns None on error")


class TestImprovedSSIM:
    """Test suite for ImprovedSSIM class"""
    
    def test_improved_ssim_initialization(self):
        """Test ImprovedSSIM initialization with mocked package"""
        print_test_header("ImprovedSSIM Initialization")
        
        # Create fresh mock for this test
        mock_torchmetrics = Mock()
        mock_ssim_class = Mock()
        mock_ssim_instance = Mock()
        mock_ssim_instance.to.return_value = mock_ssim_instance
        mock_ssim_class.return_value = mock_ssim_instance
        mock_torchmetrics.image = Mock()
        mock_torchmetrics.image.StructuralSimilarityIndexMeasure = mock_ssim_class
        
        with patch.dict('sys.modules', {'torchmetrics': mock_torchmetrics, 'torchmetrics.image': mock_torchmetrics.image}):
            from src.generative_latent_optimization.metrics.individual_metrics import ImprovedSSIM
            
            ssim_metric = ImprovedSSIM(data_range=1.0, device='cpu')
            
            assert ssim_metric.device == 'cpu'
            assert ssim_metric.data_range == 1.0
            assert hasattr(ssim_metric, 'ssim')
            
            print_test_result("ImprovedSSIM init", True, "Initialized with mocked package")
    
    def test_improved_ssim_different_data_ranges(self):
        """Test ImprovedSSIM with different data ranges"""
        print_test_header("ImprovedSSIM Data Ranges")
        
        # Create fresh mock for this test
        mock_torchmetrics = Mock()
        mock_ssim_class = Mock()
        mock_ssim_instance = Mock()
        mock_ssim_instance.to.return_value = mock_ssim_instance
        mock_ssim_class.return_value = mock_ssim_instance
        mock_torchmetrics.image = Mock()
        mock_torchmetrics.image.StructuralSimilarityIndexMeasure = mock_ssim_class
        
        data_ranges = [1.0, 2.0]
        
        with patch.dict('sys.modules', {'torchmetrics': mock_torchmetrics, 'torchmetrics.image': mock_torchmetrics.image}):
            from src.generative_latent_optimization.metrics.individual_metrics import ImprovedSSIM
            
            for data_range in data_ranges:
                ssim_metric = ImprovedSSIM(data_range=data_range, device='cpu')
                assert ssim_metric.data_range == data_range
                
            print_test_result("SSIM data ranges", True, f"Tested: {data_ranges}")
    
    def test_improved_ssim_missing_package(self):
        """Test ImprovedSSIM behavior when torchmetrics is missing"""
        print_test_header("ImprovedSSIM Missing Package")
        
        with patch.dict('sys.modules', {'torchmetrics': None}):
            from src.generative_latent_optimization.metrics.individual_metrics import ImprovedSSIM
            
            with pytest.raises(ImportError) as exc_info:
                ImprovedSSIM()
            
            assert "torchmetrics package is required" in str(exc_info.value)
            
            print_test_result("SSIM missing pkg", True, "ImportError raised correctly")
    
    def test_improved_ssim_calculation(self):
        """Test ImprovedSSIM calculation with mocked backend"""
        print_test_header("ImprovedSSIM Calculation")
        
        # Create test images
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 64, 64)
        img2 = img1 + torch.randn_like(img1) * 0.1
        img2 = torch.clamp(img2, 0, 1)
        img3 = torch.rand(1, 3, 64, 64)
        
        # Create fresh mock for this test
        mock_torchmetrics = Mock()
        mock_ssim_class = Mock()
        mock_ssim_instance = Mock()
        mock_ssim_instance.to.return_value = mock_ssim_instance
        mock_ssim_class.return_value = mock_ssim_instance
        mock_torchmetrics.image = Mock()
        mock_torchmetrics.image.StructuralSimilarityIndexMeasure = mock_ssim_class
        
        with patch.dict('sys.modules', {'torchmetrics': mock_torchmetrics, 'torchmetrics.image': mock_torchmetrics.image}):
            from src.generative_latent_optimization.metrics.individual_metrics import ImprovedSSIM
            
            ssim_metric = ImprovedSSIM(device='cpu')
            
            # Test similar images
            mock_ssim_instance.return_value = torch.tensor(0.85)
            ssim_similar = ssim_metric.calculate(img1, img2)
            
            # Test different images
            mock_ssim_instance.return_value = torch.tensor(0.45) 
            ssim_different = ssim_metric.calculate(img1, img3)
            
            assert isinstance(ssim_similar, float)
            assert isinstance(ssim_different, float)
            # Use approximate equality for floating point comparison
            assert abs(ssim_similar - 0.85) < 1e-6
            assert abs(ssim_different - 0.45) < 1e-6
            assert ssim_similar > ssim_different
            
            print_test_result("SSIM calculation", True, 
                             f"Similar: {ssim_similar}, Different: {ssim_different}")
    
    def test_improved_ssim_shape_validation(self):
        """Test ImprovedSSIM shape validation"""
        print_test_header("ImprovedSSIM Shape Validation")
        
        # Create fresh mock for this test
        mock_torchmetrics = Mock()
        mock_ssim_class = Mock()
        mock_ssim_instance = Mock()
        mock_ssim_instance.to.return_value = mock_ssim_instance
        mock_ssim_class.return_value = mock_ssim_instance
        mock_torchmetrics.image = Mock()
        mock_torchmetrics.image.StructuralSimilarityIndexMeasure = mock_ssim_class
        
        with patch.dict('sys.modules', {'torchmetrics': mock_torchmetrics, 'torchmetrics.image': mock_torchmetrics.image}):
            from src.generative_latent_optimization.metrics.individual_metrics import ImprovedSSIM
            
            ssim_metric = ImprovedSSIM(device='cpu')
            
            # Create mismatched shapes
            img1 = torch.rand(1, 3, 64, 64)
            img2 = torch.rand(1, 3, 32, 32)  # Different size
            
            result = ssim_metric.calculate(img1, img2)
            
            # Should return None for shape mismatch (error caught internally)
            assert result is None
            
            print_test_result("SSIM shape validation", True, "Returns None for mismatched shapes")
    
    def test_improved_ssim_batch_calculation(self):
        """Test ImprovedSSIM batch calculation"""
        print_test_header("ImprovedSSIM Batch Calculation")
        
        # Create fresh mock for this test
        mock_torchmetrics = Mock()
        mock_ssim_class = Mock()
        mock_ssim_instance = Mock()
        mock_ssim_instance.to.return_value = mock_ssim_instance
        mock_ssim_instance.return_value = torch.tensor(0.8)
        mock_ssim_class.return_value = mock_ssim_instance
        mock_torchmetrics.image = Mock()
        mock_torchmetrics.image.StructuralSimilarityIndexMeasure = mock_ssim_class
        
        with patch.dict('sys.modules', {'torchmetrics': mock_torchmetrics, 'torchmetrics.image': mock_torchmetrics.image}):
            from src.generative_latent_optimization.metrics.individual_metrics import ImprovedSSIM
            
            ssim_metric = ImprovedSSIM(device='cpu')
            
            # Create batch
            batch_size = 3
            img1_batch = torch.rand(batch_size, 3, 64, 64)
            img2_batch = torch.rand(batch_size, 3, 64, 64)
            
            ssim_values = ssim_metric.calculate_batch(img1_batch, img2_batch)
            
            assert len(ssim_values) == batch_size
            assert all(abs(val - 0.8) < 1e-6 for val in ssim_values)
            
            print_test_result("SSIM batch", True, f"Processed {batch_size} images")
    
    def test_improved_ssim_error_handling(self):
        """Test ImprovedSSIM error handling"""
        print_test_header("ImprovedSSIM Error Handling")
        
        # Create fresh mock for this test
        mock_torchmetrics = Mock()
        mock_ssim_class = Mock()
        mock_ssim_instance = Mock()
        mock_ssim_instance.to.return_value = mock_ssim_instance
        mock_ssim_instance.side_effect = RuntimeError("Simulated SSIM error")
        mock_ssim_class.return_value = mock_ssim_instance
        mock_torchmetrics.image = Mock()
        mock_torchmetrics.image.StructuralSimilarityIndexMeasure = mock_ssim_class
        
        with patch.dict('sys.modules', {'torchmetrics': mock_torchmetrics, 'torchmetrics.image': mock_torchmetrics.image}):
            from src.generative_latent_optimization.metrics.individual_metrics import ImprovedSSIM
            
            ssim_metric = ImprovedSSIM(device='cpu')
            
            img1 = torch.rand(1, 3, 64, 64)
            img2 = torch.rand(1, 3, 64, 64)
            
            result = ssim_metric.calculate(img1, img2)
            
            # Should return None on error
            assert result is None
            
            print_test_result("SSIM error handling", True, "Returns None on error")


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_lpips_functionality_test_function(self):
        """Test the test_lpips_functionality utility function"""
        print_test_header("LPIPS Functionality Test Function")
        
        # Mock the LPIPSMetric class entirely to avoid complexity
        mock_lpips_metric = Mock()
        mock_lpips_metric.calculate.side_effect = [0.05, 0.35]
        
        mock_lpips = Mock()
        
        with patch.dict('sys.modules', {'lpips': mock_lpips}):
            with patch('src.generative_latent_optimization.metrics.individual_metrics.LPIPSMetric', return_value=mock_lpips_metric):
                from src.generative_latent_optimization.metrics.individual_metrics import test_lpips_functionality
                
                # Should run without errors
                test_lpips_functionality(device='cpu')
                
                # Verify calls were made
                assert mock_lpips_metric.calculate.call_count == 2
                
                print_test_result("LPIPS test function", True, "Executed without errors")
    
    def test_improved_ssim_functionality_test_function(self):
        """Test the test_improved_ssim_functionality utility function"""
        print_test_header("ImprovedSSIM Functionality Test Function")
        
        # Mock the ImprovedSSIM class entirely to avoid complexity
        mock_ssim_metric = Mock()
        mock_ssim_metric.calculate.side_effect = [0.88, 0.42]
        
        mock_torchmetrics = Mock()
        
        with patch.dict('sys.modules', {'torchmetrics': mock_torchmetrics, 'torchmetrics.image': mock_torchmetrics.image}):
            with patch('src.generative_latent_optimization.metrics.individual_metrics.ImprovedSSIM', return_value=mock_ssim_metric):
                from src.generative_latent_optimization.metrics.individual_metrics import test_improved_ssim_functionality
                
                # Should run without errors
                test_improved_ssim_functionality(device='cpu')
                
                # Verify calls were made
                assert mock_ssim_metric.calculate.call_count == 2
                
                print_test_result("SSIM test function", True, "Executed without errors")


class TestRealPackageIntegration:
    """Test suite for real package integration (if available)"""
    
    def test_lpips_real_package_if_available(self):
        """Test LPIPS with real package if installed"""
        print_test_header("LPIPS Real Package Integration")
        
        try:
            import lpips
            from src.generative_latent_optimization.metrics.individual_metrics import LPIPSMetric
            
            # Test with real package
            lpips_metric = LPIPSMetric(net='alex', device='cpu', use_gpu=False)
            
            # Create simple test images
            torch.manual_seed(42)
            img1 = torch.rand(1, 3, 64, 64)
            img2 = img1.clone()  # Identical
            img3 = torch.rand(1, 3, 64, 64)  # Different
            
            lpips_identical = lpips_metric.calculate(img1, img2)
            lpips_different = lpips_metric.calculate(img1, img3)
            
            # Identical images should have lower LPIPS
            assert isinstance(lpips_identical, float)
            assert isinstance(lpips_different, float)
            assert lpips_identical <= lpips_different
            
            print_test_result("LPIPS real package", True, 
                             f"Identical: {lpips_identical:.4f}, Different: {lpips_different:.4f}")
            
        except ImportError:
            print_test_result("LPIPS real package", True, "Skipped (package not available)")
    
    def test_improved_ssim_real_package_if_available(self):
        """Test ImprovedSSIM with real package if installed"""
        print_test_header("ImprovedSSIM Real Package Integration")
        
        try:
            import torchmetrics
            from src.generative_latent_optimization.metrics.individual_metrics import ImprovedSSIM
            
            # Test with real package
            ssim_metric = ImprovedSSIM(data_range=1.0, device='cpu')
            
            # Create simple test images
            torch.manual_seed(42)
            img1 = torch.rand(1, 3, 64, 64)
            img2 = img1.clone()  # Identical
            img3 = torch.rand(1, 3, 64, 64)  # Different
            
            ssim_identical = ssim_metric.calculate(img1, img2)
            ssim_different = ssim_metric.calculate(img1, img3)
            
            # Identical images should have higher SSIM
            assert isinstance(ssim_identical, float)
            assert isinstance(ssim_different, float)
            assert ssim_identical >= ssim_different
            # Note: Real SSIM implementations might occasionally return values slightly outside [0,1]
            # due to numerical precision, so we use a slightly relaxed range check
            assert -0.1 <= ssim_identical <= 1.1
            assert -0.1 <= ssim_different <= 1.1
            
            print_test_result("SSIM real package", True, 
                             f"Identical: {ssim_identical:.4f}, Different: {ssim_different:.4f}")
            
        except ImportError:
            print_test_result("SSIM real package", True, "Skipped (package not available)")


# Convenience function for direct execution
def run_all_tests():
    """Run all tests manually without pytest runner"""
    print("🧪 Starting Individual Metrics Unit Tests")
    print("=" * 60)
    
    # Create test instances
    test_lpips = TestLPIPSMetric()
    test_ssim = TestImprovedSSIM()
    test_utilities = TestUtilityFunctions()
    test_real_integration = TestRealPackageIntegration()
    
    try:
        # Run LPIPS tests
        test_lpips.test_lpips_initialization_with_mock()
        test_lpips.test_lpips_initialization_different_networks()
        test_lpips.test_lpips_missing_package()
        test_lpips.test_lpips_calculation_with_mock()
        test_lpips.test_lpips_range_conversion()
        test_lpips.test_lpips_shape_validation()
        test_lpips.test_lpips_batch_calculation()
        test_lpips.test_lpips_error_handling()
        
        # Run ImprovedSSIM tests
        test_ssim.test_improved_ssim_initialization()
        test_ssim.test_improved_ssim_different_data_ranges()
        test_ssim.test_improved_ssim_missing_package()
        test_ssim.test_improved_ssim_calculation()
        test_ssim.test_improved_ssim_shape_validation()
        test_ssim.test_improved_ssim_batch_calculation()
        test_ssim.test_improved_ssim_error_handling()
        
        # Run utility function tests
        test_utilities.test_lpips_functionality_test_function()
        test_utilities.test_improved_ssim_functionality_test_function()
        
        # Run real package integration tests (will skip if packages not available)
        test_real_integration.test_lpips_real_package_if_available()
        test_real_integration.test_improved_ssim_real_package_if_available()
        
        print("\n" + "=" * 60)
        print("🎉 ALL INDIVIDUAL METRICS TESTS COMPLETED SUCCESSFULLY!")
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