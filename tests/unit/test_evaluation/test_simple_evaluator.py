#!/usr/bin/env python3
"""
Unit tests for SimpleAllMetricsEvaluator functionality
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, call
import pytest
import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from generative_latent_optimization.evaluation.simple_evaluator import SimpleAllMetricsEvaluator
from generative_latent_optimization.metrics.image_metrics import AllMetricsResults, IndividualImageMetrics
from ...fixtures.evaluation_mocks import mock_evaluation_dependencies
from ...fixtures.assertion_helpers import assert_float_approximately_equal, assert_statistics_equal


# mock_evaluation_dependencies はevaluation_mocks.pyのmock_evaluation_dependenciesに統一


@pytest.fixture
def sample_individual_metrics():
    """Sample individual metrics results"""
    return [
        IndividualImageMetrics(
            psnr_db=25.0,
            ssim=0.85,
            mse=0.1,
            mae=0.05,
            lpips=0.08,
            ssim_improved=0.88
        ),
        IndividualImageMetrics(
            psnr_db=30.0,
            ssim=0.9,
            mse=0.05,
            mae=0.03,
            lpips=0.06,
            ssim_improved=0.92
        ),
        IndividualImageMetrics(
            psnr_db=22.0,
            ssim=0.8,
            mse=0.15,
            mae=0.07,
            lpips=None,  # Missing LPIPS
            ssim_improved=0.85
        )
    ]


@pytest.fixture
def temp_image_directories():
    """Create temporary directories with mock image files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create original dataset directory
        original_dir = temp_path / "original"
        original_dir.mkdir()
        
        # Create created dataset directory
        created_dir = temp_path / "created"
        created_dir.mkdir()
        
        # Create matching image files
        for i in range(3):
            (original_dir / f"img_{i:03d}.jpg").touch()
            (created_dir / f"img_{i:03d}.png").touch()
        
        # Create some non-matching files
        (original_dir / "extra_original.jpg").touch()
        (created_dir / "extra_created.png").touch()
        
        yield {
            'original': original_dir,
            'created': created_dir,
            'temp_root': temp_path
        }


class TestSimpleAllMetricsEvaluator:
    """Test cases for SimpleAllMetricsEvaluator class"""
    
    def test_initialization_default(self, mock_evaluation_dependencies):
        """Test evaluator initialization with defaults"""
        evaluator = SimpleAllMetricsEvaluator()
        
        assert evaluator.device == 'cuda'
        assert hasattr(evaluator, 'individual_calculator')
        assert hasattr(evaluator, 'fid_evaluator')
    
    def test_initialization_custom_parameters(self, mock_evaluation_dependencies):
        """Test evaluator initialization with custom parameters"""
        evaluator = SimpleAllMetricsEvaluator(
            device='cpu',
            enable_lpips=False,
            enable_improved_ssim=True
        )
        
        assert evaluator.device == 'cpu'
        
        # Verify individual calculator was initialized with correct parameters
        individual_calc_init = mock_evaluation_dependencies['IndividualMetricsCalculator']
        individual_calc_init.assert_called_with(
            device='cpu',
            enable_lpips=False,
            enable_improved_ssim=True
        )
    
    def test_get_image_files_directory(self, mock_evaluation_dependencies, temp_image_directories):
        """Test _get_image_files with directory input"""
        evaluator = SimpleAllMetricsEvaluator()
        
        files = evaluator._get_image_files(temp_image_directories['original'])
        
        # Should find all jpg files
        assert len(files) == 4  # 3 matching + 1 extra
        for file_path in files:
            assert file_path.suffix.lower() in ['.jpg', '.jpeg']
    
    def test_get_image_files_single_file(self, mock_evaluation_dependencies):
        """Test _get_image_files with single file input"""
        evaluator = SimpleAllMetricsEvaluator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            single_image = Path(temp_dir) / "test.png"
            single_image.touch()
            
            files = evaluator._get_image_files(single_image)
            
            assert len(files) == 1
            assert files[0] == single_image
    
    def test_get_image_files_non_image_file(self, mock_evaluation_dependencies):
        """Test _get_image_files with non-image file"""
        evaluator = SimpleAllMetricsEvaluator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            text_file = Path(temp_dir) / "document.txt"
            text_file.touch()
            
            files = evaluator._get_image_files(text_file)
            
            assert len(files) == 0
    
    def test_match_image_pairs(self, mock_evaluation_dependencies):
        """Test _match_image_pairs method"""
        evaluator = SimpleAllMetricsEvaluator()
        
        original_images = [
            Path("/original/img_001.jpg"),
            Path("/original/img_002.jpg"),
            Path("/original/img_extra.jpg")
        ]
        
        created_images = [
            Path("/created/img_001.png"),
            Path("/created/img_002.png"),
            Path("/created/img_different.png")
        ]
        
        pairs = evaluator._match_image_pairs(original_images, created_images)
        
        # Should match by stem (filename without extension)
        assert len(pairs) == 2
        
        # Verify matching
        stems = [(orig.stem, created.stem) for orig, created in pairs]
        assert ('img_001', 'img_001') in stems
        assert ('img_002', 'img_002') in stems
    
    def test_match_image_pairs_no_matches(self, mock_evaluation_dependencies):
        """Test _match_image_pairs with no matching files"""
        evaluator = SimpleAllMetricsEvaluator()
        
        original_images = [Path("/original/img_001.jpg")]
        created_images = [Path("/created/different_name.png")]
        
        pairs = evaluator._match_image_pairs(original_images, created_images)
        
        assert len(pairs) == 0
    
    def test_create_default_metrics(self, mock_evaluation_dependencies):
        """Test _create_default_metrics method"""
        evaluator = SimpleAllMetricsEvaluator()
        
        default_metrics = evaluator._create_default_metrics()
        
        assert isinstance(default_metrics, IndividualImageMetrics)
        assert_float_approximately_equal(default_metrics.psnr_db, 0.0)
        assert_float_approximately_equal(default_metrics.ssim, 0.0)
        assert_float_approximately_equal(default_metrics.mse, 1.0)
        assert_float_approximately_equal(default_metrics.mae, 1.0)
        assert default_metrics.lpips is None
        assert default_metrics.ssim_improved is None
    
    def test_compute_metric_stats(self, mock_evaluation_dependencies):
        """Test _compute_metric_stats method"""
        evaluator = SimpleAllMetricsEvaluator()
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        stats = evaluator._compute_metric_stats(values, "Test Metric")
        
        assert_float_approximately_equal(stats['mean'], 3.0)
        assert_float_approximately_equal(stats['min'], 1.0)
        assert_float_approximately_equal(stats['max'], 5.0)
        assert_float_approximately_equal(stats['median'], 3.0)
        assert stats['count'] == 5
        assert stats['std'] > 0  # Should compute standard deviation
    
    def test_compute_metric_stats_empty_values(self, mock_evaluation_dependencies):
        """Test _compute_metric_stats with empty values"""
        evaluator = SimpleAllMetricsEvaluator()
        
        stats = evaluator._compute_metric_stats([], "Empty Metric")
        
        assert stats == {}
    
    def test_calculate_statistics(self, mock_evaluation_dependencies, sample_individual_metrics):
        """Test _calculate_statistics method"""
        evaluator = SimpleAllMetricsEvaluator()
        
        stats = evaluator._calculate_statistics(sample_individual_metrics)
        
        # Verify structure
        assert 'psnr' in stats
        assert 'ssim' in stats
        assert 'mse' in stats
        assert 'mae' in stats
        assert 'lpips' in stats  # Should be present (2 out of 3 samples have LPIPS)
        assert 'ssim_improved' in stats  # All samples have this
        
        # Verify PSNR statistics
        psnr_stats = stats['psnr']
        assert psnr_stats['mean'] == pytest.approx(25.67, rel=1e-2)  # (25+30+22)/3
        assert_float_approximately_equal(psnr_stats['min'], 22.0)
        assert_float_approximately_equal(psnr_stats['max'], 30.0)
        assert psnr_stats['count'] == 3
        
        # Verify LPIPS statistics (only 2 values, third is None)
        lpips_stats = stats['lpips']
        assert lpips_stats['count'] == 2
        assert_float_approximately_equal(lpips_stats['mean'], 0.07)  # (0.08 + 0.06) / 2
    
    def test_calculate_statistics_empty_results(self, mock_evaluation_dependencies):
        """Test _calculate_statistics with empty results"""
        evaluator = SimpleAllMetricsEvaluator()
        
        stats = evaluator._calculate_statistics([])
        
        assert stats == {}
    
    @patch('generative_latent_optimization.evaluation.simple_evaluator.tqdm')
    def test_calculate_individual_metrics_for_all(self, mock_tqdm, mock_evaluation_dependencies):
        """Test _calculate_individual_metrics_for_all method"""
        mock_tqdm.side_effect = lambda x, **kwargs: x  # Mock tqdm to return original iterable
        
        evaluator = SimpleAllMetricsEvaluator()
        
        # Create sample image pairs
        image_pairs = [
            (torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)),
            (torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))
        ]
        
        # Mock individual calculator response
        mock_metrics = IndividualImageMetrics(
            psnr_db=25.0, ssim=0.85, mse=0.1, mae=0.05, lpips=0.08, ssim_improved=0.88
        )
        evaluator.individual_calculator.calculate_all_individual_metrics.return_value = mock_metrics
        
        results = evaluator._calculate_individual_metrics_for_all(image_pairs)
        
        # Should return list of metrics
        assert len(results) == 2
        assert all(isinstance(m, IndividualImageMetrics) for m in results)
        
        # Verify individual calculator was called for each pair
        assert evaluator.individual_calculator.calculate_all_individual_metrics.call_count == 2
    
    def test_calculate_individual_metrics_for_all_with_errors(self, mock_evaluation_dependencies):
        """Test _calculate_individual_metrics_for_all with computation errors"""
        evaluator = SimpleAllMetricsEvaluator()
        
        image_pairs = [
            (torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))
        ]
        
        # Mock calculator to raise exception
        evaluator.individual_calculator.calculate_all_individual_metrics.side_effect = Exception("Calculation failed")
        
        with patch.object(evaluator, '_create_default_metrics') as mock_default:
            mock_default.return_value = IndividualImageMetrics(
                psnr_db=0.0, ssim=0.0, mse=1.0, mae=1.0, lpips=None, ssim_improved=None
            )
            
            results = evaluator._calculate_individual_metrics_for_all(image_pairs)
        
        # Should use default metrics when calculation fails
        assert len(results) == 1
        mock_default.assert_called_once()
    
    def test_calculate_dataset_fid_success(self, mock_evaluation_dependencies):
        """Test _calculate_dataset_fid successful computation"""
        evaluator = SimpleAllMetricsEvaluator()
        
        # Mock FID evaluation result
        mock_fid_result = Mock()
        mock_fid_result.fid_score = 15.5
        evaluator.fid_evaluator.evaluate_created_dataset_vs_original.return_value = mock_fid_result
        
        fid_score = evaluator._calculate_dataset_fid(Path("/created"), Path("/original"))
        
        assert_float_approximately_equal(fid_score, 15.5)
        evaluator.fid_evaluator.evaluate_created_dataset_vs_original.assert_called_once()
    
    def test_calculate_dataset_fid_error(self, mock_evaluation_dependencies):
        """Test _calculate_dataset_fid with computation error"""
        evaluator = SimpleAllMetricsEvaluator()
        
        # Mock FID evaluator to raise exception
        evaluator.fid_evaluator.evaluate_created_dataset_vs_original.side_effect = Exception("FID failed")
        
        fid_score = evaluator._calculate_dataset_fid(Path("/created"), Path("/original"))
        
        # Should return infinity on error
        assert fid_score == float('inf')
    
    @patch('vae_toolkit.load_and_preprocess_image')
    def test_load_image_as_tensor_with_vae_toolkit(self, mock_load, mock_evaluation_dependencies):
        """Test _load_image_as_tensor using vae-toolkit"""
        evaluator = SimpleAllMetricsEvaluator()
        
        # Mock vae-toolkit response
        mock_tensor = torch.randn(1, 3, 512, 512)
        mock_load.return_value = (mock_tensor, Mock())
        
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.png"
            image_path.touch()
            
            result = evaluator._load_image_as_tensor(image_path)
        
        # Should use vae-toolkit
        mock_load.assert_called_once_with(str(image_path), target_size=512)
        assert torch.equal(result, mock_tensor)
    
    def test_load_image_as_tensor_fallback(self, mock_evaluation_dependencies):
        """Test _load_image_as_tensor fallback to manual loading"""
        evaluator = SimpleAllMetricsEvaluator()
        
        # Mock vae-toolkit to not be available
        with patch('vae_toolkit.load_and_preprocess_image', 
                  side_effect=ImportError("vae-toolkit not available")):
            
            with patch.object(evaluator, '_manual_load_image') as mock_manual:
                mock_manual.return_value = torch.randn(1, 3, 512, 512)
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    image_path = Path(temp_dir) / "test.png"
                    image_path.touch()
                    
                    result = evaluator._load_image_as_tensor(image_path)
                
                # Should fallback to manual loading
                mock_manual.assert_called_once_with(image_path)
    
    @patch('generative_latent_optimization.evaluation.simple_evaluator.Image')
    @patch('torchvision.transforms')
    def test_manual_load_image(self, mock_transforms, mock_image, mock_evaluation_dependencies):
        """Test _manual_load_image method"""
        evaluator = SimpleAllMetricsEvaluator()
        
        # Mock PIL Image
        mock_pil_img = Mock()
        mock_pil_img.size = (256, 256)  # Different from 512x512
        mock_pil_img.resize.return_value = mock_pil_img
        mock_image.open.return_value.convert.return_value = mock_pil_img
        
        # Mock transforms
        mock_transform = Mock()
        mock_tensor = torch.randn(3, 512, 512)
        mock_transform.return_value = mock_tensor
        mock_transforms.Compose.return_value = mock_transform
        
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.png"
            image_path.touch()
            
            result = evaluator._manual_load_image(image_path)
        
        # Verify image loading process
        mock_image.open.assert_called_once_with(image_path)
        mock_pil_img.resize.assert_called_once_with((512, 512), mock_image.LANCZOS)
        
        # Verify tensor has batch dimension
        assert result.shape[0] == 1  # Batch dimension added
    
    def test_load_image_pairs_success(self, mock_evaluation_dependencies, temp_image_directories):
        """Test _load_image_pairs successful loading"""
        evaluator = SimpleAllMetricsEvaluator()
        
        with patch.object(evaluator, '_load_image_as_tensor') as mock_load_tensor:
            mock_load_tensor.return_value = torch.randn(1, 3, 512, 512)
            
            pairs = evaluator._load_image_pairs(
                temp_image_directories['created'],
                temp_image_directories['original']
            )
        
        # Should find 3 matching pairs (img_000, img_001, img_002)
        assert len(pairs) == 3
        
        # Each pair should be (original_tensor, created_tensor)
        for original, created in pairs:
            assert isinstance(original, torch.Tensor)
            assert isinstance(created, torch.Tensor)
    
    def test_load_image_pairs_no_matches(self, mock_evaluation_dependencies):
        """Test _load_image_pairs when no matching pairs exist"""
        evaluator = SimpleAllMetricsEvaluator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directories with non-matching files
            original_dir = Path(temp_dir) / "original"
            created_dir = Path(temp_dir) / "created"
            original_dir.mkdir()
            created_dir.mkdir()
            
            (original_dir / "original_only.jpg").touch()
            (created_dir / "created_only.png").touch()
            
            with pytest.raises(ValueError, match="No matching image pairs found"):
                evaluator._load_image_pairs(created_dir, original_dir)
    
    def test_load_image_pairs_with_loading_errors(self, mock_evaluation_dependencies, temp_image_directories):
        """Test _load_image_pairs with some loading errors"""
        evaluator = SimpleAllMetricsEvaluator()
        
        # Mock loader to fail on some images
        def mock_load_with_errors(path):
            if "img_001" in str(path):
                raise Exception("Failed to load")
            return torch.randn(1, 3, 512, 512)
        
        with patch.object(evaluator, '_load_image_as_tensor', side_effect=mock_load_with_errors):
            pairs = evaluator._load_image_pairs(
                temp_image_directories['created'],
                temp_image_directories['original']
            )
        
        # Should skip failed pairs but return successful ones
        assert len(pairs) == 2  # 1 failed, 2 successful


class TestSimpleEvaluatorIntegration:
    """Integration tests for complete evaluation flow"""
    
    @patch('generative_latent_optimization.evaluation.simple_evaluator.datetime')
    def test_evaluate_dataset_all_metrics_success(self, mock_datetime, mock_evaluation_dependencies, 
                                                 temp_image_directories, sample_individual_metrics):
        """Test complete evaluate_dataset_all_metrics flow"""
        mock_datetime.datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        evaluator = SimpleAllMetricsEvaluator()
        
        # Mock all internal methods
        with patch.object(evaluator, '_load_image_pairs') as mock_load_pairs, \
             patch.object(evaluator, '_calculate_individual_metrics_for_all') as mock_individual, \
             patch.object(evaluator, '_calculate_dataset_fid') as mock_fid, \
             patch.object(evaluator, '_calculate_statistics') as mock_stats:
            
            # Configure mocks
            mock_load_pairs.return_value = [(torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512))]
            mock_individual.return_value = sample_individual_metrics
            mock_fid.return_value = 15.5
            mock_stats.return_value = {'psnr': {'mean': 25.0}}
            
            results = evaluator.evaluate_dataset_all_metrics(
                temp_image_directories['created'],
                temp_image_directories['original']
            )
        
        # Verify result structure
        assert isinstance(results, AllMetricsResults)
        assert results.individual_metrics == sample_individual_metrics
        assert_float_approximately_equal(results.fid_score, 15.5)
        assert results.statistics == {'psnr': {'mean': 25.0}}
        assert results.total_images == len(sample_individual_metrics)
        assert results.evaluation_timestamp == "2024-01-01T12:00:00"
    
    def test_evaluate_dataset_all_metrics_with_path_strings(self, mock_evaluation_dependencies):
        """Test evaluate_dataset_all_metrics with string paths"""
        evaluator = SimpleAllMetricsEvaluator()
        
        with patch.object(evaluator, '_load_image_pairs') as mock_load_pairs, \
             patch.object(evaluator, '_calculate_individual_metrics_for_all') as mock_individual, \
             patch.object(evaluator, '_calculate_dataset_fid') as mock_fid, \
             patch.object(evaluator, '_calculate_statistics') as mock_stats:
            
            mock_load_pairs.return_value = []
            mock_individual.return_value = []
            mock_fid.return_value = 0.0
            mock_stats.return_value = {}
            
            # Should handle string paths correctly
            results = evaluator.evaluate_dataset_all_metrics(
                "/path/to/created", "/path/to/original"
            )
        
        # Verify Path objects were created and passed
        load_pairs_call = mock_load_pairs.call_args[0]
        assert isinstance(load_pairs_call[0], Path)
        assert isinstance(load_pairs_call[1], Path)
    
    def test_print_summary(self, mock_evaluation_dependencies, sample_individual_metrics):
        """Test print_summary method"""
        evaluator = SimpleAllMetricsEvaluator()
        
        # Create mock AllMetricsResults
        mock_results = Mock(spec=AllMetricsResults)
        mock_results.created_dataset_path = "/path/to/created"
        mock_results.original_dataset_path = "/path/to/original"
        mock_results.total_images = 100
        mock_results.evaluation_timestamp = "2024-01-01T12:00:00"
        mock_results.fid_score = 15.5
        mock_results.statistics = {
            'psnr': {'mean': 28.5, 'std': 3.2, 'min': 22.0, 'max': 35.0},
            'ssim': {'mean': 0.85, 'std': 0.05, 'min': 0.75, 'max': 0.95},
            'lpips': {'mean': 0.08, 'std': 0.02, 'min': 0.05, 'max': 0.12}
        }
        mock_results.get_metric_summary.return_value = "3 metrics computed successfully"
        
        # Should not raise any exceptions when printing
        try:
            evaluator.print_summary(mock_results)
            print_successful = True
        except Exception:
            print_successful = False
        
        assert print_successful
    
    def test_print_summary_with_missing_metrics(self, mock_evaluation_dependencies):
        """Test print_summary with missing optional metrics"""
        evaluator = SimpleAllMetricsEvaluator()
        
        # Create mock results with minimal statistics
        mock_results = Mock(spec=AllMetricsResults)
        mock_results.created_dataset_path = "/path/to/created"
        mock_results.original_dataset_path = "/path/to/original"
        mock_results.total_images = 50
        mock_results.evaluation_timestamp = "2024-01-01T12:00:00"
        mock_results.fid_score = 25.0
        mock_results.statistics = {
            'psnr': {'mean': 25.0, 'std': 2.0, 'min': 20.0, 'max': 30.0}
            # Missing SSIM, LPIPS, etc.
        }
        mock_results.get_metric_summary.return_value = "1 metric computed"
        
        # Should handle missing metrics gracefully
        try:
            evaluator.print_summary(mock_results)
            print_successful = True
        except Exception:
            print_successful = False
        
        assert print_successful
    
    def test_print_summary_fid_quality_interpretation(self, mock_evaluation_dependencies):
        """Test print_summary FID quality interpretation"""
        evaluator = SimpleAllMetricsEvaluator()
        
        # Test different FID scores and quality interpretations
        test_cases = [
            (5.0, "Excellent"),
            (25.0, "Good"),
            (75.0, "Fair"),
            (150.0, "Poor")
        ]
        
        for fid_score, expected_quality in test_cases:
            mock_results = Mock(spec=AllMetricsResults)
            mock_results.created_dataset_path = "/test"
            mock_results.original_dataset_path = "/test"
            mock_results.total_images = 10
            mock_results.evaluation_timestamp = "2024-01-01T12:00:00"
            mock_results.fid_score = fid_score
            mock_results.statistics = {}
            mock_results.get_metric_summary.return_value = "test"
            
            # Capture printed output
            import io
            from contextlib import redirect_stdout
            
            output = io.StringIO()
            with redirect_stdout(output):
                evaluator.print_summary(mock_results)
            
            printed_text = output.getvalue()
            assert expected_quality in printed_text


class TestSimpleEvaluatorCompleteFlow:
    """Test complete evaluation flow scenarios"""
    
    def test_complete_evaluation_flow_mocked(self, mock_evaluation_dependencies):
        """Test complete evaluation flow with fully mocked dependencies"""
        evaluator = SimpleAllMetricsEvaluator(device='cpu')
        
        # Create realistic mock data
        mock_image_pairs = [
            (torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512)),
            (torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512))
        ]
        
        mock_individual_results = [
            IndividualImageMetrics(psnr_db=25.0, ssim=0.85, mse=0.1, mae=0.05, lpips=0.08, ssim_improved=0.88),
            IndividualImageMetrics(psnr_db=30.0, ssim=0.9, mse=0.05, mae=0.03, lpips=0.06, ssim_improved=0.92)
        ]
        
        with patch.object(evaluator, '_load_image_pairs') as mock_load_pairs, \
             patch.object(evaluator, '_calculate_individual_metrics_for_all') as mock_individual, \
             patch.object(evaluator, '_calculate_dataset_fid') as mock_fid, \
             patch.object(evaluator, '_calculate_statistics') as mock_stats:
            
            # Configure mocks
            mock_load_pairs.return_value = mock_image_pairs
            mock_individual.return_value = mock_individual_results
            mock_fid.return_value = 12.5
            mock_stats.return_value = {
                'psnr': {'mean': 27.5, 'std': 2.5, 'min': 25.0, 'max': 30.0, 'count': 2}
            }
            
            # Execute evaluation
            results = evaluator.evaluate_dataset_all_metrics("/created", "/original")
        
        # Verify all steps were executed
        mock_load_pairs.assert_called_once()
        mock_individual.assert_called_once()
        mock_fid.assert_called_once()
        mock_stats.assert_called_once()
        
        # Verify results
        assert isinstance(results, AllMetricsResults)
        assert results.individual_metrics == mock_individual_results
        assert_float_approximately_equal(results.fid_score, 12.5)
        assert results.total_images == 2
    
    def test_evaluation_with_minimal_dataset(self, mock_evaluation_dependencies):
        """Test evaluation with minimal dataset (single image)"""
        evaluator = SimpleAllMetricsEvaluator()
        
        single_pair = [(torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512))]
        single_result = [IndividualImageMetrics(psnr_db=25.0, ssim=0.85, mse=0.1, mae=0.05, lpips=None, ssim_improved=None)]
        
        with patch.object(evaluator, '_load_image_pairs') as mock_load_pairs, \
             patch.object(evaluator, '_calculate_individual_metrics_for_all') as mock_individual, \
             patch.object(evaluator, '_calculate_dataset_fid') as mock_fid, \
             patch.object(evaluator, '_calculate_statistics') as mock_stats:
            
            mock_load_pairs.return_value = single_pair
            mock_individual.return_value = single_result
            mock_fid.return_value = 20.0
            mock_stats.return_value = {'psnr': {'mean': 25.0, 'count': 1}}
            
            results = evaluator.evaluate_dataset_all_metrics("/created", "/original")
        
        assert results.total_images == 1
        assert len(results.individual_metrics) == 1
        assert_float_approximately_equal(results.fid_score, 20.0)


class TestSimpleEvaluatorErrorHandling:
    """Test error handling and edge cases"""
    
    def test_device_parameter_propagation(self, mock_evaluation_dependencies):
        """Test that device parameter is properly propagated"""
        device = 'cpu'
        evaluator = SimpleAllMetricsEvaluator(device=device)
        
        # Verify device was passed to component initializers
        individual_calc_init = mock_evaluation_dependencies['IndividualMetricsCalculator']
        fid_eval_init = mock_evaluation_dependencies['DatasetFIDEvaluator']
        
        individual_calc_init.assert_called_with(
            device=device, enable_lpips=True, enable_improved_ssim=True
        )
        fid_eval_init.assert_called_with(device=device)
    
    def test_evaluation_with_computation_failures(self, mock_evaluation_dependencies):
        """Test evaluation when some computations fail"""
        evaluator = SimpleAllMetricsEvaluator()
        
        with patch.object(evaluator, '_load_image_pairs') as mock_load_pairs, \
             patch.object(evaluator, '_calculate_individual_metrics_for_all') as mock_individual, \
             patch.object(evaluator, '_calculate_dataset_fid') as mock_fid, \
             patch.object(evaluator, '_calculate_statistics') as mock_stats:
            
            # Configure some methods to fail
            mock_load_pairs.return_value = [(torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512))]
            mock_individual.return_value = [evaluator._create_default_metrics()]  # Default metrics due to failure
            mock_fid.return_value = float('inf')  # FID computation failed
            mock_stats.return_value = {}  # Empty stats
            
            results = evaluator.evaluate_dataset_all_metrics("/created", "/original")
        
        # Should complete despite failures
        assert isinstance(results, AllMetricsResults)
        assert results.fid_score == float('inf')
        assert len(results.individual_metrics) == 1
    
    def test_numpy_import_in_compute_metric_stats(self, mock_evaluation_dependencies):
        """Test that numpy operations work in _compute_metric_stats"""
        evaluator = SimpleAllMetricsEvaluator()
        
        # Test with real numpy operations
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Should not raise import errors
        stats = evaluator._compute_metric_stats(values, "Test")
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
        assert stats['count'] == 5
    
    def test_evaluation_logging_behavior(self, mock_evaluation_dependencies):
        """Test that evaluation produces appropriate logging"""
        evaluator = SimpleAllMetricsEvaluator()
        
        with patch.object(evaluator, '_load_image_pairs') as mock_load_pairs, \
             patch.object(evaluator, '_calculate_individual_metrics_for_all') as mock_individual, \
             patch.object(evaluator, '_calculate_dataset_fid') as mock_fid, \
             patch.object(evaluator, '_calculate_statistics') as mock_stats:
            
            mock_load_pairs.return_value = []
            mock_individual.return_value = []
            mock_fid.return_value = 0.0
            mock_stats.return_value = {}
            
            # Should complete without logging errors
            results = evaluator.evaluate_dataset_all_metrics("/created", "/original")
            
            assert results.total_images == 0


class TestSimpleEvaluatorEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_evaluation_with_different_image_sizes(self, mock_evaluation_dependencies):
        """Test evaluation when images have different sizes"""
        evaluator = SimpleAllMetricsEvaluator()
        
        # Create tensors with different sizes
        different_size_pairs = [
            (torch.randn(1, 3, 256, 256), torch.randn(1, 3, 512, 512))  # Different sizes
        ]
        
        with patch.object(evaluator, '_load_image_pairs') as mock_load_pairs:
            mock_load_pairs.return_value = different_size_pairs
            
            # Mock individual calculator to handle size mismatch
            evaluator.individual_calculator.calculate_all_individual_metrics.side_effect = ValueError("Size mismatch")
            
            with patch.object(evaluator, '_create_default_metrics') as mock_default:
                mock_default.return_value = IndividualImageMetrics(
                    psnr_db=0.0, ssim=0.0, mse=1.0, mae=1.0, lpips=None, ssim_improved=None
                )
                
                with patch.object(evaluator, '_calculate_dataset_fid') as mock_fid, \
                     patch.object(evaluator, '_calculate_statistics') as mock_stats:
                    
                    mock_fid.return_value = float('inf')
                    mock_stats.return_value = {}
                    
                    results = evaluator.evaluate_dataset_all_metrics("/created", "/original")
        
        # Should handle size mismatch gracefully with default metrics
        assert len(results.individual_metrics) == 1
        assert_float_approximately_equal(results.individual_metrics[0].psnr_db, 0.0)  # Default value
    
    def test_empty_directories_evaluation(self, mock_evaluation_dependencies):
        """Test evaluation with empty directories"""
        evaluator = SimpleAllMetricsEvaluator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_created = Path(temp_dir) / "empty_created"
            empty_original = Path(temp_dir) / "empty_original"
            empty_created.mkdir()
            empty_original.mkdir()
            
            # Should raise ValueError due to no matching pairs
            with pytest.raises(ValueError, match="No matching image pairs found"):
                evaluator.evaluate_dataset_all_metrics(empty_created, empty_original)
    
    def test_path_normalization(self, mock_evaluation_dependencies):
        """Test that paths are properly normalized"""
        evaluator = SimpleAllMetricsEvaluator()
        
        with patch.object(evaluator, '_load_image_pairs') as mock_load_pairs, \
             patch.object(evaluator, '_calculate_individual_metrics_for_all') as mock_individual, \
             patch.object(evaluator, '_calculate_dataset_fid') as mock_fid, \
             patch.object(evaluator, '_calculate_statistics') as mock_stats:
            
            mock_load_pairs.return_value = []
            mock_individual.return_value = []
            mock_fid.return_value = 0.0
            mock_stats.return_value = {}
            
            # Test with relative path strings
            results = evaluator.evaluate_dataset_all_metrics("./created", "../original")
        
        # Should convert to absolute paths
        assert Path(results.created_dataset_path).is_absolute()
        assert Path(results.original_dataset_path).is_absolute()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])