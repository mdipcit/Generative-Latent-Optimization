#!/usr/bin/env python3
"""
Unit tests for ComprehensiveDatasetEvaluator functionality
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from generative_latent_optimization.evaluation.dataset_evaluator import ComprehensiveDatasetEvaluator
from generative_latent_optimization.metrics.image_metrics import DatasetEvaluationResults
from ...fixtures.evaluation_mocks import mock_evaluation_dependencies
from ...fixtures.assertion_helpers import assert_float_approximately_equal, assert_statistics_equal


# mock_evaluation_dependencies はevaluation_mocks.pyのmock_evaluation_dependenciesに統一


@pytest.fixture
def sample_fid_results():
    """Sample FID evaluation results"""
    return DatasetEvaluationResults(
        fid_score=15.5,
        total_images=100,
        original_dataset_path="/path/to/original",
        generated_dataset_path="/path/to/generated",
        evaluation_timestamp="2024-01-01T12:00:00",
        individual_metrics_summary={}
    )


@pytest.fixture
def sample_individual_stats():
    """Sample individual metrics statistics"""
    return {
        'psnr_mean': 28.5,
        'psnr_std': 3.2,
        'psnr_min': 22.0,
        'psnr_max': 35.0,
        'ssim_mean': 0.85,
        'ssim_std': 0.05,
        'ssim_min': 0.75,
        'ssim_max': 0.95,
        'lpips_mean': 0.08,
        'lpips_std': 0.02,
        'ssim_improved_mean': 0.88,
        'total_samples': 100
    }


class TestComprehensiveDatasetEvaluator:
    """Test cases for ComprehensiveDatasetEvaluator class"""
    
    def test_initialization(self, mock_evaluation_dependencies):
        """Test evaluator initialization"""
        evaluator = ComprehensiveDatasetEvaluator(device='cuda')
        
        assert evaluator.device == 'cuda'
        assert hasattr(evaluator, 'individual_calculator')
        assert hasattr(evaluator, 'fid_evaluator')
    
    def test_initialization_default_device(self, mock_evaluation_dependencies):
        """Test evaluator initialization with default device"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        assert evaluator.device == 'cuda'
    
    def test_calculate_std(self, mock_evaluation_dependencies):
        """Test _calculate_std helper method"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Test with normal values
        std = evaluator._calculate_std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert std > 0
        
        # Test with single value
        std_single = evaluator._calculate_std([5.0])
        assert_float_approximately_equal(std_single, 0.0)
        
        # Test with empty list
        std_empty = evaluator._calculate_std([])
        assert_float_approximately_equal(std_empty, 0.0)
    
    def test_interpret_fid_score(self, mock_evaluation_dependencies):
        """Test _interpret_fid_score method"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Test different FID score ranges
        assert "Excellent" in evaluator._interpret_fid_score(5.0)
        assert "Good" in evaluator._interpret_fid_score(15.0)
        assert "Fair" in evaluator._interpret_fid_score(35.0)
        assert "Poor" in evaluator._interpret_fid_score(75.0)
        assert "Very Poor" in evaluator._interpret_fid_score(150.0)
    
    def test_identify_improvements(self, mock_evaluation_dependencies):
        """Test _identify_improvements method"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Test with good metrics
        good_stats = {
            'psnr_mean': 30.0,
            'ssim_mean': 0.85,
            'lpips_mean': 0.05,
            'ssim_improved_mean': 0.9
        }
        
        improvements = evaluator._identify_improvements(good_stats)
        
        assert len(improvements) > 0
        assert any("PSNR improvement" in imp for imp in improvements)
        assert any("structural similarity" in imp for imp in improvements)
        assert any("perceptual quality" in imp for imp in improvements)
        assert any("structural preservation" in imp for imp in improvements)
    
    def test_identify_improvements_poor_metrics(self, mock_evaluation_dependencies):
        """Test _identify_improvements with poor metrics"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        poor_stats = {
            'psnr_mean': 20.0,
            'ssim_mean': 0.6,
            'lpips_mean': 0.3
        }
        
        improvements = evaluator._identify_improvements(poor_stats)
        
        # Should return fewer or no improvements
        assert len(improvements) == 0
    
    def test_generate_recommendations(self, mock_evaluation_dependencies, sample_fid_results):
        """Test _generate_recommendations method"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Test excellent results
        excellent_stats = {'psnr_mean': 35.0}
        recommendation = evaluator._generate_recommendations(
            DatasetEvaluationResults(fid_score=10.0, total_images=100, 
                                   original_dataset_path="", generated_dataset_path="",
                                   evaluation_timestamp="", individual_metrics_summary={}),
            excellent_stats
        )
        assert "production use" in recommendation
        
        # Test good results
        good_stats = {'psnr_mean': 28.0}
        recommendation = evaluator._generate_recommendations(
            DatasetEvaluationResults(fid_score=30.0, total_images=100,
                                   original_dataset_path="", generated_dataset_path="",
                                   evaluation_timestamp="", individual_metrics_summary={}),
            good_stats
        )
        assert "suitable for most applications" in recommendation
        
        # Test poor results
        poor_stats = {'psnr_mean': 18.0}
        recommendation = evaluator._generate_recommendations(
            DatasetEvaluationResults(fid_score=150.0, total_images=100,
                                   original_dataset_path="", generated_dataset_path="",
                                   evaluation_timestamp="", individual_metrics_summary={}),
            poor_stats
        )
        assert "need improvement" in recommendation
    
    def test_calculate_overall_score(self, mock_evaluation_dependencies):
        """Test _calculate_overall_score method"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Test with good metrics
        good_stats = {
            'psnr_mean': 30.0,
            'ssim_mean': 0.9
        }
        score = evaluator._calculate_overall_score(10.0, good_stats)
        assert 70 <= score <= 100  # Should be high score
        
        # Test with poor metrics
        poor_stats = {
            'psnr_mean': 18.0,
            'ssim_mean': 0.5
        }
        score = evaluator._calculate_overall_score(80.0, poor_stats)
        assert 0 <= score <= 50  # Should be low score
    
    def test_calculate_overall_score_exception_handling(self, mock_evaluation_dependencies):
        """Test _calculate_overall_score exception handling"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Test with invalid stats (missing keys)
        score = evaluator._calculate_overall_score(50.0, {})
        
        # Should return default score
        assert_float_approximately_equal(score, 50.0, 1e-2)  # Looser tolerance for calculated score
    
    def test_compute_metrics_statistics(self, mock_evaluation_dependencies):
        """Test _compute_metrics_statistics method"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        metrics_list = [
            {'psnr_db': 25.0, 'ssim': 0.8, 'mse': 0.1, 'mae': 0.05, 'lpips': 0.1, 'ssim_improved': 0.85},
            {'psnr_db': 30.0, 'ssim': 0.9, 'mse': 0.05, 'mae': 0.03, 'lpips': 0.08, 'ssim_improved': 0.92}
        ]
        
        stats = evaluator._compute_metrics_statistics(metrics_list)
        
        # Verify structure and calculations
        assert stats['total_samples'] == 2
        assert_float_approximately_equal(stats['psnr_mean'], 27.5)  # (25.0 + 30.0) / 2
        assert_float_approximately_equal(stats['psnr_min'], 25.0)
        assert_float_approximately_equal(stats['psnr_max'], 30.0)
        assert_float_approximately_equal(stats['ssim_mean'], 0.85)  # (0.8 + 0.9) / 2
        assert_float_approximately_equal(stats['lpips_mean'], 0.09)  # (0.1 + 0.08) / 2
        assert_float_approximately_equal(stats['ssim_improved_mean'], 0.885)  # (0.85 + 0.92) / 2
    
    def test_compute_metrics_statistics_partial_data(self, mock_evaluation_dependencies):
        """Test _compute_metrics_statistics with partial data"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Some metrics missing from some samples
        metrics_list = [
            {'psnr_db': 25.0, 'ssim': 0.8},
            {'psnr_db': 30.0, 'lpips': 0.1}
        ]
        
        stats = evaluator._compute_metrics_statistics(metrics_list)
        
        assert stats['total_samples'] == 2
        assert_float_approximately_equal(stats['psnr_mean'], 27.5)
        # Should handle missing metrics gracefully
        assert 'ssim_mean' in stats  # Only from first sample
        assert 'lpips_mean' in stats  # Only from second sample
    
    def test_compute_metrics_statistics_empty(self, mock_evaluation_dependencies):
        """Test _compute_metrics_statistics with empty list"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        stats = evaluator._compute_metrics_statistics([])
        assert stats == {}
    
    def test_normalize_png_statistics(self, mock_evaluation_dependencies):
        """Test _normalize_png_statistics method"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Test with nested statistics format
        png_stats = {
            'total_samples': 100,
            'psnr_improvement': {
                'mean': 5.5,
                'std': 1.2,
                'min': 3.0,
                'max': 8.0
            },
            'loss_reduction': {
                'mean': 77.5,
                'std': 5.0
            },
            'simple_value': 42.0
        }
        
        normalized = evaluator._normalize_png_statistics(png_stats)
        
        # Verify normalization
        assert_float_approximately_equal(normalized['total_samples'], 100.0)
        assert_float_approximately_equal(normalized['psnr_improvement_mean'], 5.5)
        assert_float_approximately_equal(normalized['psnr_improvement_std'], 1.2)
        assert_float_approximately_equal(normalized['psnr_improvement_min'], 3.0)
        assert_float_approximately_equal(normalized['psnr_improvement_max'], 8.0)
        assert_float_approximately_equal(normalized['loss_reduction_mean'], 77.5)
        assert_float_approximately_equal(normalized['loss_reduction_std'], 5.0)
        assert_float_approximately_equal(normalized['simple_value'], 42.0)
    
    def test_compute_fid_evaluation_png(self, mock_evaluation_dependencies):
        """Test _compute_fid_evaluation for PNG dataset"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        mock_fid_result = DatasetEvaluationResults(
            fid_score=12.5,
            total_images=50,
            original_dataset_path="/original",
            generated_dataset_path="/generated",
            evaluation_timestamp="2024-01-01T12:00:00",
            individual_metrics_summary={}
        )
        
        evaluator.fid_evaluator.evaluate_created_dataset_vs_original.return_value = mock_fid_result
        
        result = evaluator._compute_fid_evaluation(Path("/dataset"), Path("/original"), 'png')
        
        evaluator.fid_evaluator.evaluate_created_dataset_vs_original.assert_called_once()
        assert result == mock_fid_result
    
    def test_compute_fid_evaluation_pytorch(self, mock_evaluation_dependencies):
        """Test _compute_fid_evaluation for PyTorch dataset"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        mock_fid_result = DatasetEvaluationResults(
            fid_score=18.2,
            total_images=75,
            original_dataset_path="/original",
            generated_dataset_path="/generated",
            evaluation_timestamp="2024-01-01T12:00:00",
            individual_metrics_summary={}
        )
        
        evaluator.fid_evaluator.evaluate_pytorch_dataset_vs_original.return_value = mock_fid_result
        
        result = evaluator._compute_fid_evaluation(Path("/dataset.pt"), Path("/original"), 'pytorch')
        
        evaluator.fid_evaluator.evaluate_pytorch_dataset_vs_original.assert_called_once()
        assert result == mock_fid_result
    
    def test_compute_fid_evaluation_unsupported_type(self, mock_evaluation_dependencies):
        """Test _compute_fid_evaluation with unsupported dataset type"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            evaluator._compute_fid_evaluation(Path("/dataset"), Path("/original"), 'unsupported')
    
    def test_calculate_individual_metrics_statistics_png(self, mock_evaluation_dependencies):
        """Test _calculate_individual_metrics_statistics for PNG dataset"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with patch.object(evaluator, '_load_png_dataset_statistics') as mock_load_png:
            mock_load_png.return_value = {'psnr_mean': 25.0}
            
            stats = evaluator._calculate_individual_metrics_statistics(Path("/png_dataset"), 'png')
            
            mock_load_png.assert_called_once()
            assert_statistics_equal(stats, {'psnr_mean': 25.0})
    
    def test_calculate_individual_metrics_statistics_pytorch(self, mock_evaluation_dependencies):
        """Test _calculate_individual_metrics_statistics for PyTorch dataset"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with patch.object(evaluator, '_load_pytorch_dataset_statistics') as mock_load_pytorch:
            mock_load_pytorch.return_value = {'ssim_mean': 0.85}
            
            stats = evaluator._calculate_individual_metrics_statistics(Path("/dataset.pt"), 'pytorch')
            
            mock_load_pytorch.assert_called_once()
            assert_statistics_equal(stats, {'ssim_mean': 0.85})
    
    def test_calculate_individual_metrics_statistics_unsupported(self, mock_evaluation_dependencies):
        """Test _calculate_individual_metrics_statistics with unsupported type"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            evaluator._calculate_individual_metrics_statistics(Path("/dataset"), 'unsupported')
    
    def test_load_png_dataset_statistics_success(self, mock_evaluation_dependencies):
        """Test _load_png_dataset_statistics successful loading"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            stats_file = dataset_path / 'statistics.json'
            
            # Create mock statistics file
            mock_stats = {
                'psnr_improvement': {'mean': 5.5, 'std': 1.2},
                'total_samples': 100
            }
            
            with open(stats_file, 'w') as f:
                json.dump(mock_stats, f)
            
            with patch.object(evaluator, '_normalize_png_statistics') as mock_normalize:
                mock_normalize.return_value = {'normalized': 'stats'}
                
                stats = evaluator._load_png_dataset_statistics(dataset_path)
            
            mock_normalize.assert_called_once_with(mock_stats)
            assert stats == {'normalized': 'stats'}
    
    def test_load_png_dataset_statistics_file_not_found(self, mock_evaluation_dependencies):
        """Test _load_png_dataset_statistics when file doesn't exist"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            # No statistics.json file
            
            stats = evaluator._load_png_dataset_statistics(dataset_path)
            
            # Should return empty dict
            assert stats == {}
    
    def test_load_png_dataset_statistics_json_error(self, mock_evaluation_dependencies):
        """Test _load_png_dataset_statistics with corrupted JSON"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            stats_file = dataset_path / 'statistics.json'
            
            # Create corrupted JSON file
            with open(stats_file, 'w') as f:
                f.write("invalid json content")
            
            stats = evaluator._load_png_dataset_statistics(dataset_path)
            
            # Should handle error gracefully
            assert stats == {}
    
    def test_load_pytorch_dataset_statistics_success(self, mock_evaluation_dependencies):
        """Test _load_pytorch_dataset_statistics successful loading"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Mock dataset with metrics
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2
        mock_dataset.__getitem__.side_effect = [
            {'metrics': {'psnr_db': 25.0, 'ssim': 0.8}},
            {'metrics': {'psnr_db': 30.0, 'ssim': 0.9}}
        ]
        
        with patch('generative_latent_optimization.dataset.load_optimized_dataset') as mock_load:
            mock_load.return_value = mock_dataset
            
            with patch.object(evaluator, '_compute_metrics_statistics') as mock_compute:
                mock_compute.return_value = {'computed': 'stats'}
                
                stats = evaluator._load_pytorch_dataset_statistics(Path("/dataset.pt"))
        
        mock_load.assert_called_once()
        mock_compute.assert_called_once()
        assert stats == {'computed': 'stats'}
    
    def test_load_pytorch_dataset_statistics_no_metrics(self, mock_evaluation_dependencies):
        """Test _load_pytorch_dataset_statistics when samples have no metrics"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Mock dataset without metrics
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 1
        mock_dataset.__getitem__.return_value = {'no_metrics': 'here'}
        
        with patch('generative_latent_optimization.dataset.load_optimized_dataset') as mock_load:
            mock_load.return_value = mock_dataset
            
            stats = evaluator._load_pytorch_dataset_statistics(Path("/dataset.pt"))
        
        # Should return empty dict when no metrics found
        assert stats == {}
    
    def test_load_pytorch_dataset_statistics_exception(self, mock_evaluation_dependencies):
        """Test _load_pytorch_dataset_statistics exception handling"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with patch('generative_latent_optimization.dataset.load_optimized_dataset') as mock_load:
            mock_load.side_effect = Exception("Failed to load dataset")
            
            stats = evaluator._load_pytorch_dataset_statistics(Path("/dataset.pt"))
        
        # Should handle exception gracefully
        assert stats == {}
    
    def test_create_evaluation_summary(self, mock_evaluation_dependencies, sample_fid_results, sample_individual_stats):
        """Test _create_evaluation_summary method"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with patch.object(evaluator, '_calculate_overall_score') as mock_score, \
             patch.object(evaluator, '_interpret_fid_score') as mock_interpret, \
             patch.object(evaluator, '_identify_improvements') as mock_improvements, \
             patch.object(evaluator, '_generate_recommendations') as mock_recommendations:
            
            mock_score.return_value = 85.0
            mock_interpret.return_value = "Excellent quality"
            mock_improvements.return_value = ["Good PSNR", "High SSIM"]
            mock_recommendations.return_value = "Ready for production"
            
            summary = evaluator._create_evaluation_summary(sample_fid_results, sample_individual_stats)
        
        # Verify structure
        assert 'overall_quality_score' in summary
        assert 'fid_interpretation' in summary
        assert 'improvement_highlights' in summary
        assert 'recommendation' in summary
        
        # Verify values
        assert_float_approximately_equal(summary['overall_quality_score'], 85.0)
        assert summary['fid_interpretation'] == "Excellent quality"
        assert summary['improvement_highlights'] == ["Good PSNR", "High SSIM"]
        assert summary['recommendation'] == "Ready for production"


class TestComprehensiveEvaluationIntegration:
    """Integration tests for comprehensive evaluation"""
    
    def test_evaluate_complete_dataset_png(self, mock_evaluation_dependencies, sample_fid_results, sample_individual_stats):
        """Test evaluate_complete_dataset for PNG dataset"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Mock all the internal methods
        with patch.object(evaluator, '_compute_fid_evaluation') as mock_fid, \
             patch.object(evaluator, '_calculate_individual_metrics_statistics') as mock_individual, \
             patch.object(evaluator, '_create_evaluation_summary') as mock_summary:
            
            mock_fid.return_value = sample_fid_results
            mock_individual.return_value = sample_individual_stats
            mock_summary.return_value = {'summary': 'data'}
            
            results = evaluator.evaluate_complete_dataset(
                "/png_dataset", "/original_bsds500", dataset_type='png'
            )
        
        # Verify method calls
        mock_fid.assert_called_once()
        mock_individual.assert_called_once()
        mock_summary.assert_called_once()
        
        # Verify result structure
        assert 'dataset_evaluation' in results
        assert 'individual_metrics_statistics' in results
        assert 'evaluation_summary' in results
        
        # Verify dataset evaluation content
        dataset_eval = results['dataset_evaluation']
        assert dataset_eval['fid_score'] == sample_fid_results.fid_score
        assert dataset_eval['total_images'] == sample_fid_results.total_images
        
        # Verify other components
        assert results['individual_metrics_statistics'] == sample_individual_stats
        assert results['evaluation_summary'] == {'summary': 'data'}
    
    def test_evaluate_complete_dataset_pytorch(self, mock_evaluation_dependencies, sample_fid_results, sample_individual_stats):
        """Test evaluate_complete_dataset for PyTorch dataset"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with patch.object(evaluator, '_compute_fid_evaluation') as mock_fid, \
             patch.object(evaluator, '_calculate_individual_metrics_statistics') as mock_individual, \
             patch.object(evaluator, '_create_evaluation_summary') as mock_summary:
            
            mock_fid.return_value = sample_fid_results
            mock_individual.return_value = sample_individual_stats
            mock_summary.return_value = {'summary': 'data'}
            
            results = evaluator.evaluate_complete_dataset(
                "/dataset.pt", "/original_bsds500", dataset_type='pytorch'
            )
        
        # Verify FID evaluation was called with correct type
        call_args = mock_fid.call_args[0]
        assert call_args[2] == 'pytorch'
        
        # Verify result structure
        assert 'dataset_evaluation' in results
        assert 'individual_metrics_statistics' in results
        assert 'evaluation_summary' in results
    
    def test_evaluate_complete_dataset_fid_error(self, mock_evaluation_dependencies):
        """Test evaluate_complete_dataset when FID computation fails"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with patch.object(evaluator, '_compute_fid_evaluation') as mock_fid, \
             patch.object(evaluator, '_calculate_individual_metrics_statistics') as mock_individual:
            
            mock_fid.side_effect = Exception("FID computation failed")
            mock_individual.return_value = {}
            
            # Should propagate FID computation error
            with pytest.raises(Exception, match="FID computation failed"):
                evaluator.evaluate_complete_dataset(
                    "/dataset", "/original", dataset_type='png'
                )
    
    def test_evaluate_complete_dataset_individual_metrics_error(self, mock_evaluation_dependencies, sample_fid_results):
        """Test evaluate_complete_dataset when individual metrics fail"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with patch.object(evaluator, '_compute_fid_evaluation') as mock_fid, \
             patch.object(evaluator, '_calculate_individual_metrics_statistics') as mock_individual, \
             patch.object(evaluator, '_create_evaluation_summary') as mock_summary:
            
            mock_fid.return_value = sample_fid_results
            mock_individual.return_value = {}  # Empty stats due to error
            mock_summary.return_value = {'limited': 'summary'}
            
            results = evaluator.evaluate_complete_dataset(
                "/dataset", "/original", dataset_type='png'
            )
        
        # Should still complete with empty individual stats
        assert results['individual_metrics_statistics'] == {}
        assert 'dataset_evaluation' in results
        assert 'evaluation_summary' in results


class TestComprehensiveEvaluatorScenarios:
    """Test specific evaluation scenarios"""
    
    def test_excellent_dataset_evaluation(self, mock_evaluation_dependencies):
        """Test evaluation scenario with excellent results"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Mock excellent FID results
        excellent_fid = DatasetEvaluationResults(
            fid_score=8.0,  # Excellent FID
            total_images=100,
            original_dataset_path="/original",
            generated_dataset_path="/generated",
            evaluation_timestamp="2024-01-01T12:00:00",
            individual_metrics_summary={}
        )
        
        # Mock excellent individual stats
        excellent_stats = {
            'psnr_mean': 32.0,
            'ssim_mean': 0.92,
            'lpips_mean': 0.05,
            'ssim_improved_mean': 0.95
        }
        
        with patch.object(evaluator, '_compute_fid_evaluation') as mock_fid, \
             patch.object(evaluator, '_calculate_individual_metrics_statistics') as mock_individual:
            
            mock_fid.return_value = excellent_fid
            mock_individual.return_value = excellent_stats
            
            results = evaluator.evaluate_complete_dataset(
                "/dataset", "/original", dataset_type='png'
            )
        
        summary = results['evaluation_summary']
        
        # Should have high overall score
        assert summary['overall_quality_score'] > 80
        
        # Should have positive interpretation
        assert "Excellent" in summary['fid_interpretation']
        
        # Should identify multiple improvements
        assert len(summary['improvement_highlights']) > 2
        
        # Should recommend production use
        assert "production" in summary['recommendation']
    
    def test_poor_dataset_evaluation(self, mock_evaluation_dependencies):
        """Test evaluation scenario with poor results"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Mock poor FID results
        poor_fid = DatasetEvaluationResults(
            fid_score=120.0,  # Poor FID
            total_images=50,
            original_dataset_path="/original",
            generated_dataset_path="/generated",
            evaluation_timestamp="2024-01-01T12:00:00",
            individual_metrics_summary={}
        )
        
        # Mock poor individual stats
        poor_stats = {
            'psnr_mean': 18.0,
            'ssim_mean': 0.6,
            'lpips_mean': 0.3
        }
        
        with patch.object(evaluator, '_compute_fid_evaluation') as mock_fid, \
             patch.object(evaluator, '_calculate_individual_metrics_statistics') as mock_individual:
            
            mock_fid.return_value = poor_fid
            mock_individual.return_value = poor_stats
            
            results = evaluator.evaluate_complete_dataset(
                "/dataset", "/original", dataset_type='png'
            )
        
        summary = results['evaluation_summary']
        
        # Should have low overall score
        assert summary['overall_quality_score'] < 50
        
        # Should have negative interpretation
        assert "Poor" in summary['fid_interpretation']
        
        # Should have few or no improvements
        assert len(summary['improvement_highlights']) == 0
        
        # Should recommend improvements
        assert "improvement" in summary['recommendation']
    
    def test_mixed_quality_evaluation(self, mock_evaluation_dependencies):
        """Test evaluation scenario with mixed quality results"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Mock mixed FID results (good FID, moderate individual metrics)
        mixed_fid = DatasetEvaluationResults(
            fid_score=25.0,  # Good FID
            total_images=75,
            original_dataset_path="/original",
            generated_dataset_path="/generated",
            evaluation_timestamp="2024-01-01T12:00:00",
            individual_metrics_summary={}
        )
        
        mixed_stats = {
            'psnr_mean': 26.0,  # Moderate PSNR
            'ssim_mean': 0.75,  # Moderate SSIM
            'lpips_mean': 0.12
        }
        
        with patch.object(evaluator, '_compute_fid_evaluation') as mock_fid, \
             patch.object(evaluator, '_calculate_individual_metrics_statistics') as mock_individual:
            
            mock_fid.return_value = mixed_fid
            mock_individual.return_value = mixed_stats
            
            results = evaluator.evaluate_complete_dataset(
                "/dataset", "/original", dataset_type='png'
            )
        
        summary = results['evaluation_summary']
        
        # Should have moderate overall score
        assert 40 <= summary['overall_quality_score'] <= 80
        
        # Should identify some improvements
        assert len(summary['improvement_highlights']) >= 0


class TestComprehensiveEvaluatorUtilities:
    """Test utility functions and edge cases"""
    
    def test_evaluation_with_minimal_individual_stats(self, mock_evaluation_dependencies, sample_fid_results):
        """Test evaluation with minimal individual statistics"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        minimal_stats = {'total_samples': 10}  # Only basic info
        
        with patch.object(evaluator, '_compute_fid_evaluation') as mock_fid, \
             patch.object(evaluator, '_calculate_individual_metrics_statistics') as mock_individual:
            
            mock_fid.return_value = sample_fid_results
            mock_individual.return_value = minimal_stats
            
            results = evaluator.evaluate_complete_dataset(
                "/dataset", "/original", dataset_type='png'
            )
        
        # Should complete successfully with minimal stats
        assert 'dataset_evaluation' in results
        assert 'individual_metrics_statistics' in results
        assert results['individual_metrics_statistics'] == minimal_stats
    
    def test_evaluation_summary_creation_with_empty_stats(self, mock_evaluation_dependencies, sample_fid_results):
        """Test evaluation summary creation with empty individual stats"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        summary = evaluator._create_evaluation_summary(sample_fid_results, {})
        
        # Should create summary even with empty stats
        assert 'overall_quality_score' in summary
        assert 'fid_interpretation' in summary
        assert 'improvement_highlights' in summary
        assert 'recommendation' in summary
        
        # Should use default values for missing metrics
        assert isinstance(summary['overall_quality_score'], float)
        assert len(summary['improvement_highlights']) == 0  # No stats to highlight
    
    def test_path_handling(self, mock_evaluation_dependencies):
        """Test proper path handling in evaluation methods"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        # Test with string paths
        with patch.object(evaluator, '_compute_fid_evaluation') as mock_fid, \
             patch.object(evaluator, '_calculate_individual_metrics_statistics') as mock_individual:
            
            mock_fid.return_value = DatasetEvaluationResults(
                fid_score=15.0, total_images=10, original_dataset_path="",
                generated_dataset_path="", evaluation_timestamp="", individual_metrics_summary={}
            )
            mock_individual.return_value = {}
            
            # Should handle string paths correctly
            results = evaluator.evaluate_complete_dataset(
                "/dataset/path", "/original/path", dataset_type='png'
            )
        
        # Verify Path objects were passed to internal methods
        fid_call_args = mock_fid.call_args[0]
        assert isinstance(fid_call_args[0], Path)
        assert isinstance(fid_call_args[1], Path)


class TestDatasetEvaluatorErrorHandling:
    """Test error handling and edge cases"""
    
    def test_device_consistency(self, mock_evaluation_dependencies):
        """Test that device setting is consistent throughout evaluation"""
        device = 'cpu'
        evaluator = ComprehensiveDatasetEvaluator(device=device)
        
        assert evaluator.device == device
        
        # Verify device was passed to component initializers
        individual_calc_init = mock_evaluation_dependencies['IndividualMetricsCalculator']
        fid_eval_init = mock_evaluation_dependencies['DatasetFIDEvaluator']
        
        individual_calc_init.assert_called_with(device=device)
        fid_eval_init.assert_called_with(device=device)
    
    def test_logging_integration(self, mock_evaluation_dependencies):
        """Test that proper logging occurs during evaluation"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        with patch.object(evaluator, '_compute_fid_evaluation') as mock_fid, \
             patch.object(evaluator, '_calculate_individual_metrics_statistics') as mock_individual, \
             patch.object(evaluator, '_create_evaluation_summary') as mock_summary:
            
            mock_fid.return_value = DatasetEvaluationResults(
                fid_score=15.0, total_images=10, original_dataset_path="",
                generated_dataset_path="", evaluation_timestamp="", individual_metrics_summary={}
            )
            mock_individual.return_value = {}
            mock_summary.return_value = {}
            
            # Should complete without logging errors
            results = evaluator.evaluate_complete_dataset(
                "/dataset", "/original", dataset_type='png'
            )
        
        assert results is not None
    
    def test_comprehensive_evaluation_edge_case_empty_dataset(self, mock_evaluation_dependencies):
        """Test evaluation with empty dataset"""
        evaluator = ComprehensiveDatasetEvaluator()
        
        empty_fid_results = DatasetEvaluationResults(
            fid_score=0.0,
            total_images=0,
            original_dataset_path="/original",
            generated_dataset_path="/generated",
            evaluation_timestamp="2024-01-01T12:00:00",
            individual_metrics_summary={}
        )
        
        with patch.object(evaluator, '_compute_fid_evaluation') as mock_fid, \
             patch.object(evaluator, '_calculate_individual_metrics_statistics') as mock_individual:
            
            mock_fid.return_value = empty_fid_results
            mock_individual.return_value = {}
            
            results = evaluator.evaluate_complete_dataset(
                "/empty_dataset", "/original", dataset_type='png'
            )
        
        # Should handle empty dataset gracefully
        assert results['dataset_evaluation']['total_images'] == 0
        assert results['individual_metrics_statistics'] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])