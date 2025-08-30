#!/usr/bin/env python3
"""
Evaluation System Mock Fixtures

Provides standardized mock configurations for evaluation system testing.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock, DEFAULT
from typing import Dict, Any, List
from pathlib import Path


@pytest.fixture
def mock_evaluation_dependencies():
    """統一Mock設定 - 評価システム全体"""
    # Individual metrics calculator mock
    mock_individual_calc = MagicMock()
    mock_individual_calc.calculate_individual_metrics.return_value = {
        'psnr_db': 25.0,
        'ssim': 0.85,
        'mse': 0.1,
        'mae': 0.05,
        'lpips': 0.1,
        'ssim_improved': 0.87
    }
    mock_individual_calc.calculate_batch_statistics.return_value = {
        'psnr_mean': 25.0, 'psnr_std': 2.0,
        'ssim_mean': 0.85, 'ssim_std': 0.05
    }
    
    # FID evaluator mock  
    mock_fid_evaluator = MagicMock()
    mock_fid_evaluator.evaluate_created_dataset_vs_original.return_value.fid_score = 15.0
    mock_fid_evaluator.evaluate_pytorch_dataset_vs_original.return_value.fid_score = 12.0
    
    with patch.multiple(
        'generative_latent_optimization.evaluation.simple_evaluator',
        IndividualMetricsCalculator=mock_individual_calc,
        DatasetFIDEvaluator=mock_fid_evaluator
    ), patch.multiple(
        'generative_latent_optimization.evaluation.dataset_evaluator',
        IndividualMetricsCalculator=mock_individual_calc,
        DatasetFIDEvaluator=mock_fid_evaluator
    ):
        yield {
            'IndividualMetricsCalculator': mock_individual_calc,
            'DatasetFIDEvaluator': mock_fid_evaluator
        }


@pytest.fixture
def mock_simple_evaluator_methods():
    """SimpleAllMetricsEvaluator内部メソッド用Mock"""
    mock_load_pairs = MagicMock()
    mock_load_pairs.return_value = [
        (torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512))
        for _ in range(5)
    ]
    
    mock_individual_metrics = MagicMock()
    mock_individual_metrics.return_value = [
        {'psnr_db': 25.0, 'ssim': 0.85, 'mse': 0.1}
        for _ in range(5)
    ]
    
    mock_fid_calculation = MagicMock()
    mock_fid_calculation.return_value = 15.0
    
    mock_statistics = MagicMock()
    mock_statistics.return_value = {
        'total_images': 5,
        'psnr_mean': 25.0,
        'ssim_mean': 0.85
    }
    
    return {
        '_load_image_pairs': mock_load_pairs,
        '_calculate_individual_metrics_for_all': mock_individual_metrics,
        '_calculate_dataset_fid': mock_fid_calculation,
        '_calculate_statistics': mock_statistics
    }


@pytest.fixture
def mock_image_loading():
    """画像読み込み処理用Mock"""
    mock_tensor = torch.randn(1, 3, 512, 512)
    
    with patch('vae_toolkit.load_and_preprocess_image') as mock_load:
        mock_load.return_value = (mock_tensor, MagicMock())
        yield mock_load


@pytest.fixture
def mock_comprehensive_evaluator_methods():
    """ComprehensiveDatasetEvaluator内部メソッド用Mock"""
    mock_fid_evaluation = MagicMock()
    mock_fid_evaluation.return_value.fid_score = 15.0
    mock_fid_evaluation.return_value.total_images = 100
    mock_fid_evaluation.return_value.original_dataset_path = "/mock/original"
    mock_fid_evaluation.return_value.generated_dataset_path = "/mock/generated"
    mock_fid_evaluation.return_value.evaluation_timestamp = "2025-08-30T10:00:00"
    
    mock_individual_stats = MagicMock()
    mock_individual_stats.return_value = {
        'total_samples': 100,
        'psnr_mean': 25.0,
        'psnr_std': 2.0,
        'ssim_mean': 0.85,
        'ssim_std': 0.05
    }
    
    return {
        '_compute_fid_evaluation': mock_fid_evaluation,
        '_calculate_individual_metrics_statistics': mock_individual_stats
    }


def create_mock_individual_metrics_result() -> Dict[str, float]:
    """標準的な個別メトリクス結果Mock"""
    return {
        'psnr_db': 25.0,
        'ssim': 0.85,
        'mse': 0.1,
        'mae': 0.05,
        'lpips': 0.1,
        'ssim_improved': 0.87
    }


def create_mock_fid_evaluation_result():
    """標準的なFID評価結果Mock"""
    mock_result = MagicMock()
    mock_result.fid_score = 15.0
    mock_result.total_images = 100
    mock_result.original_dataset_path = "/mock/original"
    mock_result.generated_dataset_path = "/mock/generated"
    mock_result.evaluation_timestamp = "2025-08-30T10:00:00"
    return mock_result


def create_mock_dataset_statistics() -> Dict[str, Any]:
    """標準的なデータセット統計Mock"""
    return {
        'total_samples': 100,
        'psnr_mean': 25.0,
        'psnr_std': 2.0,
        'psnr_min': 20.0,
        'psnr_max': 30.0,
        'ssim_mean': 0.85,
        'ssim_std': 0.05,
        'ssim_min': 0.75,
        'ssim_max': 0.95,
        'mse_mean': 0.1,
        'mae_mean': 0.05
    }