#!/usr/bin/env python3
"""
Assertion Helper Functions

Provides standardized assertion functions for testing, especially for 
handling floating point comparisons and statistical data validation.
"""

import math
import torch
import numpy as np
from typing import Dict, Any, Union, Optional, List
from pathlib import Path


def assert_float_approximately_equal(actual: float, expected: float, 
                                   tolerance: float = 1e-10) -> None:
    """
    浮動小数点の近似比較
    
    Args:
        actual: 実際の値
        expected: 期待値  
        tolerance: 許容誤差
    """
    if math.isnan(expected):
        assert math.isnan(actual), f"Expected NaN, got {actual}"
    elif math.isinf(expected):
        assert math.isinf(actual) and math.copysign(1, actual) == math.copysign(1, expected), \
            f"Expected {'+' if expected > 0 else '-'}Inf, got {actual}"
    else:
        diff = abs(actual - expected)
        assert diff < tolerance, \
            f"Expected {expected}±{tolerance}, got {actual} (diff: {diff})"


def assert_statistics_equal(actual_stats: Dict[str, Any], 
                          expected_stats: Dict[str, Any],
                          float_tolerance: float = 1e-6) -> None:
    """
    統計データの安全な比較
    
    Args:
        actual_stats: 実際の統計データ
        expected_stats: 期待される統計データ
        float_tolerance: 浮動小数点の許容誤差
    """
    for key, expected_value in expected_stats.items():
        assert key in actual_stats, f"Missing key: {key}"
        
        if isinstance(expected_value, float):
            assert_float_approximately_equal(
                actual_stats[key], expected_value, float_tolerance
            )
        elif isinstance(expected_value, dict):
            assert_statistics_equal(
                actual_stats[key], expected_value, float_tolerance
            )
        elif isinstance(expected_value, list) and len(expected_value) > 0 and isinstance(expected_value[0], float):
            assert len(actual_stats[key]) == len(expected_value), \
                f"List length mismatch for {key}: expected {len(expected_value)}, got {len(actual_stats[key])}"
            for i, (actual_item, expected_item) in enumerate(zip(actual_stats[key], expected_value)):
                assert_float_approximately_equal(
                    actual_item, expected_item, float_tolerance
                )
        else:
            assert actual_stats[key] == expected_value, \
                f"Mismatch for {key}: expected {expected_value}, got {actual_stats[key]}"


def assert_metrics_approximately_equal(actual_metrics: Dict[str, float],
                                     expected_metrics: Dict[str, float],
                                     tolerance: float = 1e-6) -> None:
    """
    メトリクス辞書の近似比較
    
    Args:
        actual_metrics: 実際のメトリクス
        expected_metrics: 期待されるメトリクス
        tolerance: 許容誤差
    """
    assert set(actual_metrics.keys()) == set(expected_metrics.keys()), \
        f"Key mismatch: actual {set(actual_metrics.keys())} vs expected {set(expected_metrics.keys())}"
    
    for key in expected_metrics:
        assert_float_approximately_equal(
            actual_metrics[key], expected_metrics[key], tolerance
        )


def assert_tensor_approximately_equal(actual: torch.Tensor, expected: torch.Tensor,
                                    tolerance: float = 1e-6) -> None:
    """
    テンソルの近似比較
    
    Args:
        actual: 実際のテンソル
        expected: 期待されるテンソル
        tolerance: 許容誤差
    """
    assert actual.shape == expected.shape, \
        f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}"
    
    assert actual.device == expected.device, \
        f"Device mismatch: actual {actual.device} vs expected {expected.device}"
    
    diff = torch.abs(actual - expected).max().item()
    assert diff < tolerance, \
        f"Tensor values differ by {diff}, tolerance {tolerance}"


def assert_path_approximately_equal(actual: Union[str, Path], expected: Union[str, Path]) -> None:
    """
    パスの近似比較（絶対パス変換後比較）
    
    Args:
        actual: 実際のパス
        expected: 期待されるパス
    """
    from pathlib import Path
    
    actual_path = Path(actual).resolve()
    expected_path = Path(expected).resolve()
    
    assert actual_path == expected_path, \
        f"Path mismatch: actual {actual_path} vs expected {expected_path}"


# 統計計算用のヘルパー関数
def create_test_metrics_list(count: int = 5) -> List[Dict[str, float]]:
    """テスト用メトリクスリスト作成"""
    return [
        {
            'psnr_db': 25.0 + i,
            'ssim': 0.8 + i * 0.02,
            'mse': 0.1 - i * 0.01,
            'mae': 0.05 - i * 0.005,
            'lpips': 0.1 - i * 0.01 if i < 3 else None,  # 一部None値
            'ssim_improved': 0.85 + i * 0.02 if i < 4 else None
        }
        for i in range(count)
    ]


def create_expected_statistics_from_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """メトリクスリストから期待される統計値を計算"""
    result = {}
    
    # 各メトリクスについて統計計算
    for metric_name in ['psnr_db', 'ssim', 'mse', 'mae', 'lpips', 'ssim_improved']:
        values = [m[metric_name] for m in metrics_list if m.get(metric_name) is not None]
        
        if values:
            result[f'{metric_name}_mean'] = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((x - result[f'{metric_name}_mean']) ** 2 for x in values) / len(values)
                result[f'{metric_name}_std'] = math.sqrt(variance)
            result[f'{metric_name}_min'] = min(values)
            result[f'{metric_name}_max'] = max(values)
    
    result['total_samples'] = len(metrics_list)
    return result


# NumPy and Dataset Processing Helpers

def assert_numpy_array_equal(actual: np.ndarray, expected: np.ndarray, 
                           tolerance: float = 1e-6) -> None:
    """
    NumPy配列の近似比較
    
    Args:
        actual: 実際の配列
        expected: 期待される配列
        tolerance: 許容誤差
    """
    assert actual.shape == expected.shape, \
        f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}"
    
    # NaN/Inf考慮した比較
    if np.any(np.isnan(expected)):
        nan_mask = np.isnan(expected)
        assert np.array_equal(np.isnan(actual), nan_mask), "NaN pattern mismatch"
        # NaN以外の値を比較
        if not np.all(nan_mask):
            non_nan_mask = ~nan_mask
            actual_vals = actual[non_nan_mask]
            expected_vals = expected[non_nan_mask] 
            diff = np.abs(actual_vals - expected_vals).max()
            assert diff < tolerance, f"Non-NaN values differ by {diff}, tolerance {tolerance}"
    else:
        diff = np.abs(actual - expected).max()
        assert diff < tolerance, f"Arrays differ by {diff}, tolerance {tolerance}"


def assert_dataset_sample_structure(sample: Dict[str, Any], 
                                  required_keys: List[str] = None) -> None:
    """
    データセットサンプルの構造検証
    
    Args:
        sample: 検証するサンプル
        required_keys: 必須キーリスト
    """
    if required_keys is None:
        required_keys = ['image_name', 'split', 'metrics', 'files']
    
    for key in required_keys:
        assert key in sample, f"Missing required key: {key}"
    
    # metrics構造検証
    if 'metrics' in sample and sample['metrics']:
        metrics = sample['metrics']
        expected_metrics = ['psnr_improvement', 'ssim_improvement', 'loss_reduction']
        for metric in expected_metrics:
            if metric in metrics:
                assert isinstance(metrics[metric], (int, float)), \
                    f"Metric {metric} must be numeric, got {type(metrics[metric])}"


def assert_processing_results_structure(results: Dict[str, Any]) -> None:
    """
    処理結果の構造検証
    
    Args:
        results: 検証する処理結果
    """
    required_keys = [
        'successful_count', 'failed_count', 'average_psnr_improvement',
        'average_ssim_improvement', 'average_loss_reduction', 'processing_time_seconds'
    ]
    
    for key in required_keys:
        assert key in results, f"Missing required key: {key}"
        assert isinstance(results[key], (int, float)), \
            f"Key {key} must be numeric, got {type(results[key])}"


def safe_calculate_statistics(values: List[float], metric_name: str = "metric") -> Dict[str, float]:
    """
    安全な統計計算（NaN/空配列処理）
    
    Args:
        values: 値のリスト
        metric_name: メトリクス名
        
    Returns:
        統計情報辞書
    """
    if not values:
        return {
            f'{metric_name}_mean': 0.0,
            f'{metric_name}_std': 0.0,
            f'{metric_name}_min': 0.0,
            f'{metric_name}_max': 0.0,
            f'{metric_name}_count': 0
        }
    
    # NaN値を除外
    clean_values = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    
    if not clean_values:
        return {
            f'{metric_name}_mean': float('nan'),
            f'{metric_name}_std': float('nan'),
            f'{metric_name}_min': float('nan'),
            f'{metric_name}_max': float('nan'),
            f'{metric_name}_count': 0
        }
    
    mean_val = sum(clean_values) / len(clean_values)
    
    if len(clean_values) > 1:
        variance = sum((x - mean_val) ** 2 for x in clean_values) / len(clean_values)
        std_val = math.sqrt(variance)
    else:
        std_val = 0.0
    
    return {
        f'{metric_name}_mean': mean_val,
        f'{metric_name}_std': std_val,
        f'{metric_name}_min': min(clean_values),
        f'{metric_name}_max': max(clean_values),
        f'{metric_name}_count': len(clean_values)
    }


def safe_calculate_median(values: List[float]) -> float:
    """
    安全な中央値計算
    
    Args:
        values: 値のリスト
        
    Returns:
        中央値
    """
    if not values:
        return 0.0
    
    # NaN値を除外してソート
    clean_values = sorted([v for v in values if not math.isnan(v) and not math.isinf(v)])
    
    if not clean_values:
        return float('nan')
    
    n = len(clean_values)
    if n % 2 == 0:
        # 偶数個の場合は中央2要素の平均
        return (clean_values[n//2 - 1] + clean_values[n//2]) / 2.0
    else:
        # 奇数個の場合は中央要素
        return clean_values[n//2]


def assert_path_operations_mock(mock_path: Any, expected_calls: Dict[str, Any]) -> None:
    """
    Path操作Mockの検証
    
    Args:
        mock_path: MockされたPathオブジェクト
        expected_calls: 期待される呼び出し
    """
    for method_name, expected_result in expected_calls.items():
        if hasattr(mock_path, method_name):
            method = getattr(mock_path, method_name)
            if callable(method):
                method.assert_called()
            else:
                assert method == expected_result, \
                    f"Path.{method_name} expected {expected_result}, got {method}"