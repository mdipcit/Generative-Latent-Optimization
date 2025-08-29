"""
Evaluation Module

Provides comprehensive dataset-level evaluation capabilities.
"""

from .dataset_evaluator import ComprehensiveDatasetEvaluator
from .simple_evaluator import SimpleAllMetricsEvaluator

__all__ = [
    'ComprehensiveDatasetEvaluator',
    'SimpleAllMetricsEvaluator'
]