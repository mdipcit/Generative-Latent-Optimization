"""Core module with base classes and interfaces for the Generative Latent Optimization package."""

from .base_classes import BaseMetric, BaseEvaluator, BaseDataset

__all__ = [
    'BaseMetric',
    'BaseEvaluator', 
    'BaseDataset',
]