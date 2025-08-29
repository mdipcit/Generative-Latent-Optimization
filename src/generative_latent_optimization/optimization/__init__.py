"""VAE Latent Optimization Module"""

from .latent_optimizer import (
    LatentOptimizer,
    OptimizationConfig, 
    OptimizationResult
)

__all__ = [
    'LatentOptimizer',
    'OptimizationConfig',
    'OptimizationResult'
]