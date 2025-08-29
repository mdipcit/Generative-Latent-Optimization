"""
Generative Latent Optimization Package

A toolkit for optimizing VAE latent representations to improve
image reconstruction quality for Stable Diffusion-based image completion.
"""

# Import core modules
from . import optimization
from . import metrics
from . import utils

# Import commonly used classes and functions
from .optimization import LatentOptimizer, OptimizationConfig, OptimizationResult
from .metrics import ImageMetrics, calculate_psnr
from .utils import IOUtils, save_image_tensor

__version__ = "0.2.0"

__all__ = [
    # Modules
    'optimization',
    'metrics', 
    'utils',
    # Classes
    'LatentOptimizer',
    'OptimizationConfig', 
    'OptimizationResult',
    'ImageMetrics',
    'IOUtils',
    # Functions
    'calculate_psnr',
    'save_image_tensor'
]


def hello() -> str:
    return "Hello from generative-latent-optimization!"
