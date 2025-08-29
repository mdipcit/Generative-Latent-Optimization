"""
Generative Latent Optimization Package

A toolkit for optimizing VAE latent representations to improve
image reconstruction quality for Stable Diffusion-based image completion.
"""

# Import core modules
from . import optimization
from . import metrics
from . import utils
from . import evaluation

# Import commonly used classes and functions
from .optimization import LatentOptimizer, OptimizationConfig, OptimizationResult
from .metrics import (
    ImageMetrics, 
    calculate_psnr,
    # Phase 3: Enhanced metrics
    IndividualMetricsCalculator,
    LPIPSMetric,
    ImprovedSSIM,
    DatasetFIDEvaluator,
    IndividualImageMetrics,
    DatasetEvaluationResults
)
from .evaluation import ComprehensiveDatasetEvaluator, SimpleAllMetricsEvaluator
from .utils import IOUtils, save_image_tensor

__version__ = "0.2.0"

__all__ = [
    # Modules
    'optimization',
    'metrics', 
    'utils',
    'evaluation',
    # Core Classes
    'LatentOptimizer',
    'OptimizationConfig', 
    'OptimizationResult',
    'IOUtils',
    # Legacy Metrics
    'ImageMetrics',
    'calculate_psnr',
    # Phase 3: Enhanced Metrics
    'IndividualMetricsCalculator',
    'LPIPSMetric',
    'ImprovedSSIM', 
    'DatasetFIDEvaluator',
    'ComprehensiveDatasetEvaluator',
    'SimpleAllMetricsEvaluator',
    # Data Structures
    'IndividualImageMetrics',
    'DatasetEvaluationResults',
    # Functions
    'save_image_tensor'
]


def hello() -> str:
    return "Hello from generative-latent-optimization!"
