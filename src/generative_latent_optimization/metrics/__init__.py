"""Image Quality Metrics Module"""

# Legacy metrics (backward compatibility)
from .image_metrics import (
    ImageMetrics,
    MetricResults,
    MetricsTracker,
    calculate_psnr,
    # New data structures
    IndividualImageMetrics,
    DatasetEvaluationResults
)

# Individual image metrics
from .individual_metrics import (
    LPIPSMetric,
    ImprovedSSIM
)

# Dataset-level metrics  
from .dataset_metrics import (
    DatasetFIDEvaluator
)

# Integrated calculators
from .metrics_integration import (
    IndividualMetricsCalculator
)

__all__ = [
    # Legacy (backward compatibility)
    'ImageMetrics',
    'MetricResults', 
    'MetricsTracker',
    'calculate_psnr',
    # New data structures
    'IndividualImageMetrics',
    'DatasetEvaluationResults',
    # Individual metrics
    'LPIPSMetric',
    'ImprovedSSIM',
    # Dataset metrics
    'DatasetFIDEvaluator',
    # Integrated calculators
    'IndividualMetricsCalculator'
]