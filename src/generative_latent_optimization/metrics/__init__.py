"""Image Quality Metrics Module"""

# torch-image-metrics integration
try:
    import torch_image_metrics as tim
    
    # 互換性エイリアス
    UnifiedMetricsCalculator = tim.Calculator
    IndividualMetricsCalculator = tim.Calculator
    
    # 高速API関数のエクスポート
    quick_psnr = tim.quick_psnr
    quick_ssim = tim.quick_ssim
    quick_all_metrics = tim.quick_all_metrics
    
except ImportError as e:
    raise ImportError(
        "torch-image-metrics is required but not installed. "
        "Install with: uv add torch-image-metrics[full]"
    ) from e

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

# 削除されたインポート（torch-image-metricsに移行済み）
# from .metrics_integration import (
#     IndividualMetricsCalculator
# )
# from .unified_calculator import (
#     UnifiedMetricsCalculator
# )

__all__ = [
    # 既存データ構造（保持）
    'ImageMetrics',              # 基本実装は一時保持
    'MetricResults', 
    'MetricsTracker',
    'calculate_psnr',
    'IndividualImageMetrics', 
    'DatasetEvaluationResults',
    
    # Individual metrics (段階的削除予定)
    'LPIPSMetric',
    'ImprovedSSIM',
    
    # Dataset metrics (段階的削除予定)
    'DatasetFIDEvaluator',
    
    # torch-image-metrics エイリアス
    'UnifiedMetricsCalculator',  # -> tim.Calculator
    'IndividualMetricsCalculator', # -> tim.Calculator
    
    # 高速API
    'quick_psnr',
    'quick_ssim', 
    'quick_all_metrics',
]