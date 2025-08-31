"""Base classes and interfaces for the Generative Latent Optimization package."""

from abc import ABC, abstractmethod
from typing import Union, Dict, Any, List, Optional
import torch
from pathlib import Path


class BaseMetric(ABC):
    """Base class for all metric calculation classes.
    
    Provides unified device management and interface for metric calculations.
    """
    
    def __init__(self, device: str = 'cuda'):
        """Initialize the base metric class.
        
        Args:
            device: Device to use for computations ('cuda' or 'cpu')
        """
        self.device = self._validate_and_setup_device(device)
        self._setup_metric_specific_resources()
    
    def _validate_and_setup_device(self, device: str) -> str:
        """Validate and setup the computation device.
        
        Args:
            device: Requested device string
            
        Returns:
            Valid device string
        """
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            return 'cpu'
        return device
    
    @abstractmethod
    def calculate(self, img1: torch.Tensor, img2: torch.Tensor) -> Union[float, Dict[str, float]]:
        """Calculate metrics between two images.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            
        Returns:
            Metric value(s)
        """
        pass
    
    @abstractmethod
    def _setup_metric_specific_resources(self) -> None:
        """Setup metric-specific resources and initialization."""
        pass
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the configured device.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor on the configured device
        """
        return tensor.to(self.device)


class BaseEvaluator(ABC):
    """Base class for all evaluation classes."""
    
    def __init__(self, device: str = 'cuda', **kwargs):
        """Initialize the base evaluator class.
        
        Args:
            device: Device to use for computations
            **kwargs: Additional configuration parameters
        """
        from .device_manager import DeviceManager
        self.device_manager = DeviceManager(device)
        self.device = self.device_manager.device
        self.metrics_calculator = self._create_metrics_calculator(**kwargs)
        self._setup_statistics_calculator()
    
    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """Execute evaluation.
        
        Args:
            **kwargs: Evaluation parameters
            
        Returns:
            Evaluation results dictionary
        """
        pass
    
    @abstractmethod
    def _create_metrics_calculator(self, **kwargs):
        """Create the metrics calculator instance.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            Metrics calculator instance
        """
        pass
    
    def _setup_statistics_calculator(self) -> None:
        """Setup the statistics calculator."""
        from ..utils.io_utils import StatisticsCalculator
        self.statistics_calculator = StatisticsCalculator()
    
    def _generate_evaluation_report(self, results: Dict) -> str:
        """Generate a formatted evaluation report.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "📊 Evaluation Report",
            "=" * 50
        ]
        
        for key, value in results.items():
            if isinstance(value, float):
                report_lines.append(f"{key}: {value:.4f}")
            elif isinstance(value, dict):
                report_lines.append(f"\n{key}:")
                for sub_key, sub_value in value.items():
                    report_lines.append(f"  {sub_key}: {sub_value:.4f}")
            else:
                report_lines.append(f"{key}: {value}")
        
        return "\n".join(report_lines)


class BaseDataset(ABC):
    """Base class for dataset processing."""
    
    def __init__(self, device: str = 'cuda'):
        """Initialize the base dataset class.
        
        Args:
            device: Device to use for processing
        """
        from .device_manager import DeviceManager
        self.device_manager = DeviceManager(device)
        self.device = self.device_manager.device
        self._setup_dataset_specific_resources()
    
    @abstractmethod
    def process_batch(self, batch_data: Any) -> Any:
        """Process a batch of data.
        
        Args:
            batch_data: Input batch data
            
        Returns:
            Processed batch data
        """
        pass
    
    @abstractmethod
    def _setup_dataset_specific_resources(self) -> None:
        """Setup dataset-specific resources."""
        pass
    
    def validate_input_path(self, path: Union[str, Path]) -> Path:
        """Validate and resolve input path.
        
        Args:
            path: Input path
            
        Returns:
            Resolved Path object
            
        Raises:
            FileNotFoundError: If path doesn't exist
        """
        resolved_path = Path(path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Path not found: {resolved_path}")
        return resolved_path
    
    def ensure_output_path(self, path: Union[str, Path]) -> Path:
        """Ensure output path exists.
        
        Args:
            path: Output path
            
        Returns:
            Resolved Path object
        """
        resolved_path = Path(path).resolve()
        resolved_path.mkdir(parents=True, exist_ok=True)
        return resolved_path