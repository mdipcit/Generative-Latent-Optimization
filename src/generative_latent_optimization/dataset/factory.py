"""
Dataset Factory Module

Provides a unified interface for creating different types of datasets
(PNG format, PyTorch format) with consistent API.
"""

from pathlib import Path
from typing import Union, Dict, Any, Tuple, Literal, Optional, List
from dataclasses import dataclass

from .pytorch_dataset import (
    DatasetBuilder as PyTorchDatasetBuilder,
    OptimizedLatentsDataset,
    DatasetMetadata
)
from .png_dataset import (
    PNGDatasetBuilder,
    PNGDatasetMetadata
)
from ..core.base_classes import BaseDataset


class DatasetFactory:
    """
    Factory class for creating different types of datasets
    
    Provides a unified interface for creating PNG and PyTorch format datasets
    with consistent configuration and API.
    """
    
    @staticmethod
    def create_dataset(format_type: Literal['png', 'pytorch'],
                      output_path: Union[str, Path],
                      **kwargs) -> Union[PNGDatasetBuilder, OptimizedLatentsDataset]:
        """
        Create a dataset of the specified format
        
        Args:
            format_type: Type of dataset to create ('png' or 'pytorch')
            output_path: Path where dataset will be saved
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            Dataset builder or dataset instance based on format
            
        Raises:
            ValueError: If unsupported format_type is provided
        """
        output_path = Path(output_path).resolve()
        
        if format_type == 'png':
            return DatasetFactory._create_png_dataset(output_path, **kwargs)
        elif format_type == 'pytorch':
            return DatasetFactory._create_pytorch_dataset(output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset format: {format_type}. "
                           f"Supported formats are: 'png', 'pytorch'")
    
    @staticmethod
    def _create_png_dataset(output_path: Path, **kwargs) -> PNGDatasetBuilder:
        """
        Create a PNG format dataset
        
        Args:
            output_path: Directory to save PNG files
            **kwargs: Additional configuration
            
        Returns:
            PNGDatasetBuilder instance
        """
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract PNG-specific parameters
        save_original = kwargs.get('save_original', True)
        save_initial = kwargs.get('save_initial', True)
        save_optimized = kwargs.get('save_optimized', True)
        save_metadata = kwargs.get('save_metadata', True)
        
        # Create and configure builder
        builder = PNGDatasetBuilder(
            output_dir=output_path,
            save_original=save_original,
            save_initial=save_initial,
            save_optimized=save_optimized,
            save_metadata=save_metadata
        )
        
        return builder
    
    @staticmethod
    def _create_pytorch_dataset(output_path: Path, **kwargs) -> PyTorchDatasetBuilder:
        """
        Create a PyTorch format dataset
        
        Args:
            output_path: File path for the .pt dataset file
            **kwargs: Additional configuration
            
        Returns:
            PyTorchDatasetBuilder instance
        """
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract PyTorch-specific parameters
        save_tensors = kwargs.get('save_tensors', True)
        save_metadata = kwargs.get('save_metadata', True)
        compression = kwargs.get('compression', None)
        
        # Create and configure builder
        builder = PyTorchDatasetBuilder(
            output_path=output_path,
            save_tensors=save_tensors,
            save_metadata=save_metadata,
            compression=compression
        )
        
        return builder
    
    @staticmethod
    def create_dual_dataset(output_path: Union[str, Path],
                           **kwargs) -> Tuple[PNGDatasetBuilder, PyTorchDatasetBuilder]:
        """
        Create both PNG and PyTorch format datasets simultaneously
        
        Args:
            output_path: Base path for datasets
            **kwargs: Configuration for both datasets
            
        Returns:
            Tuple of (PNGDatasetBuilder, PyTorchDatasetBuilder)
        """
        output_path = Path(output_path).resolve()
        
        # Create subdirectories for each format
        png_path = output_path / 'png_dataset'
        pytorch_path = output_path / 'pytorch_dataset.pt'
        
        # Create both dataset builders
        png_builder = DatasetFactory.create_dataset(
            'png', png_path, **kwargs
        )
        pytorch_builder = DatasetFactory.create_dataset(
            'pytorch', pytorch_path, **kwargs
        )
        
        return png_builder, pytorch_builder
    
    @staticmethod
    def load_dataset(dataset_path: Union[str, Path],
                    format_type: Optional[Literal['png', 'pytorch']] = None) -> Any:
        """
        Load an existing dataset
        
        Args:
            dataset_path: Path to the dataset
            format_type: Type of dataset (auto-detected if None)
            
        Returns:
            Loaded dataset object
            
        Raises:
            ValueError: If dataset format cannot be determined
        """
        dataset_path = Path(dataset_path).resolve()
        
        # Auto-detect format if not specified
        if format_type is None:
            if dataset_path.is_dir():
                # Check if it's a PNG dataset directory
                if (dataset_path / 'metadata.json').exists():
                    format_type = 'png'
                else:
                    raise ValueError(f"Cannot determine dataset format for directory: {dataset_path}")
            elif dataset_path.suffix in ['.pt', '.pth']:
                format_type = 'pytorch'
            else:
                raise ValueError(f"Cannot determine dataset format for file: {dataset_path}")
        
        # Load based on format
        if format_type == 'png':
            return DatasetFactory._load_png_dataset(dataset_path)
        elif format_type == 'pytorch':
            return DatasetFactory._load_pytorch_dataset(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset format: {format_type}")
    
    @staticmethod
    def _load_png_dataset(dataset_path: Path) -> Dict[str, Any]:
        """
        Load a PNG format dataset
        
        Args:
            dataset_path: Directory containing PNG dataset
            
        Returns:
            Dictionary with dataset information
        """
        import json
        
        if not dataset_path.is_dir():
            raise ValueError(f"PNG dataset path must be a directory: {dataset_path}")
        
        # Load metadata
        metadata_path = dataset_path / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get image directories
        original_dir = dataset_path / 'original' if (dataset_path / 'original').exists() else None
        initial_dir = dataset_path / 'initial_reconstruction' if (dataset_path / 'initial_reconstruction').exists() else None
        optimized_dir = dataset_path / 'optimized' if (dataset_path / 'optimized').exists() else None
        
        return {
            'format': 'png',
            'path': str(dataset_path),
            'metadata': metadata,
            'original_dir': str(original_dir) if original_dir else None,
            'initial_dir': str(initial_dir) if initial_dir else None,
            'optimized_dir': str(optimized_dir) if optimized_dir else None,
            'num_samples': metadata.get('num_samples', 0)
        }
    
    @staticmethod
    def _load_pytorch_dataset(dataset_path: Path) -> OptimizedLatentsDataset:
        """
        Load a PyTorch format dataset
        
        Args:
            dataset_path: Path to .pt file
            
        Returns:
            OptimizedLatentsDataset instance
        """
        import torch
        from .pytorch_dataset import load_optimized_dataset
        
        if not dataset_path.is_file():
            raise ValueError(f"PyTorch dataset path must be a file: {dataset_path}")
        
        return load_optimized_dataset(dataset_path)
    
    @staticmethod
    def get_dataset_info(dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a dataset without fully loading it
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            Dictionary with dataset information
        """
        dataset_path = Path(dataset_path).resolve()
        
        if dataset_path.is_dir():
            # PNG dataset
            metadata_path = dataset_path / 'metadata.json'
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return {
                    'format': 'png',
                    'path': str(dataset_path),
                    'num_samples': metadata.get('num_samples', 0),
                    'creation_date': metadata.get('creation_date', 'unknown'),
                    'size_on_disk': sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
                }
        elif dataset_path.suffix in ['.pt', '.pth']:
            # PyTorch dataset
            import torch
            data = torch.load(dataset_path, map_location='cpu')
            metadata = data.get('metadata', {})
            return {
                'format': 'pytorch',
                'path': str(dataset_path),
                'num_samples': metadata.get('num_samples', len(data.get('samples', []))),
                'creation_date': metadata.get('creation_date', 'unknown'),
                'size_on_disk': dataset_path.stat().st_size
            }
        
        raise ValueError(f"Cannot determine dataset format for: {dataset_path}")


class UnifiedDatasetBuilder(BaseDataset):
    """
    Unified dataset builder that can create multiple dataset formats simultaneously
    
    This builder manages the creation of both PNG and PyTorch datasets
    from the same source data, ensuring consistency across formats.
    """
    
    def __init__(self, output_base_path: Union[str, Path],
                 formats: List[Literal['png', 'pytorch']] = None):
        """
        Initialize unified dataset builder
        
        Args:
            output_base_path: Base path for all dataset outputs
            formats: List of formats to create (default: ['png', 'pytorch'])
        """
        super().__init__()
        self.output_base_path = Path(output_base_path).resolve()
        self.formats = formats or ['png', 'pytorch']
        self.builders = {}
        
        # Initialize builders for each format
        self._initialize_builders()
    
    def _initialize_builders(self):
        """Initialize dataset builders for specified formats"""
        if 'png' in self.formats:
            self.builders['png'] = DatasetFactory.create_dataset(
                'png', self.output_base_path / 'png_dataset'
            )
        
        if 'pytorch' in self.formats:
            self.builders['pytorch'] = DatasetFactory.create_dataset(
                'pytorch', self.output_base_path / 'pytorch_dataset.pt'
            )
    
    def add_sample(self, sample_data: Dict[str, Any]):
        """
        Add a sample to all dataset builders
        
        Args:
            sample_data: Dictionary containing sample data
        """
        for format_name, builder in self.builders.items():
            if hasattr(builder, 'add_sample'):
                builder.add_sample(sample_data)
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize all datasets and return paths
        
        Returns:
            Dictionary mapping format names to dataset paths
        """
        results = {}
        
        for format_name, builder in self.builders.items():
            if hasattr(builder, 'finalize'):
                results[format_name] = builder.finalize()
            elif hasattr(builder, 'save'):
                results[format_name] = builder.save()
        
        return results
    
    def process_batch(self, batch_data: Any) -> Any:
        """
        Process a batch of data (required by BaseDataset)
        
        Args:
            batch_data: Batch data to process
            
        Returns:
            Processed batch data
        """
        # This is primarily used for adding samples to the builders
        if isinstance(batch_data, list):
            for sample in batch_data:
                self.add_sample(sample)
        else:
            self.add_sample(batch_data)
        
        return batch_data