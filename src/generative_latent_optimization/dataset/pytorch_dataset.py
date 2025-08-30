#!/usr/bin/env python3
"""
PyTorch Dataset Module

Provides PyTorch Dataset classes for optimized latent representations
and utilities for loading and managing dataset files.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
import time
import json

from ..utils import IOUtils


@dataclass
class DatasetMetadata:
    """Metadata for optimized latents dataset"""
    total_samples: int
    splits_count: Dict[str, int]
    optimization_config: Dict[str, Any]
    processing_statistics: Dict[str, float]
    creation_timestamp: str
    dataset_version: str = "1.0"
    source_dataset: str = "BSDS500"
    
    
@dataclass
class SampleData:
    """Structure for individual dataset sample"""
    image_name: str
    split: str
    initial_latents: torch.Tensor
    optimized_latents: torch.Tensor
    metrics: Dict[str, float]


class OptimizedLatentsDataset(Dataset):
    """
    PyTorch Dataset for optimized VAE latent representations
    
    Loads and provides access to optimized latent representations,
    initial latents, and associated metrics.
    """
    
    def __init__(self, dataset_path: Union[str, Path]):
        """
        Initialize dataset from .pt file
        
        Args:
            dataset_path: Path to .pt dataset file
        """
        self.dataset_path = Path(dataset_path).resolve()
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Load dataset
        self.data = torch.load(self.dataset_path)
        self.samples = self.data['samples']
        self.metadata = DatasetMetadata(**self.data['metadata'])
        
        # Create index mapping
        self._create_indices()
    
    def _create_indices(self):
        """Create index mappings for efficient access"""
        self.split_indices = {}
        for i, sample in enumerate(self.samples):
            split = sample['split']
            if split not in self.split_indices:
                self.split_indices[split] = []
            self.split_indices[split].append(i)
    
    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor, Dict]]:
        """
        Get sample by index
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample data
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        sample = self.samples[idx]
        return {
            'image_name': sample['image_name'],
            'split': sample['split'],
            'initial_latents': sample['initial_latents'],
            'optimized_latents': sample['optimized_latents'],
            'metrics': sample['metrics'],
            'index': idx
        }
    
    def get_by_split(self, split: str) -> List[Dict[str, Any]]:
        """Get all samples from a specific split"""
        if split not in self.split_indices:
            return []
        
        return [self[idx] for idx in self.split_indices[split]]
    
    def get_metadata(self) -> DatasetMetadata:
        """Get dataset metadata"""
        return self.metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            'total_samples': len(self),
            'splits': {split: len(indices) for split, indices in self.split_indices.items()},
            'metadata': asdict(self.metadata)
        }
    
    def create_dataloader(self, split: Optional[str] = None, batch_size: int = 32, 
                         shuffle: bool = True, **kwargs) -> DataLoader:
        """
        Create PyTorch DataLoader
        
        Args:
            split: Specific split to load (None for all)
            batch_size: Batch size
            shuffle: Whether to shuffle data
            **kwargs: Additional DataLoader arguments
            
        Returns:
            PyTorch DataLoader
        """
        if split is not None:
            # Create subset for specific split
            indices = self.split_indices.get(split, [])
            if not indices:
                # Return an empty dataloader for non-existent splits
                from torch.utils.data import Dataset
                class EmptyDataset(Dataset):
                    def __len__(self): return 0
                    def __getitem__(self, idx): raise IndexError("Empty dataset")
                return DataLoader(EmptyDataset(), batch_size=batch_size, shuffle=False, **kwargs)
            subset = torch.utils.data.Subset(self, indices)
            return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        else:
            return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def save_subset(self, output_path: Union[str, Path], split: Optional[str] = None,
                   max_samples: Optional[int] = None) -> str:
        """
        Save subset of dataset to new file
        
        Args:
            output_path: Output path for subset
            split: Specific split to save (None for all)  
            max_samples: Maximum samples to include
            
        Returns:
            Path to saved subset
        """
        output_path = Path(output_path).resolve()
        
        # Determine samples to include
        if split is not None:
            sample_indices = self.split_indices.get(split, [])
        else:
            sample_indices = list(range(len(self)))
        
        if max_samples is not None:
            sample_indices = sample_indices[:max_samples]
        
        # Create subset data
        subset_samples = [self.samples[i] for i in sample_indices]
        
        # Update metadata
        subset_metadata = asdict(self.metadata)
        subset_metadata['total_samples'] = len(subset_samples)
        subset_metadata['creation_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if split is not None:
            subset_metadata['splits_count'] = {split: len(subset_samples)}
        else:
            # Recalculate split counts
            split_counts = {}
            for sample in subset_samples:
                sample_split = sample['split']
                split_counts[sample_split] = split_counts.get(sample_split, 0) + 1
            subset_metadata['splits_count'] = split_counts
        
        # Save subset
        subset_data = {
            'samples': subset_samples,
            'metadata': subset_metadata
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(subset_data, output_path)
        
        print(f"💾 Saved dataset subset to: {output_path}")
        print(f"   Samples: {len(subset_samples)}")
        if split:
            print(f"   Split: {split}")
        
        return str(output_path)


class DatasetBuilder:
    """Builder class for creating PyTorch datasets from processing results"""
    
    def __init__(self):
        self.io_utils = IOUtils()
    
    def create_pytorch_dataset(self, processed_data_dir: Union[str, Path],
                             output_path: Union[str, Path],
                             optimization_config: Dict[str, Any]) -> str:
        """
        Create PyTorch dataset from processed optimization results
        
        Args:
            processed_data_dir: Directory containing processed results
            output_path: Output path for .pt dataset file
            optimization_config: Configuration used for optimization
            
        Returns:
            Path to created dataset file
        """
        processed_dir = Path(processed_data_dir).resolve()
        output_path = Path(output_path).resolve()
        
        print(f"🏗️ Creating PyTorch dataset from: {processed_dir}")
        
        # Collect all processed results
        dataset_entries = []
        splits = ['train', 'val', 'test']
        
        for split in splits:
            split_dir = processed_dir / split
            if not split_dir.exists():
                print(f"⚠️ Split directory not found: {split_dir}")
                continue
            
            # Load detailed results
            results_file = split_dir / "detailed_results.json"
            if results_file.exists():
                results = self.io_utils.load_json(results_file)
                
                for result in results:
                    image_name = result['image_name']
                    image_result_dir = split_dir / image_name
                    
                    if image_result_dir.exists():
                        # Load latents and other data
                        try:
                            initial_latents_path = image_result_dir / f"{image_name}_initial_latents.pt"
                            optimized_latents_path = image_result_dir / f"{image_name}_optimized_latents.pt"
                            
                            if initial_latents_path.exists() and optimized_latents_path.exists():
                                entry = {
                                    'image_name': image_name,
                                    'split': split,
                                    'initial_latents': torch.load(initial_latents_path),
                                    'optimized_latents': torch.load(optimized_latents_path),
                                    'metrics': {
                                        'initial_psnr': result['initial_psnr'],
                                        'final_psnr': result['final_psnr'],
                                        'psnr_improvement': result['psnr_improvement'],
                                        'initial_ssim': result['initial_ssim'], 
                                        'final_ssim': result['final_ssim'],
                                        'ssim_improvement': result['ssim_improvement'],
                                        'loss_reduction': result['loss_reduction'],
                                        'optimization_iterations': result['optimization_iterations'],
                                        'convergence_iteration': result.get('convergence_iteration'),
                                        'initial_loss': result['initial_loss'],
                                        'final_loss': result['final_loss']
                                    }
                                }
                                
                                dataset_entries.append(entry)
                        except Exception as e:
                            print(f"⚠️ Failed to load data for {image_name}: {e}")
        
        print(f"📊 Collected {len(dataset_entries)} samples")
        
        # Calculate statistics
        processing_stats = self._calculate_processing_statistics(dataset_entries)
        
        # Create dataset metadata
        metadata = DatasetMetadata(
            total_samples=len(dataset_entries),
            splits_count={
                split: len([e for e in dataset_entries if e['split'] == split])
                for split in splits
            },
            optimization_config=optimization_config,
            processing_statistics=processing_stats,
            creation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Create dataset structure
        dataset = {
            'samples': dataset_entries,
            'metadata': asdict(metadata)
        }
        
        # Save dataset
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, output_path)
        
        print(f"💾 PyTorch dataset saved to: {output_path}")
        print(f"   Total samples: {metadata.total_samples}")
        print(f"   Splits: {metadata.splits_count}")
        print(f"   Avg PSNR improvement: {processing_stats['avg_psnr_improvement']:.2f} dB")
        
        return str(output_path)
    
    def _calculate_processing_statistics(self, dataset_entries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate processing statistics across all samples"""
        if not dataset_entries:
            return {}
        
        psnr_improvements = [entry['metrics']['psnr_improvement'] for entry in dataset_entries]
        ssim_improvements = [entry['metrics']['ssim_improvement'] for entry in dataset_entries]
        loss_reductions = [entry['metrics']['loss_reduction'] for entry in dataset_entries]
        iterations = [entry['metrics']['optimization_iterations'] for entry in dataset_entries]
        
        return {
            'avg_psnr_improvement': sum(psnr_improvements) / len(psnr_improvements),
            'std_psnr_improvement': self._std(psnr_improvements),
            'max_psnr_improvement': max(psnr_improvements),
            'min_psnr_improvement': min(psnr_improvements),
            'avg_ssim_improvement': sum(ssim_improvements) / len(ssim_improvements),
            'std_ssim_improvement': self._std(ssim_improvements),
            'avg_loss_reduction': sum(loss_reductions) / len(loss_reductions),
            'std_loss_reduction': self._std(loss_reductions),
            'avg_iterations': sum(iterations) / len(iterations),
            'convergence_rate': len([e for e in dataset_entries 
                                   if e['metrics']['convergence_iteration'] is not None]) / len(dataset_entries) * 100
        }
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


# Utility functions
def load_optimized_dataset(dataset_path: Union[str, Path]) -> OptimizedLatentsDataset:
    """Load optimized latents dataset from file"""
    return OptimizedLatentsDataset(dataset_path)


def create_dataset_from_results(processed_dir: Union[str, Path], 
                               output_path: Union[str, Path],
                               optimization_config: Dict[str, Any]) -> str:
    """Create PyTorch dataset from processing results directory"""
    builder = DatasetBuilder()
    return builder.create_pytorch_dataset(processed_dir, output_path, optimization_config)