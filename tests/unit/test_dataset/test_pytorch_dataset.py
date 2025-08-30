#!/usr/bin/env python3
"""
Unit tests for PyTorch Dataset functionality
"""

import os
import sys
import tempfile
import json
import pickle
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import pytest
import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from generative_latent_optimization.dataset.pytorch_dataset import (
    OptimizedLatentsDataset, DatasetBuilder, DatasetMetadata, SampleData,
    load_optimized_dataset, create_dataset_from_results
)
from ...fixtures.dataset_mocks import mock_dataset_dependencies
from ...fixtures.assertion_helpers import (
    assert_float_approximately_equal, safe_calculate_statistics, assert_dataset_sample_structure
)


@pytest.fixture
def mock_io_utils():
    """Mock IOUtils dependency"""
    with patch('generative_latent_optimization.dataset.pytorch_dataset.IOUtils') as mock_io:
        yield mock_io


@pytest.fixture
def sample_optimization_config():
    """Sample optimization configuration"""
    return {
        'iterations': 100,
        'learning_rate': 0.1,
        'loss_function': 'mse',
        'device': 'cuda'
    }


@pytest.fixture
def sample_dataset_data():
    """Sample dataset data structure"""
    return {
        'samples': [
            {
                'image_name': 'img_001',
                'split': 'train',
                'initial_latents': torch.randn(1, 4, 64, 64),
                'optimized_latents': torch.randn(1, 4, 64, 64),
                'metrics': {
                    'initial_psnr': 20.0,
                    'final_psnr': 25.0,
                    'psnr_improvement': 5.0,
                    'initial_ssim': 0.7,
                    'final_ssim': 0.85,
                    'ssim_improvement': 0.15,
                    'loss_reduction': 80.0,
                    'optimization_iterations': 50,
                    'convergence_iteration': 45,
                    'initial_loss': 1.0,
                    'final_loss': 0.2
                }
            },
            {
                'image_name': 'img_002',
                'split': 'val',
                'initial_latents': torch.randn(1, 4, 64, 64),
                'optimized_latents': torch.randn(1, 4, 64, 64),
                'metrics': {
                    'initial_psnr': 18.0,
                    'final_psnr': 24.0,
                    'psnr_improvement': 6.0,
                    'initial_ssim': 0.65,
                    'final_ssim': 0.8,
                    'ssim_improvement': 0.15,
                    'loss_reduction': 75.0,
                    'optimization_iterations': 45,
                    'convergence_iteration': 40,
                    'initial_loss': 1.2,
                    'final_loss': 0.3
                }
            },
            {
                'image_name': 'img_003',
                'split': 'train',
                'initial_latents': torch.randn(1, 4, 64, 64),
                'optimized_latents': torch.randn(1, 4, 64, 64),
                'metrics': {
                    'initial_psnr': 22.0,
                    'final_psnr': 26.0,
                    'psnr_improvement': 4.0,
                    'initial_ssim': 0.75,
                    'final_ssim': 0.88,
                    'ssim_improvement': 0.13,
                    'loss_reduction': 70.0,
                    'optimization_iterations': 40,
                    'convergence_iteration': 38,
                    'initial_loss': 0.8,
                    'final_loss': 0.24
                }
            }
        ],
        'metadata': {
            'total_samples': 3,
            'splits_count': {'train': 2, 'val': 1},
            'optimization_config': {
                'iterations': 100,
                'learning_rate': 0.1
            },
            'processing_statistics': {
                'avg_psnr_improvement': 5.0,
                'std_psnr_improvement': 1.0
            },
            'creation_timestamp': "2024-01-01 12:00:00",
            'dataset_version': "1.0",
            'source_dataset': "BSDS500"
        }
    }


@pytest.fixture
def temp_dataset_file(sample_dataset_data):
    """Create temporary dataset file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = Path(temp_dir) / "test_dataset.pt"
        torch.save(sample_dataset_data, dataset_path)
        yield dataset_path


class TestDatasetMetadata:
    """Test cases for DatasetMetadata dataclass"""
    
    def test_metadata_creation(self, sample_optimization_config):
        """Test DatasetMetadata creation"""
        metadata = DatasetMetadata(
            total_samples=100,
            splits_count={'train': 60, 'val': 25, 'test': 15},
            optimization_config=sample_optimization_config,
            processing_statistics={'avg_psnr': 5.0},
            creation_timestamp="2024-01-01 12:00:00"
        )
        
        assert metadata.total_samples == 100
        assert metadata.splits_count == {'train': 60, 'val': 25, 'test': 15}
        assert metadata.optimization_config == sample_optimization_config
        assert metadata.processing_statistics == {'avg_psnr': 5.0}
        assert metadata.creation_timestamp == "2024-01-01 12:00:00"
        assert metadata.dataset_version == "1.0"  # Default
        assert metadata.source_dataset == "BSDS500"  # Default
    
    def test_metadata_custom_defaults(self, sample_optimization_config):
        """Test metadata with custom default values"""
        metadata = DatasetMetadata(
            total_samples=50,
            splits_count={'train': 50},
            optimization_config=sample_optimization_config,
            processing_statistics={},
            creation_timestamp="2024-01-01 12:00:00",
            dataset_version="2.0",
            source_dataset="Custom"
        )
        
        assert metadata.dataset_version == "2.0"
        assert metadata.source_dataset == "Custom"


class TestSampleData:
    """Test cases for SampleData dataclass"""
    
    def test_sample_data_creation(self):
        """Test SampleData creation"""
        initial_latents = torch.randn(1, 4, 64, 64)
        optimized_latents = torch.randn(1, 4, 64, 64)
        metrics = {'psnr': 25.0, 'ssim': 0.85}
        
        sample = SampleData(
            image_name="test_img",
            split="train",
            initial_latents=initial_latents,
            optimized_latents=optimized_latents,
            metrics=metrics
        )
        
        assert sample.image_name == "test_img"
        assert sample.split == "train"
        assert torch.equal(sample.initial_latents, initial_latents)
        assert torch.equal(sample.optimized_latents, optimized_latents)
        assert sample.metrics == metrics


class TestOptimizedLatentsDataset:
    """Test cases for OptimizedLatentsDataset class"""
    
    def test_dataset_initialization_success(self, temp_dataset_file):
        """Test successful dataset initialization"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        assert dataset.dataset_path == temp_dataset_file
        assert len(dataset.samples) == 3
        assert isinstance(dataset.metadata, DatasetMetadata)
        assert dataset.metadata.total_samples == 3
    
    def test_dataset_initialization_file_not_found(self):
        """Test dataset initialization with non-existent file"""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            OptimizedLatentsDataset("/nonexistent/dataset.pt")
    
    def test_dataset_length(self, temp_dataset_file):
        """Test __len__ method"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        assert len(dataset) == 3
    
    def test_dataset_getitem_valid_index(self, temp_dataset_file):
        """Test __getitem__ with valid index"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        sample = dataset[0]
        
        # Verify structure
        assert isinstance(sample, dict)
        required_keys = ['image_name', 'split', 'initial_latents', 'optimized_latents', 'metrics', 'index']
        for key in required_keys:
            assert key in sample
        
        # Verify data types
        assert isinstance(sample['image_name'], str)
        assert isinstance(sample['split'], str)
        assert isinstance(sample['initial_latents'], torch.Tensor)
        assert isinstance(sample['optimized_latents'], torch.Tensor)
        assert isinstance(sample['metrics'], dict)
        assert sample['index'] == 0
        
        # Verify content
        assert sample['image_name'] == 'img_001'
        assert sample['split'] == 'train'
    
    def test_dataset_getitem_out_of_range(self, temp_dataset_file):
        """Test __getitem__ with out-of-range index"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        with pytest.raises(IndexError, match="Index .* out of range"):
            dataset[10]  # Only 3 samples available
    
    def test_get_by_split(self, temp_dataset_file):
        """Test get_by_split method"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        # Test train split (should have 2 samples)
        train_samples = dataset.get_by_split('train')
        assert len(train_samples) == 2
        for sample in train_samples:
            assert sample['split'] == 'train'
        
        # Test val split (should have 1 sample)
        val_samples = dataset.get_by_split('val')
        assert len(val_samples) == 1
        assert val_samples[0]['split'] == 'val'
        
        # Test non-existent split
        test_samples = dataset.get_by_split('test')
        assert len(test_samples) == 0
    
    def test_get_metadata(self, temp_dataset_file):
        """Test get_metadata method"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        metadata = dataset.get_metadata()
        
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.total_samples == 3
        assert metadata.splits_count == {'train': 2, 'val': 1}
    
    def test_get_statistics(self, temp_dataset_file):
        """Test get_statistics method"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        stats = dataset.get_statistics()
        
        assert isinstance(stats, dict)
        assert stats['total_samples'] == 3
        assert 'splits' in stats
        assert stats['splits']['train'] == 2
        assert stats['splits']['val'] == 1
        assert 'metadata' in stats
        assert isinstance(stats['metadata'], dict)
    
    def test_create_dataloader_all_data(self, temp_dataset_file):
        """Test create_dataloader for all data"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        dataloader = dataset.create_dataloader(batch_size=2, shuffle=False)
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 2
        
        # Test iteration
        batches = list(dataloader)
        assert len(batches) == 2  # ceil(3/2) = 2
        
        # Check first batch
        first_batch = batches[0]
        assert isinstance(first_batch, dict)
        assert first_batch['image_name'][0] == 'img_001'  # First sample
    
    def test_create_dataloader_specific_split(self, temp_dataset_file):
        """Test create_dataloader for specific split"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        # Create dataloader for train split only
        train_dataloader = dataset.create_dataloader(split='train', batch_size=1, shuffle=False)
        
        # Should only get train samples
        batches = list(train_dataloader)
        assert len(batches) == 2  # 2 train samples
        
        for batch in batches:
            assert batch['split'][0] == 'train'
    
    def test_create_dataloader_nonexistent_split(self, temp_dataset_file):
        """Test create_dataloader for non-existent split"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        # Create dataloader for non-existent split
        test_dataloader = dataset.create_dataloader(split='test', batch_size=1)
        
        # Should create empty dataloader
        batches = list(test_dataloader)
        assert len(batches) == 0
    
    @patch('generative_latent_optimization.dataset.pytorch_dataset.time.strftime')
    def test_save_subset_all_data(self, mock_strftime, temp_dataset_file):
        """Test save_subset for all data"""
        mock_strftime.return_value = "2024-01-01 12:00:00"
        
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subset.pt"
            
            result_path = dataset.save_subset(output_path)
            
            assert result_path == str(output_path)
            assert output_path.exists()
            
            # Load and verify subset
            subset_data = torch.load(output_path)
            assert len(subset_data['samples']) == 3
            assert subset_data['metadata']['total_samples'] == 3
            assert subset_data['metadata']['creation_timestamp'] == "2024-01-01 12:00:00"
    
    @patch('generative_latent_optimization.dataset.pytorch_dataset.time.strftime')
    def test_save_subset_specific_split(self, mock_strftime, temp_dataset_file):
        """Test save_subset for specific split"""
        mock_strftime.return_value = "2024-01-01 12:00:00"
        
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "train_subset.pt"
            
            result_path = dataset.save_subset(output_path, split='train')
            
            assert result_path == str(output_path)
            assert output_path.exists()
            
            # Load and verify subset
            subset_data = torch.load(output_path)
            assert len(subset_data['samples']) == 2  # Only train samples
            assert subset_data['metadata']['total_samples'] == 2
            assert subset_data['metadata']['splits_count'] == {'train': 2}
            
            # Verify all samples are from train split
            for sample in subset_data['samples']:
                assert sample['split'] == 'train'
    
    def test_save_subset_with_max_samples(self, temp_dataset_file):
        """Test save_subset with max_samples limit"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "limited_subset.pt"
            
            result_path = dataset.save_subset(output_path, max_samples=2)
            
            assert output_path.exists()
            
            # Load and verify subset
            subset_data = torch.load(output_path)
            assert len(subset_data['samples']) == 2  # Limited to 2 samples
            assert subset_data['metadata']['total_samples'] == 2
    
    def test_save_subset_nonexistent_split(self, temp_dataset_file):
        """Test save_subset for non-existent split"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty_subset.pt"
            
            result_path = dataset.save_subset(output_path, split='test')
            
            assert output_path.exists()
            
            # Load and verify empty subset
            subset_data = torch.load(output_path)
            assert len(subset_data['samples']) == 0
            assert subset_data['metadata']['total_samples'] == 0
            assert subset_data['metadata']['splits_count'] == {'test': 0}
    
    def test_create_indices(self, temp_dataset_file):
        """Test _create_indices method"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        # Verify split indices were created correctly
        assert 'train' in dataset.split_indices
        assert 'val' in dataset.split_indices
        assert len(dataset.split_indices['train']) == 2
        assert len(dataset.split_indices['val']) == 1
        
        # Verify indices point to correct samples
        train_indices = dataset.split_indices['train']
        for idx in train_indices:
            assert dataset.samples[idx]['split'] == 'train'


class TestDatasetBuilder:
    """Test cases for DatasetBuilder class"""
    
    def test_initialization(self, mock_io_utils):
        """Test DatasetBuilder initialization"""
        builder = DatasetBuilder()
        assert hasattr(builder, 'io_utils')
    
    def test_std_calculation(self, mock_io_utils):
        """Test _std helper method"""
        builder = DatasetBuilder()
        
        # Test with normal values
        std = builder._std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert std > 0
        
        # Test with empty list
        std_empty = builder._std([])
        assert std_empty == 0.0
        
        # Test with single value
        std_single = builder._std([5.0])
        assert std_single == 0.0
    
    def test_calculate_processing_statistics(self, mock_io_utils):
        """Test _calculate_processing_statistics method"""
        builder = DatasetBuilder()
        
        dataset_entries = [
            {
                'metrics': {
                    'psnr_improvement': 5.0,
                    'ssim_improvement': 0.15,
                    'loss_reduction': 80.0,
                    'optimization_iterations': 50,
                    'convergence_iteration': 45
                }
            },
            {
                'metrics': {
                    'psnr_improvement': 6.0,
                    'ssim_improvement': 0.10,
                    'loss_reduction': 75.0,
                    'optimization_iterations': 40,
                    'convergence_iteration': 35
                }
            }
        ]
        
        stats = builder._calculate_processing_statistics(dataset_entries)
        
        # Verify statistical calculations
        assert_float_approximately_equal(stats['avg_psnr_improvement'], 5.5)  # (5.0 + 6.0) / 2
        assert_float_approximately_equal(stats['max_psnr_improvement'], 6.0)
        assert_float_approximately_equal(stats['min_psnr_improvement'], 5.0)
        assert_float_approximately_equal(stats['avg_ssim_improvement'], 0.125)  # (0.15 + 0.10) / 2
        assert_float_approximately_equal(stats['avg_loss_reduction'], 77.5)  # (80.0 + 75.0) / 2
        assert_float_approximately_equal(stats['avg_iterations'], 45.0)  # (50 + 40) / 2
        assert_float_approximately_equal(stats['convergence_rate'], 100.0)  # Both converged now that we fixed None values
    
    def test_calculate_processing_statistics_empty(self, mock_io_utils):
        """Test _calculate_processing_statistics with empty entries"""
        builder = DatasetBuilder()
        
        stats = builder._calculate_processing_statistics([])
        assert stats == {}
    
    def test_create_pytorch_dataset_success(self, mock_io_utils, sample_optimization_config):
        """Test successful PyTorch dataset creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create processed data structure
            processed_dir = Path(temp_dir) / "processed"
            train_dir = processed_dir / "train"
            train_dir.mkdir(parents=True)
            
            # Create results file with actual JSON file
            results = [
                {
                    'image_name': 'test_img',
                    'initial_psnr': 20.0,
                    'final_psnr': 25.0,
                    'psnr_improvement': 5.0,
                    'initial_ssim': 0.7,
                    'final_ssim': 0.85,
                    'ssim_improvement': 0.15,
                    'loss_reduction': 80.0,
                    'optimization_iterations': 50,
                    'convergence_iteration': 45,
                    'initial_loss': 1.0,
                    'final_loss': 0.2
                }
            ]
            
            # Create actual results JSON file to ensure it exists
            results_file = train_dir / 'detailed_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f)
            
            # Create image result directory and latent files
            img_dir = train_dir / 'test_img'
            img_dir.mkdir()
            
            # Create mock latent tensors
            initial_latents = torch.randn(1, 4, 64, 64)
            optimized_latents = torch.randn(1, 4, 64, 64)
            torch.save(initial_latents, img_dir / "test_img_initial_latents.pt")
            torch.save(optimized_latents, img_dir / "test_img_optimized_latents.pt")
            
            # Create empty val and test directories to avoid warnings
            val_dir = processed_dir / "val"
            test_dir = processed_dir / "test"
            val_dir.mkdir()
            test_dir.mkdir()
            
            # Mock IOUtils to return the results we created
            mock_io = mock_io_utils.return_value
            mock_io.load_json.return_value = results
            
            output_path = Path(temp_dir) / "dataset.pt"
            
            builder = DatasetBuilder()
            
            with patch('generative_latent_optimization.dataset.pytorch_dataset.time.strftime', 
                      return_value="2024-01-01 12:00:00"):
                
                result_path = builder.create_pytorch_dataset(
                    processed_dir, output_path, sample_optimization_config
                )
            
            assert result_path == str(output_path)
            assert output_path.exists()
            
            # Verify created dataset
            created_data = torch.load(output_path)
            assert len(created_data['samples']) == 1
            assert created_data['metadata']['total_samples'] == 1
    
    def test_create_pytorch_dataset_no_splits(self, mock_io_utils, sample_optimization_config):
        """Test PyTorch dataset creation when no splits exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir) / "empty_processed"
            processed_dir.mkdir()
            output_path = Path(temp_dir) / "dataset.pt"
            
            builder = DatasetBuilder()
            
            # Mock empty statistics for empty dataset
            with patch.object(builder, '_calculate_processing_statistics', return_value={'avg_psnr_improvement': 0.0}):
                with patch('generative_latent_optimization.dataset.pytorch_dataset.time.strftime',
                          return_value="2024-01-01 12:00:00"):
                    
                    result_path = builder.create_pytorch_dataset(
                        processed_dir, output_path, sample_optimization_config
                    )
            
            assert result_path == str(output_path)
            assert output_path.exists()
            
            # Verify empty dataset
            created_data = torch.load(output_path)
            assert len(created_data['samples']) == 0
            assert created_data['metadata']['total_samples'] == 0
    
    def test_create_pytorch_dataset_missing_results_file(self, mock_io_utils, sample_optimization_config):
        """Test dataset creation when results file is missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir) / "processed"
            train_dir = processed_dir / "train"
            train_dir.mkdir(parents=True)
            # No detailed_results.json file created
            
            output_path = Path(temp_dir) / "dataset.pt"
            
            builder = DatasetBuilder()
            
            # Mock empty statistics for empty dataset
            with patch.object(builder, '_calculate_processing_statistics', return_value={'avg_psnr_improvement': 0.0}):
                with patch('generative_latent_optimization.dataset.pytorch_dataset.time.strftime',
                          return_value="2024-01-01 12:00:00"):
                    
                    result_path = builder.create_pytorch_dataset(
                        processed_dir, output_path, sample_optimization_config
                    )
            
            # Should still create dataset, just with no samples
            assert output_path.exists()
            created_data = torch.load(output_path)
            assert len(created_data['samples']) == 0
    
    def test_create_pytorch_dataset_missing_latent_files(self, mock_io_utils, sample_optimization_config):
        """Test dataset creation when latent files are missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir) / "processed"
            train_dir = processed_dir / "train"
            train_dir.mkdir(parents=True)
            
            # Create results file
            results = [{'image_name': 'test_img'}]
            
            # Create image directory but no latent files
            img_dir = train_dir / 'test_img'
            img_dir.mkdir()
            
            mock_io = mock_io_utils.return_value
            mock_io.load_json.return_value = results
            
            output_path = Path(temp_dir) / "dataset.pt"
            
            builder = DatasetBuilder()
            
            # Mock empty statistics for empty dataset
            with patch.object(builder, '_calculate_processing_statistics', return_value={'avg_psnr_improvement': 0.0}):
                with patch('generative_latent_optimization.dataset.pytorch_dataset.time.strftime',
                          return_value="2024-01-01 12:00:00"):
                    
                    result_path = builder.create_pytorch_dataset(
                        processed_dir, output_path, sample_optimization_config
                    )
            
            # Should create dataset but skip samples with missing latents
            assert output_path.exists()
            created_data = torch.load(output_path)
            assert len(created_data['samples']) == 0  # No valid samples
    
    def test_create_pytorch_dataset_latent_loading_error(self, mock_io_utils, sample_optimization_config):
        """Test dataset creation when latent loading fails"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir) / "processed"
            train_dir = processed_dir / "train"
            train_dir.mkdir(parents=True)
            
            # Create results file
            results = [
                {
                    'image_name': 'test_img',
                    'initial_psnr': 20.0,
                    'final_psnr': 25.0,
                    'psnr_improvement': 5.0,
                    'initial_ssim': 0.7,
                    'final_ssim': 0.85,
                    'ssim_improvement': 0.15,
                    'loss_reduction': 80.0,
                    'optimization_iterations': 50,
                    'convergence_iteration': 45,
                    'initial_loss': 1.0,
                    'final_loss': 0.2
                }
            ]
            
            # Create image directory and latent files
            img_dir = train_dir / 'test_img'
            img_dir.mkdir()
            (img_dir / "test_img_initial_latents.pt").touch()
            (img_dir / "test_img_optimized_latents.pt").touch()
            
            mock_io = mock_io_utils.return_value
            mock_io.load_json.return_value = results
            
            output_path = Path(temp_dir) / "dataset.pt"
            
            builder = DatasetBuilder()
            
            # Mock empty statistics for empty dataset and torch.load to raise exception
            with patch.object(builder, '_calculate_processing_statistics', return_value={'avg_psnr_improvement': 0.0}):
                with patch('generative_latent_optimization.dataset.pytorch_dataset.torch.load', 
                          side_effect=Exception("Failed to load latents")), \
                     patch('generative_latent_optimization.dataset.pytorch_dataset.time.strftime',
                          return_value="2024-01-01 12:00:00"):
                    
                    result_path = builder.create_pytorch_dataset(
                        processed_dir, output_path, sample_optimization_config
                )
            
            # Should handle loading errors gracefully
            assert output_path.exists()
            created_data = torch.load(output_path)
            assert len(created_data['samples']) == 0  # Failed to load sample


class TestPyTorchDatasetUtilityFunctions:
    """Test utility functions"""
    
    def test_load_optimized_dataset(self, temp_dataset_file):
        """Test load_optimized_dataset utility function"""
        dataset = load_optimized_dataset(temp_dataset_file)
        
        assert isinstance(dataset, OptimizedLatentsDataset)
        assert len(dataset) == 3
    
    def test_create_dataset_from_results(self, mock_io_utils, sample_optimization_config):
        """Test create_dataset_from_results utility function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir) / "processed"
            processed_dir.mkdir()
            output_path = Path(temp_dir) / "dataset.pt"
            
            with patch.object(DatasetBuilder, 'create_pytorch_dataset') as mock_create:
                mock_create.return_value = str(output_path)
                
                result_path = create_dataset_from_results(
                    processed_dir, output_path, sample_optimization_config
                )
            
            # Verify DatasetBuilder.create_pytorch_dataset was called
            mock_create.assert_called_once_with(
                processed_dir, output_path, sample_optimization_config
            )
            
            assert result_path == str(output_path)


class TestPyTorchDatasetIntegration:
    """Integration tests for PyTorch dataset functionality"""
    
    def test_dataset_round_trip(self, mock_io_utils, sample_optimization_config):
        """Test creating and loading dataset round trip"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal processed data structure
            processed_dir = Path(temp_dir) / "processed"
            train_dir = processed_dir / "train"
            train_dir.mkdir(parents=True)
            
            # Create results and latent data
            results = [
                {
                    'image_name': 'test_img',
                    'initial_psnr': 20.0,
                    'final_psnr': 25.0,
                    'psnr_improvement': 5.0,
                    'initial_ssim': 0.7,
                    'final_ssim': 0.85,
                    'ssim_improvement': 0.15,
                    'loss_reduction': 80.0,
                    'optimization_iterations': 50,
                    'convergence_iteration': 45,
                    'initial_loss': 1.0,
                    'final_loss': 0.2
                }
            ]
            
            # Create actual results JSON file to ensure it exists
            results_file = train_dir / 'detailed_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f)
            
            img_dir = train_dir / 'test_img'
            img_dir.mkdir()
            
            initial_latents = torch.randn(1, 4, 64, 64)
            optimized_latents = torch.randn(1, 4, 64, 64)
            torch.save(initial_latents, img_dir / "test_img_initial_latents.pt")
            torch.save(optimized_latents, img_dir / "test_img_optimized_latents.pt")
            
            # Create empty val and test directories to avoid warnings
            val_dir = processed_dir / "val"
            test_dir = processed_dir / "test"
            val_dir.mkdir()
            test_dir.mkdir()
            
            mock_io = mock_io_utils.return_value
            mock_io.load_json.return_value = results
            
            output_path = Path(temp_dir) / "dataset.pt"
            
            # Create dataset
            builder = DatasetBuilder()
            
            with patch('generative_latent_optimization.dataset.pytorch_dataset.time.strftime',
                      return_value="2024-01-01 12:00:00"):
                
                created_path = builder.create_pytorch_dataset(
                    processed_dir, output_path, sample_optimization_config
                )
            
            # Load and verify dataset
            dataset = OptimizedLatentsDataset(created_path)
            
            assert len(dataset) == 1
            sample = dataset[0]
            assert sample['image_name'] == 'test_img'
            assert sample['split'] == 'train'
            assert torch.equal(sample['initial_latents'], initial_latents)
            assert torch.equal(sample['optimized_latents'], optimized_latents)
    
    def test_dataloader_integration(self, temp_dataset_file):
        """Test DataLoader integration with dataset"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        dataloader = dataset.create_dataloader(batch_size=2, shuffle=False)
        
        # Process all batches
        total_samples = 0
        for batch in dataloader:
            batch_size = len(batch['image_name'])
            total_samples += batch_size
            
            # Verify batch structure
            assert isinstance(batch['image_name'], list)
            assert isinstance(batch['initial_latents'], torch.Tensor)
            assert isinstance(batch['optimized_latents'], torch.Tensor)
            assert batch['initial_latents'].shape[0] == batch_size
            assert batch['optimized_latents'].shape[0] == batch_size
        
        # Verify all samples were processed
        assert total_samples == len(dataset)
    
    def test_split_specific_dataloader(self, temp_dataset_file):
        """Test split-specific dataloader functionality"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        # Test train split dataloader
        train_dataloader = dataset.create_dataloader(split='train', batch_size=1, shuffle=False)
        train_batches = list(train_dataloader)
        
        assert len(train_batches) == 2  # 2 train samples
        
        for batch in train_batches:
            assert batch['split'][0] == 'train'
    
    def test_error_handling_in_dataset_loading(self):
        """Test error handling when dataset file is corrupted"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create corrupted dataset file
            corrupted_path = Path(temp_dir) / "corrupted.pt"
            with open(corrupted_path, 'w') as f:
                f.write("not a valid pytorch file")
            
            # Should raise an appropriate exception
            with pytest.raises((RuntimeError, pickle.UnpicklingError, EOFError)):
                OptimizedLatentsDataset(corrupted_path)
    
    def test_metadata_consistency(self, temp_dataset_file):
        """Test that metadata remains consistent throughout operations"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        # Get original metadata
        original_metadata = dataset.get_metadata()
        
        # Perform various operations
        dataset.get_by_split('train')
        dataset.get_statistics()
        dataset.create_dataloader()
        
        # Metadata should remain unchanged
        current_metadata = dataset.get_metadata()
        assert current_metadata.total_samples == original_metadata.total_samples
        assert current_metadata.splits_count == original_metadata.splits_count


class TestPyTorchDatasetErrorCases:
    """Test error cases and edge conditions"""
    
    def test_dataset_with_missing_metadata_fields(self):
        """Test dataset loading with incomplete metadata"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "incomplete_dataset.pt"
            
            # Create dataset with minimal metadata
            incomplete_data = {
                'samples': [],
                'metadata': {
                    'total_samples': 0,
                    'splits_count': {},
                    'optimization_config': {},
                    'processing_statistics': {},
                    'creation_timestamp': "2024-01-01 12:00:00"
                    # Missing dataset_version and source_dataset (should use defaults)
                }
            }
            
            torch.save(incomplete_data, dataset_path)
            
            # Should load successfully with defaults
            dataset = OptimizedLatentsDataset(dataset_path)
            metadata = dataset.get_metadata()
            
            assert metadata.dataset_version == "1.0"
            assert metadata.source_dataset == "BSDS500"
    
    def test_save_subset_directory_creation(self, temp_dataset_file):
        """Test that save_subset creates parent directories"""
        dataset = OptimizedLatentsDataset(temp_dataset_file)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Deep directory path that doesn't exist
            output_path = Path(temp_dir) / "deep" / "nested" / "path" / "subset.pt"
            
            result_path = dataset.save_subset(output_path, max_samples=1)
            
            assert output_path.exists()
            assert result_path == str(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])