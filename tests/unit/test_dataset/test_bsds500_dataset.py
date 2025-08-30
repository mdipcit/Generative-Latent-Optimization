#!/usr/bin/env python3
"""
Unit tests for BSDS500Dataset functionality
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from generative_latent_optimization.dataset.bsds500_dataset import BSDS500Dataset, BSDS500DataLoader
from ...fixtures.dataset_mocks import mock_dataset_dependencies
from ...fixtures.assertion_helpers import assert_float_approximately_equal


@pytest.fixture
def mock_vae_toolkit():
    """Mock vae-toolkit functionality"""
    mock_load_and_preprocess = Mock()
    mock_load_and_preprocess.return_value = (
        torch.randn(1, 3, 256, 256),  # Mock image tensor
        Mock()  # Mock PIL image
    )
    
    # Patch the function where it's imported and used
    with patch('generative_latent_optimization.dataset.bsds500_dataset.load_and_preprocess_image', 
               mock_load_and_preprocess):
        with patch('generative_latent_optimization.dataset.bsds500_dataset.VAE_TOOLKIT_AVAILABLE', True):
            yield mock_load_and_preprocess


@pytest.fixture
def temp_bsds500_structure():
    """Create temporary BSDS500-like directory structure"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create split directories
        train_dir = temp_path / 'train'
        val_dir = temp_path / 'val'
        test_dir = temp_path / 'test'
        
        train_dir.mkdir()
        val_dir.mkdir()
        test_dir.mkdir()
        
        # Create dummy JPG files
        for i in range(5):
            (train_dir / f'train_image_{i:03d}.jpg').touch()
            (val_dir / f'val_image_{i:03d}.jpg').touch()
            (test_dir / f'test_image_{i:03d}.jpg').touch()
        
        yield temp_path


class TestBSDS500Dataset:
    """Test cases for BSDS500Dataset class"""
    
    def test_initialization_with_explicit_path(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test dataset initialization with explicit path"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train',
            target_size=256,
            cache_in_memory=False
        )
        
        assert dataset.bsds500_path == temp_bsds500_structure
        assert dataset.split == 'train'
        assert dataset.target_size == 256
        assert dataset.cache_in_memory == False
        assert len(dataset) == 5
    
    def test_initialization_with_environment_variable(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test dataset initialization using environment variable"""
        with patch.dict(os.environ, {'BSDS500_PATH': str(temp_bsds500_structure)}):
            dataset = BSDS500Dataset(split='val', target_size=512)
            
            assert dataset.bsds500_path == temp_bsds500_structure
            assert dataset.split == 'val'
            assert dataset.target_size == 512
            assert len(dataset) == 5
    
    def test_initialization_without_path_raises_error(self, mock_vae_toolkit):
        """Test that missing path raises ValueError"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="BSDS500 path not provided"):
                BSDS500Dataset()
    
    def test_initialization_without_vae_toolkit_raises_error(self, temp_bsds500_structure):
        """Test that missing vae-toolkit raises ImportError"""
        with patch('generative_latent_optimization.dataset.bsds500_dataset.VAE_TOOLKIT_AVAILABLE', False):
            with pytest.raises(ImportError, match="vae-toolkit is required"):
                BSDS500Dataset(bsds500_path=temp_bsds500_structure)
    
    def test_invalid_directory_raises_error(self, mock_vae_toolkit):
        """Test that invalid directory raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            BSDS500Dataset(bsds500_path='/nonexistent/path')
    
    def test_empty_directory_raises_error(self, mock_vae_toolkit):
        """Test that empty directory raises ValueError"""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_path = Path(temp_dir)
            train_dir = empty_path / 'train'
            train_dir.mkdir()
            
            with pytest.raises(ValueError, match="No images found"):
                BSDS500Dataset(bsds500_path=empty_path, split='train')
    
    @pytest.mark.parametrize("split", ['train', 'val', 'test'])
    def test_different_splits(self, mock_vae_toolkit, temp_bsds500_structure, split):
        """Test dataset initialization with different splits"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split=split
        )
        
        assert dataset.split == split
        assert len(dataset) == 5
    
    @pytest.mark.parametrize("target_size", [128, 256, 512])
    def test_different_target_sizes(self, mock_vae_toolkit, temp_bsds500_structure, target_size):
        """Test dataset initialization with different target sizes"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train',
            target_size=target_size
        )
        
        assert dataset.target_size == target_size
    
    def test_getitem_returns_correct_format(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test __getitem__ returns correct tensor and metadata format"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        # Configure mock to return specific tensor shape
        mock_vae_toolkit.return_value = (
            torch.randn(1, 3, 256, 256),
            Mock()
        )
        
        image_tensor, metadata = dataset[0]
        
        # Verify tensor properties
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (1, 3, 256, 256)
        
        # Verify metadata
        assert isinstance(metadata, dict)
        assert 'name' in metadata
        assert 'original_path' in metadata
        assert 'target_size' in metadata
        assert 'split' in metadata
        assert metadata['target_size'] == 256
        assert metadata['split'] == 'train'
    
    def test_getitem_out_of_range_raises_error(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test __getitem__ with out-of-range index raises IndexError"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        with pytest.raises(IndexError, match="Index .* out of range"):
            dataset[10]  # Only 5 images available
    
    def test_cache_functionality(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test in-memory caching functionality"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train',
            cache_in_memory=True
        )
        
        # First access - should call vae-toolkit
        image1, metadata1 = dataset[0]
        call_count_first = mock_vae_toolkit.call_count
        
        # Second access - should use cache
        image2, metadata2 = dataset[0]
        call_count_second = mock_vae_toolkit.call_count
        
        # Verify cache was used (no additional calls to vae-toolkit)
        assert call_count_second == call_count_first
        
        # Verify same data returned
        assert torch.equal(image1, image2)
        assert metadata1 == metadata2
    
    def test_clear_cache(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test cache clearing functionality"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train',
            cache_in_memory=True
        )
        
        # Load some data to populate cache
        dataset[0]
        assert len(dataset.data_cache) == 1
        
        # Clear cache
        dataset.clear_cache()
        assert len(dataset.data_cache) == 0
    
    def test_clear_cache_without_caching(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test clear_cache when caching is disabled"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train',
            cache_in_memory=False
        )
        
        # Should not raise error
        dataset.clear_cache()
        assert dataset.data_cache is None
    
    def test_get_image_path(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test get_image_path method"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        path = dataset.get_image_path(0)
        assert isinstance(path, str)
        assert path.endswith('.jpg')
        assert 'train_image_000.jpg' in path
    
    def test_get_image_path_out_of_range(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test get_image_path with out-of-range index"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        with pytest.raises(IndexError, match="Index .* out of range"):
            dataset.get_image_path(10)
    
    def test_get_image_name(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test get_image_name method"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        name = dataset.get_image_name(0)
        assert isinstance(name, str)
        assert name == 'train_image_000'
    
    def test_get_image_name_out_of_range(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test get_image_name with out-of-range index"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        with pytest.raises(IndexError, match="Index .* out of range"):
            dataset.get_image_name(10)
    
    def test_get_batch(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test batch data retrieval"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        # Configure mock for consistent tensor shapes
        mock_vae_toolkit.return_value = (
            torch.randn(1, 3, 256, 256),
            Mock()
        )
        
        # Test batch retrieval
        batch_indices = [0, 1, 2]
        batch_tensor, batch_metadata = dataset.get_batch(batch_indices)
        
        # Verify batch tensor properties
        assert isinstance(batch_tensor, torch.Tensor)
        assert batch_tensor.shape == (3, 3, 256, 256)  # [batch_size, channels, height, width]
        
        # Verify batch metadata
        assert isinstance(batch_metadata, list)
        assert len(batch_metadata) == 3
        for metadata in batch_metadata:
            assert isinstance(metadata, dict)
            assert 'name' in metadata
    
    def test_get_batch_with_out_of_range_index(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test get_batch with out-of-range index in batch"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        with pytest.raises(IndexError):
            dataset.get_batch([0, 1, 10])  # Index 10 is out of range
    
    def test_file_list_loading_and_sorting(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test that file list is loaded and sorted correctly"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        # Verify file list properties
        assert len(dataset.file_list) == 5
        
        # Verify sorting (should be alphabetical)
        names = [info['name'] for info in dataset.file_list]
        assert names == sorted(names)
        
        # Verify file structure
        for file_info in dataset.file_list:
            assert 'image_path' in file_info
            assert 'name' in file_info
            assert isinstance(file_info['image_path'], Path)
            assert file_info['image_path'].suffix == '.jpg'


class TestBSDS500DataLoader:
    """Test cases for BSDS500DataLoader class"""
    
    def test_dataloader_initialization(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test dataloader initialization"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        dataloader = BSDS500DataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=True
        )
        
        assert dataloader.dataset == dataset
        assert dataloader.batch_size == 2
        assert dataloader.shuffle == True
        assert len(dataloader.indices) == 5
    
    def test_dataloader_length(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test dataloader __len__ method"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        # Test different batch sizes
        dataloader1 = BSDS500DataLoader(dataset, batch_size=2)
        assert len(dataloader1) == 3  # ceil(5/2) = 3
        
        dataloader2 = BSDS500DataLoader(dataset, batch_size=3)
        assert len(dataloader2) == 2  # ceil(5/3) = 2
        
        dataloader3 = BSDS500DataLoader(dataset, batch_size=10)
        assert len(dataloader3) == 1  # ceil(5/10) = 1
    
    def test_dataloader_iteration(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test dataloader iteration"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        dataloader = BSDS500DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Configure mock for consistent returns
        mock_vae_toolkit.return_value = (
            torch.randn(1, 3, 256, 256),
            Mock()
        )
        
        batches = list(dataloader)
        
        # Should have 3 batches: [2, 2, 1] images
        assert len(batches) == 3
        
        # Check first batch
        batch_tensor_0, batch_metadata_0 = batches[0]
        assert batch_tensor_0.shape[0] == 2  # batch size 2
        assert len(batch_metadata_0) == 2
        
        # Check last batch (partial)
        batch_tensor_2, batch_metadata_2 = batches[2]
        assert batch_tensor_2.shape[0] == 1  # remaining 1 image
        assert len(batch_metadata_2) == 1
    
    def test_dataloader_shuffle_functionality(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test that shuffle actually changes order"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        # Create two dataloaders with same random seed for comparison
        with patch('random.shuffle') as mock_shuffle:
            dataloader_shuffled = BSDS500DataLoader(dataset, batch_size=1, shuffle=True)
            mock_shuffle.assert_called()
        
        # Create dataloader without shuffle
        dataloader_no_shuffle = BSDS500DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Verify indices are sequential for non-shuffled
        assert dataloader_no_shuffle.indices == [0, 1, 2, 3, 4]
    
    def test_dataloader_iteration_reset(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test that iterator resets properly"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        dataloader = BSDS500DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Configure mock
        mock_vae_toolkit.return_value = (
            torch.randn(1, 3, 256, 256),
            Mock()
        )
        
        # First iteration
        batches1 = list(dataloader)
        assert len(batches1) == 3
        
        # Second iteration should work
        batches2 = list(dataloader)
        assert len(batches2) == 3
    
    def test_dataloader_stop_iteration(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test StopIteration behavior"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        dataloader = BSDS500DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Configure mock
        mock_vae_toolkit.return_value = (
            torch.randn(1, 3, 256, 256),
            Mock()
        )
        
        # Manually iterate
        iterator = iter(dataloader)
        
        # Should get 3 batches
        next(iterator)  # batch 1
        next(iterator)  # batch 2
        next(iterator)  # batch 3
        
        # Next call should raise StopIteration
        with pytest.raises(StopIteration):
            next(iterator)


class TestBSDS500Integration:
    """Integration tests for BSDS500 dataset functionality"""
    
    def test_dataset_and_dataloader_integration(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test integration between dataset and dataloader"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train',
            cache_in_memory=True
        )
        dataloader = BSDS500DataLoader(dataset, batch_size=2)
        
        # Configure mock
        mock_vae_toolkit.return_value = (
            torch.randn(1, 3, 256, 256),
            Mock()
        )
        
        # Process all batches
        total_samples = 0
        for batch_tensor, batch_metadata in dataloader:
            total_samples += batch_tensor.shape[0]
            
            # Verify batch properties
            assert batch_tensor.dim() == 4  # [B, C, H, W]
            assert batch_tensor.shape[1:] == (3, 256, 256)
            assert len(batch_metadata) == batch_tensor.shape[0]
        
        # Verify all samples were processed
        assert total_samples == len(dataset)
    
    def test_edge_case_single_image_batch(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test edge case with batch size larger than dataset"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        dataloader = BSDS500DataLoader(dataset, batch_size=10)  # Larger than dataset size
        
        # Configure mock
        mock_vae_toolkit.return_value = (
            torch.randn(1, 3, 256, 256),
            Mock()
        )
        
        batches = list(dataloader)
        
        # Should have exactly 1 batch with all 5 images
        assert len(batches) == 1
        batch_tensor, batch_metadata = batches[0]
        assert batch_tensor.shape[0] == 5
        assert len(batch_metadata) == 5
    
    def test_reproducibility_without_shuffle(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test that results are reproducible when shuffle=False"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        # Create two identical dataloaders
        dataloader1 = BSDS500DataLoader(dataset, batch_size=2, shuffle=False)
        dataloader2 = BSDS500DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Verify indices are identical
        assert dataloader1.indices == dataloader2.indices
    
    def test_vae_toolkit_integration_call_pattern(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test that vae-toolkit is called with correct parameters"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train',
            target_size=128
        )
        
        # Access first item
        dataset[0]
        
        # Verify vae-toolkit was called correctly
        mock_vae_toolkit.assert_called_once()
        call_args = mock_vae_toolkit.call_args
        
        # Check arguments
        image_path_arg = call_args[0][0]
        target_size_arg = call_args[1]['target_size']
        
        assert str(temp_bsds500_structure / 'train') in image_path_arg
        assert '.jpg' in image_path_arg
        assert target_size_arg == 128
    
    def test_error_handling_in_vae_toolkit_call(self, mock_vae_toolkit, temp_bsds500_structure):
        """Test error handling when vae-toolkit fails"""
        dataset = BSDS500Dataset(
            bsds500_path=temp_bsds500_structure,
            split='train'
        )
        
        # Configure mock to raise exception
        mock_vae_toolkit.side_effect = Exception("Mock vae-toolkit error")
        
        # Should propagate the exception
        with pytest.raises(Exception, match="Mock vae-toolkit error"):
            dataset[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])