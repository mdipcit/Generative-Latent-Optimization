#!/usr/bin/env python3
"""
Unit tests for BatchProcessor functionality
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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from generative_latent_optimization.dataset.batch_processor import (
    BatchProcessor, BatchProcessingConfig, ProcessingResults, ProcessingCheckpoint
)
from generative_latent_optimization.optimization import OptimizationConfig
from ...fixtures.dataset_mocks import mock_dataset_dependencies, mock_batch_processing
from ...fixtures.assertion_helpers import (
    assert_float_approximately_equal, safe_calculate_median
)


@pytest.fixture
def basic_batch_config():
    """Basic batch processing configuration"""
    return BatchProcessingConfig(
        batch_size=4,
        num_workers=2,
        checkpoint_dir="./test_checkpoints",
        resume_from_checkpoint=True,
        save_visualizations=False,
        max_images=10
    )


@pytest.fixture
def basic_optimization_config():
    """Basic optimization configuration"""
    return OptimizationConfig(
        iterations=50,
        learning_rate=0.1,
        device='cpu'
    )


@pytest.fixture
def temp_input_directory():
    """Create temporary input directory with mock images"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock image files
        for i in range(8):
            (temp_path / f'image_{i:03d}.jpg').touch()
            (temp_path / f'image_{i:03d}.png').touch()
        
        # Create some files with different extensions
        (temp_path / 'document.txt').touch()  # Should be ignored
        (temp_path / 'readme.md').touch()     # Should be ignored
        
        yield temp_path


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies"""
    with patch.multiple(
        'generative_latent_optimization.dataset.batch_processor',
        VAELoader=Mock(),
    ):
        # Mock VAE model
        mock_vae = Mock()
        mock_vae.encode.return_value.latent_dist.mode.return_value = torch.randn(1, 4, 64, 64)
        mock_vae.decode.return_value.sample = torch.randn(1, 3, 512, 512)
        
        # Mock optimization result
        mock_opt_result = Mock()
        mock_opt_result.optimized_latents = torch.randn(1, 4, 64, 64)
        mock_opt_result.losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        mock_opt_result.metrics = {
            'loss_reduction_percent': 80.0,
            'psnr_improvement': 5.0,
            'ssim_improvement': 0.15,
            'loss_reduction': 80.0,
            'final_psnr': 25.5,
            'initial_psnr': 20.0,
            'final_ssim': 0.85,
            'initial_ssim': 0.70
        }
        mock_opt_result.convergence_iteration = 4
        mock_opt_result.initial_loss = 1.0
        mock_opt_result.final_loss = 0.2
        
        yield {
            'mock_vae': mock_vae,
            'mock_opt_result': mock_opt_result
        }


class TestBatchProcessingConfig:
    """Test cases for BatchProcessingConfig dataclass"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = BatchProcessingConfig()
        
        assert config.batch_size == 8
        assert config.num_workers == 4
        assert config.checkpoint_dir == "./checkpoints"
        assert config.resume_from_checkpoint == True
        assert config.save_visualizations == True
        assert config.max_images is None
        assert config.image_extensions == ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = BatchProcessingConfig(
            batch_size=16,
            num_workers=8,
            checkpoint_dir="./custom_checkpoints",
            resume_from_checkpoint=False,
            save_visualizations=False,
            max_images=100,
            image_extensions=['.png', '.jpg']
        )
        
        assert config.batch_size == 16
        assert config.num_workers == 8
        assert config.checkpoint_dir == "./custom_checkpoints"
        assert config.resume_from_checkpoint == False
        assert config.save_visualizations == False
        assert config.max_images == 100
        assert config.image_extensions == ['.png', '.jpg']
    
    def test_post_init_image_extensions(self):
        """Test that post_init sets default image extensions"""
        config = BatchProcessingConfig(image_extensions=None)
        assert config.image_extensions == ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']


class TestProcessingResults:
    """Test cases for ProcessingResults dataclass"""
    
    def test_processing_results_creation(self):
        """Test ProcessingResults creation and access"""
        results = ProcessingResults(
            total_processed=100,
            successful_optimizations=95,
            failed_optimizations=5,
            average_psnr_improvement=3.5,
            average_loss_reduction=75.0,
            processing_time_seconds=3600.0,
            output_directory="/path/to/output",
            checkpoint_path="/path/to/checkpoint.pkl"
        )
        
        assert results.total_processed == 100
        assert results.successful_optimizations == 95
        assert results.failed_optimizations == 5
        assert_float_approximately_equal(results.average_psnr_improvement, 3.5)
        assert_float_approximately_equal(results.average_loss_reduction, 75.0)
        assert_float_approximately_equal(results.processing_time_seconds, 3600.0)
        assert results.output_directory == "/path/to/output"
        assert results.checkpoint_path == "/path/to/checkpoint.pkl"


class TestProcessingCheckpoint:
    """Test cases for ProcessingCheckpoint dataclass"""
    
    def test_checkpoint_creation(self, basic_batch_config, basic_optimization_config):
        """Test ProcessingCheckpoint creation"""
        checkpoint = ProcessingCheckpoint(
            processed_files=['file1.jpg', 'file2.jpg'],
            config=basic_batch_config,
            optimization_config=basic_optimization_config,
            last_processed_index=1,
            results_so_far=[{'result': 'data'}],
            timestamp="2024-01-01 12:00:00"
        )
        
        assert len(checkpoint.processed_files) == 2
        assert checkpoint.config == basic_batch_config
        assert checkpoint.optimization_config == basic_optimization_config
        assert checkpoint.last_processed_index == 1
        assert len(checkpoint.results_so_far) == 1
        assert checkpoint.timestamp == "2024-01-01 12:00:00"


class TestBatchProcessor:
    """Test cases for BatchProcessor class"""
    
    def test_initialization(self, basic_batch_config):
        """Test BatchProcessor initialization"""
        with patch('generative_latent_optimization.dataset.batch_processor.IOUtils'), \
             patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
            
            processor = BatchProcessor(basic_batch_config)
            
            assert processor.config == basic_batch_config
            assert processor.checkpoint_dir == Path("./test_checkpoints").resolve()
    
    def test_get_image_files(self, basic_batch_config, temp_input_directory):
        """Test _get_image_files method"""
        with patch('generative_latent_optimization.dataset.batch_processor.IOUtils'), \
             patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
            
            processor = BatchProcessor(basic_batch_config)
            image_files = processor._get_image_files(temp_input_directory)
            
            # Should find 16 image files (8 JPG + 8 PNG), ignore text files
            assert len(image_files) == 16
            
            # Verify all are valid image extensions
            for file_path in image_files:
                assert file_path.suffix.lower() in ['.jpg', '.png', '.jpeg']
            
            # Verify sorting
            assert image_files == sorted(image_files)
    
    def test_get_image_files_max_images_limit(self, basic_batch_config, temp_input_directory):
        """Test max_images limitation"""
        basic_batch_config.max_images = 5
        
        with patch('generative_latent_optimization.dataset.batch_processor.IOUtils'), \
             patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
            
            processor = BatchProcessor(basic_batch_config)
            image_files = processor._get_image_files(temp_input_directory)
            
            # Should return all files, max_images is applied later in process_directory
            assert len(image_files) == 16
    
    def test_calculate_std(self, basic_batch_config):
        """Test _calculate_std helper method"""
        with patch('generative_latent_optimization.dataset.batch_processor.IOUtils'), \
             patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
            
            processor = BatchProcessor(basic_batch_config)
            
            # Test with normal values
            std = processor._calculate_std([1.0, 2.0, 3.0, 4.0, 5.0])
            assert std > 0
            
            # Test with empty list
            std_empty = processor._calculate_std([])
            assert_float_approximately_equal(std_empty, 0.0)
            
            # Test with single value
            std_single = processor._calculate_std([5.0])
            assert_float_approximately_equal(std_single, 0.0)
    
    def test_combine_processing_results(self, basic_batch_config):
        """Test _combine_processing_results method"""
        with patch('generative_latent_optimization.dataset.batch_processor.IOUtils'), \
             patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
            
            processor = BatchProcessor(basic_batch_config)
            
            # Create test results
            result1 = ProcessingResults(
                total_processed=50,
                successful_optimizations=45,
                failed_optimizations=5,
                average_psnr_improvement=3.0,
                average_loss_reduction=70.0,
                processing_time_seconds=1800.0,
                output_directory="dir1"
            )
            
            result2 = ProcessingResults(
                total_processed=30,
                successful_optimizations=25,
                failed_optimizations=5,
                average_psnr_improvement=4.0,
                average_loss_reduction=80.0,
                processing_time_seconds=1200.0,
                output_directory="dir2"
            )
            
            combined = processor._combine_processing_results([result1, result2])
            
            assert combined.total_processed == 80
            assert combined.successful_optimizations == 70
            assert combined.failed_optimizations == 10
            assert_float_approximately_equal(combined.processing_time_seconds, 3000.0)
            assert combined.output_directory == "combined_results"
            
            # Verify weighted averages
            expected_psnr = (3.0 * 45 + 4.0 * 25) / 70
            expected_loss = (70.0 * 45 + 80.0 * 25) / 70
            assert abs(combined.average_psnr_improvement - expected_psnr) < 0.01
            assert abs(combined.average_loss_reduction - expected_loss) < 0.01
    
    def test_combine_processing_results_with_no_successful(self, basic_batch_config):
        """Test combining results when no successful optimizations"""
        with patch('generative_latent_optimization.dataset.batch_processor.IOUtils'), \
             patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
            
            processor = BatchProcessor(basic_batch_config)
            
            result = ProcessingResults(
                total_processed=10,
                successful_optimizations=0,
                failed_optimizations=10,
                average_psnr_improvement=0.0,
                average_loss_reduction=0.0,
                processing_time_seconds=600.0,
                output_directory="dir1"
            )
            
            combined = processor._combine_processing_results([result])
            
            assert combined.successful_optimizations == 0
            assert_float_approximately_equal(combined.average_psnr_improvement, 0.0)
            assert_float_approximately_equal(combined.average_loss_reduction, 0.0)
    
    @patch('generative_latent_optimization.dataset.batch_processor.time.strftime')
    def test_save_checkpoint(self, mock_strftime, basic_batch_config, basic_optimization_config):
        """Test checkpoint saving functionality"""
        mock_strftime.return_value = "2024-01-01 12:00:00"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            basic_batch_config.checkpoint_dir = temp_dir
            
            with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
                 patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
                
                processor = BatchProcessor(basic_batch_config)
                
                checkpoint_path = Path(temp_dir) / "test_checkpoint.pkl"
                processed_files = ['file1.jpg', 'file2.jpg']
                results_so_far = [{'result': 'data1'}, {'result': 'data2'}]
                
                processor._save_checkpoint(
                    checkpoint_path, processed_files, basic_optimization_config, 1, results_so_far
                )
                
                # Verify IOUtils.save_pickle was called
                mock_io.return_value.save_pickle.assert_called_once()
                call_args = mock_io.return_value.save_pickle.call_args
                
                checkpoint_obj = call_args[0][0]
                assert isinstance(checkpoint_obj, ProcessingCheckpoint)
                assert checkpoint_obj.processed_files == processed_files
                assert checkpoint_obj.last_processed_index == 1
                assert checkpoint_obj.timestamp == "2024-01-01 12:00:00"
    
    def test_load_checkpoint_not_exists(self, basic_batch_config, basic_optimization_config):
        """Test loading checkpoint when file doesn't exist"""
        with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
             patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
            
            processor = BatchProcessor(basic_batch_config)
            
            nonexistent_path = Path("/nonexistent/checkpoint.pkl")
            image_files = [Path("img1.jpg"), Path("img2.jpg")]
            
            processed, results, start_idx = processor._load_checkpoint(
                nonexistent_path, image_files, basic_optimization_config
            )
            
            assert processed == []
            assert results == []
            assert start_idx == 0
    
    def test_load_checkpoint_disabled(self, basic_batch_config, basic_optimization_config):
        """Test loading checkpoint when resume is disabled"""
        basic_batch_config.resume_from_checkpoint = False
        
        with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
             patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
            
            processor = BatchProcessor(basic_batch_config)
            
            # Even if checkpoint exists, should not load
            checkpoint_path = Path("existing_checkpoint.pkl")
            image_files = [Path("img1.jpg")]
            
            processed, results, start_idx = processor._load_checkpoint(
                checkpoint_path, image_files, basic_optimization_config
            )
            
            assert processed == []
            assert results == []
            assert start_idx == 0
    
    def test_load_checkpoint_valid(self, basic_batch_config, basic_optimization_config):
        """Test loading valid checkpoint"""
        with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
             patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
            
            processor = BatchProcessor(basic_batch_config)
            
            # Mock checkpoint data
            mock_checkpoint = ProcessingCheckpoint(
                processed_files=['file1.jpg', 'file2.jpg'],
                config=basic_batch_config,
                optimization_config=basic_optimization_config,
                last_processed_index=1,
                results_so_far=[{'result': 'data1'}, {'result': 'data2'}],
                timestamp="2024-01-01 12:00:00"
            )
            
            mock_io.return_value.load_pickle.return_value = mock_checkpoint
            
            checkpoint_path = Path("checkpoint.pkl")
            image_files = [Path("img1.jpg"), Path("img2.jpg")]
            
            with patch('pathlib.Path.exists', return_value=True):
                processed, results, start_idx = processor._load_checkpoint(
                    checkpoint_path, image_files, basic_optimization_config
                )
            
            assert processed == ['file1.jpg', 'file2.jpg']
            assert len(results) == 2
            assert start_idx == 2  # last_processed_index + 1
    
    def test_load_checkpoint_config_mismatch(self, basic_batch_config, basic_optimization_config):
        """Test loading checkpoint with mismatched configuration"""
        with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
             patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
            
            processor = BatchProcessor(basic_batch_config)
            
            # Create mismatched config
            mismatched_config = BatchProcessingConfig(batch_size=16)  # Different batch size
            mock_checkpoint = ProcessingCheckpoint(
                processed_files=['file1.jpg'],
                config=mismatched_config,
                optimization_config=basic_optimization_config,
                last_processed_index=0,
                results_so_far=[],
                timestamp="2024-01-01 12:00:00"
            )
            
            mock_io.return_value.load_pickle.return_value = mock_checkpoint
            
            checkpoint_path = Path("checkpoint.pkl")
            image_files = [Path("img1.jpg")]
            
            with patch('pathlib.Path.exists', return_value=True):
                processed, results, start_idx = processor._load_checkpoint(
                    checkpoint_path, image_files, basic_optimization_config
                )
            
            # Should return empty results due to config mismatch
            assert processed == []
            assert results == []
            assert start_idx == 0
    
    def test_load_checkpoint_exception_handling(self, basic_batch_config, basic_optimization_config):
        """Test checkpoint loading exception handling"""
        with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
             patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
            
            processor = BatchProcessor(basic_batch_config)
            
            # Mock IOUtils to raise exception
            mock_io.return_value.load_pickle.side_effect = Exception("Corrupted checkpoint")
            
            checkpoint_path = Path("checkpoint.pkl")
            image_files = [Path("img1.jpg")]
            
            with patch('pathlib.Path.exists', return_value=True):
                processed, results, start_idx = processor._load_checkpoint(
                    checkpoint_path, image_files, basic_optimization_config
                )
            
            # Should handle exception gracefully
            assert processed == []
            assert results == []
            assert start_idx == 0
    
    @patch('generative_latent_optimization.dataset.batch_processor.time.strftime')
    def test_save_processing_summary(self, mock_strftime, basic_batch_config):
        """Test processing summary saving"""
        mock_strftime.return_value = "2024-01-01 12:00:00"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
                 patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
                
                processor = BatchProcessor(basic_batch_config)
                
                # Mock results data
                results = [
                    {'psnr_improvement': 3.0, 'ssim_improvement': 0.1, 'loss_reduction': 80.0},
                    {'psnr_improvement': 4.0, 'ssim_improvement': 0.2, 'loss_reduction': 85.0}
                ]
                
                processor._save_processing_summary(output_dir, results, 3600.0)
                
                # Verify JSON saving was called twice (summary + detailed results)
                assert mock_io.return_value.save_json.call_count == 2
                
                # Check summary data structure
                summary_call = mock_io.return_value.save_json.call_args_list[0]
                summary_data = summary_call[0][0]
                
                assert summary_data['total_images'] == 2
                assert_float_approximately_equal(summary_data['processing_time_seconds'], 3600.0)
                assert_float_approximately_equal(summary_data['processing_time_hours'], 1.0)
                assert_float_approximately_equal(summary_data['average_psnr_improvement'], 3.5)
                assert_float_approximately_equal(summary_data['average_ssim_improvement'], 0.15)
                assert_float_approximately_equal(summary_data['average_loss_reduction'], 82.5)
                assert summary_data['timestamp'] == "2024-01-01 12:00:00"
    
    def test_save_processing_summary_empty_results(self, basic_batch_config):
        """Test processing summary with empty results"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
                 patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'):
                
                processor = BatchProcessor(basic_batch_config)
                
                processor._save_processing_summary(output_dir, [], 100.0)
                
                # Check that averages are 0 for empty results
                summary_call = mock_io.return_value.save_json.call_args_list[0]
                summary_data = summary_call[0][0]
                
                assert summary_data['total_images'] == 0
                assert_float_approximately_equal(summary_data['average_psnr_improvement'], 0)
                assert_float_approximately_equal(summary_data['average_ssim_improvement'], 0)
                assert_float_approximately_equal(summary_data['average_loss_reduction'], 0)
    
    def test_process_bsds500_dataset_single_split(self, basic_batch_config, basic_optimization_config):
        """Test process_bsds500_dataset with single split"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create BSDS500-like structure
            bsds_path = Path(temp_dir) / "bsds500"
            train_dir = bsds_path / "train"
            train_dir.mkdir(parents=True)
            
            # Create dummy images
            for i in range(3):
                (train_dir / f'train_img_{i}.jpg').touch()
            
            output_dir = Path(temp_dir) / "output"
            
            with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
                 patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'), \
                 patch.object(BatchProcessor, 'process_directory') as mock_process_dir:
                
                # Mock process_directory return value
                mock_result = ProcessingResults(
                    total_processed=3,
                    successful_optimizations=3,
                    failed_optimizations=0,
                    average_psnr_improvement=3.5,
                    average_loss_reduction=75.0,
                    processing_time_seconds=300.0,
                    output_directory=str(output_dir / "train")
                )
                mock_process_dir.return_value = mock_result
                
                processor = BatchProcessor(basic_batch_config)
                result = processor.process_bsds500_dataset(
                    bsds_path, output_dir, basic_optimization_config, split="train"
                )
                
                # Verify process_directory was called
                mock_process_dir.assert_called_once()
                call_args = mock_process_dir.call_args[0]
                assert Path(call_args[0]) == train_dir
                assert Path(call_args[1]) == output_dir / "train"
                
                # Verify result
                assert result.total_processed == 3
                assert result.successful_optimizations == 3
    
    def test_process_bsds500_dataset_all_splits(self, basic_batch_config, basic_optimization_config):
        """Test process_bsds500_dataset with all splits"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create BSDS500-like structure
            bsds_path = Path(temp_dir) / "bsds500"
            for split in ["train", "val", "test"]:
                split_dir = bsds_path / split
                split_dir.mkdir(parents=True)
                (split_dir / f'{split}_img.jpg').touch()
            
            output_dir = Path(temp_dir) / "output"
            
            with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
                 patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'), \
                 patch.object(BatchProcessor, 'process_directory') as mock_process_dir, \
                 patch.object(BatchProcessor, '_combine_processing_results') as mock_combine:
                
                # Mock individual results for each split
                mock_result = ProcessingResults(
                    total_processed=1,
                    successful_optimizations=1,
                    failed_optimizations=0,
                    average_psnr_improvement=3.0,
                    average_loss_reduction=70.0,
                    processing_time_seconds=100.0,
                    output_directory="split_output"
                )
                mock_process_dir.return_value = mock_result
                
                # Mock combined result
                combined_result = ProcessingResults(
                    total_processed=3,
                    successful_optimizations=3,
                    failed_optimizations=0,
                    average_psnr_improvement=3.0,
                    average_loss_reduction=70.0,
                    processing_time_seconds=300.0,
                    output_directory="combined_output"
                )
                mock_combine.return_value = combined_result
                
                processor = BatchProcessor(basic_batch_config)
                result = processor.process_bsds500_dataset(
                    bsds_path, output_dir, basic_optimization_config, split="all"
                )
                
                # Verify process_directory was called 3 times (for each split)
                assert mock_process_dir.call_count == 3
                
                # Verify combine was called
                mock_combine.assert_called_once()
                
                # Verify result is combined result
                assert result.total_processed == 3
    
    def test_process_bsds500_dataset_missing_split(self, basic_batch_config, basic_optimization_config):
        """Test process_bsds500_dataset with missing split directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create BSDS500-like structure without test split
            bsds_path = Path(temp_dir) / "bsds500"
            train_dir = bsds_path / "train"
            train_dir.mkdir(parents=True)
            (train_dir / 'train_img.jpg').touch()
            
            output_dir = Path(temp_dir) / "output"
            
            with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
                 patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'), \
                 patch.object(BatchProcessor, 'process_directory') as mock_process_dir:
                
                mock_result = ProcessingResults(
                    total_processed=1,
                    successful_optimizations=1,
                    failed_optimizations=0,
                    average_psnr_improvement=3.0,
                    average_loss_reduction=70.0,
                    processing_time_seconds=100.0,
                    output_directory="train_output"
                )
                mock_process_dir.return_value = mock_result
                
                processor = BatchProcessor(basic_batch_config)
                result = processor.process_bsds500_dataset(
                    bsds_path, output_dir, basic_optimization_config, split="all"
                )
                
                # Should process only the existing split (train)
                assert mock_process_dir.call_count == 1
                assert result.total_processed == 1
    
    def test_process_bsds500_dataset_no_splits_found(self, basic_batch_config, basic_optimization_config):
        """Test process_bsds500_dataset when no splits exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            bsds_path = Path(temp_dir) / "empty_bsds500"
            bsds_path.mkdir()
            output_dir = Path(temp_dir) / "output"
            
            with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
                 patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'), \
                 patch.object(BatchProcessor, 'process_directory') as mock_process_dir:
                
                processor = BatchProcessor(basic_batch_config)
                result = processor.process_bsds500_dataset(
                    bsds_path, output_dir, basic_optimization_config, split="all"
                )
                
                # Should not call process_directory
                mock_process_dir.assert_not_called()
                
                # Should return empty result
                assert result.total_processed == 0
                assert result.successful_optimizations == 0
                assert result.failed_optimizations == 0


class TestBatchProcessorMockedIntegration:
    """Integration tests with mocked dependencies"""
    
    @patch('generative_latent_optimization.dataset.batch_processor.tqdm')
    @patch('generative_latent_optimization.dataset.batch_processor.time.time')
    def test_process_directory_basic_flow(self, mock_time, mock_tqdm, 
                                        basic_batch_config, basic_optimization_config,
                                        temp_input_directory, mock_dependencies):
        """Test basic process_directory flow with mocked dependencies"""
        mock_time.side_effect = [0.0, 3600.0]  # Start and end times
        mock_tqdm.return_value = range(3)  # Mock tqdm to return simple range
        
        with tempfile.TemporaryDirectory() as temp_output:
            output_dir = Path(temp_output)
            
            with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
                 patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics') as mock_metrics, \
                 patch('generative_latent_optimization.dataset.batch_processor.VAELoader') as mock_vae_loader, \
                 patch('generative_latent_optimization.dataset.batch_processor.LatentOptimizer') as mock_optimizer_class:
                
                # Configure mocks
                mock_vae_loader.load_sd_vae_simple.return_value = (mock_dependencies['mock_vae'], 'cuda')
                mock_optimizer = Mock()
                mock_optimizer.optimize.return_value = mock_dependencies['mock_opt_result']
                mock_optimizer_class.return_value = mock_optimizer
                
                # Mock metrics results
                mock_metric_result = Mock()
                mock_metric_result.psnr_db = 25.0
                mock_metric_result.ssim = 0.85
                mock_metrics.return_value.calculate_all_metrics.return_value = mock_metric_result
                
                # Limit to 3 images for testing
                basic_batch_config.max_images = 3
                basic_batch_config.save_visualizations = False
                
                processor = BatchProcessor(basic_batch_config)
                
                # Mock _process_single_image to return predictable results
                def mock_process_single_image(*args):
                    return {
                        'image_path': 'test.jpg',
                        'image_name': 'test',
                        'psnr_improvement': 5.0,
                        'ssim_improvement': 0.15,
                        'loss_reduction': 80.0
                    }
                
                with patch.object(processor, '_process_single_image', 
                                side_effect=mock_process_single_image):
                    
                    result = processor.process_directory(
                        temp_input_directory, output_dir, basic_optimization_config
                    )
                
                # Verify result structure
                assert isinstance(result, ProcessingResults)
                assert result.total_processed == 3
                assert result.successful_optimizations == 3
                assert result.failed_optimizations == 0
                assert_float_approximately_equal(result.average_psnr_improvement, 5.0)
                assert_float_approximately_equal(result.average_loss_reduction, 80.0)
                assert_float_approximately_equal(result.processing_time_seconds, 3600.0)
    
    def test_process_single_image_with_mocked_vae(self, basic_batch_config, mock_dependencies):
        """Test _process_single_image with fully mocked VAE components"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            # Create dummy image file
            image_path = temp_path / "test_image.jpg"
            image_path.touch()
            
            with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
                 patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics') as mock_metrics, \
                 patch('generative_latent_optimization.dataset.batch_processor.load_and_preprocess_image') as mock_load:
                
                # Configure mocks
                mock_load.return_value = (torch.randn(1, 3, 512, 512), Mock())
                
                mock_metric_result = Mock()
                mock_metric_result.psnr_db = 25.0
                mock_metric_result.ssim = 0.85
                mock_metrics.return_value.calculate_all_metrics.return_value = mock_metric_result
                
                mock_optimizer = Mock()
                mock_optimizer.optimize.return_value = mock_dependencies['mock_opt_result']
                
                basic_batch_config.save_visualizations = False
                processor = BatchProcessor(basic_batch_config)
                
                result = processor._process_single_image(
                    image_path, output_dir, mock_dependencies['mock_vae'], 
                    mock_optimizer, 'cuda'
                )
                
                # Verify result structure
                assert result is not None
                assert result['image_path'] == str(image_path)
                assert result['image_name'] == 'test_image'
                assert 'psnr_improvement' in result
                assert 'loss_reduction' in result
    
    def test_process_single_image_exception_handling(self, basic_batch_config):
        """Test _process_single_image exception handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            with patch('generative_latent_optimization.dataset.batch_processor.IOUtils') as mock_io, \
                 patch('generative_latent_optimization.dataset.batch_processor.ImageMetrics'), \
                 patch('vae_toolkit.load_and_preprocess_image') as mock_load:
                
                # Configure mock to raise exception
                mock_load.side_effect = Exception("Failed to load image")
                
                processor = BatchProcessor(basic_batch_config)
                
                result = processor._process_single_image(
                    Path("nonexistent.jpg"), output_dir, Mock(), Mock(), 'cuda'
                )
                
                # Should return None on exception
                assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])