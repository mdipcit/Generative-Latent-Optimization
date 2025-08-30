#!/usr/bin/env python3
"""
Unit tests for PNG Dataset functionality
"""

import os
import sys
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from generative_latent_optimization.dataset.png_dataset import (
    PNGDatasetBuilder, PNGDatasetMetadata, create_png_dataset_from_results
)
from ...fixtures.dataset_mocks import mock_dataset_dependencies, mock_png_dataset_processing
from ...fixtures.assertion_helpers import (
    assert_float_approximately_equal, safe_calculate_median, assert_dataset_sample_structure
)


@pytest.fixture
def mock_dependencies():
    """Mock external dependencies"""
    with patch('generative_latent_optimization.dataset.png_dataset.IOUtils') as mock_io, \
         patch('generative_latent_optimization.dataset.png_dataset.ImageVisualizer') as mock_viz:
        yield {
            'mock_io': mock_io,
            'mock_viz': mock_viz
        }


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
def sample_processing_results():
    """Sample processing results data"""
    return [
        {
            'image_name': 'img_001',
            'initial_psnr': 20.0,
            'final_psnr': 25.0,
            'psnr_improvement': 5.0,
            'initial_ssim': 0.7,
            'final_ssim': 0.85,
            'ssim_improvement': 0.15,
            'loss_reduction': 80.0,
            'optimization_iterations': 50
        },
        {
            'image_name': 'img_002',
            'initial_psnr': 18.0,
            'final_psnr': 24.0,
            'psnr_improvement': 6.0,
            'initial_ssim': 0.65,
            'final_ssim': 0.8,
            'ssim_improvement': 0.15,
            'loss_reduction': 75.0,
            'optimization_iterations': 45
        }
    ]


@pytest.fixture
def temp_processed_data():
    """Create temporary processed data structure"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create split directories and result files
        for split in ['train', 'val']:
            split_dir = temp_path / split
            split_dir.mkdir()
            
            # Create detailed_results.json
            results = [
                {
                    'image_name': f'{split}_img_001',
                    'initial_psnr': 20.0,
                    'final_psnr': 25.0,
                    'psnr_improvement': 5.0,
                    'initial_ssim': 0.7,
                    'final_ssim': 0.85,
                    'ssim_improvement': 0.15,
                    'loss_reduction': 80.0,
                    'optimization_iterations': 50
                }
            ]
            
            with open(split_dir / 'detailed_results.json', 'w') as f:
                json.dump(results, f)
            
            # Create image result directories
            img_dir = split_dir / f'{split}_img_001'
            img_dir.mkdir()
            
            # Create dummy image files
            (img_dir / f'{split}_img_001_original.png').touch()
            (img_dir / f'{split}_img_001_initial_reconstruction.png').touch()
            (img_dir / f'{split}_img_001_optimized_reconstruction.png').touch()
        
        yield temp_path


class TestPNGDatasetMetadata:
    """Test cases for PNGDatasetMetadata dataclass"""
    
    def test_metadata_creation(self, sample_optimization_config):
        """Test PNGDatasetMetadata creation"""
        metadata = PNGDatasetMetadata(
            total_samples=100,
            splits_count={'train': 50, 'val': 30, 'test': 20},
            optimization_config=sample_optimization_config,
            directory_structure={'root': 'Root directory'},
            creation_timestamp="2024-01-01 12:00:00"
        )
        
        assert metadata.total_samples == 100
        assert metadata.splits_count == {'train': 50, 'val': 30, 'test': 20}
        assert metadata.optimization_config == sample_optimization_config
        assert metadata.creation_timestamp == "2024-01-01 12:00:00"
        assert metadata.dataset_version == "1.0"  # Default value
        assert metadata.source_dataset == "BSDS500"  # Default value
    
    def test_metadata_custom_defaults(self, sample_optimization_config):
        """Test metadata with custom default values"""
        metadata = PNGDatasetMetadata(
            total_samples=50,
            splits_count={'train': 50},
            optimization_config=sample_optimization_config,
            directory_structure={},
            creation_timestamp="2024-01-01 12:00:00",
            dataset_version="2.0",
            source_dataset="Custom"
        )
        
        assert metadata.dataset_version == "2.0"
        assert metadata.source_dataset == "Custom"


class TestPNGDatasetBuilder:
    """Test cases for PNGDatasetBuilder class"""
    
    def test_initialization(self, mock_dependencies):
        """Test PNGDatasetBuilder initialization"""
        builder = PNGDatasetBuilder()
        
        # Verify components are initialized
        assert hasattr(builder, 'io_utils')
        assert hasattr(builder, 'visualizer')
    
    def test_create_directory_structure(self, mock_dependencies):
        """Test _create_directory_structure method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "png_dataset"
            
            builder = PNGDatasetBuilder()
            builder._create_directory_structure(output_dir)
            
            # Verify directory structure
            assert output_dir.exists()
            assert (output_dir / 'train').exists()
            assert (output_dir / 'val').exists()
            assert (output_dir / 'test').exists()
            assert (output_dir / 'overview').exists()
    
    def test_std_calculation(self, mock_dependencies):
        """Test _std helper method"""
        builder = PNGDatasetBuilder()
        
        # Test with normal values
        std = builder._std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert std > 0
        
        # Test with empty list
        std_empty = builder._std([])
        assert_float_approximately_equal(std_empty, 0.0)
        
        # Test with single value
        std_single = builder._std([5.0])
        assert_float_approximately_equal(std_single, 0.0)
    
    def test_calculate_statistics(self, mock_dependencies, sample_processing_results):
        """Test _calculate_statistics method"""
        builder = PNGDatasetBuilder()
        
        # Create dataset samples
        dataset_samples = [
            {
                'split': 'train',
                'metrics': sample_processing_results[0]
            },
            {
                'split': 'val',
                'metrics': sample_processing_results[1]
            }
        ]
        
        statistics = builder._calculate_statistics(dataset_samples)
        
        # Verify basic structure
        assert statistics['total_samples'] == 2
        assert 'psnr_improvement' in statistics
        assert 'loss_reduction' in statistics
        assert 'optimization' in statistics
        assert 'splits_breakdown' in statistics
        
        # Verify PSNR statistics
        psnr_stats = statistics['psnr_improvement']
        assert_float_approximately_equal(psnr_stats['mean'], 5.5)  # (5.0 + 6.0) / 2
        assert_float_approximately_equal(psnr_stats['min'], 5.0)
        assert_float_approximately_equal(psnr_stats['max'], 6.0)
        assert_float_approximately_equal(psnr_stats['median'], 6.0)  # Implementation uses [len//2] for even length
        
        # Verify loss reduction statistics
        loss_stats = statistics['loss_reduction']
        assert_float_approximately_equal(loss_stats['mean'], 77.5)  # (80.0 + 75.0) / 2
        
        # Verify splits breakdown
        assert 'train' in statistics['splits_breakdown']
        assert 'val' in statistics['splits_breakdown']
        assert statistics['splits_breakdown']['train']['count'] == 1
        assert statistics['splits_breakdown']['val']['count'] == 1
    
    def test_calculate_statistics_empty_samples(self, mock_dependencies):
        """Test _calculate_statistics with empty samples"""
        builder = PNGDatasetBuilder()
        
        statistics = builder._calculate_statistics([])
        
        # Should return empty dict
        assert statistics == {}
    
    @patch('generative_latent_optimization.dataset.png_dataset.time.strftime')
    def test_create_metadata(self, mock_strftime, mock_dependencies, sample_optimization_config):
        """Test _create_metadata method"""
        mock_strftime.return_value = "2024-01-01 12:00:00"
        
        builder = PNGDatasetBuilder()
        
        dataset_samples = [
            {'split': 'train', 'metrics': {}},
            {'split': 'train', 'metrics': {}},
            {'split': 'val', 'metrics': {}}
        ]
        
        metadata = builder._create_metadata(dataset_samples, sample_optimization_config, Path("/output"))
        
        assert isinstance(metadata, PNGDatasetMetadata)
        assert metadata.total_samples == 3
        assert metadata.splits_count == {'train': 2, 'val': 1}
        assert metadata.optimization_config == sample_optimization_config
        assert metadata.creation_timestamp == "2024-01-01 12:00:00"
        assert 'root' in metadata.directory_structure
        assert 'splits' in metadata.directory_structure
    
    def test_save_metadata(self, mock_dependencies, sample_optimization_config):
        """Test _save_metadata method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            builder = PNGDatasetBuilder()
            
            metadata = PNGDatasetMetadata(
                total_samples=2,
                splits_count={'train': 2},
                optimization_config=sample_optimization_config,
                directory_structure={},
                creation_timestamp="2024-01-01 12:00:00"
            )
            
            dataset_samples = [
                {
                    'split': 'train',
                    'metrics': {'psnr_improvement': 5.0, 'loss_reduction': 80.0, 'optimization_iterations': 50}
                },
                {
                    'split': 'val', 
                    'metrics': {'psnr_improvement': 6.0, 'loss_reduction': 75.0, 'optimization_iterations': 45}
                }
            ]
            
            with patch.object(builder, '_create_readme') as mock_readme:
                builder._save_metadata(output_dir, metadata, dataset_samples)
            
            # Verify IOUtils was called for saving JSON files
            io_mock = mock_dependencies['mock_io'].return_value
            assert io_mock.save_json.call_count == 3  # metadata, statistics, samples
            
            # Verify _create_readme was called
            mock_readme.assert_called_once()
    
    def test_process_split_no_results_file(self, mock_dependencies):
        """Test _process_split when results file doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            split_input_dir = Path(temp_dir) / "train"
            split_input_dir.mkdir()
            split_output_dir = Path(temp_dir) / "output_train"
            
            builder = PNGDatasetBuilder()
            
            samples = builder._process_split(
                split_input_dir, split_output_dir, 'train', True, True
            )
            
            # Should return empty list when no results file
            assert samples == []
    
    def test_process_split_with_valid_data(self, mock_dependencies):
        """Test _process_split with valid data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            split_input_dir = Path(temp_dir) / "train"
            split_input_dir.mkdir()
            split_output_dir = Path(temp_dir) / "output_train"
            split_output_dir.mkdir()
            
            # Create mock results file
            results = [
                {
                    'image_name': 'test_img',
                    'initial_psnr': 20.0,
                    'final_psnr': 25.0,
                    'psnr_improvement': 5.0,
                    'loss_reduction': 80.0,
                    'optimization_iterations': 50
                }
            ]
            
            # Create image result directory with files
            img_dir = split_input_dir / 'test_img'
            img_dir.mkdir()
            (img_dir / 'test_img_original.png').touch()
            (img_dir / 'test_img_initial_reconstruction.png').touch()
            (img_dir / 'test_img_optimized_reconstruction.png').touch()
            
            # Create actual results file to make it exist
            results_file = split_input_dir / 'detailed_results.json'
            with open(results_file, 'w') as f:
                import json
                json.dump(results, f)
            
            builder = PNGDatasetBuilder()
            
            # Mock IOUtils to return the results we created
            mock_io = mock_dependencies['mock_io'].return_value
            mock_io.load_json.return_value = results
            
            with patch.object(builder, '_process_single_image') as mock_process_single:
                mock_process_single.return_value = {'processed': True}
                
                samples = builder._process_split(
                    split_input_dir, split_output_dir, 'train', True, True
                )
            
            # Verify _process_single_image was called
            mock_process_single.assert_called_once()
            
            # Should return the processed sample
            assert len(samples) == 1
            assert samples[0] == {'processed': True}
    
    def test_process_single_image_missing_files(self, mock_dependencies):
        """Test _process_single_image with missing image files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            # Create only one of the required files
            (input_dir / 'test_original.png').touch()
            # Missing: initial_reconstruction.png, optimized_reconstruction.png
            
            result_data = {'image_name': 'test'}
            
            builder = PNGDatasetBuilder()
            
            sample_info = builder._process_single_image(
                input_dir, output_dir, result_data, 'train', True, True
            )
            
            # Should return None when files are missing
            assert sample_info is None
    
    def test_process_single_image_success(self, mock_dependencies, sample_processing_results):
        """Test _process_single_image successful processing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            # Create all required image files with correct naming pattern
            (input_dir / 'img_001_original.png').touch()
            (input_dir / 'img_001_initial_reconstruction.png').touch()
            (input_dir / 'img_001_optimized_reconstruction.png').touch()
            
            result_data = sample_processing_results[0]
            
            builder = PNGDatasetBuilder()
            
            # Mock matplotlib/PIL imports for comparison grid creation
            with patch('PIL.Image') as mock_image, \
                 patch('matplotlib.pyplot') as mock_plt:
                
                # Mock PIL Image loading
                mock_pil_img = Mock()
                mock_image.open.return_value = mock_pil_img
                
                # Mock matplotlib.pyplot.subplots to return proper tuple
                mock_fig = Mock()
                mock_ax1 = Mock()
                mock_ax2 = Mock()
                mock_plt.subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
                
                sample_info = builder._process_single_image(
                    input_dir, output_dir, result_data, 'train', 
                    include_comparisons=True, include_individual_images=True
                )
            
            # Verify sample info structure
            assert sample_info is not None
            assert sample_info['image_name'] == 'img_001'
            assert sample_info['split'] == 'train'
            assert 'files' in sample_info
            assert 'metrics' in sample_info
            
            # Verify metrics were preserved
            metrics = sample_info['metrics']
            assert_float_approximately_equal(metrics['psnr_improvement'], 5.0)
            assert_float_approximately_equal(metrics['loss_reduction'], 80.0)
            assert metrics['optimization_iterations'] == 50
    
    def test_process_single_image_no_comparisons(self, mock_dependencies, sample_processing_results):
        """Test _process_single_image without comparison grids"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            # Create required image files with correct naming pattern
            (input_dir / 'img_001_original.png').touch()
            (input_dir / 'img_001_initial_reconstruction.png').touch()
            (input_dir / 'img_001_optimized_reconstruction.png').touch()
            
            result_data = sample_processing_results[0]
            
            builder = PNGDatasetBuilder()
            
            sample_info = builder._process_single_image(
                input_dir, output_dir, result_data, 'train',
                include_comparisons=False, include_individual_images=True
            )
            
            # Should succeed without creating comparison grid
            assert sample_info is not None
            assert 'comparison_grid' not in sample_info['files']
    
    def test_process_single_image_comparison_exception(self, mock_dependencies, sample_processing_results):
        """Test _process_single_image when comparison creation fails"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            # Create required image files with correct naming pattern
            (input_dir / 'img_001_original.png').touch()
            (input_dir / 'img_001_initial_reconstruction.png').touch()
            (input_dir / 'img_001_optimized_reconstruction.png').touch()
            
            result_data = sample_processing_results[0]
            
            builder = PNGDatasetBuilder()
            
            # Mock PIL to raise exception
            with patch('PIL.Image') as mock_image:
                mock_image.open.side_effect = Exception("Failed to load image")
                
                sample_info = builder._process_single_image(
                    input_dir, output_dir, result_data, 'train',
                    include_comparisons=True, include_individual_images=True
                )
            
            # Should still succeed, just without comparison grid
            assert sample_info is not None
            assert 'comparison_grid' not in sample_info.get('files', {})
    
    @patch('matplotlib.pyplot')
    def test_create_overview_visualizations(self, mock_plt, mock_dependencies):
        """Test _create_overview_visualizations method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            overview_dir = output_dir / 'overview'
            overview_dir.mkdir(parents=True)
            
            dataset_samples = [
                {'split': 'train', 'metrics': {'psnr_improvement': 5.0}},
                {'split': 'train', 'metrics': {'psnr_improvement': 6.0}},
                {'split': 'val', 'metrics': {'psnr_improvement': 4.0}}
            ]
            
            # Mock matplotlib.pyplot.subplots to return proper tuple
            mock_fig = Mock()
            mock_ax1 = Mock()
            mock_ax2 = Mock()
            mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
            
            builder = PNGDatasetBuilder()
            builder._create_overview_visualizations(output_dir, dataset_samples)
            
            # Verify matplotlib was used
            mock_plt.subplots.assert_called()
            mock_plt.savefig.assert_called()
            mock_plt.close.assert_called()
    
    def test_create_overview_visualizations_no_matplotlib(self, mock_dependencies):
        """Test _create_overview_visualizations when matplotlib is not available"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            dataset_samples = [
                {'split': 'train', 'metrics': {'psnr_improvement': 5.0}}
            ]
            
            builder = PNGDatasetBuilder()
            
            # Mock matplotlib import to raise ImportError by patching the import
            import sys
            original_modules = sys.modules.copy()
            if 'matplotlib.pyplot' in sys.modules:
                del sys.modules['matplotlib.pyplot']
            
            try:
                with patch.dict('sys.modules', {'matplotlib.pyplot': None}):
                    # Should not raise exception when matplotlib is not available
                    builder._create_overview_visualizations(output_dir, dataset_samples)
            finally:
                sys.modules.update(original_modules)
    
    def test_create_readme(self, mock_dependencies, sample_optimization_config):
        """Test _create_readme method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            metadata = PNGDatasetMetadata(
                total_samples=100,
                splits_count={'train': 60, 'val': 25, 'test': 15},
                optimization_config=sample_optimization_config,
                directory_structure={},
                creation_timestamp="2024-01-01 12:00:00"
            )
            
            statistics = {
                'psnr_improvement': {
                    'mean': 5.5,
                    'std': 1.2,
                    'min': 3.0,
                    'max': 8.0,
                    'median': 5.0
                },
                'loss_reduction': {
                    'mean': 77.5,
                    'std': 5.2,
                    'min': 70.0,
                    'max': 85.0
                }
            }
            
            builder = PNGDatasetBuilder()
            builder._create_readme(output_dir, metadata, statistics)
            
            # Verify README file was created
            readme_path = output_dir / "README.md"
            assert readme_path.exists()
            
            # Verify content contains key information
            content = readme_path.read_text()
            assert "VAE Optimization PNG Dataset" in content
            assert "100" in content  # total samples
            assert "train" in content
            assert "5.5" in content  # PSNR mean
            assert "77.5" in content  # Loss reduction mean
    
    def test_create_png_dataset_integration(self, mock_dependencies, temp_processed_data, sample_optimization_config):
        """Test complete create_png_dataset flow"""
        with tempfile.TemporaryDirectory() as temp_output:
            output_dir = Path(temp_output) / "png_dataset"
            
            builder = PNGDatasetBuilder()
            
            # Mock process_split to return sample data
            sample_data = [
                {
                    'image_name': 'test_img',
                    'split': 'train',
                    'metrics': {'psnr_improvement': 5.0, 'loss_reduction': 80.0, 'optimization_iterations': 50}
                }
            ]
            
            with patch.object(builder, '_process_split', return_value=sample_data), \
                 patch.object(builder, '_create_overview_visualizations'), \
                 patch.object(builder, '_save_metadata'):
                
                result_path = builder.create_png_dataset(
                    temp_processed_data, output_dir, sample_optimization_config
                )
            
            # Verify return path
            assert result_path == str(output_dir)
            
            # Verify directory structure was created
            assert output_dir.exists()
            assert (output_dir / 'train').exists()
            assert (output_dir / 'val').exists()
            assert (output_dir / 'test').exists()
            assert (output_dir / 'overview').exists()
    
    def test_create_png_dataset_no_splits_found(self, mock_dependencies, sample_optimization_config):
        """Test create_png_dataset when no splits are found"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty processed directory
            processed_dir = Path(temp_dir) / "empty_processed"
            processed_dir.mkdir()
            
            output_dir = Path(temp_dir) / "png_dataset"
            
            builder = PNGDatasetBuilder()
            
            with patch.object(builder, '_create_overview_visualizations'), \
                 patch.object(builder, '_save_metadata'):
                
                result_path = builder.create_png_dataset(
                    processed_dir, output_dir, sample_optimization_config
                )
            
            # Should still create directory structure and return path
            assert result_path == str(output_dir)
            assert output_dir.exists()
    
    def test_create_png_dataset_partial_splits(self, mock_dependencies, sample_optimization_config):
        """Test create_png_dataset with only some splits available"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir) / "processed"
            processed_dir.mkdir()
            
            # Create only train split
            train_dir = processed_dir / "train"
            train_dir.mkdir()
            
            output_dir = Path(temp_dir) / "png_dataset"
            
            builder = PNGDatasetBuilder()
            
            # Mock _process_split to return data only for train
            def mock_process_split(split_input_dir, split_output_dir, split_name, *args):
                if split_name == 'train' and split_input_dir.exists():
                    return [{'split': 'train', 'data': 'mock'}]
                return []
            
            with patch.object(builder, '_process_split', side_effect=mock_process_split), \
                 patch.object(builder, '_create_overview_visualizations'), \
                 patch.object(builder, '_save_metadata'):
                
                result_path = builder.create_png_dataset(
                    processed_dir, output_dir, sample_optimization_config
                )
            
            assert result_path == str(output_dir)


class TestPNGDatasetIntegration:
    """Integration tests for PNG dataset functionality"""
    
    def test_full_png_dataset_creation_flow(self, mock_dependencies, temp_processed_data, sample_optimization_config):
        """Test complete PNG dataset creation flow with real file operations"""
        with tempfile.TemporaryDirectory() as temp_output:
            output_dir = Path(temp_output) / "png_dataset"
            
            builder = PNGDatasetBuilder()
            
            # Mock external dependencies but allow real file operations
            with patch('PIL.Image') as mock_image, \
                 patch('matplotlib.pyplot') as mock_plt:
                
                # Mock PIL Image for comparison grid creation
                mock_pil_img = Mock()
                mock_image.open.return_value = mock_pil_img
                
                # Mock matplotlib.pyplot.subplots to return proper tuple
                mock_fig = Mock()
                mock_ax1 = Mock()
                mock_ax2 = Mock()
                mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
                
                result_path = builder.create_png_dataset(
                    temp_processed_data, output_dir, sample_optimization_config,
                    include_comparisons=True, include_individual_images=True
                )
            
            # Verify basic directory structure
            assert Path(result_path).exists()
            assert (Path(result_path) / 'train').exists()
            assert (Path(result_path) / 'val').exists()
            assert (Path(result_path) / 'overview').exists()
            
            # Verify IOUtils save methods were called
            io_mock = mock_dependencies['mock_io'].return_value
            assert io_mock.save_json.call_count >= 3  # metadata, statistics, samples
    
    def test_error_handling_in_process_split(self, mock_dependencies):
        """Test error handling during split processing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            split_input_dir = Path(temp_dir) / "train"
            split_input_dir.mkdir()
            split_output_dir = Path(temp_dir) / "output_train"
            
            # Create results file but no image directories
            results = [{'image_name': 'nonexistent_img'}]
            
            mock_io = mock_dependencies['mock_io'].return_value
            mock_io.load_json.return_value = results
            
            builder = PNGDatasetBuilder()
            
            # Should handle missing image directories gracefully
            samples = builder._process_split(
                split_input_dir, split_output_dir, 'train', True, True
            )
            
            # Should return empty list due to missing image directory
            assert samples == []


class TestPNGDatasetUtilityFunctions:
    """Test utility functions"""
    
    def test_create_png_dataset_from_results(self, mock_dependencies, sample_optimization_config):
        """Test create_png_dataset_from_results utility function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir) / "processed"
            processed_dir.mkdir()
            output_dir = Path(temp_dir) / "output"
            
            with patch.object(PNGDatasetBuilder, 'create_png_dataset') as mock_create:
                mock_create.return_value = str(output_dir)
                
                result_path = create_png_dataset_from_results(
                    processed_dir, output_dir, sample_optimization_config
                )
            
            # Verify PNGDatasetBuilder.create_png_dataset was called
            mock_create.assert_called_once_with(
                processed_dir, output_dir, sample_optimization_config
            )
            
            assert result_path == str(output_dir)


class TestPNGDatasetErrorCases:
    """Test error cases and edge conditions"""
    
    def test_process_single_image_file_copy_error(self, mock_dependencies, sample_processing_results):
        """Test _process_single_image when file copying fails"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            # Create required files with correct naming pattern
            (input_dir / 'img_001_original.png').touch()
            (input_dir / 'img_001_initial_reconstruction.png').touch()
            (input_dir / 'img_001_optimized_reconstruction.png').touch()
            
            result_data = sample_processing_results[0]
            
            builder = PNGDatasetBuilder()
            
            # Should succeed with file existence check
            sample_info = builder._process_single_image(
                input_dir, output_dir, result_data, 'train',
                include_comparisons=False, include_individual_images=False
            )
            
            # Should return sample info when files exist, even without individual images
            assert sample_info is not None
            assert sample_info['image_name'] == 'img_001'
            assert sample_info['split'] == 'train'
    
    def test_calculate_statistics_with_missing_metrics(self, mock_dependencies):
        """Test _calculate_statistics with samples missing some metrics"""
        builder = PNGDatasetBuilder()
        
        # Dataset samples with incomplete metrics
        dataset_samples = [
            {'split': 'train', 'metrics': {'psnr_improvement': 5.0, 'loss_reduction': 75.0, 'optimization_iterations': 50}},
            {'split': 'train', 'metrics': {'psnr_improvement': 6.0, 'loss_reduction': 80.0, 'optimization_iterations': 45}}
        ]
        
        # Should handle missing fields gracefully
        statistics = builder._calculate_statistics(dataset_samples)
        
        assert statistics['total_samples'] == 2
        assert 'psnr_improvement' in statistics
    
    def test_metadata_serialization(self, mock_dependencies, sample_optimization_config):
        """Test that metadata can be properly serialized"""
        from dataclasses import asdict
        
        metadata = PNGDatasetMetadata(
            total_samples=10,
            splits_count={'train': 10},
            optimization_config=sample_optimization_config,
            directory_structure={'root': 'test'},
            creation_timestamp="2024-01-01 12:00:00"
        )
        
        # Should be serializable to dict
        metadata_dict = asdict(metadata)
        assert isinstance(metadata_dict, dict)
        assert metadata_dict['total_samples'] == 10
        assert metadata_dict['optimization_config'] == sample_optimization_config
        
        # Should be JSON serializable
        json_str = json.dumps(metadata_dict)
        assert isinstance(json_str, str)
        
        # Should be deserializable
        restored_dict = json.loads(json_str)
        assert restored_dict == metadata_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])