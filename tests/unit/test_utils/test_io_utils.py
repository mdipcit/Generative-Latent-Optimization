#!/usr/bin/env python3
"""
Unit tests for IOUtils functionality
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
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from generative_latent_optimization.utils.io_utils import (
    IOUtils, ResultsSaver, save_image_tensor, create_results_saver
)


class TestIOUtils:
    """Test cases for IOUtils static methods"""
    
    def test_save_image_tensor_4d_input(self):
        """Test save_image_tensor with 4D tensor input"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_image.png"
            
            # Create 4D tensor [1, 3, 64, 64]
            tensor = torch.rand(1, 3, 64, 64)
            
            with patch('generative_latent_optimization.utils.io_utils.Image') as mock_image:
                mock_pil_img = Mock()
                mock_image.fromarray.return_value = mock_pil_img
                
                IOUtils.save_image_tensor(tensor, output_path)
            
            # Verify Image.fromarray was called
            mock_image.fromarray.assert_called_once()
            
            # Verify save was called
            mock_pil_img.save.assert_called_once_with(output_path, format="PNG")
    
    def test_save_image_tensor_3d_input(self):
        """Test save_image_tensor with 3D tensor input"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_image.jpg"
            
            # Create 3D tensor [3, 64, 64]
            tensor = torch.rand(3, 64, 64)
            
            with patch('generative_latent_optimization.utils.io_utils.Image') as mock_image:
                mock_pil_img = Mock()
                mock_image.fromarray.return_value = mock_pil_img
                
                IOUtils.save_image_tensor(tensor, output_path, format="JPEG")
            
            # Verify correct format was used
            mock_pil_img.save.assert_called_once_with(output_path, format="JPEG")
    
    def test_save_image_tensor_invalid_dimensions(self):
        """Test save_image_tensor with invalid tensor dimensions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_image.png"
            
            # Create invalid 2D tensor
            tensor = torch.rand(64, 64)
            
            with pytest.raises(ValueError, match="Expected 3D or 4D tensor"):
                IOUtils.save_image_tensor(tensor, output_path)
    
    def test_save_image_tensor_grayscale(self):
        """Test save_image_tensor with grayscale image"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "grayscale.png"
            
            # Create grayscale tensor [1, 64, 64]
            tensor = torch.rand(1, 64, 64)
            
            with patch('generative_latent_optimization.utils.io_utils.Image') as mock_image:
                mock_pil_img = Mock()
                mock_image.fromarray.return_value = mock_pil_img
                
                IOUtils.save_image_tensor(tensor, output_path)
            
            # Should handle grayscale correctly
            mock_image.fromarray.assert_called_once()
    
    def test_save_image_tensor_clipping(self):
        """Test save_image_tensor value clipping"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "clipped.png"
            
            # Create tensor with values outside [0, 1] range
            tensor = torch.tensor([[[[-0.5, 1.5], [0.3, 0.7]],
                                   [[0.2, 1.2], [-0.1, 0.8]],
                                   [[0.9, 0.1], [1.8, -0.3]]]])
            
            with patch('generative_latent_optimization.utils.io_utils.Image') as mock_image, \
                 patch('generative_latent_optimization.utils.io_utils.np') as mock_np:
                
                # Configure numpy mock properly
                mock_np.clip.side_effect = lambda arr, min_val, max_val: np.clip(arr, min_val, max_val)
                mock_np.uint8 = np.uint8  # Use real numpy uint8 type
                
                mock_pil_img = Mock()
                mock_image.fromarray.return_value = mock_pil_img
                
                IOUtils.save_image_tensor(tensor, output_path)
            
            # Verify clipping was applied
            mock_np.clip.assert_called()
            # Verify image was created
            mock_image.fromarray.assert_called_once()
    
    def test_save_tensor(self):
        """Test save_tensor method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "deep" / "nested" / "tensor.pt"
            tensor = torch.randn(4, 4, 64, 64)
            
            with patch('generative_latent_optimization.utils.io_utils.torch.save') as mock_save:
                IOUtils.save_tensor(tensor, output_path)
            
            # Verify directory was created and tensor was saved
            assert output_path.parent.exists()
            mock_save.assert_called_once_with(tensor, output_path)
    
    def test_load_tensor(self):
        """Test load_tensor method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tensor_path = Path(temp_dir) / "tensor.pt"
            original_tensor = torch.randn(2, 3, 32, 32)
            
            # Save tensor first
            torch.save(original_tensor, tensor_path)
            
            # Test loading
            loaded_tensor = IOUtils.load_tensor(tensor_path)
            
            assert torch.equal(loaded_tensor, original_tensor)
    
    def test_save_numpy(self):
        """Test save_numpy method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "array.npy"
            array = np.random.rand(10, 10)
            
            with patch('generative_latent_optimization.utils.io_utils.np.save') as mock_save:
                IOUtils.save_numpy(array, output_path)
            
            mock_save.assert_called_once_with(output_path, array)
    
    def test_load_numpy(self):
        """Test load_numpy method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            array_path = Path(temp_dir) / "array.npy"
            original_array = np.random.rand(5, 5)
            
            # Save array first
            np.save(array_path, original_array)
            
            # Test loading
            loaded_array = IOUtils.load_numpy(array_path)
            
            assert np.array_equal(loaded_array, original_array)
    
    def test_save_json(self):
        """Test save_json method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "data.json"
            data = {'key1': 'value1', 'key2': 42, 'key3': [1, 2, 3]}
            
            IOUtils.save_json(data, output_path)
            
            # Verify file was created and contains correct data
            assert output_path.exists()
            with open(output_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == data
    
    def test_save_json_with_custom_indent(self):
        """Test save_json with custom indentation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "data.json"
            data = {'key': 'value'}
            
            IOUtils.save_json(data, output_path, indent=4)
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify indentation (4 spaces)
            content = output_path.read_text()
            assert '    "key"' in content  # 4 spaces for indentation
    
    def test_save_json_with_non_serializable_data(self):
        """Test save_json with non-serializable data using default=str"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "data.json"
            
            # Include non-serializable object
            class CustomObject:
                def __str__(self):
                    return "custom_object_string"
            
            data = {'normal': 'value', 'custom': CustomObject()}
            
            IOUtils.save_json(data, output_path)
            
            # Should save successfully using str() on non-serializable objects
            assert output_path.exists()
            with open(output_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data['normal'] == 'value'
            assert loaded_data['custom'] == 'custom_object_string'
    
    def test_load_json(self):
        """Test load_json method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "data.json"
            original_data = {'test': 'data', 'number': 123}
            
            # Save JSON first
            with open(json_path, 'w') as f:
                json.dump(original_data, f)
            
            # Test loading
            loaded_data = IOUtils.load_json(json_path)
            
            assert loaded_data == original_data
    
    def test_save_pickle(self):
        """Test save_pickle method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "data.pkl"
            data = {'tensor': torch.randn(2, 2), 'list': [1, 2, 3]}
            
            IOUtils.save_pickle(data, output_path)
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify can be loaded with pickle
            with open(output_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            assert loaded_data['list'] == [1, 2, 3]
            assert torch.equal(loaded_data['tensor'], data['tensor'])
    
    def test_load_pickle(self):
        """Test load_pickle method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "data.pkl"
            original_data = {'key': 'value', 'number': 456}
            
            # Save pickle first
            with open(pickle_path, 'wb') as f:
                pickle.dump(original_data, f)
            
            # Test loading
            loaded_data = IOUtils.load_pickle(pickle_path)
            
            assert loaded_data == original_data
    
    @pytest.mark.skipif(not pytest.importorskip("h5py"), reason="h5py not available")
    def test_save_hdf5(self):
        """Test save_hdf5 method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "data.h5"
            data = {
                'array1': np.random.rand(10, 10),
                'array2': np.random.rand(5, 5)
            }
            
            IOUtils.save_hdf5(data, output_path)
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify can be loaded with h5py
            import h5py
            with h5py.File(output_path, 'r') as f:
                assert 'array1' in f
                assert 'array2' in f
                assert f['array1'].shape == (10, 10)
                assert f['array2'].shape == (5, 5)
    
    @pytest.mark.skipif(not pytest.importorskip("h5py"), reason="h5py not available")
    def test_load_hdf5_all_keys(self):
        """Test load_hdf5 method loading all keys"""
        with tempfile.TemporaryDirectory() as temp_dir:
            hdf5_path = Path(temp_dir) / "data.h5"
            original_data = {
                'test_array1': np.random.rand(8, 8),
                'test_array2': np.random.rand(3, 3)
            }
            
            # Save HDF5 first
            IOUtils.save_hdf5(original_data, hdf5_path)
            
            # Test loading all keys
            loaded_data = IOUtils.load_hdf5(hdf5_path)
            
            assert set(loaded_data.keys()) == set(original_data.keys())
            assert np.array_equal(loaded_data['test_array1'], original_data['test_array1'])
            assert np.array_equal(loaded_data['test_array2'], original_data['test_array2'])
    
    @pytest.mark.skipif(not pytest.importorskip("h5py"), reason="h5py not available")
    def test_load_hdf5_specific_keys(self):
        """Test load_hdf5 method loading specific keys"""
        with tempfile.TemporaryDirectory() as temp_dir:
            hdf5_path = Path(temp_dir) / "data.h5"
            original_data = {
                'array1': np.random.rand(6, 6),
                'array2': np.random.rand(4, 4),
                'array3': np.random.rand(2, 2)
            }
            
            # Save HDF5 first
            IOUtils.save_hdf5(original_data, hdf5_path)
            
            # Test loading specific keys
            loaded_data = IOUtils.load_hdf5(hdf5_path, keys=['array1', 'array3'])
            
            assert set(loaded_data.keys()) == {'array1', 'array3'}
            assert 'array2' not in loaded_data
            assert np.array_equal(loaded_data['array1'], original_data['array1'])
            assert np.array_equal(loaded_data['array3'], original_data['array3'])
    
    @pytest.mark.skipif(not pytest.importorskip("h5py"), reason="h5py not available")
    def test_load_hdf5_missing_keys(self):
        """Test load_hdf5 with non-existent keys"""
        with tempfile.TemporaryDirectory() as temp_dir:
            hdf5_path = Path(temp_dir) / "data.h5"
            original_data = {'existing_array': np.random.rand(3, 3)}
            
            # Save HDF5 first
            IOUtils.save_hdf5(original_data, hdf5_path)
            
            # Test loading mix of existing and non-existing keys
            loaded_data = IOUtils.load_hdf5(hdf5_path, keys=['existing_array', 'nonexistent_array'])
            
            # Should only load existing keys
            assert 'existing_array' in loaded_data
            assert 'nonexistent_array' not in loaded_data
    
    def test_create_directory(self):
        """Test create_directory method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new" / "nested" / "directory"
            
            result_path = IOUtils.create_directory(new_dir)
            
            assert new_dir.exists()
            assert new_dir.is_dir()
            assert result_path == new_dir
    
    def test_create_directory_existing(self):
        """Test create_directory with existing directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir) / "existing"
            existing_dir.mkdir()
            
            # Should not raise error for existing directory
            result_path = IOUtils.create_directory(existing_dir)
            
            assert existing_dir.exists()
            assert result_path == existing_dir
    
    def test_get_image_files(self):
        """Test get_image_files method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            
            # Create various file types
            image_files = [
                test_dir / "image1.png",
                test_dir / "image2.jpg",
                test_dir / "image3.jpeg",
                test_dir / "image4.PNG",  # Uppercase
                test_dir / "image5.JPG"   # Uppercase
            ]
            
            # Create non-image files
            non_image_files = [
                test_dir / "document.txt",
                test_dir / "script.py"
            ]
            
            # Touch all files
            for file_path in image_files + non_image_files:
                file_path.touch()
            
            # Test getting image files
            found_files = IOUtils.get_image_files(test_dir)
            
            # Should find all image files, sorted
            assert len(found_files) == 5
            assert all(f.suffix.lower() in ['.png', '.jpg', '.jpeg'] for f in found_files)
            assert found_files == sorted(found_files)
    
    def test_get_image_files_custom_extensions(self):
        """Test get_image_files with custom extensions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            
            # Create files with different extensions
            (test_dir / "image.png").touch()
            (test_dir / "image.tiff").touch()
            (test_dir / "image.bmp").touch()
            (test_dir / "image.jpg").touch()
            
            # Test with custom extensions
            found_files = IOUtils.get_image_files(test_dir, extensions=['.png', '.tiff'])
            
            # Should only find files with specified extensions
            assert len(found_files) == 2
            assert all(f.suffix in ['.png', '.tiff'] for f in found_files)
    
    def test_get_image_files_empty_directory(self):
        """Test get_image_files with empty directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            
            found_files = IOUtils.get_image_files(test_dir)
            
            assert found_files == []
    
    def test_ensure_path_exists(self):
        """Test ensure_path_exists method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "deep" / "nested" / "file.txt"
            
            result_path = IOUtils.ensure_path_exists(file_path)
            
            # Should create parent directories but not the file itself
            assert file_path.parent.exists()
            assert not file_path.exists()  # File itself should not be created
            assert result_path == file_path


class TestResultsSaver:
    """Test cases for ResultsSaver class"""
    
    def test_initialization(self):
        """Test ResultsSaver initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            
            saver = ResultsSaver(output_dir)
            
            assert saver.output_dir == output_dir
            assert output_dir.exists()  # Should be created during init
            assert hasattr(saver, 'io_utils')
    
    def test_save_optimization_results(self):
        """Test save_optimization_results method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            saver = ResultsSaver(output_dir)
            
            # Create test data
            original = torch.rand(1, 3, 64, 64)
            initial_recon = torch.rand(1, 3, 64, 64)
            optimized_recon = torch.rand(1, 3, 64, 64)
            initial_latents = torch.rand(1, 4, 8, 8)
            optimized_latents = torch.rand(1, 4, 8, 8)
            losses = [1.0, 0.8, 0.6, 0.4]
            metrics = {'psnr': 25.0, 'ssim': 0.85}
            
            with patch.object(saver.io_utils, 'save_image_tensor') as mock_save_img, \
                 patch.object(saver.io_utils, 'save_tensor') as mock_save_tensor, \
                 patch.object(saver.io_utils, 'save_json') as mock_save_json:
                
                saved_files = saver.save_optimization_results(
                    original, initial_recon, optimized_recon,
                    initial_latents, optimized_latents,
                    losses, metrics, "test_image"
                )
            
            # Verify all save methods were called
            assert mock_save_img.call_count == 3  # 3 images
            assert mock_save_tensor.call_count == 2  # 2 latent tensors
            assert mock_save_json.call_count == 2  # losses and metrics
            
            # Verify saved_files structure
            expected_keys = ['original', 'initial_recon', 'optimized_recon', 
                           'initial_latents', 'optimized_latents', 'losses', 'metrics']
            for key in expected_keys:
                assert key in saved_files
                assert isinstance(saved_files[key], Path)
    
    def test_save_optimization_results_default_name(self):
        """Test save_optimization_results with default image name"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            saver = ResultsSaver(output_dir)
            
            # Mock data
            tensor_data = torch.rand(1, 3, 32, 32)
            latent_data = torch.rand(1, 4, 4, 4)
            
            with patch.object(saver.io_utils, 'save_image_tensor'), \
                 patch.object(saver.io_utils, 'save_tensor'), \
                 patch.object(saver.io_utils, 'save_json'):
                
                saved_files = saver.save_optimization_results(
                    tensor_data, tensor_data, tensor_data,
                    latent_data, latent_data,
                    [1.0], {}, image_name="result"  # Default name
                )
            
            # Verify file names use default
            assert any("result_original" in str(path) for path in saved_files.values())
    
    def test_save_batch_results(self):
        """Test save_batch_results method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            saver = ResultsSaver(output_dir)
            
            batch_results = [
                {'image': 'img1', 'psnr': 25.0},
                {'image': 'img2', 'psnr': 30.0}
            ]
            
            with patch.object(saver.io_utils, 'save_pickle') as mock_save_pickle:
                result_path = saver.save_batch_results(batch_results, "test_batch")
            
            # Verify save_pickle was called
            mock_save_pickle.assert_called_once_with(
                batch_results, 
                output_dir / "test_batch_results.pkl"
            )
            
            # Verify return path
            assert result_path == output_dir / "test_batch_results.pkl"
    
    def test_save_batch_results_default_name(self):
        """Test save_batch_results with default batch name"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            saver = ResultsSaver(output_dir)
            
            with patch.object(saver.io_utils, 'save_pickle') as mock_save_pickle:
                result_path = saver.save_batch_results([])
            
            # Should use default "batch" name
            expected_path = output_dir / "batch_results.pkl"
            mock_save_pickle.assert_called_once_with([], expected_path)
            assert result_path == expected_path


class TestIOUtilsIntegration:
    """Integration tests for IOUtils functionality"""
    
    def test_tensor_save_load_round_trip(self):
        """Test saving and loading tensor round trip"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tensor_path = Path(temp_dir) / "tensor.pt"
            original_tensor = torch.randn(2, 3, 16, 16)
            
            # Save and load
            IOUtils.save_tensor(original_tensor, tensor_path)
            loaded_tensor = IOUtils.load_tensor(tensor_path)
            
            assert torch.equal(loaded_tensor, original_tensor)
    
    def test_numpy_save_load_round_trip(self):
        """Test saving and loading numpy array round trip"""
        with tempfile.TemporaryDirectory() as temp_dir:
            array_path = Path(temp_dir) / "array.npy"
            original_array = np.random.rand(10, 15)
            
            # Save and load
            IOUtils.save_numpy(original_array, array_path)
            loaded_array = IOUtils.load_numpy(array_path)
            
            assert np.array_equal(loaded_array, original_array)
    
    def test_json_save_load_round_trip(self):
        """Test saving and loading JSON round trip"""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "data.json"
            original_data = {
                'string': 'test',
                'number': 42,
                'float': 3.14,
                'list': [1, 2, 3],
                'nested': {'key': 'value'}
            }
            
            # Save and load
            IOUtils.save_json(original_data, json_path)
            loaded_data = IOUtils.load_json(json_path)
            
            assert loaded_data == original_data
    
    def test_pickle_save_load_round_trip(self):
        """Test saving and loading pickle round trip"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_path = Path(temp_dir) / "data.pkl"
            
            # Complex data including tensors
            original_data = {
                'tensor': torch.randn(2, 2),
                'array': np.random.rand(3, 3),
                'string': 'test',
                'complex_object': {'nested': [1, 2, 3]}
            }
            
            # Save and load
            IOUtils.save_pickle(original_data, pickle_path)
            loaded_data = IOUtils.load_pickle(pickle_path)
            
            assert loaded_data['string'] == original_data['string']
            assert loaded_data['complex_object'] == original_data['complex_object']
            assert torch.equal(loaded_data['tensor'], original_data['tensor'])
            assert np.array_equal(loaded_data['array'], original_data['array'])
    
    @pytest.mark.skipif(not pytest.importorskip("h5py"), reason="h5py not available")
    def test_hdf5_save_load_round_trip(self):
        """Test saving and loading HDF5 round trip"""
        with tempfile.TemporaryDirectory() as temp_dir:
            hdf5_path = Path(temp_dir) / "data.h5"
            original_data = {
                'large_array': np.random.rand(100, 100),
                'small_array': np.random.rand(5, 5),
                'vector': np.random.rand(50)
            }
            
            # Save and load
            IOUtils.save_hdf5(original_data, hdf5_path)
            loaded_data = IOUtils.load_hdf5(hdf5_path)
            
            assert set(loaded_data.keys()) == set(original_data.keys())
            for key in original_data.keys():
                assert np.array_equal(loaded_data[key], original_data[key])


class TestIOUtilsErrorHandling:
    """Test error handling and edge cases"""
    
    def test_load_tensor_nonexistent_file(self):
        """Test load_tensor with non-existent file"""
        with pytest.raises(FileNotFoundError):
            IOUtils.load_tensor("/nonexistent/tensor.pt")
    
    def test_load_numpy_nonexistent_file(self):
        """Test load_numpy with non-existent file"""
        with pytest.raises(FileNotFoundError):
            IOUtils.load_numpy("/nonexistent/array.npy")
    
    def test_load_json_nonexistent_file(self):
        """Test load_json with non-existent file"""
        with pytest.raises(FileNotFoundError):
            IOUtils.load_json("/nonexistent/data.json")
    
    def test_load_pickle_nonexistent_file(self):
        """Test load_pickle with non-existent file"""
        with pytest.raises(FileNotFoundError):
            IOUtils.load_pickle("/nonexistent/data.pkl")
    
    def test_load_json_invalid_format(self):
        """Test load_json with invalid JSON format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "invalid.json"
            
            # Create invalid JSON file
            with open(json_path, 'w') as f:
                f.write("invalid json content")
            
            with pytest.raises(json.JSONDecodeError):
                IOUtils.load_json(json_path)
    
    def test_save_image_tensor_directory_creation(self):
        """Test that save_image_tensor creates parent directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "deep" / "nested" / "path" / "image.png"
            tensor = torch.rand(3, 32, 32)
            
            with patch('generative_latent_optimization.utils.io_utils.Image') as mock_image:
                mock_pil_img = Mock()
                mock_image.fromarray.return_value = mock_pil_img
                
                IOUtils.save_image_tensor(tensor, nested_path)
            
            # Verify parent directories were created
            assert nested_path.parent.exists()


class TestUtilityFunctions:
    """Test standalone utility functions"""
    
    def test_save_image_tensor_function(self):
        """Test save_image_tensor standalone function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.png"
            tensor = torch.rand(3, 32, 32)
            
            with patch.object(IOUtils, 'save_image_tensor') as mock_save:
                save_image_tensor(tensor, output_path)
            
            # Should call IOUtils.save_image_tensor
            mock_save.assert_called_once_with(tensor, output_path)
    
    def test_create_results_saver_function(self):
        """Test create_results_saver factory function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            
            saver = create_results_saver(output_dir)
            
            assert isinstance(saver, ResultsSaver)
            assert saver.output_dir == output_dir
            assert output_dir.exists()


class TestResultsSaverIntegration:
    """Integration tests for ResultsSaver functionality"""
    
    def test_complete_optimization_results_saving(self):
        """Test complete optimization results saving with real I/O"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "optimization_results"
            saver = ResultsSaver(output_dir)
            
            # Create realistic test data
            original = torch.rand(1, 3, 128, 128)
            initial_recon = torch.rand(1, 3, 128, 128)
            optimized_recon = torch.rand(1, 3, 128, 128)
            initial_latents = torch.rand(1, 4, 16, 16)
            optimized_latents = torch.rand(1, 4, 16, 16)
            losses = [1.0, 0.8, 0.6, 0.4, 0.2]
            metrics = {
                'initial_psnr': 20.0,
                'final_psnr': 28.0,
                'psnr_improvement': 8.0,
                'loss_reduction': 80.0
            }
            
            # Mock Image saving to avoid actual image processing
            with patch('generative_latent_optimization.utils.io_utils.Image') as mock_image:
                mock_pil_img = Mock()
                mock_image.fromarray.return_value = mock_pil_img
                
                saved_files = saver.save_optimization_results(
                    original, initial_recon, optimized_recon,
                    initial_latents, optimized_latents,
                    losses, metrics, "complete_test"
                )
            
            # Verify all file types were saved
            assert 'original' in saved_files
            assert 'initial_recon' in saved_files
            assert 'optimized_recon' in saved_files
            assert 'initial_latents' in saved_files
            assert 'optimized_latents' in saved_files
            assert 'losses' in saved_files
            assert 'metrics' in saved_files
            
            # Verify tensor files exist (real I/O)
            assert saved_files['initial_latents'].exists()
            assert saved_files['optimized_latents'].exists()
            
            # Verify JSON files exist and contain correct data
            assert saved_files['losses'].exists()
            assert saved_files['metrics'].exists()
            
            # Load and verify JSON content
            with open(saved_files['losses'], 'r') as f:
                saved_losses = json.load(f)
            assert saved_losses['losses'] == losses
            
            with open(saved_files['metrics'], 'r') as f:
                saved_metrics = json.load(f)
            assert saved_metrics == metrics
    
    def test_batch_results_saving_and_loading(self):
        """Test batch results saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "batch_results"
            saver = ResultsSaver(output_dir)
            
            # Create batch results
            batch_results = [
                {'image': 'img1', 'psnr': 25.0, 'optimization_time': 120.0},
                {'image': 'img2', 'psnr': 28.0, 'optimization_time': 110.0},
                {'image': 'img3', 'psnr': 30.0, 'optimization_time': 100.0}
            ]
            
            # Save batch results
            saved_path = saver.save_batch_results(batch_results, "experiment_1")
            
            # Verify file exists
            assert saved_path.exists()
            assert saved_path.name == "experiment_1_results.pkl"
            
            # Load and verify content
            loaded_results = IOUtils.load_pickle(saved_path)
            assert loaded_results == batch_results


class TestIOUtilsPathHandling:
    """Test path handling across all IOUtils methods"""
    
    def test_string_path_conversion(self):
        """Test that string paths are properly converted to Path objects"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with string path
            string_path = str(Path(temp_dir) / "test_dir")
            
            result_path = IOUtils.create_directory(string_path)
            
            assert isinstance(result_path, Path)
            assert result_path.exists()
    
    def test_path_object_handling(self):
        """Test that Path objects are handled correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with Path object
            path_object = Path(temp_dir) / "test_path_object"
            
            result_path = IOUtils.create_directory(path_object)
            
            assert isinstance(result_path, Path)
            assert result_path == path_object
            assert result_path.exists()
    
    def test_relative_path_handling(self):
        """Test handling of relative paths"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Create directory with relative path
                relative_path = Path("relative") / "nested"
                result_path = IOUtils.create_directory(relative_path)
                
                assert result_path.exists()
                assert result_path.is_absolute()  # Should be converted to absolute
                
            finally:
                os.chdir(original_cwd)


class TestIOUtilsEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_save_image_tensor_edge_values(self):
        """Test save_image_tensor with edge case tensor values"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "edge_case.png"
            
            # Create tensor with edge values (0.0 and 1.0)
            tensor = torch.tensor([[[0.0, 1.0], [0.0, 1.0]],
                                  [[0.5, 0.5], [0.5, 0.5]],
                                  [[1.0, 0.0], [1.0, 0.0]]])
            
            with patch('generative_latent_optimization.utils.io_utils.Image') as mock_image:
                mock_pil_img = Mock()
                mock_image.fromarray.return_value = mock_pil_img
                
                IOUtils.save_image_tensor(tensor, output_path)
            
            # Should handle edge values correctly
            mock_image.fromarray.assert_called_once()
    
    def test_get_image_files_mixed_case_extensions(self):
        """Test get_image_files with mixed case extensions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)
            
            # Create files with mixed case extensions
            files = [
                test_dir / "image1.png",
                test_dir / "image2.PNG",
                test_dir / "image3.jpg",
                test_dir / "image4.JPG",
                test_dir / "image5.jpeg",
                test_dir / "image6.JPEG"
            ]
            
            for file_path in files:
                file_path.touch()
            
            found_files = IOUtils.get_image_files(test_dir)
            
            # Should find all files regardless of case
            assert len(found_files) == 6
    
    def test_empty_data_handling(self):
        """Test handling of empty data structures"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test empty JSON
            json_path = Path(temp_dir) / "empty.json"
            IOUtils.save_json({}, json_path)
            loaded_empty_json = IOUtils.load_json(json_path)
            assert loaded_empty_json == {}
            
            # Test empty pickle
            pickle_path = Path(temp_dir) / "empty.pkl"
            IOUtils.save_pickle([], pickle_path)
            loaded_empty_pickle = IOUtils.load_pickle(pickle_path)
            assert loaded_empty_pickle == []


class TestResultsSaverErrorHandling:
    """Test ResultsSaver error handling"""
    
    def test_results_saver_with_invalid_output_dir(self):
        """Test ResultsSaver with problematic output directory"""
        # Test with None path (should convert to Path properly)
        with patch('generative_latent_optimization.utils.io_utils.Path') as mock_path_class:
            mock_path_instance = Mock()
            mock_path_instance.mkdir = Mock()
            mock_path_class.return_value = mock_path_instance
            
            saver = ResultsSaver(None)
            
            # Should handle None path conversion
            mock_path_class.assert_called_once_with(None)
    
    def test_results_saver_io_error_handling(self):
        """Test ResultsSaver handling of I/O errors"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"
            saver = ResultsSaver(output_dir)
            
            # Create test data
            tensor_data = torch.rand(1, 3, 32, 32)
            latent_data = torch.rand(1, 4, 4, 4)
            
            # Mock io_utils to raise exception on save_tensor
            with patch.object(saver.io_utils, 'save_image_tensor'), \
                 patch.object(saver.io_utils, 'save_tensor', side_effect=Exception("Save failed")), \
                 patch.object(saver.io_utils, 'save_json'):
                
                # Should propagate the exception
                with pytest.raises(Exception, match="Save failed"):
                    saver.save_optimization_results(
                        tensor_data, tensor_data, tensor_data,
                        latent_data, latent_data,
                        [1.0], {}, "error_test"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])