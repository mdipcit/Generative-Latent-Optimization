#!/usr/bin/env python3
"""
I/O Utilities Module

Provides file input/output utilities for saving images, tensors,
and other data formats used in the VAE optimization pipeline.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Union, Any, Optional
import pickle
import h5py


class IOUtils:
    """Utility class for file I/O operations"""
    
    @staticmethod
    def save_image_tensor(tensor: torch.Tensor, path: Union[str, Path], 
                         format: str = "PNG") -> None:
        """
        Convert tensor to PIL image and save
        
        Args:
            tensor: Image tensor [1, C, H, W] or [C, H, W] in [0, 1] range
            path: Output path for the image
            format: Image format (PNG, JPEG, etc.)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle tensor dimensions
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension
        elif tensor.dim() != 3:
            raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")
        
        # Convert from [C, H, W] to [H, W, C]
        image_np = tensor.detach().cpu().permute(1, 2, 0).numpy()
        
        # Ensure values are in [0, 1] range
        image_np = np.clip(image_np, 0, 1)
        
        # Convert to uint8
        image_np = (image_np * 255).astype(np.uint8)
        
        # Handle grayscale images
        if image_np.shape[2] == 1:
            image_np = image_np.squeeze(2)
        
        # Save as PIL image
        Image.fromarray(image_np).save(path, format=format)
    
    @staticmethod
    def save_tensor(tensor: torch.Tensor, path: Union[str, Path]) -> None:
        """Save tensor to .pt file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)
    
    @staticmethod
    def load_tensor(path: Union[str, Path]) -> torch.Tensor:
        """Load tensor from .pt file"""
        return torch.load(path)
    
    @staticmethod
    def save_numpy(array: np.ndarray, path: Union[str, Path]) -> None:
        """Save numpy array to .npy file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
    
    @staticmethod
    def load_numpy(path: Union[str, Path]) -> np.ndarray:
        """Load numpy array from .npy file"""
        return np.load(path)
    
    @staticmethod
    def save_json(data: Dict[str, Any], path: Union[str, Path], 
                  indent: int = 2) -> None:
        """Save dictionary to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
    
    @staticmethod
    def load_json(path: Union[str, Path]) -> Dict[str, Any]:
        """Load dictionary from JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_pickle(data: Any, path: Union[str, Path]) -> None:
        """Save data using pickle"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load_pickle(path: Union[str, Path]) -> Any:
        """Load data using pickle"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_hdf5(data: Dict[str, np.ndarray], path: Union[str, Path]) -> None:
        """
        Save multiple arrays to HDF5 file
        
        Args:
            data: Dictionary with array names as keys and numpy arrays as values
            path: Output HDF5 file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(path, 'w') as f:
            for key, array in data.items():
                f.create_dataset(key, data=array, compression='gzip')
    
    @staticmethod
    def load_hdf5(path: Union[str, Path], keys: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Load arrays from HDF5 file
        
        Args:
            path: HDF5 file path
            keys: Specific keys to load (if None, load all)
            
        Returns:
            Dictionary with loaded arrays
        """
        data = {}
        with h5py.File(path, 'r') as f:
            load_keys = keys if keys is not None else list(f.keys())
            for key in load_keys:
                if key in f:
                    data[key] = f[key][:]
        return data
    
    @staticmethod
    def create_directory(path: Union[str, Path]) -> Path:
        """Create directory if it doesn't exist"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_image_files(directory: Union[str, Path], 
                       extensions: List[str] = ['.png', '.jpg', '.jpeg']) -> List[Path]:
        """
        Get all image files in directory
        
        Args:
            directory: Directory to search
            extensions: List of file extensions to include
            
        Returns:
            List of image file paths
        """
        directory = Path(directory)
        files = []
        
        for ext in extensions:
            files.extend(directory.glob(f'*{ext}'))
            files.extend(directory.glob(f'*{ext.upper()}'))
        
        return sorted(files)
    
    @staticmethod
    def ensure_path_exists(path: Union[str, Path]) -> Path:
        """Ensure parent directory exists for given path"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class ResultsSaver:
    """
    Specialized class for saving optimization results
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.io_utils = IOUtils()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_optimization_results(self, 
                                original_image: torch.Tensor,
                                initial_reconstruction: torch.Tensor,
                                optimized_reconstruction: torch.Tensor,
                                initial_latents: torch.Tensor,
                                optimized_latents: torch.Tensor,
                                losses: List[float],
                                metrics: Dict[str, float],
                                image_name: str = "result") -> Dict[str, Path]:
        """
        Save complete optimization results
        
        Returns:
            Dictionary with paths to saved files
        """
        saved_files = {}
        
        # Save images
        image_paths = {
            'original': self.output_dir / f"{image_name}_original.png",
            'initial_recon': self.output_dir / f"{image_name}_initial_reconstruction.png",
            'optimized_recon': self.output_dir / f"{image_name}_optimized_reconstruction.png"
        }
        
        self.io_utils.save_image_tensor(original_image, image_paths['original'])
        self.io_utils.save_image_tensor(initial_reconstruction, image_paths['initial_recon'])
        self.io_utils.save_image_tensor(optimized_reconstruction, image_paths['optimized_recon'])
        
        saved_files.update(image_paths)
        
        # Save latents
        latent_paths = {
            'initial_latents': self.output_dir / f"{image_name}_initial_latents.pt",
            'optimized_latents': self.output_dir / f"{image_name}_optimized_latents.pt"
        }
        
        self.io_utils.save_tensor(initial_latents, latent_paths['initial_latents'])
        self.io_utils.save_tensor(optimized_latents, latent_paths['optimized_latents'])
        
        saved_files.update(latent_paths)
        
        # Save metrics and losses
        data_paths = {
            'losses': self.output_dir / f"{image_name}_losses.json",
            'metrics': self.output_dir / f"{image_name}_metrics.json"
        }
        
        self.io_utils.save_json({'losses': losses}, data_paths['losses'])
        self.io_utils.save_json(metrics, data_paths['metrics'])
        
        saved_files.update(data_paths)
        
        return saved_files
    
    def save_batch_results(self, results_list: List[Dict[str, Any]], 
                          batch_name: str = "batch") -> Path:
        """
        Save batch processing results to a single file
        
        Args:
            results_list: List of dictionaries containing results
            batch_name: Name prefix for the batch file
            
        Returns:
            Path to saved batch file
        """
        batch_file = self.output_dir / f"{batch_name}_results.pkl"
        self.io_utils.save_pickle(results_list, batch_file)
        return batch_file


# Utility functions for backward compatibility
def save_image_tensor(tensor: torch.Tensor, path: Union[str, Path]) -> None:
    """Standalone function for saving image tensor"""
    IOUtils.save_image_tensor(tensor, path)


def create_results_saver(output_dir: Union[str, Path]) -> ResultsSaver:
    """Factory function for creating ResultsSaver"""
    return ResultsSaver(output_dir)