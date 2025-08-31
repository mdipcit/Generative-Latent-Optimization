"""Unified image loading utilities for the package."""

from pathlib import Path
from typing import Union, Tuple, Optional, List
import torch
from PIL import Image
import numpy as np
from .path_utils import PathUtils


class UnifiedImageLoader:
    """Unified image loading and preprocessing class.
    
    Provides consistent image loading, preprocessing, and tensor conversion
    methods to consolidate different image loading implementations across modules.
    """
    
    def __init__(self, device: str = 'cuda'):
        """Initialize the image loader.
        
        Args:
            device: Device to load tensors to
        """
        from ..core.device_manager import DeviceManager
        self.device_manager = DeviceManager(device)
        self.device = self.device_manager.device
    
    def load_image(
        self,
        path: Union[str, Path],
        target_size: Optional[int] = None,
        normalize: bool = True,
        to_device: bool = True
    ) -> torch.Tensor:
        """Load a single image as a tensor.
        
        Args:
            path: Path to the image file
            target_size: Optional target size for resizing (maintains aspect ratio)
            normalize: Whether to normalize to [0, 1] range
            to_device: Whether to move tensor to configured device
            
        Returns:
            Image tensor with shape (C, H, W)
        """
        # Validate path
        image_path = PathUtils.validate_file_exists(path)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize if needed
        if target_size:
            image = self._resize_image(image, target_size)
        
        # Convert to tensor
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).float()
        
        # Rearrange dimensions from (H, W, C) to (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        
        # Normalize if requested
        if normalize:
            image_tensor = image_tensor / 255.0
        
        # Move to device if requested
        if to_device:
            image_tensor = self.device_manager.move_to_device(image_tensor)
        
        return image_tensor
    
    def load_image_pair(
        self,
        path1: Union[str, Path],
        path2: Union[str, Path],
        target_size: Optional[int] = None,
        normalize: bool = True,
        to_device: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a pair of images as tensors.
        
        Args:
            path1: Path to the first image
            path2: Path to the second image
            target_size: Optional target size for resizing
            normalize: Whether to normalize to [0, 1] range
            to_device: Whether to move tensors to configured device
            
        Returns:
            Tuple of image tensors
        """
        img1 = self.load_image(path1, target_size, normalize, to_device)
        img2 = self.load_image(path2, target_size, normalize, to_device)
        return img1, img2
    
    def load_batch_images(
        self,
        paths: List[Union[str, Path]],
        target_size: Optional[int] = None,
        normalize: bool = True,
        to_device: bool = True,
        batch_format: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Load a batch of images.
        
        Args:
            paths: List of image paths
            target_size: Optional target size for resizing
            normalize: Whether to normalize to [0, 1] range
            to_device: Whether to move tensors to configured device
            batch_format: If True, stack into batch tensor; if False, return list
            
        Returns:
            Batch tensor with shape (B, C, H, W) or list of tensors
        """
        images = []
        for path in paths:
            img = self.load_image(path, target_size, normalize, to_device=False)
            images.append(img)
        
        if batch_format:
            # Stack into batch tensor
            batch_tensor = torch.stack(images)
            if to_device:
                batch_tensor = self.device_manager.move_to_device(batch_tensor)
            return batch_tensor
        else:
            # Return list of tensors
            if to_device:
                images = self.device_manager.move_batch_to_device(images)
            return images
    
    def save_image(
        self,
        tensor: torch.Tensor,
        path: Union[str, Path],
        denormalize: bool = True,
        quality: int = 95
    ) -> Path:
        """Save a tensor as an image.
        
        Args:
            tensor: Image tensor with shape (C, H, W) or (B, C, H, W)
            path: Output path for the image
            denormalize: Whether to denormalize from [0, 1] to [0, 255]
            quality: JPEG quality (1-100)
            
        Returns:
            Path to the saved image
        """
        # Handle batch dimension
        if tensor.dim() == 4:
            # Take first image from batch
            tensor = tensor[0]
        
        # Move to CPU if needed
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Denormalize if needed
        if denormalize:
            tensor = tensor * 255.0
        
        # Clamp values
        tensor = torch.clamp(tensor, 0, 255)
        
        # Convert to numpy and rearrange dimensions
        image_array = tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Save image
        image = Image.fromarray(image_array)
        output_path = PathUtils.resolve_path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format based on extension
        extension = output_path.suffix.lower()
        if extension in ['.jpg', '.jpeg']:
            image.save(output_path, 'JPEG', quality=quality)
        else:
            image.save(output_path)
        
        return output_path
    
    def _resize_image(self, image: Image.Image, target_size: int) -> Image.Image:
        """Resize image maintaining aspect ratio.
        
        Args:
            image: PIL Image
            target_size: Target size for the smaller dimension
            
        Returns:
            Resized PIL Image
        """
        width, height = image.size
        
        # Calculate new dimensions maintaining aspect ratio
        if width < height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
        """Convert a tensor to numpy array for visualization.
        
        Args:
            tensor: Image tensor with shape (C, H, W)
            denormalize: Whether to denormalize from [0, 1] to [0, 255]
            
        Returns:
            Numpy array with shape (H, W, C)
        """
        # Move to CPU if needed
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Denormalize if needed
        if denormalize:
            tensor = tensor * 255.0
        
        # Clamp and convert
        tensor = torch.clamp(tensor, 0, 255)
        return tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    
    @staticmethod
    def numpy_to_tensor(
        array: np.ndarray,
        normalize: bool = True,
        device: Optional[str] = None
    ) -> torch.Tensor:
        """Convert a numpy array to tensor.
        
        Args:
            array: Numpy array with shape (H, W, C)
            normalize: Whether to normalize to [0, 1] range
            device: Optional device to move tensor to
            
        Returns:
            Tensor with shape (C, H, W)
        """
        tensor = torch.from_numpy(array).float()
        
        # Rearrange dimensions
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)
        
        # Normalize if needed
        if normalize and tensor.max() > 1.0:
            tensor = tensor / 255.0
        
        # Move to device if specified
        if device:
            tensor = tensor.to(device)
        
        return tensor
    
    def validate_image_paths(
        self,
        *paths: Union[str, Path]
    ) -> List[Path]:
        """Validate multiple image paths.
        
        Args:
            *paths: Variable number of image paths
            
        Returns:
            List of validated Path objects
            
        Raises:
            FileNotFoundError: If any path doesn't exist
            ValueError: If any path is not an image file
        """
        validated_paths = []
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        for path in paths:
            validated_path = PathUtils.validate_file_exists(path)
            
            # Check if it's an image file
            if validated_path.suffix.lower() not in valid_extensions:
                raise ValueError(f"Not an image file: {validated_path}")
            
            validated_paths.append(validated_path)
        
        return validated_paths