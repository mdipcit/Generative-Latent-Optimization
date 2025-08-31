"""Unified device management for the Generative Latent Optimization package."""

from typing import Dict, Any, Optional, Union, List
import torch


class DeviceManager:
    """Unified device management class.
    
    Provides consistent device detection, validation, and tensor movement
    across all modules in the package.
    """
    
    def __init__(self, device: str = 'cuda'):
        """Initialize the device manager.
        
        Args:
            device: Requested device ('cuda', 'cpu', or specific cuda device like 'cuda:0')
        """
        self.device = self._detect_optimal_device(device)
        self._log_device_info()
    
    def _detect_optimal_device(self, requested: str) -> str:
        """Detect and validate the optimal device for computation.
        
        Args:
            requested: Requested device string
            
        Returns:
            Valid device string
        """
        # Handle specific cuda device requests
        if requested.startswith('cuda'):
            if not torch.cuda.is_available():
                print(f"Warning: CUDA requested but not available, using CPU")
                return 'cpu'
            
            # Extract device index if specified
            if ':' in requested:
                try:
                    device_idx = int(requested.split(':')[1])
                    if device_idx >= torch.cuda.device_count():
                        print(f"Warning: CUDA device {device_idx} not available, using device 0")
                        return 'cuda:0'
                    return requested
                except (ValueError, IndexError):
                    print(f"Warning: Invalid device format '{requested}', using cuda:0")
                    return 'cuda:0'
            else:
                # Use current device
                return f'cuda:{torch.cuda.current_device()}'
        
        return requested
    
    def _log_device_info(self) -> None:
        """Log device information for debugging."""
        if self.device.startswith('cuda'):
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            device_name = torch.cuda.get_device_name(device_idx)
            print(f"Using device: {self.device} ({device_name})")
        else:
            print(f"Using device: {self.device}")
    
    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the configured device.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor on the configured device
        """
        return tensor.to(self.device)
    
    def move_batch_to_device(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Move a batch of tensors to the configured device.
        
        Args:
            tensors: List of input tensors
            
        Returns:
            List of tensors on the configured device
        """
        return [self.move_to_device(t) for t in tensors]
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information.
        
        Returns:
            Dictionary containing device information
        """
        info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if self.device.startswith('cuda'):
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            info.update({
                'device_name': torch.cuda.get_device_name(device_idx),
                'total_memory': torch.cuda.get_device_properties(device_idx).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(device_idx),
                'memory_reserved': torch.cuda.memory_reserved(device_idx),
            })
        
        return info
    
    def ensure_same_device(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        """Ensure all tensors are on the same device as the manager.
        
        Args:
            *tensors: Variable number of tensors
            
        Returns:
            List of tensors all on the same device
        """
        return [self.move_to_device(t) if t.device != self.device else t for t in tensors]
    
    def get_memory_summary(self) -> Optional[str]:
        """Get a summary of GPU memory usage.
        
        Returns:
            Memory summary string if using CUDA, None otherwise
        """
        if not self.device.startswith('cuda'):
            return None
        
        device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
        allocated = torch.cuda.memory_allocated(device_idx) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device_idx) / 1024**3  # GB
        total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3  # GB
        
        return (f"GPU Memory: {allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved, {total:.2f}GB total")
    
    def synchronize(self) -> None:
        """Synchronize CUDA operations if using CUDA device."""
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)
    
    @staticmethod
    def auto_select_device() -> str:
        """Automatically select the best available device.
        
        Returns:
            Best available device string
        """
        if torch.cuda.is_available():
            # Select GPU with most free memory
            max_free_memory = 0
            best_device = 0
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_memory = props.total_memory - torch.cuda.memory_reserved(i)
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = i
            
            return f'cuda:{best_device}'
        else:
            return 'cpu'