"""Path utilities for consistent path handling across the package."""

from pathlib import Path
from typing import Union, List, Optional
import os


class PathUtils:
    """Unified path processing utility class.
    
    Provides consistent path resolution, validation, and manipulation
    methods to reduce code duplication across modules.
    """
    
    @staticmethod
    def resolve_path(path: Union[str, Path]) -> Path:
        """Resolve a path to its absolute form.
        
        Args:
            path: Input path as string or Path object
            
        Returns:
            Resolved absolute Path object
        """
        return Path(path).resolve()
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            Resolved directory Path object
        """
        resolved_path = PathUtils.resolve_path(path)
        resolved_path.mkdir(parents=True, exist_ok=True)
        return resolved_path
    
    @staticmethod
    def validate_file_exists(path: Union[str, Path]) -> Path:
        """Validate that a file exists.
        
        Args:
            path: File path
            
        Returns:
            Resolved file Path object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        resolved_path = PathUtils.resolve_path(path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {resolved_path}")
        if not resolved_path.is_file():
            raise ValueError(f"Path is not a file: {resolved_path}")
        return resolved_path
    
    @staticmethod
    def validate_directory_exists(path: Union[str, Path]) -> Path:
        """Validate that a directory exists.
        
        Args:
            path: Directory path
            
        Returns:
            Resolved directory Path object
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
        """
        resolved_path = PathUtils.resolve_path(path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Directory not found: {resolved_path}")
        if not resolved_path.is_dir():
            raise ValueError(f"Path is not a directory: {resolved_path}")
        return resolved_path
    
    @staticmethod
    def get_file_extension(path: Union[str, Path]) -> str:
        """Get the file extension from a path.
        
        Args:
            path: File path
            
        Returns:
            File extension (including the dot)
        """
        resolved_path = PathUtils.resolve_path(path)
        return resolved_path.suffix
    
    @staticmethod
    def get_filename_without_extension(path: Union[str, Path]) -> str:
        """Get the filename without extension.
        
        Args:
            path: File path
            
        Returns:
            Filename without extension
        """
        resolved_path = PathUtils.resolve_path(path)
        return resolved_path.stem
    
    @staticmethod
    def list_files_with_extension(
        directory: Union[str, Path],
        extension: str,
        recursive: bool = False
    ) -> List[Path]:
        """List all files with a specific extension in a directory.
        
        Args:
            directory: Directory path
            extension: File extension (e.g., '.png', '.jpg')
            recursive: Whether to search recursively
            
        Returns:
            List of file paths
        """
        dir_path = PathUtils.validate_directory_exists(directory)
        
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        if recursive:
            pattern = f'**/*{extension}'
        else:
            pattern = f'*{extension}'
        
        return sorted(dir_path.glob(pattern))
    
    @staticmethod
    def create_output_path(
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        suffix: Optional[str] = None,
        extension: Optional[str] = None
    ) -> Path:
        """Create an output path based on an input path.
        
        Args:
            input_path: Input file path
            output_dir: Output directory
            suffix: Optional suffix to add to filename
            extension: Optional new extension (replaces original)
            
        Returns:
            Output file path
        """
        input_path = PathUtils.resolve_path(input_path)
        output_dir = PathUtils.ensure_directory(output_dir)
        
        # Get base filename
        filename = input_path.stem
        if suffix:
            filename = f"{filename}{suffix}"
        
        # Get extension
        if extension:
            if not extension.startswith('.'):
                extension = f'.{extension}'
        else:
            extension = input_path.suffix
        
        return output_dir / f"{filename}{extension}"
    
    @staticmethod
    def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:
        """Get the relative path from a base directory.
        
        Args:
            path: Target path
            base: Base directory
            
        Returns:
            Relative path from base to target
        """
        target = PathUtils.resolve_path(path)
        base_path = PathUtils.resolve_path(base)
        
        try:
            return target.relative_to(base_path)
        except ValueError:
            # Paths don't have a common base
            return target
    
    @staticmethod
    def safe_remove_file(path: Union[str, Path]) -> bool:
        """Safely remove a file if it exists.
        
        Args:
            path: File path to remove
            
        Returns:
            True if file was removed, False if it didn't exist
        """
        try:
            resolved_path = PathUtils.resolve_path(path)
            if resolved_path.exists() and resolved_path.is_file():
                resolved_path.unlink()
                return True
        except Exception as e:
            print(f"Warning: Could not remove file {path}: {e}")
        return False
    
    @staticmethod
    def get_file_size(path: Union[str, Path]) -> int:
        """Get the size of a file in bytes.
        
        Args:
            path: File path
            
        Returns:
            File size in bytes
        """
        resolved_path = PathUtils.validate_file_exists(path)
        return resolved_path.stat().st_size
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string (e.g., "1.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"