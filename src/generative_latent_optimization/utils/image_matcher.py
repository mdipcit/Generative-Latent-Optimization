"""Image matching utilities for dataset evaluation."""

from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageMatcher:
    """Handles matching of image pairs between datasets by filename."""
    
    def __init__(self, match_strategy: str = 'stem'):
        """Initialize image matcher.
        
        Args:
            match_strategy: Strategy for matching images ('stem' or 'full_name')
        """
        self.match_strategy = match_strategy
        
    def get_image_files(self, path: Path) -> List[Path]:
        """Get all image files from a directory.
        
        Args:
            path: Directory path to search for images
            
        Returns:
            List of image file paths
        """
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = []
        
        if path.is_file():
            if path.suffix.lower() in image_extensions:
                return [path]
            else:
                raise ValueError(f"File {path} is not a supported image format")
        
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a valid file or directory")
        
        for ext in image_extensions:
            image_files.extend(path.glob(f"*{ext}"))
            image_files.extend(path.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def match_image_pairs(self, original_images: List[Path], created_images: List[Path]) -> List[Tuple[Path, Path]]:
        """Match images between original and created datasets.
        
        Args:
            original_images: List of original image paths
            created_images: List of created image paths
            
        Returns:
            List of (original_path, created_path) pairs
        """
        if self.match_strategy == 'stem':
            return self._match_by_stem(original_images, created_images)
        elif self.match_strategy == 'full_name':
            return self._match_by_full_name(original_images, created_images)
        else:
            raise ValueError(f"Unknown match strategy: {self.match_strategy}")
    
    def _match_by_stem(self, original_images: List[Path], created_images: List[Path]) -> List[Tuple[Path, Path]]:
        """Match images by filename stem (without extension).
        
        Args:
            original_images: List of original image paths
            created_images: List of created image paths
            
        Returns:
            List of (original_path, created_path) pairs
        """
        # Create dictionaries for efficient lookup
        original_dict = {img.stem: img for img in original_images}
        created_dict = {img.stem: img for img in created_images}
        
        # Find matching pairs
        pairs = []
        for stem in original_dict.keys():
            if stem in created_dict:
                pairs.append((original_dict[stem], created_dict[stem]))
        
        return pairs
    
    def _match_by_full_name(self, original_images: List[Path], created_images: List[Path]) -> List[Tuple[Path, Path]]:
        """Match images by full filename (including extension).
        
        Args:
            original_images: List of original image paths
            created_images: List of created image paths
            
        Returns:
            List of (original_path, created_path) pairs
        """
        # Create dictionaries for efficient lookup
        original_dict = {img.name: img for img in original_images}
        created_dict = {img.name: img for img in created_images}
        
        # Find matching pairs
        pairs = []
        for name in original_dict.keys():
            if name in created_dict:
                pairs.append((original_dict[name], created_dict[name]))
        
        return pairs
    
    def find_image_pairs(self, created_path: Path, original_path: Path) -> List[Tuple[Path, Path]]:
        """Complete workflow to find matching image pairs between two datasets.
        
        Args:
            created_path: Path to created dataset
            original_path: Path to original dataset
            
        Returns:
            List of (original_path, created_path) pairs
            
        Raises:
            ValueError: If no matching pairs are found
        """
        # Get image files from both datasets
        created_images = self.get_image_files(created_path)
        original_images = self.get_image_files(original_path)
        
        logger.info(f"Found {len(created_images)} created images")
        logger.info(f"Found {len(original_images)} original images")
        
        # Match images by filename
        image_pairs = self.match_image_pairs(original_images, created_images)
        
        if not image_pairs:
            raise ValueError(f"No matching image pairs found between {created_path} and {original_path}")
        
        logger.info(f"Matched {len(image_pairs)} image pairs using strategy '{self.match_strategy}'")
        
        return image_pairs
    
    def get_matching_statistics(self, created_path: Path, original_path: Path) -> Dict[str, int]:
        """Get statistics about matching between two datasets.
        
        Args:
            created_path: Path to created dataset
            original_path: Path to original dataset
            
        Returns:
            Dictionary with matching statistics
        """
        created_images = self.get_image_files(created_path)
        original_images = self.get_image_files(original_path)
        image_pairs = self.match_image_pairs(original_images, created_images)
        
        return {
            'total_original': len(original_images),
            'total_created': len(created_images),
            'matched_pairs': len(image_pairs),
            'unmatched_original': len(original_images) - len(image_pairs),
            'unmatched_created': len(created_images) - len(image_pairs)
        }