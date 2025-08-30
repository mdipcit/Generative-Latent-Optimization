#!/usr/bin/env python3
"""
Dataset-Level Metrics Module

Provides dataset-level quality evaluation metrics, specifically FID (Fréchet Inception Distance).
These metrics compare entire datasets rather than individual images.
"""

import os
import datetime
import tempfile
import shutil
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging

import torch
from torchvision.utils import save_image

try:
    from .image_metrics import DatasetEvaluationResults
except ImportError:
    # For direct execution as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from image_metrics import DatasetEvaluationResults

logger = logging.getLogger(__name__)


class DatasetFIDEvaluator:
    """
    Dataset-level FID evaluation class
    
    Compares entire datasets using Fréchet Inception Distance (FID).
    Designed for evaluating VAE-optimized datasets against original BSDS500 dataset.
    NOT used for individual image evaluation.
    """
    
    def __init__(self, batch_size=50, dims=2048, device='cuda', num_workers=4):
        """
        Initialize FID evaluator
        
        Args:
            batch_size: Batch size for FID computation
            dims: Inception feature dimensions (2048 recommended)
            device: Computation device
            num_workers: Number of dataloader workers
        """
        self.batch_size = batch_size
        self.dims = dims
        self.device = device
        self.num_workers = num_workers
        
        # Verify pytorch-fid is available
        try:
            import pytorch_fid.fid_score as fid_score
            self.fid_score = fid_score
            logger.info(f"FID evaluator initialized: batch_size={batch_size}, dims={dims}, device={device}")
        except ImportError:
            raise ImportError("pytorch-fid package is required but not installed. Install with: pip install pytorch-fid")
    
    def evaluate_created_dataset_vs_original(self, 
                                           created_dataset_path: Union[str, Path],
                                           original_dataset_path: Union[str, Path]) -> DatasetEvaluationResults:
        """
        Compare created dataset directory against original dataset directory
        
        Args:
            created_dataset_path: VAE-optimized dataset directory (containing PNG images)
            original_dataset_path: Original BSDS500 dataset directory
        
        Returns:
            DatasetEvaluationResults: Complete FID evaluation results
        """
        created_path = Path(created_dataset_path).resolve()
        original_path = Path(original_dataset_path).resolve()
        
        # Validate dataset paths
        self._validate_dataset_paths(created_path, original_path)
        
        # Execute FID computation
        logger.info("Starting dataset FID evaluation...")
        logger.info(f"  Created dataset: {created_path}")
        logger.info(f"  Original dataset: {original_path}")
        
        try:
            fid_value = self.fid_score.calculate_fid_given_paths(
                paths=[str(original_path), str(created_path)],
                batch_size=self.batch_size,
                device=self.device,
                dims=self.dims,
                num_workers=self.num_workers
            )
            
            logger.info(f"FID computation completed: {fid_value:.2f}")
            
        except Exception as e:
            logger.error(f"FID computation failed: {e}")
            raise
        
        # Collect result metadata
        total_created = len(list(created_path.glob('**/*.png')))
        total_original = len(list(original_path.glob('**/*.png')))
        
        return DatasetEvaluationResults(
            fid_score=fid_value,
            total_images=total_created,
            original_dataset_path=str(original_path),
            generated_dataset_path=str(created_path),
            evaluation_timestamp=datetime.datetime.now().isoformat(),
            individual_metrics_summary={}  # To be populated separately
        )
    
    def evaluate_pytorch_dataset_vs_original(self,
                                           pytorch_dataset_path: Union[str, Path],
                                           original_bsds500_path: Union[str, Path]) -> DatasetEvaluationResults:
        """
        Extract images from PyTorch dataset (.pt) and compare with original dataset
        
        Args:
            pytorch_dataset_path: Created .pt dataset file
            original_bsds500_path: Original BSDS500 directory
        
        Returns:
            DatasetEvaluationResults: Complete FID evaluation results
        """
        pytorch_path = Path(pytorch_dataset_path).resolve()
        original_path = Path(original_bsds500_path).resolve()
        
        if not pytorch_path.exists():
            raise FileNotFoundError(f"PyTorch dataset not found: {pytorch_path}")
        if not original_path.exists():
            raise FileNotFoundError(f"Original dataset not found: {original_path}")
        
        logger.info("Loading PyTorch dataset for FID evaluation...")
        
        # Load PyTorch dataset
        try:
            # Import here to avoid circular imports
            try:
                from ..dataset import load_optimized_dataset
            except ImportError:
                # For direct execution, try alternative import paths
                import sys
                from pathlib import Path
                parent_path = Path(__file__).parent.parent
                sys.path.append(str(parent_path))
                from dataset import load_optimized_dataset
            
            dataset = load_optimized_dataset(pytorch_dataset_path)
            logger.info(f"Loaded dataset with {len(dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load PyTorch dataset: {e}")
            raise
        
        # Extract reconstructed images to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            reconstructed_dir = Path(temp_dir) / 'reconstructed'
            reconstructed_dir.mkdir()
            
            logger.info(f"Extracting reconstructed images to temporary directory...")
            
            for i in range(len(dataset)):
                try:
                    sample = dataset[i]
                    
                    # Try to get reconstructed image from various possible keys
                    reconstructed_img = None
                    for key in ['reconstructed_image', 'final_reconstructed', 'optimized_reconstructed']:
                        if key in sample:
                            reconstructed_img = sample[key]
                            break
                    
                    if reconstructed_img is None:
                        # If no reconstructed image found, decode from optimized latents
                        if 'optimized_latents' in sample:
                            logger.warning(f"No reconstructed image found for sample {i}, using optimized latents")
                            # This would require VAE decoder - for now, skip
                            continue
                        else:
                            logger.warning(f"No usable image data found for sample {i}")
                            continue
                    
                    # Save reconstructed image
                    save_image(reconstructed_img, 
                             reconstructed_dir / f'{i:05d}.png',
                             normalize=True,
                             value_range=(-1, 1) if reconstructed_img.min() < 0 else (0, 1))
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {i}: {e}")
                    continue
            
            # Check if we have enough images for meaningful FID calculation
            extracted_images = list(reconstructed_dir.glob('*.png'))
            if len(extracted_images) < 10:
                logger.warning(f"Only {len(extracted_images)} images extracted, FID may not be reliable")
            
            # Perform FID evaluation
            return self.evaluate_created_dataset_vs_original(
                reconstructed_dir, 
                original_path
            )
    
    def _validate_dataset_paths(self, created_path: Path, original_path: Path):
        """
        Validate dataset paths and log statistics
        
        Args:
            created_path: Path to created dataset
            original_path: Path to original dataset
            
        Raises:
            ValueError: If datasets are invalid or empty
        """
        if not created_path.exists():
            raise ValueError(f"Created dataset not found: {created_path}")
        if not original_path.exists():
            raise ValueError(f"Original dataset not found: {original_path}")
        
        # Count images in datasets
        created_images = list(created_path.glob('**/*.png'))
        created_images.extend(list(created_path.glob('**/*.jpg')))
        created_images.extend(list(created_path.glob('**/*.jpeg')))
        
        original_images = list(original_path.glob('**/*.png'))
        original_images.extend(list(original_path.glob('**/*.jpg')))
        original_images.extend(list(original_path.glob('**/*.jpeg')))
        
        if len(created_images) == 0:
            raise ValueError(f"No images found in created dataset: {created_path}")
        if len(original_images) == 0:
            raise ValueError(f"No images found in original dataset: {original_path}")
        
        logger.info("Dataset validation successful:")
        logger.info(f"  Created dataset: {len(created_images)} images")
        logger.info(f"  Original dataset: {len(original_images)} images")
        
        # Check if datasets have reasonable sizes for FID computation
        if len(created_images) < 10:
            logger.warning(f"Created dataset has only {len(created_images)} images - FID may not be reliable")
        if len(original_images) < 50:
            logger.warning(f"Original dataset has only {len(original_images)} images - consider using full dataset")


# Utility functions for testing and validation
def test_fid_evaluator_with_dummy_data(device='cuda'):
    """
    Test FID evaluator functionality with dummy data
    
    Args:
        device: Computation device
    """
    print("Testing FID evaluator with dummy data...")
    
    try:
        evaluator = DatasetFIDEvaluator(batch_size=10, device=device)
        
        # Create temporary directories with dummy images
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy dataset 1
            dataset1_dir = temp_path / 'dataset1'
            dataset1_dir.mkdir()
            
            # Create dummy dataset 2  
            dataset2_dir = temp_path / 'dataset2'
            dataset2_dir.mkdir()
            
            # Generate dummy images
            print("  Generating dummy images...")
            for i in range(20):
                # Dataset 1: Random images
                img1 = torch.rand(3, 64, 64)
                save_image(img1, dataset1_dir / f'img_{i:03d}.png', normalize=True)
                
                # Dataset 2: Slightly different random images
                img2 = torch.rand(3, 64, 64)
                save_image(img2, dataset2_dir / f'img_{i:03d}.png', normalize=True)
            
            # Test FID computation
            print("  Computing FID between dummy datasets...")
            result = evaluator.evaluate_created_dataset_vs_original(
                dataset2_dir, dataset1_dir
            )
            
            print(f"  FID Score: {result.fid_score:.2f}")
            print(f"  Total images: {result.total_images}")
            print(f"  Test passed: FID computed successfully")
            
    except Exception as e:
        print(f"  FID evaluator test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run functionality tests
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running FID evaluator tests on device: {device}")
    
    test_fid_evaluator_with_dummy_data(device)
    
    print("Dataset metrics module tests completed.")