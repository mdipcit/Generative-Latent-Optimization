#!/usr/bin/env python3
"""
Simple All Metrics Evaluator

Provides a straightforward way to compute all metrics for a dataset.
Focuses on practical utility rather than complex analysis.
"""

import os
import datetime
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
import logging
from tqdm import tqdm

import torch
from PIL import Image

try:
    from ..metrics.metrics_integration import IndividualMetricsCalculator
    from ..metrics.dataset_metrics import DatasetFIDEvaluator
    from ..metrics.image_metrics import AllMetricsResults, IndividualImageMetrics
except ImportError:
    # For direct execution as script
    import sys
    parent_path = Path(__file__).parent.parent
    sys.path.append(str(parent_path))
    from metrics.metrics_integration import IndividualMetricsCalculator
    from metrics.dataset_metrics import DatasetFIDEvaluator
    from metrics.image_metrics import AllMetricsResults, IndividualImageMetrics

logger = logging.getLogger(__name__)


class SimpleAllMetricsEvaluator:
    """
    Simple All Metrics Evaluator
    
    A straightforward evaluator that computes all metrics for a dataset
    and displays results in an easy-to-understand format.
    
    Supported metrics:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)  
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - Improved SSIM (TorchMetrics implementation)
    - MSE/MAE (Mean Squared/Absolute Error)
    - FID (Fréchet Inception Distance)
    """
    
    def __init__(self, device='cuda', enable_lpips=True, enable_improved_ssim=True):
        """
        Initialize the simple evaluator
        
        Args:
            device: Computation device ('cuda' or 'cpu')
            enable_lpips: Whether to enable LPIPS calculation
            enable_improved_ssim: Whether to enable improved SSIM calculation
        """
        self.device = device
        
        # Initialize individual metrics calculator
        self.individual_calculator = IndividualMetricsCalculator(
            device=device,
            enable_lpips=enable_lpips,
            enable_improved_ssim=enable_improved_ssim
        )
        
        # Initialize FID evaluator
        self.fid_evaluator = DatasetFIDEvaluator(device=device)
        
        logger.info(f"Simple All Metrics Evaluator initialized on {device}")
        logger.info(f"  LPIPS: {'enabled' if enable_lpips else 'disabled'}")
        logger.info(f"  Improved SSIM: {'enabled' if enable_improved_ssim else 'disabled'}")
    
    def evaluate_dataset_all_metrics(self, 
                                   created_dataset_path: Union[str, Path],
                                   original_dataset_path: Union[str, Path]) -> AllMetricsResults:
        """
        Evaluate a dataset with all available metrics
        
        Args:
            created_dataset_path: Path to VAE-optimized dataset (PNG images)
            original_dataset_path: Path to original BSDS500 dataset (PNG images)
        
        Returns:
            AllMetricsResults: Complete evaluation results
        """
        created_path = Path(created_dataset_path).resolve()
        original_path = Path(original_dataset_path).resolve()
        
        logger.info("=== Starting All Metrics Evaluation ===")
        logger.info(f"Created dataset: {created_path}")
        logger.info(f"Original dataset: {original_path}")
        
        # Step 1: Load image pairs
        logger.info("Step 1: Loading image pairs...")
        image_pairs = self._load_image_pairs(created_path, original_path)
        logger.info(f"Loaded {len(image_pairs)} image pairs")
        
        # Step 2: Calculate individual metrics for all images
        logger.info("Step 2: Computing individual metrics...")
        individual_results = self._calculate_individual_metrics_for_all(image_pairs)
        
        # Step 3: Calculate dataset-level FID
        logger.info("Step 3: Computing FID...")
        fid_score = self._calculate_dataset_fid(created_path, original_path)
        logger.info(f"FID Score: {fid_score:.2f}")
        
        # Step 4: Calculate statistics
        logger.info("Step 4: Computing statistics...")
        statistics = self._calculate_statistics(individual_results)
        
        # Step 5: Create results object
        all_results = AllMetricsResults(
            individual_metrics=individual_results,
            fid_score=fid_score,
            statistics=statistics,
            total_images=len(individual_results),
            evaluation_timestamp=datetime.datetime.now().isoformat(),
            created_dataset_path=str(created_path),
            original_dataset_path=str(original_path)
        )
        
        logger.info("=== All Metrics Evaluation Completed ===")
        return all_results
    
    def _load_image_pairs(self, created_path: Path, original_path: Path) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load image pairs from both datasets
        
        Args:
            created_path: Path to created dataset
            original_path: Path to original dataset
            
        Returns:
            List of (original_image, created_image) tensor pairs
        """
        # Get image files from both datasets
        created_images = self._get_image_files(created_path)
        original_images = self._get_image_files(original_path)
        
        logger.info(f"Found {len(created_images)} created images")
        logger.info(f"Found {len(original_images)} original images")
        
        # Match images by filename
        image_pairs = self._match_image_pairs(original_images, created_images)
        
        if not image_pairs:
            raise ValueError("No matching image pairs found between datasets")
        
        logger.info(f"Matched {len(image_pairs)} image pairs")
        
        # Load and preprocess image pairs
        tensor_pairs = []
        
        print("Loading images...")
        for orig_path, created_path in tqdm(image_pairs):
            try:
                # Load using vae-toolkit compatible method
                orig_tensor = self._load_image_as_tensor(orig_path)
                created_tensor = self._load_image_as_tensor(created_path)
                
                tensor_pairs.append((orig_tensor, created_tensor))
                
            except Exception as e:
                logger.warning(f"Failed to load image pair {orig_path.name}: {e}")
                continue
        
        return tensor_pairs
    
    def _get_image_files(self, path: Path) -> List[Path]:
        """Get all image files from a directory"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = []
        
        if path.is_file():
            if path.suffix.lower() in image_extensions:
                image_files.append(path)
        else:
            # Recursively find all image files
            for ext in image_extensions:
                image_files.extend(path.glob(f'**/*{ext}'))
                image_files.extend(path.glob(f'**/*{ext.upper()}'))
        
        return sorted(image_files)
    
    def _match_image_pairs(self, original_images: List[Path], created_images: List[Path]) -> List[Tuple[Path, Path]]:
        """
        Match images between original and created datasets by filename
        
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
    
    def _load_image_as_tensor(self, image_path: Path) -> torch.Tensor:
        """
        Load image as tensor compatible with vae-toolkit
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image tensor in [1, 3, H, W] format, range [-1, 1]
        """
        try:
            # Try using vae-toolkit if available
            from vae_toolkit import load_and_preprocess_image
            tensor, _ = load_and_preprocess_image(str(image_path), target_size=512)
            return tensor
        except ImportError:
            # Fallback to manual loading
            logger.warning("vae-toolkit not available, using manual image loading")
            return self._manual_load_image(image_path)
    
    def _manual_load_image(self, image_path: Path) -> torch.Tensor:
        """
        Manually load and preprocess image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image tensor in [1, 3, H, W] format, range [-1, 1]
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to 512x512 if needed
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.LANCZOS)
        
        # Convert to tensor
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor
    
    def _calculate_individual_metrics_for_all(self, image_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[IndividualImageMetrics]:
        """
        Calculate individual metrics for all image pairs
        
        Args:
            image_pairs: List of (original, created) tensor pairs
            
        Returns:
            List of IndividualImageMetrics for each pair
        """
        results = []
        
        print(f"Computing individual metrics for {len(image_pairs)} image pairs...")
        
        for i, (original, created) in enumerate(tqdm(image_pairs, desc="Individual metrics")):
            try:
                # Move tensors to device
                original = original.to(self.device)
                created = created.to(self.device)
                
                # Calculate all individual metrics
                metrics = self.individual_calculator.calculate_all_individual_metrics(
                    original, created
                )
                results.append(metrics)
                
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for image pair {i}: {e}")
                # Add default metrics to maintain list consistency
                default_metrics = self._create_default_metrics()
                results.append(default_metrics)
        
        return results
    
    def _calculate_dataset_fid(self, created_path: Path, original_path: Path) -> float:
        """
        Calculate FID score between datasets
        
        Args:
            created_path: Path to created dataset
            original_path: Path to original dataset
            
        Returns:
            FID score
        """
        try:
            fid_results = self.fid_evaluator.evaluate_created_dataset_vs_original(
                created_path, original_path
            )
            return fid_results.fid_score
        except Exception as e:
            logger.error(f"FID calculation failed: {e}")
            return float('inf')  # Return infinity to indicate failure
    
    def _create_default_metrics(self) -> IndividualImageMetrics:
        """Create default metrics for failed calculations"""
        return IndividualImageMetrics(
            psnr_db=0.0,
            ssim=0.0,
            mse=1.0,
            mae=1.0,
            lpips=None,
            ssim_improved=None
        )
    
    def _calculate_statistics(self, individual_results: List[IndividualImageMetrics]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for all individual metrics
        
        Args:
            individual_results: List of IndividualImageMetrics
            
        Returns:
            Dictionary with statistics for each metric type
        """
        if not individual_results:
            return {}
        
        statistics = {}
        
        # Basic metrics (always available)
        psnr_values = [m.psnr_db for m in individual_results]
        ssim_values = [m.ssim for m in individual_results]
        mse_values = [m.mse for m in individual_results]
        mae_values = [m.mae for m in individual_results]
        
        statistics['psnr'] = self._compute_metric_stats(psnr_values, 'PSNR')
        statistics['ssim'] = self._compute_metric_stats(ssim_values, 'SSIM')
        statistics['mse'] = self._compute_metric_stats(mse_values, 'MSE')
        statistics['mae'] = self._compute_metric_stats(mae_values, 'MAE')
        
        # Optional metrics
        lpips_values = [m.lpips for m in individual_results if m.lpips is not None]
        if lpips_values:
            statistics['lpips'] = self._compute_metric_stats(lpips_values, 'LPIPS')
        
        ssim_improved_values = [m.ssim_improved for m in individual_results if m.ssim_improved is not None]
        if ssim_improved_values:
            statistics['ssim_improved'] = self._compute_metric_stats(ssim_improved_values, 'SSIM_Improved')
        
        return statistics
    
    def _compute_metric_stats(self, values: List[float], metric_name: str) -> Dict[str, float]:
        """
        Compute statistical summary for a single metric
        
        Args:
            values: List of metric values
            metric_name: Name of the metric for logging
            
        Returns:
            Dictionary with mean, std, min, max, median
        """
        if not values:
            return {}
        
        import numpy as np
        
        values_array = np.array(values)
        
        stats = {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array)),
            'count': len(values)
        }
        
        logger.info(f"    {metric_name}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")
        
        return stats
    
    def print_summary(self, results: AllMetricsResults):
        """
        Print user-friendly summary of evaluation results
        
        Args:
            results: Complete evaluation results
        """
        print("\n" + "="*60)
        print("📊 All Metrics Evaluation Summary")
        print("="*60)
        
        # Basic info
        print(f"📁 Created Dataset: {results.created_dataset_path}")
        print(f"📁 Original Dataset: {results.original_dataset_path}")
        print(f"🖼️  Total Images: {results.total_images}")
        print(f"⏰ Evaluation Time: {results.evaluation_timestamp}")
        print()
        
        # Individual metrics summary
        print("📈 Individual Metrics Summary:")
        stats = results.statistics
        
        if 'psnr' in stats:
            psnr_stats = stats['psnr']
            print(f"   PSNR: {psnr_stats['mean']:.2f} ± {psnr_stats['std']:.2f} dB (range: {psnr_stats['min']:.2f} - {psnr_stats['max']:.2f})")
        
        if 'ssim' in stats:
            ssim_stats = stats['ssim']
            print(f"   SSIM: {ssim_stats['mean']:.4f} ± {ssim_stats['std']:.4f} (range: {ssim_stats['min']:.4f} - {ssim_stats['max']:.4f})")
        
        if 'lpips' in stats:
            lpips_stats = stats['lpips']
            print(f"  LPIPS: {lpips_stats['mean']:.4f} ± {lpips_stats['std']:.4f} (range: {lpips_stats['min']:.4f} - {lpips_stats['max']:.4f})")
        
        if 'ssim_improved' in stats:
            ssim_imp_stats = stats['ssim_improved']
            print(f"SSIM++: {ssim_imp_stats['mean']:.4f} ± {ssim_imp_stats['std']:.4f} (range: {ssim_imp_stats['min']:.4f} - {ssim_imp_stats['max']:.4f})")
        
        # FID score
        print()
        print(f"🎯 Dataset-level FID Score: {results.fid_score:.2f}")
        
        # Quality interpretation
        if results.fid_score < 20:
            quality = "Excellent ✨"
        elif results.fid_score < 50:
            quality = "Good ✅"
        elif results.fid_score < 100:
            quality = "Fair ⚠️"
        else:
            quality = "Poor ❌"
        
        print(f"🏆 Overall Quality: {quality}")
        
        # Quick metrics summary
        print()
        print("📋 Quick Summary:")
        print(f"   {results.get_metric_summary()}")
        
        print("="*60)
        print()