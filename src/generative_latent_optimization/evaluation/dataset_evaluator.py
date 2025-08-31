#!/usr/bin/env python3
"""
Comprehensive Dataset Evaluator

Provides complete dataset evaluation after dataset creation is finished.
Combines individual metrics statistics with dataset-level FID evaluation.
"""

import json
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
import logging

try:
    from ..metrics.metrics_integration import IndividualMetricsCalculator
    from ..metrics.dataset_metrics import DatasetFIDEvaluator
    from ..metrics.image_metrics import DatasetEvaluationResults
except ImportError:
    # For direct execution as script
    import sys
    parent_path = Path(__file__).parent.parent
    sys.path.append(str(parent_path))
    from metrics.metrics_integration import IndividualMetricsCalculator
    from metrics.dataset_metrics import DatasetFIDEvaluator
    from metrics.image_metrics import DatasetEvaluationResults

logger = logging.getLogger(__name__)


class ComprehensiveDatasetEvaluator:
    """
    Complete dataset evaluation after dataset creation is finished
    
    Combines:
    1. Individual image metrics statistics (from existing data)
    2. Dataset-level FID evaluation (computed fresh)
    3. Overall quality assessment and recommendations
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize comprehensive evaluator
        
        Args:
            device: Computation device
        """
        self.device = device
        
        # Initialize individual metrics calculator (for potential re-calculation)
        self.individual_calculator = IndividualMetricsCalculator(device=device)
        
        # Initialize FID evaluator
        self.fid_evaluator = DatasetFIDEvaluator(device=device)
        
        logger.info(f"Comprehensive dataset evaluator initialized on {device}")
    
    def evaluate_complete_dataset(self, 
                                dataset_path: Union[str, Path],
                                original_bsds500_path: Union[str, Path],
                                dataset_type: str = 'png') -> Dict[str, Any]:
        """
        Execute comprehensive evaluation of completed dataset
        
        Args:
            dataset_path: Path to created dataset
            original_bsds500_path: Path to original BSDS500 dataset  
            dataset_type: 'png' or 'pytorch'
        
        Returns:
            Complete evaluation results including individual metrics stats + FID score
        """
        # Normalize paths
        dataset_path = Path(dataset_path).resolve()
        original_bsds500_path = Path(original_bsds500_path).resolve()
        
        logger.info("=== Starting comprehensive dataset evaluation ===")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Original: {original_bsds500_path}")
        logger.info(f"Type: {dataset_type}")
        
        # Step 1: FID dataset evaluation
        logger.info("1. Computing FID dataset evaluation...")
        fid_results = self._compute_fid_evaluation(
            dataset_path, original_bsds500_path, dataset_type
        )
        logger.info(f"   FID Score: {fid_results.fid_score:.2f}")
        
        # Step 2: Individual metrics statistics
        logger.info("2. Computing individual metrics statistics...")
        individual_stats = self._calculate_individual_metrics_statistics(
            dataset_path, dataset_type
        )
        
        # Step 3: Create evaluation summary
        logger.info("3. Creating evaluation summary...")
        evaluation_summary = self._create_evaluation_summary(fid_results, individual_stats)
        
        # Step 4: Combine results
        comprehensive_results = {
            'dataset_evaluation': {
                'fid_score': fid_results.fid_score,
                'total_images': fid_results.total_images,
                'original_dataset_path': fid_results.original_dataset_path,
                'generated_dataset_path': fid_results.generated_dataset_path,
                'evaluation_timestamp': fid_results.evaluation_timestamp
            },
            'individual_metrics_statistics': individual_stats,
            'evaluation_summary': evaluation_summary
        }
        
        logger.info("=== Comprehensive evaluation completed ===")
        return comprehensive_results
    
    def _compute_fid_evaluation(self, dataset_path: Path, original_path: Path, 
                              dataset_type: str) -> DatasetEvaluationResults:
        """Compute FID evaluation based on dataset type"""
        if dataset_type == 'png':
            return self.fid_evaluator.evaluate_created_dataset_vs_original(
                dataset_path, original_path
            )
        elif dataset_type == 'pytorch':
            return self.fid_evaluator.evaluate_pytorch_dataset_vs_original(
                dataset_path, original_path
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def _calculate_individual_metrics_statistics(self, dataset_path: Path, 
                                               dataset_type: str) -> Dict[str, float]:
        """
        Calculate individual metrics statistics from existing dataset
        
        Args:
            dataset_path: Path to dataset
            dataset_type: 'png' or 'pytorch'
            
        Returns:
            Dictionary with individual metrics statistics
        """
        if dataset_type == 'pytorch':
            return self._load_pytorch_dataset_statistics(dataset_path)
        elif dataset_type == 'png':
            return self._load_png_dataset_statistics(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def _load_pytorch_dataset_statistics(self, dataset_path: Path) -> Dict[str, float]:
        """Load individual metrics statistics from PyTorch dataset"""
        try:
            # Load dataset and extract metrics
            try:
                from ..dataset import load_optimized_dataset
            except ImportError:
                import sys
                parent_path = Path(__file__).parent.parent
                sys.path.append(str(parent_path))
                from dataset import load_optimized_dataset
            
            dataset = load_optimized_dataset(dataset_path)
            
            # Extract metrics from all samples
            all_metrics = []
            for i in range(len(dataset)):
                sample = dataset[i]
                if 'metrics' in sample:
                    all_metrics.append(sample['metrics'])
            
            if not all_metrics:
                logger.warning("No metrics found in PyTorch dataset")
                return {}
            
            # Calculate statistics
            return self._compute_metrics_statistics(all_metrics)
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch dataset statistics: {e}")
            return {}
    
    def _load_png_dataset_statistics(self, dataset_path: Path) -> Dict[str, float]:
        """Load individual metrics statistics from PNG dataset statistics.json"""
        try:
            stats_file = Path(dataset_path) / 'statistics.json'
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                # Convert to standard format
                return self._normalize_png_statistics(stats)
            else:
                logger.warning(f"Statistics file not found: {stats_file}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to load PNG dataset statistics: {e}")
            return {}
    
    def _compute_metrics_statistics(self, metrics_list: list) -> Dict[str, float]:
        """Compute statistics from list of metrics dictionaries"""
        if not metrics_list:
            return {}
        
        # Extract values for each metric
        psnr_values = [m.get('psnr_db', 0) for m in metrics_list if 'psnr_db' in m]
        ssim_values = [m.get('ssim', 0) for m in metrics_list if 'ssim' in m]
        mse_values = [m.get('mse', 0) for m in metrics_list if 'mse' in m]
        mae_values = [m.get('mae', 0) for m in metrics_list if 'mae' in m]
        
        statistics = {}
        
        if psnr_values:
            statistics.update({
                'psnr_mean': sum(psnr_values) / len(psnr_values),
                'psnr_std': self._calculate_std(psnr_values),
                'psnr_min': min(psnr_values),
                'psnr_max': max(psnr_values)
            })
        
        if ssim_values:
            statistics.update({
                'ssim_mean': sum(ssim_values) / len(ssim_values),
                'ssim_std': self._calculate_std(ssim_values),
                'ssim_min': min(ssim_values),
                'ssim_max': max(ssim_values)
            })
        
        # Add LPIPS if available
        lpips_values = [m.get('lpips', 0) for m in metrics_list if m.get('lpips') is not None]
        if lpips_values:
            statistics.update({
                'lpips_mean': sum(lpips_values) / len(lpips_values),
                'lpips_std': self._calculate_std(lpips_values),
                'lpips_min': min(lpips_values),
                'lpips_max': max(lpips_values)
            })
        
        # Add improved SSIM if available
        ssim_improved_values = [m.get('ssim_improved', 0) for m in metrics_list if m.get('ssim_improved') is not None]
        if ssim_improved_values:
            statistics.update({
                'ssim_improved_mean': sum(ssim_improved_values) / len(ssim_improved_values),
                'ssim_improved_std': self._calculate_std(ssim_improved_values),
                'ssim_improved_min': min(ssim_improved_values),
                'ssim_improved_max': max(ssim_improved_values)
            })
        
        statistics['total_samples'] = len(metrics_list)
        return statistics
    
    def _normalize_png_statistics(self, stats: Dict) -> Dict[str, float]:
        """Normalize PNG statistics to standard format"""
        # This depends on the actual format of PNG statistics.json
        # Adapt based on the actual structure from PNG dataset creation
        normalized = {}
        
        # Try to extract common statistics
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                normalized[key] = float(value)
            elif isinstance(value, dict) and 'mean' in value:
                normalized[f"{key}_mean"] = float(value['mean'])
                if 'std' in value:
                    normalized[f"{key}_std"] = float(value['std'])
                if 'min' in value:
                    normalized[f"{key}_min"] = float(value['min'])
                if 'max' in value:
                    normalized[f"{key}_max"] = float(value['max'])
        
        return normalized
    
    def _create_evaluation_summary(self, fid_results: DatasetEvaluationResults, 
                                 individual_stats: Dict[str, float]) -> Dict[str, Any]:
        """Create comprehensive evaluation summary with interpretations and recommendations"""
        summary = {
            'overall_quality_score': self._calculate_overall_score(fid_results.fid_score, individual_stats),
            'fid_interpretation': self._interpret_fid_score(fid_results.fid_score),
            'improvement_highlights': self._identify_improvements(individual_stats),
            'recommendation': self._generate_recommendations(fid_results, individual_stats)
        }
        
        return summary
    
    def _calculate_overall_score(self, fid_score: float, individual_stats: Dict[str, float]) -> float:
        """Calculate overall quality score (0-100 scale)"""
        try:
            # FID component (lower is better, typical range 0-200)
            fid_component = max(0, 100 - fid_score / 2)  # Normalize roughly to 0-100
            
            # PSNR component (higher is better, typical range 15-45 dB)
            psnr_mean = individual_stats.get('psnr_mean', 20)
            psnr_component = min(100, max(0, (psnr_mean - 15) * 3.33))  # Normalize to 0-100
            
            # SSIM component (higher is better, range 0-1)
            ssim_mean = individual_stats.get('ssim_mean', 0.5)
            ssim_component = ssim_mean * 100  # Convert to 0-100
            
            # Weighted combination
            overall_score = (fid_component * 0.4 + psnr_component * 0.3 + ssim_component * 0.3)
            return min(100, max(0, overall_score))
            
        except Exception as e:
            logger.warning(f"Failed to calculate overall score: {e}")
            return 50.0  # Default neutral score
    
    def _interpret_fid_score(self, fid_score: float) -> str:
        """Provide interpretation of FID score"""
        if fid_score < 10:
            return "Excellent - Very high dataset quality, nearly indistinguishable from original"
        elif fid_score < 20:
            return "Good - High dataset quality with minor perceptual differences"
        elif fid_score < 50:
            return "Fair - Moderate dataset quality, noticeable but acceptable differences"
        elif fid_score < 100:
            return "Poor - Low dataset quality with significant perceptual differences"
        else:
            return "Very Poor - Dataset quality is substantially different from original"
    
    def _identify_improvements(self, individual_stats: Dict[str, float]) -> List[str]:
        """Identify key improvements achieved"""
        improvements = []
        
        psnr_mean = individual_stats.get('psnr_mean', 0)
        if psnr_mean > 25:
            improvements.append(f"Strong PSNR improvement: {psnr_mean:.1f} dB average")
        
        ssim_mean = individual_stats.get('ssim_mean', 0)
        if ssim_mean > 0.8:
            improvements.append(f"High structural similarity: {ssim_mean:.3f} average SSIM")
        
        if 'lpips_mean' in individual_stats:
            lpips_mean = individual_stats['lpips_mean']
            if lpips_mean < 0.1:
                improvements.append(f"Good perceptual quality: {lpips_mean:.3f} average LPIPS")
        
        if 'ssim_improved_mean' in individual_stats:
            ssim_improved = individual_stats['ssim_improved_mean']
            if ssim_improved > 0.85:
                improvements.append(f"Excellent structural preservation: {ssim_improved:.3f} improved SSIM")
        
        return improvements
    
    def _generate_recommendations(self, fid_results: DatasetEvaluationResults, 
                                individual_stats: Dict[str, float]) -> str:
        """Generate recommendations based on evaluation results"""
        fid_score = fid_results.fid_score
        psnr_mean = individual_stats.get('psnr_mean', 0)
        
        if fid_score < 20 and psnr_mean > 30:
            return "Excellent results - Dataset is ready for production use"
        elif fid_score < 50 and psnr_mean > 25:
            return "Good results - Dataset suitable for most applications with minor quality trade-offs"
        elif fid_score < 100:
            return "Moderate results - Consider additional optimization iterations or parameter tuning"
        else:
            return "Results need improvement - Review optimization parameters and increase iterations"
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation using StatisticsCalculator"""
        from ..utils.io_utils import StatisticsCalculator
        stats = StatisticsCalculator.calculate_basic_stats(values)
        return stats.get('std', 0.0) if stats else 0.0


# Utility functions for testing
def test_comprehensive_evaluator():
    """Test comprehensive evaluator with mock data"""
    print("Testing comprehensive dataset evaluator...")
    
    try:
        evaluator = ComprehensiveDatasetEvaluator(device='cpu')  # Use CPU for testing
        
        # This would require actual datasets to test fully
        print("  Comprehensive evaluator initialized successfully")
        print("  Note: Full testing requires actual dataset files")
        
    except Exception as e:
        print(f"  Comprehensive evaluator test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_comprehensive_evaluator()
    print("Dataset evaluator module tests completed.")