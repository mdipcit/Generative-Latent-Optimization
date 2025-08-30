#!/usr/bin/env python3
"""
Batch Processing Workflow Module

High-level workflow functions for batch processing of image datasets,
specifically optimized for BSDS500 dataset processing and PyTorch dataset creation.
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time

from ..optimization import OptimizationConfig
from ..dataset import BatchProcessor, BatchProcessingConfig, ProcessingResults
from ..dataset import DatasetBuilder, PNGDatasetBuilder
from ..utils import IOUtils


def _config_to_dict(optimization_config):
    """Convert optimization config to dictionary"""
    if isinstance(optimization_config, dict):
        return optimization_config
    elif hasattr(optimization_config, '__dataclass_fields__'):
        # It's a dataclass
        return {
            'iterations': getattr(optimization_config, 'iterations', 150),
            'learning_rate': getattr(optimization_config, 'learning_rate', 0.4),
            'loss_function': getattr(optimization_config, 'loss_function', 'mse'),
            'convergence_threshold': getattr(optimization_config, 'convergence_threshold', 1e-6),
            'checkpoint_interval': getattr(optimization_config, 'checkpoint_interval', 20)
        }
    elif hasattr(optimization_config, '__dict__'):
        # Fallback to __dict__
        return optimization_config.__dict__
    else:
        # Last resort - create basic dict
        return {
            'iterations': 150,
            'learning_rate': 0.4,
            'loss_function': 'mse'
        }


def _calculate_average_metrics(dataset_entries):
    """Calculate average metrics from dataset entries"""
    if not dataset_entries:
        return {}
    
    psnr_improvements = [entry['metrics']['psnr_improvement'] for entry in dataset_entries]
    ssim_improvements = [entry['metrics']['ssim_improvement'] for entry in dataset_entries]
    loss_reductions = [entry['metrics']['loss_reduction'] for entry in dataset_entries]
    
    return {
        'avg_psnr_improvement': sum(psnr_improvements) / len(psnr_improvements),
        'avg_ssim_improvement': sum(ssim_improvements) / len(ssim_improvements),
        'avg_loss_reduction': sum(loss_reductions) / len(loss_reductions),
        'max_psnr_improvement': max(psnr_improvements),
        'min_psnr_improvement': min(psnr_improvements)
    }


def process_bsds500_dataset(bsds500_path: Union[str, Path],
                           output_path: Union[str, Path],
                           config: OptimizationConfig,
                           batch_size: int = 8,
                           max_images_per_split: Optional[int] = None,
                           save_visualizations: bool = True,
                           vae_model: str = "sd14",
                           create_pytorch_dataset: bool = True,
                           create_png_dataset: bool = True) -> Dict[str, str]:
    """
    Process entire BSDS500 dataset and create optimized latent datasets
    
    Args:
        bsds500_path: Path to BSDS500 dataset (or $BSDS500_PATH)
        output_path: Base path for dataset outputs
        config: Optimization configuration
        batch_size: Batch size for processing
        max_images_per_split: Max images per split (for testing)
        save_visualizations: Whether to save comparison images
        vae_model: VAE model to use
        create_pytorch_dataset: Whether to create PyTorch (.pt) dataset
        create_png_dataset: Whether to create PNG dataset
        
    Returns:
        Dictionary with paths to created datasets
    """
    print("🚀 Starting BSDS500 dataset optimization...")
    
    # Handle environment variable
    if isinstance(bsds500_path, str) and bsds500_path.startswith("$"):
        env_var = bsds500_path[1:]  # Remove $
        bsds500_path = os.environ.get(env_var)
        if not bsds500_path:
            raise ValueError(f"Environment variable {env_var} not set")
    
    bsds500_path = Path(bsds500_path).resolve()
    output_path = Path(output_path).resolve()
    
    print(f"Input: {bsds500_path}")
    print(f"Output: {output_path}")
    
    # Validate input
    if not bsds500_path.exists():
        raise FileNotFoundError(f"BSDS500 path not found: {bsds500_path}")
    
    # Create temporary processing directory
    temp_dir = output_path.parent / f"temp_processing_{int(time.time())}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Configure batch processor
        batch_config = BatchProcessingConfig(
            batch_size=batch_size,
            checkpoint_dir=str(temp_dir / "checkpoints"),
            save_visualizations=save_visualizations,
            max_images=max_images_per_split
        )
        
        processor = BatchProcessor(batch_config)
        
        # Process all splits
        print("\n📁 Processing BSDS500 splits...")
        results = processor.process_bsds500_dataset(
            bsds500_path, temp_dir / "processed", config, split="all"
        )
        
        print(f"\n✅ Processing completed!")
        print(f"   Total processed: {results.total_processed}")
        print(f"   Successful: {results.successful_optimizations}")
        print(f"   Failed: {results.failed_optimizations}")
        print(f"   Avg PSNR improvement: {results.average_psnr_improvement:.2f} dB")
        print(f"   Processing time: {results.processing_time_seconds/3600:.1f} hours")
        
        # Create datasets based on parameters
        created_datasets = {}
        
        # Create PyTorch dataset
        if create_pytorch_dataset:
            print("\n📦 Creating PyTorch dataset...")
            pytorch_path = output_path if str(output_path).endswith('.pt') else f"{output_path}.pt"
            # Call the function directly using globals() to avoid name conflict
            dataset_path = globals()['create_pytorch_dataset'](temp_dir / "processed", pytorch_path, config)
            created_datasets['pytorch'] = dataset_path
            print(f"✅ PyTorch dataset created: {dataset_path}")
        
        # Create PNG dataset
        if create_png_dataset:
            print("\n🖼️ Creating PNG dataset...")
            png_path = f"{output_path}_png" if not str(output_path).endswith('_png') else str(output_path)
            # Call the function directly using globals() to avoid name conflict
            png_dataset_path = globals()['create_png_dataset'](temp_dir / "processed", png_path, config)
            created_datasets['png'] = png_dataset_path
            print(f"✅ PNG dataset created: {png_dataset_path}")
        
        print(f"\n🎉 Dataset creation completed!")
        for dataset_type, path in created_datasets.items():
            print(f"   {dataset_type.upper()}: {path}")
        
        # Cleanup temp directory (optional)
        # shutil.rmtree(temp_dir)  # Uncomment to clean up
        
        return created_datasets
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise


def create_pytorch_dataset(processed_data_dir: Union[str, Path],
                          output_path: Union[str, Path],
                          optimization_config: Union[OptimizationConfig, Dict[str, Any]]) -> str:
    """
    Create PyTorch dataset from processed optimization results
    
    Args:
        processed_data_dir: Directory containing processed results
        output_path: Output path for .pt dataset file
        optimization_config: Configuration used for optimization
        
    Returns:
        Path to created dataset file
    """
    processed_dir = Path(processed_data_dir).resolve()
    output_path = Path(output_path).resolve()
    
    print(f"Creating dataset from: {processed_dir}")
    
    # Collect all processed results
    dataset_entries = []
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = processed_dir / split
        if not split_dir.exists():
            print(f"⚠️ Split directory not found: {split_dir}")
            continue
        
        # Load detailed results
        results_file = split_dir / "detailed_results.json"
        if results_file.exists():
            io_utils = IOUtils()
            results = io_utils.load_json(results_file)
            
            for result in results:
                image_name = result['image_name']
                image_result_dir = split_dir / image_name
                
                if image_result_dir.exists():
                    # Load latents and other data
                    try:
                        initial_latents_path = image_result_dir / f"{image_name}_initial_latents.pt"
                        optimized_latents_path = image_result_dir / f"{image_name}_optimized_latents.pt"
                        
                        if initial_latents_path.exists() and optimized_latents_path.exists():
                            entry = {
                                'image_name': image_name,
                                'split': split,
                                'initial_latents': torch.load(initial_latents_path),
                                'optimized_latents': torch.load(optimized_latents_path),
                                'metrics': {
                                    'initial_psnr': result['initial_psnr'],
                                    'final_psnr': result['final_psnr'],
                                    'psnr_improvement': result['psnr_improvement'],
                                    'initial_ssim': result['initial_ssim'], 
                                    'final_ssim': result['final_ssim'],
                                    'ssim_improvement': result['ssim_improvement'],
                                    'loss_reduction': result['loss_reduction']
                                }
                            }
                            
                            dataset_entries.append(entry)
                    except Exception as e:
                        print(f"⚠️ Failed to load data for {image_name}: {e}")
    
    print(f"Collected {len(dataset_entries)} samples")
    
    # Create dataset structure
    dataset = {
        'samples': dataset_entries,
        'metadata': {
            'total_samples': len(dataset_entries),
            'splits_count': {
                split: len([e for e in dataset_entries if e['split'] == split])
                for split in splits
            },
            'optimization_config': _config_to_dict(optimization_config),
            'creation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'average_metrics': _calculate_average_metrics(dataset_entries)
        }
    }
    
    # Save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    
    print(f"💾 Dataset saved to: {output_path}")
    print(f"   Total samples: {dataset['metadata']['total_samples']}")
    print(f"   Splits: {dataset['metadata']['splits_count']}")
    print(f"   Avg PSNR improvement: {dataset['metadata']['average_metrics']['avg_psnr_improvement']:.2f} dB")
    
    return str(output_path)


def create_png_dataset(processed_data_dir: Union[str, Path],
                      output_dir: Union[str, Path],
                      optimization_config: OptimizationConfig) -> str:
    """
    Create PNG dataset from processed optimization results
    
    Args:
        processed_data_dir: Directory containing processed results
        output_dir: Output directory for PNG dataset
        optimization_config: Configuration used for optimization
        
    Returns:
        Path to created PNG dataset directory
    """
    # Convert optimization config to dict
    if hasattr(optimization_config, '__dataclass_fields__'):
        # It's a dataclass
        config_dict = {
            'iterations': optimization_config.iterations,
            'learning_rate': optimization_config.learning_rate,
            'loss_function': optimization_config.loss_function,
            'convergence_threshold': optimization_config.convergence_threshold,
            'checkpoint_interval': optimization_config.checkpoint_interval
        }
    else:
        # Fallback to __dict__
        config_dict = optimization_config.__dict__
    
    builder = PNGDatasetBuilder()
    return builder.create_png_dataset(processed_data_dir, output_dir, config_dict)


def create_dual_datasets(processed_data_dir: Union[str, Path],
                        base_output_path: Union[str, Path],
                        optimization_config: OptimizationConfig) -> Dict[str, str]:
    """
    Create both PyTorch and PNG datasets from processing results
    
    Args:
        processed_data_dir: Directory containing processed results
        base_output_path: Base path for outputs
        optimization_config: Configuration used for optimization
        
    Returns:
        Dictionary with paths to both datasets
    """
    base_path = Path(base_output_path).resolve()
    
    # Prepare paths
    pytorch_path = f"{base_path}.pt"
    png_path = f"{base_path}_png"
    
    print("🏗️ Creating dual datasets (PyTorch + PNG)...")
    
    # Create PyTorch dataset
    pytorch_dataset_path = create_pytorch_dataset(
        processed_data_dir, pytorch_path, optimization_config
    )
    
    # Create PNG dataset  
    png_dataset_path = create_png_dataset(
        processed_data_dir, png_path, optimization_config
    )
    
    datasets = {
        'pytorch': pytorch_dataset_path,
        'png': png_dataset_path
    }
    
    print(f"✅ Dual datasets created successfully!")
    for dataset_type, path in datasets.items():
        print(f"   {dataset_type.upper()}: {path}")
    
    return datasets


def process_single_directory(input_dir: Union[str, Path],
                           output_dir: Union[str, Path], 
                           config: OptimizationConfig,
                           batch_size: int = 8,
                           max_images: Optional[int] = None) -> ProcessingResults:
    """
    Process a single directory of images
    
    Args:
        input_dir: Input directory with images
        output_dir: Output directory for results
        config: Optimization configuration
        batch_size: Batch processing size
        max_images: Maximum images to process
        
    Returns:
        ProcessingResults with statistics
    """
    batch_config = BatchProcessingConfig(
        batch_size=batch_size,
        max_images=max_images,
        save_visualizations=True
    )
    
    processor = BatchProcessor(batch_config)
    
    return processor.process_directory(input_dir, output_dir, config)


def quick_test_processing(test_dir: Union[str, Path],
                         output_dir: Union[str, Path],
                         num_images: int = 5,
                         iterations: int = 50) -> ProcessingResults:
    """
    Quick test processing for a small number of images
    
    Args:
        test_dir: Directory with test images  
        output_dir: Output directory
        num_images: Number of images to process
        iterations: Optimization iterations
        
    Returns:
        ProcessingResults
    """
    config = OptimizationConfig(
        iterations=iterations,
        learning_rate=0.4,
        checkpoint_interval=10
    )
    
    return process_single_directory(test_dir, output_dir, config, max_images=num_images)


def _calculate_average_metrics(dataset_entries: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate average metrics across all samples"""
    if not dataset_entries:
        return {}
    
    psnr_improvements = [entry['metrics']['psnr_improvement'] for entry in dataset_entries]
    ssim_improvements = [entry['metrics']['ssim_improvement'] for entry in dataset_entries]
    loss_reductions = [entry['metrics']['loss_reduction'] for entry in dataset_entries]
    
    return {
        'avg_psnr_improvement': sum(psnr_improvements) / len(psnr_improvements),
        'avg_ssim_improvement': sum(ssim_improvements) / len(ssim_improvements),
        'avg_loss_reduction': sum(loss_reductions) / len(loss_reductions),
        'max_psnr_improvement': max(psnr_improvements),
        'min_psnr_improvement': min(psnr_improvements)
    }


# Convenience functions for common workflows

def optimize_bsds500_full(output_path: Union[str, Path] = "./optimized_bsds500_dataset",
                         iterations: int = 150,
                         learning_rate: float = 0.4,
                         create_pytorch: bool = True,
                         create_png: bool = True) -> Dict[str, str]:
    """
    Optimize full BSDS500 dataset with default settings
    
    Args:
        output_path: Base output path for datasets
        iterations: Optimization iterations
        learning_rate: Learning rate
        create_pytorch: Whether to create PyTorch dataset
        create_png: Whether to create PNG dataset
        
    Returns:
        Dictionary with paths to created datasets
    """
    bsds500_path = os.environ.get("BSDS500_PATH")
    if not bsds500_path:
        raise ValueError("BSDS500_PATH environment variable not set")
    
    config = OptimizationConfig(
        iterations=iterations,
        learning_rate=learning_rate,
        checkpoint_interval=20
    )
    
    return process_bsds500_dataset(
        bsds500_path, output_path, config,
        create_pytorch_dataset=create_pytorch,
        create_png_dataset=create_png
    )


def optimize_bsds500_test(output_path: Union[str, Path] = "./test_bsds500_dataset",
                         max_images: int = 10,
                         create_pytorch: bool = True,
                         create_png: bool = True) -> Dict[str, str]:
    """
    Quick test optimization of BSDS500 with limited images
    
    Args:
        output_path: Base output path for datasets
        max_images: Maximum images per split to process
        create_pytorch: Whether to create PyTorch dataset
        create_png: Whether to create PNG dataset
        
    Returns:
        Dictionary with paths to created datasets
    """
    bsds500_path = os.environ.get("BSDS500_PATH")
    if not bsds500_path:
        raise ValueError("BSDS500_PATH environment variable not set")
    
    config = OptimizationConfig(
        iterations=50,
        learning_rate=0.4,
        checkpoint_interval=10
    )
    
    return process_bsds500_dataset(
        bsds500_path, output_path, config,
        max_images_per_split=max_images,
        save_visualizations=True,
        create_pytorch_dataset=create_pytorch,
        create_png_dataset=create_png
    )