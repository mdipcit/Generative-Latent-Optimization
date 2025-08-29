#!/usr/bin/env python3
"""
PNG Dataset Module

Creates organized PNG dataset from optimization results with
proper directory structure, comparison images, and metadata.
"""

import torch
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import time

from ..utils import IOUtils
from ..visualization import ImageVisualizer


@dataclass
class PNGDatasetMetadata:
    """Metadata for PNG dataset"""
    total_samples: int
    splits_count: Dict[str, int]
    optimization_config: Dict[str, Any]
    directory_structure: Dict[str, str]
    creation_timestamp: str
    dataset_version: str = "1.0"
    source_dataset: str = "BSDS500"


class PNGDatasetBuilder:
    """
    Builder for creating PNG datasets from optimization results
    
    Creates organized directory structure with original images,
    reconstructions, comparison grids, and metadata.
    """
    
    def __init__(self):
        self.io_utils = IOUtils()
        self.visualizer = ImageVisualizer()
    
    def create_png_dataset(self, processed_data_dir: Union[str, Path],
                          output_dir: Union[str, Path],
                          optimization_config: Dict[str, Any],
                          include_comparisons: bool = True,
                          include_individual_images: bool = True) -> str:
        """
        Create PNG dataset from processed optimization results
        
        Args:
            processed_data_dir: Directory containing processed results
            output_dir: Output directory for PNG dataset
            optimization_config: Configuration used for optimization
            include_comparisons: Whether to create comparison grids
            include_individual_images: Whether to include individual images
            
        Returns:
            Path to created PNG dataset directory
        """
        processed_dir = Path(processed_data_dir)
        output_dir = Path(output_dir)
        
        print(f"🖼️ Creating PNG dataset from: {processed_dir}")
        print(f"   Output directory: {output_dir}")
        
        # Create output directory structure
        self._create_directory_structure(output_dir)
        
        # Process all splits
        splits = ['train', 'val', 'test']
        dataset_samples = []
        
        for split in splits:
            split_dir = processed_dir / split
            if not split_dir.exists():
                print(f"⚠️ Split directory not found: {split_dir}")
                continue
            
            print(f"📁 Processing {split} split...")
            split_samples = self._process_split(
                split_dir, output_dir / split, split,
                include_comparisons, include_individual_images
            )
            dataset_samples.extend(split_samples)
        
        print(f"📊 Processed {len(dataset_samples)} samples total")
        
        # Create metadata
        metadata = self._create_metadata(dataset_samples, optimization_config, output_dir)
        
        # Save metadata files
        self._save_metadata(output_dir, metadata, dataset_samples)
        
        # Create overview visualizations
        self._create_overview_visualizations(output_dir, dataset_samples)
        
        print(f"✅ PNG dataset created successfully!")
        print(f"   Total samples: {metadata.total_samples}")
        print(f"   Directory: {output_dir}")
        
        return str(output_dir)
    
    def _create_directory_structure(self, output_dir: Path):
        """Create the directory structure for PNG dataset"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (output_dir / split).mkdir(exist_ok=True)
        
        # Create overview directory
        (output_dir / 'overview').mkdir(exist_ok=True)
    
    def _process_split(self, split_input_dir: Path, split_output_dir: Path,
                      split_name: str, include_comparisons: bool,
                      include_individual_images: bool) -> List[Dict[str, Any]]:
        """Process a single split directory"""
        
        # Load split results
        results_file = split_input_dir / "detailed_results.json"
        if not results_file.exists():
            print(f"⚠️ No results file found for {split_name}: {results_file}")
            return []
        
        results = self.io_utils.load_json(results_file)
        split_samples = []
        
        for result in results:
            image_name = result['image_name']
            image_result_dir = split_input_dir / image_name
            
            if not image_result_dir.exists():
                continue
            
            # Create output directory for this image
            image_output_dir = split_output_dir / image_name
            image_output_dir.mkdir(exist_ok=True)
            
            try:
                # Process this image
                sample_info = self._process_single_image(
                    image_result_dir, image_output_dir, result, split_name,
                    include_comparisons, include_individual_images
                )
                
                if sample_info:
                    split_samples.append(sample_info)
                    
            except Exception as e:
                print(f"⚠️ Failed to process {image_name}: {e}")
        
        return split_samples
    
    def _process_single_image(self, input_dir: Path, output_dir: Path,
                            result: Dict[str, Any], split_name: str,
                            include_comparisons: bool, 
                            include_individual_images: bool) -> Optional[Dict[str, Any]]:
        """Process a single image's optimization results"""
        
        image_name = result['image_name']
        
        # Load image files
        image_files = {
            'original': input_dir / f"{image_name}_original.png",
            'initial_recon': input_dir / f"{image_name}_initial_reconstruction.png",
            'optimized_recon': input_dir / f"{image_name}_optimized_reconstruction.png"
        }
        
        # Check if all required files exist
        missing_files = [name for name, path in image_files.items() if not path.exists()]
        if missing_files:
            print(f"⚠️ Missing files for {image_name}: {missing_files}")
            return None
        
        # Copy individual images if requested
        output_files = {}
        if include_individual_images:
            for image_type, input_path in image_files.items():
                output_path = output_dir / f"{image_type}.png"
                shutil.copy2(input_path, output_path)
                output_files[image_type] = str(output_path.relative_to(output_dir.parent.parent))
        
        # Create comparison grid if requested
        if include_comparisons:
            # Load tensors for comparison grid
            try:
                # Load original tensor (recreate from PNG if needed)
                # For now, we'll create comparison from existing PNG files
                comparison_path = output_dir / "comparison_grid.png"
                
                # Use PIL to load and create comparison
                from PIL import Image
                import matplotlib.pyplot as plt
                
                # Load images
                orig_img = Image.open(image_files['original'])
                init_img = Image.open(image_files['initial_recon'])
                opt_img = Image.open(image_files['optimized_recon'])
                
                # Create comparison grid using matplotlib
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(orig_img)
                axes[0].set_title('Original', fontweight='bold')
                axes[0].axis('off')
                
                axes[1].imshow(init_img)
                axes[1].set_title('Initial Reconstruction', fontweight='bold')
                axes[1].axis('off')
                
                axes[2].imshow(opt_img)
                axes[2].set_title('Optimized Reconstruction', fontweight='bold')
                axes[2].axis('off')
                
                # Add metrics information
                metrics_text = []
                if 'psnr_improvement' in result:
                    metrics_text.append(f"PSNR improvement: +{result['psnr_improvement']:.2f} dB")
                if 'loss_reduction' in result:
                    metrics_text.append(f"Loss reduction: {result['loss_reduction']:.1f}%")
                
                if metrics_text:
                    fig.text(0.5, 0.02, " | ".join(metrics_text), 
                            ha='center', va='bottom', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
                
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.15)
                plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                output_files['comparison_grid'] = str(comparison_path.relative_to(output_dir.parent.parent))
                
            except Exception as e:
                print(f"⚠️ Failed to create comparison grid for {image_name}: {e}")
        
        # Save individual metrics
        metrics_path = output_dir / "metrics.json"
        self.io_utils.save_json(result, metrics_path)
        output_files['metrics'] = str(metrics_path.relative_to(output_dir.parent.parent))
        
        # Prepare sample information
        sample_info = {
            'image_name': image_name,
            'split': split_name,
            'directory': str(output_dir.relative_to(output_dir.parent.parent)),
            'files': output_files,
            'metrics': {
                'initial_psnr': result['initial_psnr'],
                'final_psnr': result['final_psnr'],
                'psnr_improvement': result['psnr_improvement'],
                'initial_ssim': result.get('initial_ssim', 0),
                'final_ssim': result.get('final_ssim', 0),
                'ssim_improvement': result.get('ssim_improvement', 0),
                'loss_reduction': result['loss_reduction'],
                'optimization_iterations': result['optimization_iterations']
            }
        }
        
        return sample_info
    
    def _create_metadata(self, dataset_samples: List[Dict[str, Any]],
                        optimization_config: Dict[str, Any],
                        output_dir: Path) -> PNGDatasetMetadata:
        """Create metadata for the PNG dataset"""
        
        splits_count = {}
        for sample in dataset_samples:
            split = sample['split']
            splits_count[split] = splits_count.get(split, 0) + 1
        
        directory_structure = {
            "root": "PNG dataset root directory",
            "splits": "train/, val/, test/ - Split directories",
            "images": "Individual image directories with original, reconstructions, and comparisons",
            "overview": "Dataset overview and summary visualizations",
            "metadata.json": "Dataset metadata and configuration",
            "statistics.json": "Processing statistics and metrics summary"
        }
        
        metadata = PNGDatasetMetadata(
            total_samples=len(dataset_samples),
            splits_count=splits_count,
            optimization_config=optimization_config,
            directory_structure=directory_structure,
            creation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return metadata
    
    def _save_metadata(self, output_dir: Path, metadata: PNGDatasetMetadata,
                      dataset_samples: List[Dict[str, Any]]):
        """Save metadata files"""
        
        # Save main metadata
        self.io_utils.save_json(asdict(metadata), output_dir / "metadata.json")
        
        # Calculate and save statistics
        statistics = self._calculate_statistics(dataset_samples)
        self.io_utils.save_json(statistics, output_dir / "statistics.json")
        
        # Save detailed sample information
        self.io_utils.save_json(dataset_samples, output_dir / "samples.json")
        
        # Create README
        self._create_readme(output_dir, metadata, statistics)
    
    def _calculate_statistics(self, dataset_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate dataset statistics"""
        
        if not dataset_samples:
            return {}
        
        psnr_improvements = [s['metrics']['psnr_improvement'] for s in dataset_samples]
        loss_reductions = [s['metrics']['loss_reduction'] for s in dataset_samples]
        iterations = [s['metrics']['optimization_iterations'] for s in dataset_samples]
        
        statistics = {
            'total_samples': len(dataset_samples),
            'psnr_improvement': {
                'mean': sum(psnr_improvements) / len(psnr_improvements),
                'std': self._std(psnr_improvements),
                'min': min(psnr_improvements),
                'max': max(psnr_improvements),
                'median': sorted(psnr_improvements)[len(psnr_improvements) // 2]
            },
            'loss_reduction': {
                'mean': sum(loss_reductions) / len(loss_reductions),
                'std': self._std(loss_reductions),
                'min': min(loss_reductions),
                'max': max(loss_reductions)
            },
            'optimization': {
                'mean_iterations': sum(iterations) / len(iterations),
                'total_iterations': sum(iterations)
            },
            'splits_breakdown': {}
        }
        
        # Per-split statistics
        for split in set(s['split'] for s in dataset_samples):
            split_samples = [s for s in dataset_samples if s['split'] == split]
            split_psnr = [s['metrics']['psnr_improvement'] for s in split_samples]
            
            statistics['splits_breakdown'][split] = {
                'count': len(split_samples),
                'avg_psnr_improvement': sum(split_psnr) / len(split_psnr) if split_psnr else 0
            }
        
        return statistics
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _create_overview_visualizations(self, output_dir: Path, dataset_samples: List[Dict[str, Any]]):
        """Create overview visualizations for the dataset"""
        
        overview_dir = output_dir / 'overview'
        
        # Create histogram of PSNR improvements
        try:
            import matplotlib.pyplot as plt
            
            psnr_improvements = [s['metrics']['psnr_improvement'] for s in dataset_samples]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # PSNR improvement histogram
            ax1.hist(psnr_improvements, bins=20, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('PSNR Improvement (dB)')
            ax1.set_ylabel('Number of Images')
            ax1.set_title('Distribution of PSNR Improvements')
            ax1.grid(True, alpha=0.3)
            
            # Split comparison
            splits = ['train', 'val', 'test']
            split_avg_psnr = []
            split_counts = []
            
            for split in splits:
                split_samples = [s for s in dataset_samples if s['split'] == split]
                if split_samples:
                    avg_psnr = sum(s['metrics']['psnr_improvement'] for s in split_samples) / len(split_samples)
                    split_avg_psnr.append(avg_psnr)
                    split_counts.append(len(split_samples))
                else:
                    split_avg_psnr.append(0)
                    split_counts.append(0)
            
            ax2.bar(splits, split_avg_psnr, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])
            ax2.set_ylabel('Average PSNR Improvement (dB)')
            ax2.set_title('PSNR Improvement by Split')
            ax2.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for i, (split, count) in enumerate(zip(splits, split_counts)):
                if count > 0:
                    ax2.text(i, split_avg_psnr[i] + 0.1, f'n={count}', 
                            ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(overview_dir / 'statistics_overview.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Overview visualizations saved to: {overview_dir}")
            
        except ImportError:
            print("⚠️ matplotlib not available, skipping overview visualizations")
    
    def _create_readme(self, output_dir: Path, metadata: PNGDatasetMetadata,
                      statistics: Dict[str, Any]):
        """Create README file for the dataset"""
        
        readme_content = f"""# VAE Optimization PNG Dataset

## Overview
This dataset contains optimized VAE latent representations for image reconstruction,
created from {metadata.source_dataset} using generative latent optimization.

## Dataset Information
- **Total Samples**: {metadata.total_samples}
- **Creation Date**: {metadata.creation_timestamp}
- **Dataset Version**: {metadata.dataset_version}

## Split Information
"""
        
        for split, count in metadata.splits_count.items():
            readme_content += f"- **{split.capitalize()}**: {count} images\n"
        
        readme_content += f"""
## Directory Structure
```
{output_dir.name}/
├── train/          # Training images
├── val/            # Validation images  
├── test/           # Test images
├── overview/       # Dataset visualizations
├── metadata.json   # Dataset metadata
├── statistics.json # Processing statistics
├── samples.json    # Detailed sample information
└── README.md       # This file
```

## Image Directory Structure
Each image has its own directory containing:
- `original.png` - Original input image
- `initial_reconstruction.png` - Initial VAE reconstruction  
- `optimized_reconstruction.png` - Optimized reconstruction
- `comparison_grid.png` - Side-by-side comparison
- `metrics.json` - Optimization metrics

## Performance Statistics
"""
        
        if 'psnr_improvement' in statistics:
            psnr_stats = statistics['psnr_improvement']
            readme_content += f"""
- **Average PSNR Improvement**: {psnr_stats['mean']:.2f} ± {psnr_stats['std']:.2f} dB
- **PSNR Range**: {psnr_stats['min']:.2f} to {psnr_stats['max']:.2f} dB
- **Median PSNR Improvement**: {psnr_stats['median']:.2f} dB
"""
        
        if 'loss_reduction' in statistics:
            loss_stats = statistics['loss_reduction']
            readme_content += f"""
- **Average Loss Reduction**: {loss_stats['mean']:.1f} ± {loss_stats['std']:.1f}%
- **Loss Reduction Range**: {loss_stats['min']:.1f} to {loss_stats['max']:.1f}%
"""
        
        readme_content += f"""
## Optimization Configuration
- **Iterations**: {metadata.optimization_config.get('iterations', 'N/A')}
- **Learning Rate**: {metadata.optimization_config.get('learning_rate', 'N/A')}
- **Loss Function**: {metadata.optimization_config.get('loss_function', 'N/A')}

## Usage
This dataset can be used for:
- Evaluating VAE reconstruction quality
- Training models on optimized latent representations
- Comparing optimization algorithms
- Visual analysis of reconstruction improvements

## File Formats
- Images: PNG format (RGB, 8-bit)
- Metadata: JSON format
- Documentation: Markdown format
"""
        
        # Save README
        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)


# Utility functions
def create_png_dataset_from_results(processed_dir: Union[str, Path],
                                   output_dir: Union[str, Path],
                                   optimization_config: Dict[str, Any]) -> str:
    """Create PNG dataset from processing results directory"""
    builder = PNGDatasetBuilder()
    return builder.create_png_dataset(processed_dir, output_dir, optimization_config)