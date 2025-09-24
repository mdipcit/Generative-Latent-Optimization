#!/usr/bin/env python3
"""
Batch Processing Module

Provides batch processing capabilities for optimizing multiple images
from directories, with checkpointing and progress tracking.
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
import json
import pickle
from tqdm import tqdm
import time

import torch_image_metrics as tim

from vae_toolkit import VAELoader, load_and_preprocess_image
from ..optimization import LatentOptimizer, OptimizationConfig, OptimizationResult
from ..metrics import ImageMetrics, MetricResults
from ..utils import IOUtils, ResultsSaver, StatisticsCalculator


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""
    batch_size: int = 8
    num_workers: int = 4
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: bool = True
    save_visualizations: bool = True
    max_images: Optional[int] = None  # Limit number of images to process
    image_extensions: List[str] = None
    
    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']


@dataclass 
class ProcessingResults:
    """Results from batch processing"""
    total_processed: int
    successful_optimizations: int
    failed_optimizations: int
    average_psnr_improvement: float
    average_loss_reduction: float
    processing_time_seconds: float
    output_directory: str
    checkpoint_path: Optional[str] = None


@dataclass
class ProcessingCheckpoint:
    """Checkpoint data for resuming processing"""
    processed_files: List[str]
    config: BatchProcessingConfig
    optimization_config: OptimizationConfig
    last_processed_index: int
    results_so_far: List[Dict[str, Any]]
    timestamp: str


@dataclass
class ProcessingSetup:
    """Setup data for batch processing"""
    input_dir: Path
    output_dir: Path
    image_files: List[Path]
    checkpoint_path: Path
    processed_files: List[str]
    results_so_far: List[Dict[str, Any]]
    start_index: int
    vae: Any
    optimizer: LatentOptimizer
    device: str


@dataclass
class ProcessingState:
    """Current state during batch processing"""
    successful: int = 0
    failed: int = 0
    psnr_improvements: List[float] = None
    loss_reductions: List[float] = None
    
    def __post_init__(self):
        if self.psnr_improvements is None:
            self.psnr_improvements = []
        if self.loss_reductions is None:
            self.loss_reductions = []


class BatchProcessor:
    """
    Batch processing engine for VAE latent optimization
    
    Processes directories of images with checkpointing, progress tracking,
    and error handling capabilities.
    """
    
    def __init__(self, config: BatchProcessingConfig):
        self.config = config
        self.io_utils = IOUtils()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        # torch-image-metricsのCalculatorを使用
        self.metrics_calc = tim.Calculator(device='cpu')
        
        # 互換性のため既存calculatorも保持（段階的移行）
        try:
            self.legacy_metrics_calc = ImageMetrics()
        except:
            # 既存実装が削除された場合のフォールバック
            self.legacy_metrics_calc = None
        
    def process_directory(self, input_dir: Union[str, Path], 
                         output_dir: Union[str, Path],
                         optimization_config: OptimizationConfig,
                         vae_model_name: str = "sd14") -> ProcessingResults:
        """
        Process all images in a directory with VAE latent optimization
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            optimization_config: Configuration for optimization
            vae_model_name: VAE model to use ('sd14', 'sd15', etc.)
            
        Returns:
            ProcessingResults with statistics and output information
        """
        start_time = time.time()
        
        # Setup processing environment
        setup = self._setup_processing_environment(
            input_dir, output_dir, optimization_config, vae_model_name
        )
        
        # Execute batch processing
        state = self._execute_batch_processing_loop(setup, optimization_config)
        
        # Generate final results
        processing_time = time.time() - start_time
        results = self._generate_processing_report(
            setup, state, processing_time, optimization_config
        )
        
        return results
    
    def _setup_processing_environment(self, input_dir: Union[str, Path],
                                     output_dir: Union[str, Path],
                                     optimization_config: OptimizationConfig,
                                     vae_model_name: str) -> ProcessingSetup:
        """
        Setup the processing environment including directories, files, and models
        """
        # Prepare directories
        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting batch processing: {input_dir} -> {output_dir}")
        
        # Get list of image files
        image_files = self._get_image_files(input_dir)
        if self.config.max_images:
            image_files = image_files[:self.config.max_images]
        
        print(f"Found {len(image_files)} images to process")
        
        # Setup checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{input_dir.name}_{output_dir.name}.pkl"
        processed_files, results_so_far, start_index = self._load_checkpoint(
            checkpoint_path, image_files, optimization_config
        )
        
        if start_index > 0:
            print(f"📂 Resuming from checkpoint: {start_index}/{len(image_files)} images already processed")
        
        # Load models
        print("Loading VAE model...")
        device = optimization_config.device if hasattr(optimization_config, 'device') else 'cuda'
        vae, device = VAELoader.load_sd_vae_simple(vae_model_name, device)
        optimizer = LatentOptimizer(optimization_config)
        
        return ProcessingSetup(
            input_dir=input_dir,
            output_dir=output_dir,
            image_files=image_files,
            checkpoint_path=checkpoint_path,
            processed_files=processed_files,
            results_so_far=results_so_far,
            start_index=start_index,
            vae=vae,
            optimizer=optimizer,
            device=device
        )
    
    def _execute_batch_processing_loop(self, setup: ProcessingSetup,
                                      optimization_config: OptimizationConfig) -> ProcessingState:
        """
        Execute the main batch processing loop
        """
        state = ProcessingState()
        
        for i in tqdm(range(setup.start_index, len(setup.image_files)), desc="Processing images"):
            image_path = setup.image_files[i]
            
            try:
                # Process single image
                result = self._process_single_image(
                    image_path, setup.output_dir, setup.vae, setup.optimizer, setup.device
                )
                
                if result:
                    state.successful += 1
                    state.psnr_improvements.append(result['psnr_improvement'])
                    state.loss_reductions.append(result['loss_reduction'])
                    setup.results_so_far.append(result)
                else:
                    state.failed += 1
                
                # Checkpoint management
                if (i + 1) % 10 == 0:
                    self._manage_checkpoint(
                        setup, optimization_config, i, state
                    )
                    
            except Exception as e:
                print(f"❌ Failed to process {image_path}: {e}")
                state.failed += 1
                continue
        
        return state
    
    def _manage_checkpoint(self, setup: ProcessingSetup,
                          optimization_config: OptimizationConfig,
                          current_index: int, state: ProcessingState):
        """
        Manage checkpoint saving during processing
        """
        self._save_checkpoint(
            setup.checkpoint_path,
            setup.processed_files + [str(p) for p in setup.image_files[:current_index+1]],
            optimization_config,
            current_index,
            setup.results_so_far
        )
    
    def _generate_processing_report(self, setup: ProcessingSetup,
                                   state: ProcessingState,
                                   processing_time: float,
                                   optimization_config: OptimizationConfig) -> ProcessingResults:
        """
        Generate final processing report and clean up
        """
        # Calculate statistics
        total_processed = state.successful + state.failed
        avg_psnr = sum(state.psnr_improvements) / len(state.psnr_improvements) if state.psnr_improvements else 0
        avg_loss = sum(state.loss_reductions) / len(state.loss_reductions) if state.loss_reductions else 0
        
        # Clean up checkpoint if completed successfully
        if setup.checkpoint_path.exists() and state.failed == 0:
            setup.checkpoint_path.unlink()
        
        # Save summary results
        self._save_processing_summary(setup.output_dir, setup.results_so_far, processing_time)
        
        return ProcessingResults(
            total_processed=total_processed,
            successful_optimizations=state.successful,
            failed_optimizations=state.failed,
            average_psnr_improvement=avg_psnr,
            average_loss_reduction=avg_loss,
            processing_time_seconds=processing_time,
            output_directory=str(setup.output_dir),
            checkpoint_path=str(setup.checkpoint_path) if setup.checkpoint_path.exists() else None
        )
    
    def process_bsds500_dataset(self, bsds500_path: Union[str, Path],
                               output_dir: Union[str, Path],
                               optimization_config: OptimizationConfig,
                               split: str = "all") -> ProcessingResults:
        """
        Process BSDS500 dataset specifically
        
        Args:
            bsds500_path: Path to BSDS500 dataset (should have train/val/test dirs)
            output_dir: Output directory
            optimization_config: Optimization configuration
            split: Which split to process ('train', 'val', 'test', 'all')
            
        Returns:
            ProcessingResults
        """
        bsds500_path = Path(bsds500_path).resolve()
        output_dir = Path(output_dir).resolve()
        
        if split == "all":
            splits_to_process = ["train", "val", "test"]
        else:
            splits_to_process = [split]
        
        all_results = []
        
        for split_name in splits_to_process:
            split_input = bsds500_path / split_name
            split_output = output_dir / split_name
            
            if not split_input.exists():
                print(f"⚠️ Split directory not found: {split_input}")
                continue
            
            print(f"\n📁 Processing {split_name} split...")
            result = self.process_directory(split_input, split_output, optimization_config)
            all_results.append(result)
        
        # Combine results if processing multiple splits
        if len(all_results) > 1:
            combined_result = self._combine_processing_results(all_results)
            return combined_result
        else:
            return all_results[0] if all_results else ProcessingResults(
                total_processed=0, successful_optimizations=0, failed_optimizations=0,
                average_psnr_improvement=0, average_loss_reduction=0,
                processing_time_seconds=0, output_directory=str(output_dir)
            )
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in directory"""
        image_files = []
        for ext in self.config.image_extensions:
            image_files.extend(directory.glob(f'*{ext}'))
        return sorted(image_files)
    
    def _process_single_image(self, image_path: Path, output_dir: Path, 
                            vae, optimizer: LatentOptimizer, device: str) -> Optional[Dict[str, Any]]:
        """Process a single image"""
        try:
            # Load and preprocess image
            original_tensor, original_pil = load_and_preprocess_image(str(image_path), target_size=512)
            original_tensor = original_tensor.to(device)
            
            # Convert target image to [0,1] range to match reconstructed output
            target_tensor = (original_tensor + 1.0) / 2.0  # [-1,1] → [0,1]
            
            # Initial VAE encoding
            with torch.no_grad():
                initial_latents = vae.encode(original_tensor).latent_dist.mode()
                initial_reconstruction = vae.decode(initial_latents)
                initial_recon_tensor = (initial_reconstruction.sample / 2 + 0.5).clamp(0, 1)
            
            # Calculate initial metrics (using consistent [0,1] range)
            initial_metrics = self.metrics_calc.compute_all_metrics(target_tensor, initial_recon_tensor)
            
            # Perform optimization
            opt_result = optimizer.optimize(vae, initial_latents, target_tensor)
            
            # Final reconstruction
            with torch.no_grad():
                final_reconstruction = vae.decode(opt_result.optimized_latents)
                final_recon_tensor = (final_reconstruction.sample / 2 + 0.5).clamp(0, 1)
            
            # Calculate final metrics (using consistent [0,1] range)
            final_metrics = self.metrics_calc.compute_all_metrics(target_tensor, final_recon_tensor)
            
            # Save results if requested
            if self.config.save_visualizations:
                image_output_dir = output_dir / image_path.stem
                image_output_dir.mkdir(exist_ok=True)
                
                results_saver = ResultsSaver(image_output_dir)
                saved_files = results_saver.save_optimization_results(
                    target_tensor, initial_recon_tensor, final_recon_tensor,
                    initial_latents, opt_result.optimized_latents,
                    opt_result.losses, opt_result.metrics, image_path.stem
                )
            
            # Prepare result record
            result = {
                'image_path': str(image_path),
                'image_name': image_path.stem,
                'initial_psnr': initial_metrics.psnr_db,
                'final_psnr': final_metrics.psnr_db,
                'psnr_improvement': final_metrics.psnr_db - initial_metrics.psnr_db,
                'initial_ssim': initial_metrics.ssim,
                'final_ssim': final_metrics.ssim,
                'ssim_improvement': final_metrics.ssim - initial_metrics.ssim,
                'loss_reduction': opt_result.metrics['loss_reduction_percent'],
                'optimization_iterations': len(opt_result.losses),
                'convergence_iteration': opt_result.convergence_iteration,
                'initial_loss': opt_result.initial_loss,
                'final_loss': opt_result.final_loss
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def _load_checkpoint(self, checkpoint_path: Path, image_files: List[Path],
                        optimization_config: OptimizationConfig) -> Tuple[List[str], List[Dict], int]:
        """Load checkpoint if it exists and is valid"""
        if not self.config.resume_from_checkpoint or not checkpoint_path.exists():
            return [], [], 0
        
        try:
            checkpoint = self.io_utils.load_pickle(checkpoint_path)
            
            # Validate checkpoint compatibility
            if (checkpoint.config.batch_size != self.config.batch_size or
                checkpoint.optimization_config.iterations != optimization_config.iterations):
                print("⚠️ Checkpoint configuration mismatch, starting fresh")
                return [], [], 0
            
            print(f"📂 Loaded checkpoint with {len(checkpoint.processed_files)} processed files")
            return checkpoint.processed_files, checkpoint.results_so_far, checkpoint.last_processed_index + 1
            
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            return [], [], 0
    
    def _save_checkpoint(self, checkpoint_path: Path, processed_files: List[str],
                        optimization_config: OptimizationConfig, last_index: int,
                        results_so_far: List[Dict[str, Any]]):
        """Save processing checkpoint"""
        checkpoint = ProcessingCheckpoint(
            processed_files=processed_files,
            config=self.config,
            optimization_config=optimization_config,
            last_processed_index=last_index,
            results_so_far=results_so_far,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.io_utils.save_pickle(checkpoint, checkpoint_path)
    
    def _save_processing_summary(self, output_dir: Path, results: List[Dict[str, Any]], 
                               processing_time: float):
        """Save processing summary and statistics"""
        summary = {
            'total_images': len(results),
            'processing_time_seconds': processing_time,
            'processing_time_hours': processing_time / 3600,
            'average_psnr_improvement': sum(r['psnr_improvement'] for r in results) / len(results) if results else 0,
            'average_ssim_improvement': sum(r['ssim_improvement'] for r in results) / len(results) if results else 0,
            'average_loss_reduction': sum(r['loss_reduction'] for r in results) / len(results) if results else 0,
            'psnr_improvement_std': StatisticsCalculator.calculate_basic_stats([r['psnr_improvement'] for r in results]).get('std', 0),
            'best_psnr_improvement': max([r['psnr_improvement'] for r in results]) if results else 0,
            'worst_psnr_improvement': min([r['psnr_improvement'] for r in results]) if results else 0,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save summary
        self.io_utils.save_json(summary, output_dir / "processing_summary.json")
        
        # Save detailed results
        self.io_utils.save_json(results, output_dir / "detailed_results.json")
        
        print(f"\n📊 Processing Summary:")
        print(f"  Total images: {summary['total_images']}")
        print(f"  Processing time: {summary['processing_time_hours']:.1f} hours")
        print(f"  Average PSNR improvement: {summary['average_psnr_improvement']:.2f} dB")
        print(f"  Average SSIM improvement: {summary['average_ssim_improvement']:.3f}")
        print(f"  Average loss reduction: {summary['average_loss_reduction']:.1f}%")
    
    def _combine_processing_results(self, results_list: List[ProcessingResults]) -> ProcessingResults:
        """Combine multiple processing results"""
        total_processed = sum(r.total_processed for r in results_list)
        successful = sum(r.successful_optimizations for r in results_list)
        failed = sum(r.failed_optimizations for r in results_list)
        
        # Weighted averages
        total_time = sum(r.processing_time_seconds for r in results_list)
        
        weighted_psnr = sum(r.average_psnr_improvement * r.successful_optimizations for r in results_list)
        weighted_psnr /= successful if successful > 0 else 1
        
        weighted_loss = sum(r.average_loss_reduction * r.successful_optimizations for r in results_list)
        weighted_loss /= successful if successful > 0 else 1
        
        return ProcessingResults(
            total_processed=total_processed,
            successful_optimizations=successful,
            failed_optimizations=failed,
            average_psnr_improvement=weighted_psnr,
            average_loss_reduction=weighted_loss,
            processing_time_seconds=total_time,
            output_directory="combined_results"
        )