#!/usr/bin/env python3
"""
Dataset Processing Mock Fixtures

Provides standardized mock configurations for dataset processing system testing.
"""

import pytest
import torch
import json
from unittest.mock import patch, MagicMock, Mock, DEFAULT
from typing import Dict, Any, List
from pathlib import Path


@pytest.fixture
def mock_dataset_dependencies():
    """統一Mock設定 - データセット処理システム全体"""
    # VAE model mock
    mock_vae = MagicMock()
    mock_vae.encode.return_value = MagicMock()
    mock_vae.encode.return_value.latent_dist = MagicMock()
    mock_vae.encode.return_value.latent_dist.sample.return_value = torch.randn(1, 4, 64, 64)
    mock_vae.decode.return_value = MagicMock()
    mock_vae.decode.return_value.sample = torch.randn(1, 3, 512, 512)
    
    # LatentOptimizer mock
    mock_optimizer = MagicMock()
    mock_optimizer.optimize.return_value = create_mock_optimization_result()
    
    # File I/O mocks
    mock_file_ops = {
        'load_json': MagicMock(return_value=create_mock_optimization_results_json()),
        'save_json': MagicMock(),
        'copy_file': MagicMock(),
        'ensure_directory': MagicMock()
    }
    
    # Image processing mocks
    mock_image_ops = {
        'Image': create_mock_pil_image(),
        'plt': create_mock_matplotlib()
    }
    
    return {
        'vae': mock_vae,
        'optimizer': mock_optimizer,
        'file_ops': mock_file_ops,
        'image_ops': mock_image_ops
    }


@pytest.fixture
def mock_batch_processing():
    """バッチ処理専用Mock設定"""
    # Processing results mock
    mock_results = create_mock_processing_results()
    
    # VAE loader mock
    mock_vae_loader = MagicMock()
    mock_vae_loader.load_vae.return_value = (create_mock_vae(), 'cuda')
    
    # File system operations mock
    mock_fs_ops = create_mock_filesystem_operations()
    
    return {
        'processing_results': mock_results,
        'vae_loader': mock_vae_loader,
        'filesystem': mock_fs_ops
    }


@pytest.fixture 
def mock_png_dataset_processing():
    """PNG Dataset処理専用Mock設定"""
    # Dataset samples mock
    mock_samples = create_mock_dataset_samples()
    
    # External library mocks
    mock_external = {
        'PIL_Image': create_mock_pil_image(),
        'matplotlib_plt': create_mock_matplotlib(),
        'load_and_preprocess_image': MagicMock(return_value=(torch.randn(1, 3, 512, 512), {}))
    }
    
    # File operations mock
    mock_file_ops = create_mock_file_operations()
    
    return {
        'samples': mock_samples,
        'external': mock_external,
        'file_ops': mock_file_ops
    }


@pytest.fixture
def mock_pytorch_dataset_processing():
    """PyTorch Dataset処理専用Mock設定"""
    # Dataset creation mock
    mock_dataset_creation = {
        'samples': create_mock_pytorch_samples(),
        'metadata': create_mock_dataset_metadata(),
        'latents': create_mock_latent_tensors()
    }
    
    # Optimization results mock
    mock_optimization = create_mock_optimization_results_data()
    
    return {
        'dataset_creation': mock_dataset_creation,
        'optimization': mock_optimization
    }


# Helper functions for creating standardized mock data

def create_mock_optimization_result():
    """標準的な最適化結果Mock"""
    mock_result = MagicMock()
    mock_result.optimized_latents = torch.randn(1, 4, 64, 64)
    mock_result.initial_loss = 1.0
    mock_result.final_loss = 0.2
    mock_result.loss_history = [1.0, 0.8, 0.5, 0.2]
    mock_result.iterations = 50
    mock_result.converged = True
    mock_result.metrics = {
        'psnr_improvement': 5.0,
        'ssim_improvement': 0.15,
        'loss_reduction': 80.0,
        'initial_psnr': 20.0,
        'final_psnr': 25.0,
        'initial_ssim': 0.70,
        'final_ssim': 0.85
    }
    return mock_result


def create_mock_processing_results():
    """バッチ処理結果Mock"""
    return {
        'successful_count': 2,
        'failed_count': 0,
        'average_psnr_improvement': 3.5,
        'average_ssim_improvement': 0.15,
        'average_loss_reduction': 75.0,
        'processing_time_seconds': 3600.0,
        'results': [
            {
                'image_name': 'img1.jpg',
                'psnr_improvement': 3.0,
                'ssim_improvement': 0.10,
                'loss_reduction': 70.0,
                'iterations': 50,
                'converged': True
            },
            {
                'image_name': 'img2.jpg', 
                'psnr_improvement': 4.0,
                'ssim_improvement': 0.20,
                'loss_reduction': 80.0,
                'iterations': 45,
                'converged': True
            }
        ]
    }


def create_mock_optimization_results_json():
    """最適化結果JSON Mock"""
    return {
        'successful_optimizations': 2,
        'failed_optimizations': 0,
        'results': [
            {
                'image_name': 'img1.jpg',
                'psnr_improvement': 5.0,
                'ssim_improvement': 0.15,
                'loss_reduction': 80.0,
                'initial_loss': 1.0,
                'final_loss': 0.2,
                'iterations': 50,
                'converged': True
            }
        ]
    }


def create_mock_dataset_samples():
    """Dataset samples Mock"""
    return [
        {
            'image_name': 'img1.jpg',
            'split': 'train',
            'metrics': {
                'psnr_improvement': 5.0,
                'ssim_improvement': 0.15,
                'loss_reduction': 80.0
            },
            'files': {
                'original': 'original/img1.jpg',
                'initial_recon': 'initial/img1.jpg', 
                'optimized_recon': 'optimized/img1.jpg'
            }
        },
        {
            'image_name': 'img2.jpg',
            'split': 'train',
            'metrics': {
                'psnr_improvement': 6.0,
                'ssim_improvement': 0.20,
                'loss_reduction': 75.0
            },
            'files': {
                'original': 'original/img2.jpg',
                'initial_recon': 'initial/img2.jpg',
                'optimized_recon': 'optimized/img2.jpg'
            }
        }
    ]


def create_mock_pytorch_samples():
    """PyTorch samples Mock"""
    return {
        'train': [
            {
                'original_latents': torch.randn(4, 64, 64),
                'optimized_latents': torch.randn(4, 64, 64),
                'metrics': {'psnr_improvement': 5.0, 'loss_reduction': 80.0}
            }
        ],
        'val': [
            {
                'original_latents': torch.randn(4, 64, 64),
                'optimized_latents': torch.randn(4, 64, 64), 
                'metrics': {'psnr_improvement': 6.0, 'loss_reduction': 75.0}
            }
        ]
    }


def create_mock_dataset_metadata():
    """Dataset metadata Mock"""
    return {
        'creation_timestamp': '2025-08-30T10:00:00',
        'total_samples': 100,
        'processing_statistics': {'avg_psnr': 5.0},
        'dataset_version': '1.0'
    }


def create_mock_latent_tensors():
    """Latent tensors Mock"""
    return {
        'original': torch.randn(1, 4, 64, 64),
        'optimized': torch.randn(1, 4, 64, 64)
    }


def create_mock_optimization_results_data():
    """最適化結果データMock"""
    return [
        {
            'image_name': 'img1.jpg',
            'original_latents_path': 'latents/original/img1.pt',
            'optimized_latents_path': 'latents/optimized/img1.pt', 
            'psnr_improvement': 5.0,
            'loss_reduction': 80.0,
            'iterations': 50,
            'converged': True
        }
    ]


def create_mock_vae():
    """VAE model Mock"""
    mock_vae = MagicMock()
    
    # Encoder mock
    mock_encode_result = MagicMock()
    mock_encode_result.latent_dist = MagicMock()
    mock_encode_result.latent_dist.sample.return_value = torch.randn(1, 4, 64, 64)
    mock_vae.encode.return_value = mock_encode_result
    
    # Decoder mock  
    mock_decode_result = MagicMock()
    mock_decode_result.sample = torch.randn(1, 3, 512, 512)
    mock_vae.decode.return_value = mock_decode_result
    
    return mock_vae


def create_mock_pil_image():
    """PIL.Image Mock"""
    mock_image = MagicMock()
    
    # Image.open mock
    mock_img_instance = MagicMock()
    mock_img_instance.size = (512, 512)
    mock_img_instance.mode = 'RGB'
    mock_image.open.return_value = mock_img_instance
    
    # Image.fromarray mock
    mock_image.fromarray.return_value = mock_img_instance
    
    return mock_image


def create_mock_matplotlib():
    """matplotlib.pyplot Mock"""
    mock_plt = MagicMock()
    
    # Plotting methods
    mock_plt.figure.return_value = MagicMock()
    mock_plt.subplot.return_value = MagicMock()
    mock_plt.imshow.return_value = MagicMock()
    mock_plt.savefig.return_value = None
    mock_plt.close.return_value = None
    
    return mock_plt


def create_mock_filesystem_operations():
    """ファイルシステム操作Mock"""
    return {
        'Path_exists': True,
        'Path_mkdir': None,
        'Path_is_file': True,
        'Path_is_dir': True,
        'copy2': None,
        'json_dump': None,
        'json_load': create_mock_optimization_results_json()
    }


def create_mock_file_operations():
    """ファイル操作Mock"""
    mock_ops = MagicMock()
    
    # JSON operations
    mock_ops.load_json.return_value = create_mock_optimization_results_json()
    mock_ops.save_json.return_value = None
    
    # File copy operations
    mock_ops.copy_file.return_value = None
    mock_ops.ensure_directory.return_value = None
    
    return mock_ops