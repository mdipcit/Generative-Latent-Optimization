#!/usr/bin/env python3
"""
BSDS500 FID Optimization Datasets - Usage Example

This script demonstrates how to load and use the BSDS500 FID optimization datasets.
Released as part of Generative Latent Optimization project v1.0.0.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def load_dataset(dataset_path):
    """
    Load a BSDS500 FID optimization dataset.
    
    Args:
        dataset_path (str): Path to the .pt file
        
    Returns:
        dict: Dataset containing 'train', 'val', 'test' splits
    """
    try:
        dataset = torch.load(dataset_path, map_location='cpu')
        print(f"✅ Successfully loaded dataset from {dataset_path}")
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def inspect_dataset(dataset, dataset_name=""):
    """
    Inspect the structure and content of a dataset.
    
    Args:
        dataset (dict): Loaded dataset
        dataset_name (str): Name for display
    """
    if dataset is None:
        return
        
    print(f"\n📊 Dataset Inspection: {dataset_name}")
    print("=" * 50)
    
    # Check keys
    print(f"Available splits: {list(dataset.keys())}")
    
    # Check sizes
    for split in ['train', 'val', 'test']:
        if split in dataset:
            print(f"{split.capitalize()} size: {len(dataset[split])} images")
    
    # Sample image inspection
    if 'train' in dataset and len(dataset['train']) > 0:
        sample = dataset['train'][0]
        if isinstance(sample, dict):
            print(f"Sample structure: {list(sample.keys())}")
            if 'optimized' in sample:
                print(f"Optimized image shape: {sample['optimized'].shape}")
            if 'original' in sample:
                print(f"Original image shape: {sample['original'].shape}")
        else:
            print(f"Sample shape: {sample.shape}")

def create_dataloader(dataset_split, batch_size=16, shuffle=True):
    """
    Create a PyTorch DataLoader from a dataset split.
    
    Args:
        dataset_split: Dataset split (train/val/test)
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    return DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )

def visualize_sample_comparison(dataset, split='train', num_samples=4):
    """
    Visualize comparison between original and optimized images.
    
    Args:
        dataset (dict): Loaded dataset
        split (str): Dataset split to use
        num_samples (int): Number of samples to show
    """
    if split not in dataset:
        print(f"Split '{split}' not found in dataset")
        return
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle(f'Original vs Optimized Images ({split} split)', fontsize=16)
    
    for i in range(min(num_samples, len(dataset[split]))):
        sample = dataset[split][i]
        
        if isinstance(sample, dict):
            # If sample is a dictionary with 'original' and 'optimized' keys
            if 'original' in sample and 'optimized' in sample:
                original = sample['original']
                optimized = sample['optimized']
            else:
                print("Sample doesn't contain 'original' and 'optimized' keys")
                return
        else:
            # If sample is just the optimized image
            optimized = sample
            original = sample  # Fallback - showing same image
        
        # Convert to numpy for visualization
        if isinstance(original, torch.Tensor):
            original = original.permute(1, 2, 0).numpy()
        if isinstance(optimized, torch.Tensor):
            optimized = optimized.permute(1, 2, 0).numpy()
        
        # Normalize to [0, 1] if needed
        original = np.clip(original, 0, 1)
        optimized = np.clip(optimized, 0, 1)
        
        axes[0, i].imshow(original)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(optimized)
        axes[1, i].set_title(f'Optimized {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_basic_statistics(dataset_split):
    """
    Calculate basic statistics for a dataset split.
    
    Args:
        dataset_split: Dataset split to analyze
        
    Returns:
        dict: Basic statistics
    """
    if len(dataset_split) == 0:
        return {}
    
    # Get sample to determine structure
    sample = dataset_split[0]
    
    if isinstance(sample, dict) and 'optimized' in sample:
        images = torch.stack([item['optimized'] for item in dataset_split])
    else:
        images = torch.stack(dataset_split)
    
    stats = {
        'mean': images.mean().item(),
        'std': images.std().item(),
        'min': images.min().item(),
        'max': images.max().item(),
        'shape': images.shape
    }
    
    return stats

def main():
    """
    Main function demonstrating dataset usage.
    """
    print("🚀 BSDS500 FID Optimization Datasets - Usage Example")
    print("=" * 60)
    
    # Example dataset paths (adjust as needed)
    dataset_paths = {
        'LPIPS': 'bsds500_lpips_dataset.pt',
        'PSNR': 'bsds500_psnr_dataset.pt', 
        'Improved SSIM': 'bsds500_improved_ssim_dataset.pt'
    }
    
    # Load and inspect datasets
    datasets = {}
    for name, path in dataset_paths.items():
        print(f"\n📂 Loading {name} dataset...")
        dataset = load_dataset(path)
        if dataset:
            datasets[name] = dataset
            inspect_dataset(dataset, name)
    
    # Example usage with the first available dataset
    if datasets:
        first_dataset_name = list(datasets.keys())[0]
        first_dataset = datasets[first_dataset_name]
        
        print(f"\n🔬 Detailed analysis of {first_dataset_name} dataset:")
        
        # Create DataLoaders
        if 'train' in first_dataset:
            train_loader = create_dataloader(first_dataset['train'], batch_size=8)
            print(f"Train DataLoader created: {len(train_loader)} batches")
            
            # Process first batch
            for batch_idx, batch in enumerate(train_loader):
                print(f"First batch shape: {batch.shape if isinstance(batch, torch.Tensor) else 'Complex structure'}")
                break
        
        # Calculate statistics
        for split in ['train', 'val', 'test']:
            if split in first_dataset:
                stats = calculate_basic_statistics(first_dataset[split])
                print(f"{split.capitalize()} statistics: {stats}")
        
        # Visualize samples (uncomment if matplotlib is available)
        # visualize_sample_comparison(first_dataset, split='train', num_samples=4)
    
    print("\n✨ Example completed! Check the documentation for more advanced usage patterns.")

if __name__ == "__main__":
    main()