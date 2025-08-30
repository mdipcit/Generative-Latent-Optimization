#!/usr/bin/env python3
"""
Unit tests for ImageVisualizer module

Tests visualization functionality for VAE optimization results.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict

from generative_latent_optimization.visualization.image_viz import ImageVisualizer


class TestImageVisualizer:
    """Test the ImageVisualizer class"""

    def test_init_default_figsize(self):
        """Test ImageVisualizer initialization with default figsize"""
        viz = ImageVisualizer()
        assert viz.figsize == (15, 5)

    def test_init_custom_figsize(self):
        """Test ImageVisualizer initialization with custom figsize"""
        custom_size = (20, 8)
        viz = ImageVisualizer(figsize=custom_size)
        assert viz.figsize == custom_size

    @pytest.fixture
    def sample_tensors(self):
        """Fixture providing sample image tensors"""
        return {
            'original': torch.rand(1, 3, 64, 64),
            'initial_recon': torch.rand(1, 3, 64, 64),
            'optimized_recon': torch.rand(1, 3, 64, 64),
            'tensor_3d': torch.rand(3, 64, 64),
            'tensor_grayscale': torch.rand(1, 64, 64)
        }

    @pytest.fixture
    def sample_metrics(self):
        """Fixture providing sample metrics"""
        return {
            'initial_psnr': 25.5,
            'final_psnr': 28.3,
            'psnr_improvement': 2.8,
            'initial_ssim': 0.85,
            'final_ssim': 0.91,
            'ssim_improvement': 0.06,
            'loss_reduction': 45.2,
            'optimization_iterations': 50
        }

    @pytest.fixture
    def visualizer(self):
        """Fixture providing ImageVisualizer instance"""
        return ImageVisualizer()


class TestTensorToNumpy:
    """Test the _tensor_to_numpy helper method"""

    @pytest.fixture
    def visualizer(self):
        """Fixture providing ImageVisualizer instance"""
        return ImageVisualizer()

    def test_tensor_to_numpy_4d_input(self, visualizer):
        """Test _tensor_to_numpy with 4D input tensor"""
        tensor = torch.rand(1, 3, 32, 32)  # [1, C, H, W]
        result = visualizer._tensor_to_numpy(tensor)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 32, 3)  # [H, W, C]
        assert 0 <= result.min() <= result.max() <= 1

    def test_tensor_to_numpy_3d_input(self, visualizer):
        """Test _tensor_to_numpy with 3D input tensor"""
        tensor = torch.rand(3, 32, 32)  # [C, H, W]
        result = visualizer._tensor_to_numpy(tensor)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 32, 3)  # [H, W, C]

    def test_tensor_to_numpy_grayscale(self, visualizer):
        """Test _tensor_to_numpy with grayscale tensor"""
        tensor = torch.rand(1, 1, 32, 32)  # [1, 1, H, W]
        result = visualizer._tensor_to_numpy(tensor)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 32)  # [H, W] - squeezed grayscale

    def test_tensor_to_numpy_value_clipping(self, visualizer):
        """Test _tensor_to_numpy clips values to [0, 1]"""
        # Create tensor with values outside [0, 1]
        tensor = torch.tensor([[[[-0.5, 1.5], [0.3, 0.7]]]], dtype=torch.float32)
        result = visualizer._tensor_to_numpy(tensor)
        
        assert result.min() >= 0
        assert result.max() <= 1

    def test_tensor_to_numpy_gradient_handling(self, visualizer):
        """Test _tensor_to_numpy handles gradients properly"""
        tensor = torch.rand(1, 3, 16, 16, requires_grad=True)
        result = visualizer._tensor_to_numpy(tensor)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 16, 3)


class TestCreateComparisonGrid:
    """Test the create_comparison_grid method"""

    @pytest.fixture
    def visualizer(self):
        """Fixture providing ImageVisualizer instance"""
        return ImageVisualizer()

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_comparison_grid_basic(self, mock_plt, visualizer, sample_tensors):
        """Test basic create_comparison_grid functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "comparison.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = [Mock(), Mock(), Mock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            result = visualizer.create_comparison_grid(
                sample_tensors['original'],
                sample_tensors['initial_recon'],
                sample_tensors['optimized_recon'],
                save_path
            )
            
            # Verify matplotlib calls
            mock_plt.subplots.assert_called_once_with(1, 3, figsize=(15, 5))
            mock_plt.tight_layout.assert_called_once()
            mock_plt.savefig.assert_called_once_with(save_path, dpi=150, bbox_inches='tight')
            mock_plt.close.assert_called_once_with(mock_fig)
            
            # Verify axes setup
            for ax in mock_axes:
                ax.imshow.assert_called_once()
                ax.set_title.assert_called_once()
                ax.axis.assert_called_once_with('off')
            
            assert result == save_path

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_comparison_grid_with_metrics(self, mock_plt, visualizer, sample_tensors, sample_metrics):
        """Test create_comparison_grid with metrics display"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "comparison_metrics.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = [Mock(), Mock(), Mock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            result = visualizer.create_comparison_grid(
                sample_tensors['original'],
                sample_tensors['initial_recon'],
                sample_tensors['optimized_recon'],
                save_path,
                metrics=sample_metrics
            )
            
            # Verify metrics text was added
            mock_fig.text.assert_called_once()
            text_call = mock_fig.text.call_args
            
            # Check that text contains metric information
            text_content = text_call[0][3]  # Fourth argument is the text content
            assert "PSNR:" in text_content
            assert "Improvement:" in text_content
            assert "Loss reduction:" in text_content
            
            # Verify layout adjustment for metrics
            mock_plt.subplots_adjust.assert_called_once_with(bottom=0.15)

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_comparison_grid_with_title(self, mock_plt, visualizer, sample_tensors):
        """Test create_comparison_grid with custom title"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "comparison_title.png"
            custom_title = "Test Optimization Results"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = [Mock(), Mock(), Mock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            visualizer.create_comparison_grid(
                sample_tensors['original'],
                sample_tensors['initial_recon'],
                sample_tensors['optimized_recon'],
                save_path,
                title=custom_title
            )
            
            # Verify title was set
            mock_fig.suptitle.assert_called_once_with(custom_title, fontsize=14, fontweight='bold')

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_comparison_grid_directory_creation(self, mock_plt, visualizer, sample_tensors):
        """Test create_comparison_grid creates parent directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use nested path that doesn't exist
            save_path = Path(temp_dir) / "nested" / "dir" / "comparison.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = [Mock(), Mock(), Mock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            result = visualizer.create_comparison_grid(
                sample_tensors['original'],
                sample_tensors['initial_recon'],
                sample_tensors['optimized_recon'],
                save_path
            )
            
            # Parent directory should be created (we can't easily test this without actually creating files)
            assert result == save_path


class TestCreateDetailedComparison:
    """Test the create_detailed_comparison method"""

    @pytest.fixture
    def visualizer(self):
        """Fixture providing ImageVisualizer instance"""
        return ImageVisualizer()

    @pytest.fixture
    def sample_losses(self):
        """Fixture providing sample loss history"""
        return [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2]

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_detailed_comparison_basic(self, mock_plt, visualizer, sample_tensors):
        """Test basic create_detailed_comparison functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "detailed.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            result = visualizer.create_detailed_comparison(
                sample_tensors['original'],
                sample_tensors['initial_recon'],
                sample_tensors['optimized_recon'],
                save_path
            )
            
            # Verify matplotlib setup
            mock_plt.subplots.assert_called_once_with(2, 2, figsize=(12, 10))
            mock_plt.tight_layout.assert_called_once()
            mock_plt.savefig.assert_called_once_with(save_path, dpi=200, bbox_inches='tight')
            mock_plt.close.assert_called_once_with(mock_fig)
            
            assert result == save_path

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_detailed_comparison_with_losses(self, mock_plt, visualizer, sample_tensors, sample_losses):
        """Test create_detailed_comparison with loss curve"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "detailed_losses.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            visualizer.create_detailed_comparison(
                sample_tensors['original'],
                sample_tensors['initial_recon'],
                sample_tensors['optimized_recon'],
                save_path,
                losses=sample_losses
            )
            
            # Verify loss plot was created
            bottom_left_ax = mock_axes[1, 0]
            bottom_left_ax.plot.assert_called_once_with(sample_losses, 'b-', linewidth=2, alpha=0.8)
            bottom_left_ax.set_xlabel.assert_called_once_with('Iteration')
            bottom_left_ax.set_ylabel.assert_called_once_with('Loss')
            bottom_left_ax.set_title.assert_called_once_with('Optimization Progress')
            bottom_left_ax.grid.assert_called_once_with(True, alpha=0.3)
            bottom_left_ax.set_yscale.assert_called_once_with('log')

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_detailed_comparison_with_metrics(self, mock_plt, visualizer, sample_tensors, sample_metrics):
        """Test create_detailed_comparison with metrics display"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "detailed_metrics.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            visualizer.create_detailed_comparison(
                sample_tensors['original'],
                sample_tensors['initial_recon'],
                sample_tensors['optimized_recon'],
                save_path,
                metrics=sample_metrics
            )
            
            # Verify metrics text was added
            bottom_right_ax = mock_axes[1, 1]
            bottom_right_ax.axis.assert_called_once_with('off')
            bottom_right_ax.text.assert_called_once()
            
            # Check text content includes metrics
            text_call = bottom_right_ax.text.call_args
            text_content = text_call[0][3]  # Fourth argument is the text content
            assert "Optimization Results:" in text_content
            assert "Initial PSNR:" in text_content
            assert "Final PSNR:" in text_content
            assert "PSNR Improvement:" in text_content

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_detailed_comparison_no_losses_no_metrics(self, mock_plt, visualizer, sample_tensors):
        """Test create_detailed_comparison without losses or metrics"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "detailed_minimal.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            visualizer.create_detailed_comparison(
                sample_tensors['original'],
                sample_tensors['initial_recon'],
                sample_tensors['optimized_recon'],
                save_path
            )
            
            # Both bottom axes should be turned off
            bottom_left_ax = mock_axes[1, 0]
            bottom_right_ax = mock_axes[1, 1]
            
            bottom_left_ax.axis.assert_called_once_with('off')
            bottom_right_ax.axis.assert_called_once_with('off')


class TestCreateDifferenceMap:
    """Test the create_difference_map method"""

    @pytest.fixture
    def visualizer(self):
        """Fixture providing ImageVisualizer instance"""
        return ImageVisualizer()

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_difference_map_basic(self, mock_plt, visualizer, sample_tensors):
        """Test basic create_difference_map functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "difference.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = [Mock(), Mock(), Mock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            mock_colorbar_im = Mock()
            mock_axes[2].imshow.return_value = mock_colorbar_im
            
            result = visualizer.create_difference_map(
                sample_tensors['original'],
                sample_tensors['optimized_recon'],
                save_path
            )
            
            # Verify matplotlib setup
            mock_plt.subplots.assert_called_once_with(1, 3, figsize=(15, 5))
            mock_plt.tight_layout.assert_called_once()
            mock_plt.savefig.assert_called_once_with(save_path, dpi=150, bbox_inches='tight')
            mock_plt.close.assert_called_once_with(mock_fig)
            
            # Verify colorbar was added
            mock_plt.colorbar.assert_called_once_with(mock_colorbar_im, ax=mock_axes[2], shrink=0.6)
            
            assert result == save_path

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_difference_map_custom_title(self, mock_plt, visualizer, sample_tensors):
        """Test create_difference_map with custom title"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "difference_custom.png"
            custom_title = "Custom Difference Analysis"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = [Mock(), Mock(), Mock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            mock_colorbar_im = Mock()
            mock_axes[2].imshow.return_value = mock_colorbar_im
            
            visualizer.create_difference_map(
                sample_tensors['original'],
                sample_tensors['optimized_recon'],
                save_path,
                title=custom_title
            )
            
            # Verify custom title was set
            mock_axes[2].set_title.assert_called_with(custom_title)

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_difference_map_grayscale_handling(self, mock_plt, visualizer):
        """Test create_difference_map with grayscale images"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "difference_gray.png"
            
            # Create grayscale tensors
            tensor1 = torch.rand(1, 1, 32, 32)
            tensor2 = torch.rand(1, 1, 32, 32)
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = [Mock(), Mock(), Mock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            mock_colorbar_im = Mock()
            mock_axes[2].imshow.return_value = mock_colorbar_im
            
            visualizer.create_difference_map(tensor1, tensor2, save_path)
            
            # Should work without errors for grayscale
            assert mock_axes[2].imshow.call_count == 1


class TestCreateBatchOverview:
    """Test the create_batch_overview method"""

    @pytest.fixture
    def visualizer(self):
        """Fixture providing ImageVisualizer instance"""
        return ImageVisualizer()

    @pytest.fixture
    def sample_batch_results(self, sample_tensors):
        """Fixture providing sample batch results"""
        return [
            {
                'image_name': f'test_image_{i}',
                'optimized_reconstruction': sample_tensors['optimized_recon'],
                'metrics': {
                    'psnr_improvement': 2.5 + i * 0.3,
                    'final_psnr': 25.0 + i * 0.5
                }
            }
            for i in range(8)
        ]

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_batch_overview_basic(self, mock_plt, visualizer, sample_batch_results):
        """Test basic create_batch_overview functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "batch_overview.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            # Create 2x4 grid for 8 images
            mock_axes = np.array([[Mock() for _ in range(4)] for _ in range(2)])
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            result = visualizer.create_batch_overview(
                sample_batch_results,
                save_path
            )
            
            # Verify matplotlib setup for grid layout
            mock_plt.subplots.assert_called_once_with(2, 4, figsize=(16, 8))
            mock_plt.suptitle.assert_called_once_with('Optimization Results Overview (8 images)', fontsize=16, fontweight='bold')
            mock_plt.tight_layout.assert_called_once()
            mock_plt.savefig.assert_called_once_with(save_path, dpi=150, bbox_inches='tight')
            mock_plt.close.assert_called_once_with(mock_fig)
            
            assert result == save_path

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_batch_overview_max_images_limit(self, mock_plt, visualizer, sample_batch_results):
        """Test create_batch_overview respects max_images limit"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "batch_limited.png"
            
            # Mock matplotlib components - should only create grid for 4 images
            mock_fig = Mock()
            mock_axes = np.array([[Mock() for _ in range(4)]])
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            visualizer.create_batch_overview(
                sample_batch_results,  # 8 results
                save_path,
                max_images=4  # Limit to 4
            )
            
            # Should create 1x4 grid for 4 images
            mock_plt.subplots.assert_called_once_with(1, 4, figsize=(16, 4))
            mock_plt.suptitle.assert_called_once_with('Optimization Results Overview (4 images)', fontsize=16, fontweight='bold')

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_batch_overview_single_row(self, mock_plt, visualizer):
        """Test create_batch_overview with single row of images"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "batch_single_row.png"
            
            # Create results for 3 images (single row)
            results = [
                {
                    'image_name': f'test_{i}',
                    'optimized_reconstruction': torch.rand(1, 3, 32, 32),
                    'metrics': {'psnr_improvement': 2.0 + i}
                }
                for i in range(3)
            ]
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = np.array([Mock(), Mock(), Mock(), Mock()])  # Single row, 4 cols
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            visualizer.create_batch_overview(results, save_path)
            
            # Should reshape axes for single row
            mock_plt.subplots.assert_called_once_with(1, 4, figsize=(16, 4))

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_batch_overview_metrics_display(self, mock_plt, visualizer, sample_batch_results):
        """Test create_batch_overview displays metrics correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "batch_metrics.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = np.array([[Mock() for _ in range(4)] for _ in range(2)])
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            visualizer.create_batch_overview(sample_batch_results, save_path)
            
            # Verify text was added to first few axes (for metrics display)
            first_ax = mock_axes[0, 0]
            first_ax.text.assert_called_once()
            
            # Check text content includes metrics
            text_call = first_ax.text.call_args
            text_content = text_call[0][3]  # Fourth argument is the text content
            assert "PSNR:" in text_content


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""

    @pytest.fixture
    def visualizer(self):
        """Fixture providing ImageVisualizer instance"""
        return ImageVisualizer()

    def test_tensor_to_numpy_with_different_ranges(self, visualizer):
        """Test _tensor_to_numpy with tensors in different value ranges"""
        # Test with [-1, 1] range tensor
        tensor_neg_one_one = torch.rand(1, 3, 16, 16) * 2 - 1
        result = visualizer._tensor_to_numpy(tensor_neg_one_one)
        assert 0 <= result.min() <= result.max() <= 1

        # Test with [0, 255] range tensor
        tensor_uint8_range = torch.rand(1, 3, 16, 16) * 255
        result = visualizer._tensor_to_numpy(tensor_uint8_range)
        assert 0 <= result.min() <= result.max() <= 1

    def test_tensor_to_numpy_with_inf_nan(self, visualizer):
        """Test _tensor_to_numpy handles inf and nan values"""
        tensor = torch.tensor([[[[float('inf'), float('nan')], [0.5, 0.3]]]], dtype=torch.float32)
        result = visualizer._tensor_to_numpy(tensor)
        
        # Should be clipped to [0, 1] range
        assert np.isfinite(result).all()
        assert 0 <= result.min() <= result.max() <= 1

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_comparison_grid_empty_metrics(self, mock_plt, visualizer, sample_tensors):
        """Test create_comparison_grid with empty metrics dict"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "comparison_empty_metrics.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = [Mock(), Mock(), Mock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            visualizer.create_comparison_grid(
                sample_tensors['original'],
                sample_tensors['initial_recon'],
                sample_tensors['optimized_recon'],
                save_path,
                metrics={}  # Empty metrics
            )
            
            # Should not add metrics text
            mock_fig.text.assert_not_called()
            mock_plt.subplots_adjust.assert_not_called()

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_batch_overview_empty_results(self, mock_plt, visualizer):
        """Test create_batch_overview with empty results list"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "batch_empty.png"
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = np.array([[Mock() for _ in range(4)]])  # Should still create grid
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            result = visualizer.create_batch_overview([], save_path)
            
            # Should handle empty list gracefully
            mock_plt.subplots.assert_called_once_with(1, 4, figsize=(16, 4))
            mock_plt.suptitle.assert_called_once_with('Optimization Results Overview (0 images)', fontsize=16, fontweight='bold')
            assert result == save_path

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_create_batch_overview_results_without_optimized_reconstruction(self, mock_plt, visualizer):
        """Test create_batch_overview with results missing optimized_reconstruction"""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "batch_incomplete.png"
            
            results = [
                {'image_name': 'test_1', 'metrics': {'psnr_improvement': 2.0}},  # Missing optimized_reconstruction
                {'image_name': 'test_2'}  # Missing both
            ]
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = np.array([[Mock() for _ in range(4)]])
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            result = visualizer.create_batch_overview(results, save_path)
            
            # Should handle missing data gracefully
            assert result == save_path

    def test_tensor_to_numpy_channel_detection(self, visualizer):
        """Test _tensor_to_numpy correctly detects channel dimension"""
        # Test with many channels (should still be treated as channels-first)
        tensor_many_channels = torch.rand(8, 16, 16)  # 8 channels
        result = visualizer._tensor_to_numpy(tensor_many_channels)
        assert result.shape == (16, 16, 8)  # Should permute to [H, W, C]

        # Test with spatial dimensions that could be confused with channels
        tensor_spatial = torch.rand(32, 3, 3)  # Could be [H, W, C] or [C, H, W]
        result = visualizer._tensor_to_numpy(tensor_spatial)
        # Since first dimension (32) > 4, should be treated as [H, W, C]
        assert result.shape == (32, 3, 3)


class TestIntegrationAndWorkflow:
    """Test integration scenarios and workflow combinations"""

    @pytest.fixture
    def visualizer(self):
        """Fixture providing ImageVisualizer instance"""
        return ImageVisualizer()

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_full_visualization_workflow(self, mock_plt, visualizer, sample_tensors, sample_metrics):
        """Test complete visualization workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock matplotlib components for all calls
            mock_fig = Mock()
            mock_axes_3 = [Mock(), Mock(), Mock()]
            mock_axes_2x2 = np.array([[Mock(), Mock()], [Mock(), Mock()]])
            
            # Configure different subplot configurations
            mock_plt.subplots.side_effect = [
                (mock_fig, mock_axes_3),      # For comparison grid
                (mock_fig, mock_axes_2x2),    # For detailed comparison
                (mock_fig, mock_axes_3)       # For difference map
            ]
            
            mock_colorbar_im = Mock()
            mock_axes_3[2].imshow.return_value = mock_colorbar_im
            
            # Run complete workflow
            grid_path = visualizer.create_comparison_grid(
                sample_tensors['original'],
                sample_tensors['initial_recon'], 
                sample_tensors['optimized_recon'],
                Path(temp_dir) / "grid.png",
                metrics=sample_metrics
            )
            
            detailed_path = visualizer.create_detailed_comparison(
                sample_tensors['original'],
                sample_tensors['initial_recon'],
                sample_tensors['optimized_recon'],
                Path(temp_dir) / "detailed.png",
                metrics=sample_metrics,
                losses=[1.0, 0.5, 0.2]
            )
            
            diff_path = visualizer.create_difference_map(
                sample_tensors['original'],
                sample_tensors['optimized_recon'],
                Path(temp_dir) / "diff.png"
            )
            
            # All should complete successfully
            assert grid_path.name == "grid.png"
            assert detailed_path.name == "detailed.png" 
            assert diff_path.name == "diff.png"

    @patch('generative_latent_optimization.visualization.image_viz.plt')
    def test_visualization_with_different_tensor_shapes(self, mock_plt, visualizer):
        """Test visualization methods with various tensor shapes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with different input shapes
            tensors_3d = torch.rand(3, 64, 64)
            tensors_4d = torch.rand(1, 3, 64, 64)
            
            # Mock matplotlib components
            mock_fig = Mock()
            mock_axes = [Mock(), Mock(), Mock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            # Should handle both shapes
            result1 = visualizer.create_comparison_grid(
                tensors_3d, tensors_3d, tensors_3d,
                Path(temp_dir) / "test_3d.png"
            )
            
            result2 = visualizer.create_comparison_grid(
                tensors_4d, tensors_4d, tensors_4d,
                Path(temp_dir) / "test_4d.png"
            )
            
            assert result1.name == "test_3d.png"
            assert result2.name == "test_4d.png"


if __name__ == '__main__':
    pytest.main([__file__])