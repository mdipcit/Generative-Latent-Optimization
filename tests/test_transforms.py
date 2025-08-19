"""
画像変換関数のテスト
"""

import unittest
import tempfile
import os
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# プロジェクトルートをパスに追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import (
    resize_short_edge,
    center_crop,
    resize_and_crop,
    normalize_to_range,
    denormalize_from_range,
    validate_image_properties,
    compute_resize_params,
    ImageTransformPipeline
)


@unittest.skipIf(not PIL_AVAILABLE, "PIL not available")
class TestImageTransforms(unittest.TestCase):
    """画像変換関数のテストクラス"""
    
    def setUp(self):
        """テスト用画像の準備"""
        # テスト用のRGB画像を作成
        self.test_image = Image.new('RGB', (600, 400), color='red')
        self.square_image = Image.new('RGB', (512, 512), color='blue')
        self.small_image = Image.new('RGB', (100, 150), color='green')
    
    def test_resize_short_edge(self):
        """短辺基準リサイズのテスト"""
        # 400が短辺なので、400 -> 512にリサイズ
        # 600 * (512/400) = 768
        resized = resize_short_edge(self.test_image, 512)
        
        self.assertEqual(resized.size, (768, 512))
        self.assertEqual(resized.mode, 'RGB')
    
    def test_resize_short_edge_square(self):
        """正方形画像のリサイズテスト"""
        resized = resize_short_edge(self.square_image, 256)
        self.assertEqual(resized.size, (256, 256))
    
    def test_center_crop(self):
        """センタークロップのテスト"""
        # 600x400から512x512をクロップ（不可能なのでエラーになるはず）
        with self.assertRaises(ValueError):
            center_crop(self.test_image, 512)
        
        # 600x400から300x200をクロップ
        cropped = center_crop(self.test_image, (300, 200))
        self.assertEqual(cropped.size, (300, 200))
    
    def test_center_crop_square(self):
        """正方形クロップのテスト"""
        cropped = center_crop(self.test_image, 300)
        self.assertEqual(cropped.size, (300, 300))
    
    def test_resize_and_crop(self):
        """リサイズ+クロップのテスト"""
        result = resize_and_crop(self.test_image, 512)
        self.assertEqual(result.size, (512, 512))
        self.assertEqual(result.mode, 'RGB')
    
    def test_normalize_to_range(self):
        """正規化のテスト"""
        # デフォルト範囲 [-1, 1]
        normalized = normalize_to_range(self.test_image)
        
        self.assertEqual(normalized.shape, (400, 600, 3))
        self.assertEqual(normalized.dtype, np.float32)
        self.assertGreaterEqual(normalized.min(), -1.0)
        self.assertLessEqual(normalized.max(), 1.0)
    
    def test_normalize_custom_range(self):
        """カスタム範囲での正規化テスト"""
        normalized = normalize_to_range(self.test_image, (0.0, 2.0))
        
        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 2.0)
    
    def test_denormalize_from_range(self):
        """逆正規化のテスト"""
        # 正規化 -> 逆正規化
        normalized = normalize_to_range(self.test_image, (-1, 1))
        denormalized = denormalize_from_range(normalized, (-1, 1))
        
        self.assertEqual(denormalized.size, self.test_image.size)
        self.assertEqual(denormalized.mode, 'RGB')
    
    def test_validate_image_properties(self):
        """画像プロパティ検証のテスト"""
        # 有効な画像
        self.assertTrue(validate_image_properties(self.test_image))
        
        # 最小サイズチェック
        self.assertTrue(validate_image_properties(self.test_image, min_size=300))
        self.assertFalse(validate_image_properties(self.small_image, min_size=200))
        
        # 最大サイズチェック
        self.assertFalse(validate_image_properties(self.test_image, max_size=500))
        
        # モードチェック
        self.assertTrue(validate_image_properties(self.test_image, required_mode='RGB'))
        self.assertFalse(validate_image_properties(self.test_image, required_mode='L'))
    
    def test_compute_resize_params(self):
        """リサイズパラメータ計算のテスト"""
        params = compute_resize_params((600, 400), 512)
        
        # 期待値の検証
        self.assertEqual(params['original_size'], (600, 400))
        self.assertEqual(params['target_size'], 512)
        self.assertEqual(params['scale_factor'], 512 / 400)  # 1.28
        self.assertEqual(params['resized_size'], (768, 512))
        self.assertEqual(params['final_size'], (512, 512))
        
        # クロップ座標
        crop_coords = params['crop_coordinates']
        self.assertEqual(crop_coords, (128, 0, 640, 512))  # 左右から128ピクセルクロップ
    
    def test_image_transform_pipeline(self):
        """変換パイプラインのテスト"""
        pipeline = ImageTransformPipeline(target_size=256, normalize_range=(-1, 1))
        
        result, metadata = pipeline(self.test_image)
        
        # 結果検証
        self.assertEqual(result.shape, (256, 256, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(result.min(), -1.0)
        self.assertLessEqual(result.max(), 1.0)
        
        # メタデータ検証
        self.assertEqual(metadata['original_size'], (600, 400))
        self.assertEqual(metadata['target_size'], 256)
        self.assertEqual(metadata['final_size'], (256, 256))
        self.assertEqual(metadata['normalize_range'], (-1, 1))


if __name__ == '__main__':
    # ログレベルを設定
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # テスト実行
    unittest.main()