"""
画像読み込み機能のテスト
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

from src.data.loader import ImageLoader, BSDS500ImageLoader


@unittest.skipIf(not PIL_AVAILABLE, "PIL not available")
class TestImageLoader(unittest.TestCase):
    """画像読み込み機能のテストクラス"""
    
    def setUp(self):
        """テスト用の一時ファイルを準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # テスト用画像を作成
        self.test_image = Image.new('RGB', (300, 200), color='red')
        self.test_image_path = self.temp_path / "test_image.jpg"
        self.test_image.save(self.test_image_path)
        
        # 無効なファイル
        self.invalid_file = self.temp_path / "invalid.txt"
        with open(self.invalid_file, 'w') as f:
            f.write("This is not an image")
        
        self.loader = ImageLoader()
    
    def tearDown(self):
        """テンポラリファイルのクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_valid_image(self):
        """有効な画像の読み込みテスト"""
        loaded_image = self.loader.load_image(self.test_image_path)
        
        self.assertIsNotNone(loaded_image)
        self.assertEqual(loaded_image.mode, 'RGB')
        self.assertEqual(loaded_image.size, (300, 200))
    
    def test_load_nonexistent_file(self):
        """存在しないファイルの読み込みテスト"""
        nonexistent_path = self.temp_path / "nonexistent.jpg"
        loaded_image = self.loader.load_image(nonexistent_path)
        
        self.assertIsNone(loaded_image)
    
    def test_load_invalid_format(self):
        """無効なフォーマットのファイル読み込みテスト"""
        loaded_image = self.loader.load_image(self.invalid_file)
        
        self.assertIsNone(loaded_image)
    
    def test_validate_image(self):
        """画像検証のテスト"""
        # 有効な画像
        self.assertTrue(self.loader.validate_image(self.test_image))
        
        # 小さすぎる画像
        small_image = Image.new('RGB', (10, 10), color='blue')
        self.assertFalse(self.loader.validate_image(small_image))
    
    def test_get_image_info(self):
        """画像情報取得のテスト"""
        info = self.loader.get_image_info(self.test_image)
        
        expected_info = {
            'size': (300, 200),
            'mode': 'RGB',
            'format': None,
            'width': 300,
            'height': 200
        }
        
        self.assertEqual(info, expected_info)


@unittest.skipIf(not PIL_AVAILABLE, "PIL not available")
class TestBSDS500ImageLoader(unittest.TestCase):
    """BSDS500画像読み込み機能のテストクラス"""
    
    def setUp(self):
        """テスト用のディレクトリ構造を作成"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # BSDS500風のディレクトリ構造を作成
        for split in ['train', 'val', 'test']:
            split_dir = self.temp_path / split
            split_dir.mkdir()
            
            # 各splitに2枚のテスト画像を作成
            for i in range(2):
                image = Image.new('RGB', (300, 200), color='red')
                image_path = split_dir / f"{split}_image_{i:03d}.jpg"
                image.save(image_path)
    
    def tearDown(self):
        """テンポラリファイルのクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization_with_path(self):
        """パス指定での初期化テスト"""
        loader = BSDS500ImageLoader(dataset_path=self.temp_path)
        self.assertEqual(loader.dataset_path, self.temp_path)
    
    def test_get_image_paths(self):
        """画像パス取得のテスト"""
        loader = BSDS500ImageLoader(dataset_path=self.temp_path)
        
        for split in ['train', 'val', 'test']:
            paths = loader.get_image_paths(split)
            self.assertEqual(len(paths), 2)
            
            # パスが正しいかチェック
            for path in paths:
                self.assertTrue(path.exists())
                self.assertEqual(path.suffix, '.jpg')
    
    def test_invalid_split(self):
        """無効なsplitの指定テスト"""
        loader = BSDS500ImageLoader(dataset_path=self.temp_path)
        
        with self.assertRaises(ValueError):
            loader.get_image_paths('invalid_split')
    
    def test_load_split_images(self):
        """split画像の読み込みテスト"""
        loader = BSDS500ImageLoader(dataset_path=self.temp_path)
        
        loaded_images = loader.load_split_images('train')
        
        self.assertEqual(len(loaded_images), 2)
        
        for path, image in loaded_images:
            self.assertIsInstance(path, Path)
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.mode, 'RGB')
    
    def test_get_dataset_statistics(self):
        """データセット統計取得のテスト"""
        loader = BSDS500ImageLoader(dataset_path=self.temp_path)
        
        stats = loader.get_dataset_statistics()
        
        # 基本構造の確認
        self.assertIn('splits', stats)
        self.assertIn('total_images', stats)
        self.assertEqual(stats['total_images'], 6)  # 各split 2枚 × 3 splits
        
        # 各splitの確認
        for split in ['train', 'val', 'test']:
            self.assertIn(split, stats['splits'])
            self.assertEqual(stats['splits'][split]['count'], 2)


class TestLoaderWithoutPIL(unittest.TestCase):
    """PIL未インストール時のテスト"""
    
    def test_import_error_on_initialization(self):
        """PIL未インストール時の初期化エラーテスト"""
        # PIL_AVAILABLEを一時的にFalseに設定
        import src.data.loader as loader_module
        original_pil_available = loader_module.PIL_AVAILABLE
        loader_module.PIL_AVAILABLE = False
        
        try:
            with self.assertRaises(ImportError):
                ImageLoader()
        finally:
            # 元の状態に戻す
            loader_module.PIL_AVAILABLE = original_pil_available


if __name__ == '__main__':
    # ログレベルを設定
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # テスト実行
    unittest.main()