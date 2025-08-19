"""
画像前処理パイプライン統合テスト

前処理システム全体の動作を検証します。
"""

import unittest
import tempfile
import shutil
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

from src.data.preprocessor import ImagePreprocessor
from src.data.dataset import BSDS500Dataset, BSDS500DataLoader


@unittest.skipIf(not PIL_AVAILABLE, "PIL not available")
class TestImagePreprocessingIntegration(unittest.TestCase):
    """画像前処理統合テストクラス"""
    
    def setUp(self):
        """テスト用のディレクトリとデータを準備"""
        # 一時ディレクトリ作成
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # BSDS500風の入力ディレクトリ構造を作成
        self.input_dir = self.temp_path / "bsds_input"
        for split in ['train', 'val', 'test']:
            split_dir = self.input_dir / split
            split_dir.mkdir(parents=True)
            
            # 各splitに2枚のテスト画像を作成
            for i in range(2):
                # 異なるサイズの画像を作成
                sizes = [(600, 400), (480, 320)]
                colors = ['red', 'blue']
                
                size = sizes[i % 2]
                color = colors[i % 2]
                
                image = Image.new('RGB', size, color=color)
                image_path = split_dir / f"{split}_image_{i:03d}.jpg"
                image.save(image_path, 'JPEG')
        
        # 出力ディレクトリ
        self.output_dir = self.temp_path / "processed_output"
    
    def tearDown(self):
        """テンポラリファイルのクリーンアップ"""
        shutil.rmtree(self.temp_dir)
    
    def test_full_preprocessing_pipeline(self):
        """完全な前処理パイプラインのテスト"""
        # 前処理器初期化
        preprocessor = ImagePreprocessor(
            target_size=256,
            normalize_range=(-1, 1),
            output_dir=self.output_dir,
            save_metadata=True,
            overwrite=True
        )
        
        # データセット前処理実行
        results = preprocessor.process_dataset(
            dataset_path=self.input_dir,
            splits=['train', 'val'],  # testは除外してテスト
            batch_size=2,
            num_workers=2
        )
        
        # 結果検証
        self.assertIn('stats', results)
        self.assertIn('results', results)
        self.assertIn('summary', results)
        
        # 統計情報検証
        stats = results['stats']
        self.assertEqual(stats['processed'], 4)  # train: 2, val: 2
        self.assertEqual(stats['failed'], 0)
        
        # 出力ファイルの存在確認
        for split in ['train', 'val']:
            images_dir = self.output_dir / split / 'images'
            metadata_dir = self.output_dir / split / 'metadata'
            
            self.assertTrue(images_dir.exists())
            self.assertTrue(metadata_dir.exists())
            
            # NPZファイルとJSONファイルの確認
            npz_files = list(images_dir.glob("*.npz"))
            json_files = list(metadata_dir.glob("*.json"))
            
            self.assertEqual(len(npz_files), 2)
            self.assertEqual(len(json_files), 2)
    
    def test_dataset_loading(self):
        """前処理済みデータセットの読み込みテスト"""
        # まず前処理を実行
        preprocessor = ImagePreprocessor(
            target_size=128,
            normalize_range=(0, 1),
            output_dir=self.output_dir,
            save_metadata=True
        )
        
        preprocessor.process_dataset(
            dataset_path=self.input_dir,
            splits=['train'],
            num_workers=1
        )
        
        # 前処理済みデータセットを読み込み
        dataset = BSDS500Dataset(
            processed_data_dir=self.output_dir,
            split='train',
            load_metadata=True,
            cache_in_memory=True
        )
        
        # データセット基本検証
        self.assertEqual(len(dataset), 2)
        
        # 個別データ取得テスト
        image, metadata = dataset[0]
        
        self.assertEqual(image.shape, (128, 128, 3))
        self.assertEqual(image.dtype, np.float32)
        self.assertGreaterEqual(image.min(), 0.0)
        self.assertLessEqual(image.max(), 1.0)
        self.assertIsInstance(metadata, dict)
        
        # バッチデータ取得テスト
        batch_images, batch_metadata = dataset.get_batch([0, 1])
        
        self.assertEqual(batch_images.shape, (2, 128, 128, 3))
        self.assertEqual(len(batch_metadata), 2)
    
    def test_data_loader(self):
        """データローダーのテスト"""
        # 前処理実行
        preprocessor = ImagePreprocessor(
            target_size=64,
            output_dir=self.output_dir
        )
        
        preprocessor.process_dataset(
            dataset_path=self.input_dir,
            splits=['train'],
            num_workers=1
        )
        
        # データセットとローダー作成
        dataset = BSDS500Dataset(
            processed_data_dir=self.output_dir,
            split='train',
            load_metadata=False
        )
        
        dataloader = BSDS500DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False
        )
        
        # データローダーのイテレーション
        batches = list(dataloader)
        
        self.assertEqual(len(batches), 2)  # 2個のサンプルで batch_size=1
        
        for batch in batches:
            self.assertEqual(batch.shape, (1, 64, 64, 3))
    
    def test_validation(self):
        """出力検証のテスト"""
        # 前処理実行
        preprocessor = ImagePreprocessor(
            target_size=256,
            normalize_range=(-1, 1),
            output_dir=self.output_dir
        )
        
        preprocessor.process_dataset(
            dataset_path=self.input_dir,
            splits=['train'],
            num_workers=1
        )
        
        # 出力検証実行
        validation_result = preprocessor.validate_output('train', sample_size=2)
        
        # 検証結果確認
        self.assertIn('total_files', validation_result)
        self.assertIn('validation_rate', validation_result)
        self.assertEqual(validation_result['total_files'], 2)
        self.assertEqual(validation_result['validation_rate'], 1.0)  # 全て有効
        
        # 詳細結果確認
        details = validation_result['details']
        for detail in details:
            self.assertTrue(detail['valid'])
            self.assertEqual(detail['shape'], (256, 256, 3))
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 存在しないディレクトリでの初期化
        with self.assertRaises(Exception):
            ImagePreprocessor().process_dataset(
                dataset_path=Path("/nonexistent/path")
            )
        
        # 空のディレクトリでの処理
        empty_dir = self.temp_path / "empty"
        empty_dir.mkdir()
        
        preprocessor = ImagePreprocessor(output_dir=self.output_dir)
        result = preprocessor.process_dataset(dataset_path=empty_dir)
        
        # エラーが適切に処理されることを確認
        self.assertIn('error', result)
    
    def test_memory_usage(self):
        """メモリ使用量のテスト"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # キャッシュ有りでデータセット作成
        preprocessor = ImagePreprocessor(
            target_size=128,
            output_dir=self.output_dir
        )
        
        preprocessor.process_dataset(
            dataset_path=self.input_dir,
            splits=['train'],
            num_workers=1
        )
        
        dataset = BSDS500Dataset(
            processed_data_dir=self.output_dir,
            split='train',
            cache_in_memory=True
        )
        
        # 全データを読み込み（キャッシュに保存される）
        for i in range(len(dataset)):
            _ = dataset[i]
        
        cached_memory = process.memory_info().rss
        
        # キャッシュクリア
        dataset.clear_cache()
        
        cleared_memory = process.memory_info().rss
        
        # メモリ使用量が適切に管理されていることを確認
        memory_increase = cached_memory - initial_memory
        memory_after_clear = cleared_memory - initial_memory
        
        self.assertGreater(memory_increase, 0)  # キャッシュでメモリ増加
        self.assertLess(memory_after_clear, memory_increase)  # クリア後に減少


if __name__ == '__main__':
    # ログレベルを設定
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    unittest.main()