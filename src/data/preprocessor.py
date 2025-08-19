"""
画像前処理クラス

BSDS500データセットの前処理を統合管理するメインクラスです。
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:
    from PIL import Image
    import numpy as np

try:
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
    PIL_AVAILABLE = True
    TQDM_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    TQDM_AVAILABLE = False
    Image = None
    np = None
    tqdm = None

from .loader import BSDS500ImageLoader
from .transforms import ImageTransformPipeline


class ImagePreprocessor:
    """画像前処理統合クラス"""
    
    def __init__(self, 
                 target_size: int = 512,
                 normalize_range: Tuple[float, float] = (-1.0, 1.0),
                 output_dir: Union[str, Path] = "./processed_data",
                 save_metadata: bool = True,
                 overwrite: bool = False):
        """
        Args:
            target_size: 前処理後の画像サイズ（正方形）
            normalize_range: 正規化範囲
            output_dir: 出力ディレクトリ
            save_metadata: メタデータを保存するかどうか
            overwrite: 既存ファイルを上書きするかどうか
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL and numpy are required for image preprocessing")
        
        self.target_size = target_size
        self.normalize_range = normalize_range
        self.output_dir = Path(output_dir)
        self.save_metadata = save_metadata
        self.overwrite = overwrite
        
        # ロガー設定（最初に設定）
        self.logger = logging.getLogger(__name__)
        
        # 出力ディレクトリの作成
        self._create_output_directories()
        
        # 変換パイプラインの初期化
        self.transform_pipeline = ImageTransformPipeline(
            target_size=target_size,
            normalize_range=normalize_range
        )
        
        # 統計情報
        self.stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _create_output_directories(self):
        """出力ディレクトリ構造を作成"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            if self.save_metadata:
                (self.output_dir / split / 'metadata').mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / 'stats').mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created output directories in {self.output_dir}")
    
    def process_single_image(self, 
                           image_path: Path, 
                           output_split: str) -> Dict:
        """
        単一画像の前処理
        
        Args:
            image_path: 入力画像パス
            output_split: 出力先split（train/val/test）
            
        Returns:
            Dict: 処理結果情報
        """
        try:
            # 出力ファイルパスを生成
            output_filename = image_path.stem + '.npz'
            output_path = self.output_dir / output_split / 'images' / output_filename
            
            # 既存ファイルのスキップチェック
            if output_path.exists() and not self.overwrite:
                self.logger.debug(f"Skipping existing file: {output_filename}")
                return {
                    'status': 'skipped',
                    'input_path': str(image_path),
                    'output_path': str(output_path),
                    'reason': 'file_exists'
                }
            
            # 画像読み込み
            with Image.open(image_path) as image:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 前処理実行
                processed_array, metadata = self.transform_pipeline(image)
                
                # 処理済み画像を保存
                np.savez_compressed(
                    output_path,
                    image=processed_array,
                    metadata=json.dumps(metadata)
                )
                
                # メタデータファイルの保存
                if self.save_metadata:
                    metadata_path = self.output_dir / output_split / 'metadata' / (image_path.stem + '.json')
                    enhanced_metadata = {
                        **metadata,
                        'input_path': str(image_path),
                        'output_path': str(output_path),
                        'processing_timestamp': time.time(),
                        'target_size': self.target_size,
                        'normalize_range': self.normalize_range
                    }
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(enhanced_metadata, f, indent=2)
                
                self.logger.debug(f"Processed: {image_path.name} -> {output_filename}")
                
                return {
                    'status': 'success',
                    'input_path': str(image_path),
                    'output_path': str(output_path),
                    'metadata': metadata,
                    'output_shape': processed_array.shape
                }
        
        except Exception as e:
            self.logger.error(f"Failed to process {image_path}: {str(e)}")
            return {
                'status': 'failed',
                'input_path': str(image_path),
                'error': str(e)
            }
    
    def process_batch(self, 
                     image_paths: List[Path], 
                     output_split: str,
                     batch_size: int = 32,
                     num_workers: int = 4) -> List[Dict]:
        """
        バッチ画像前処理
        
        Args:
            image_paths: 処理する画像パスのリスト
            output_split: 出力先split
            batch_size: バッチサイズ（表示用）
            num_workers: 並列処理のワーカー数
            
        Returns:
            List[Dict]: 各画像の処理結果リスト
        """
        results = []
        
        # プログレスバー設定
        if TQDM_AVAILABLE:
            progress = tqdm(total=len(image_paths), 
                          desc=f"Processing {output_split}", 
                          unit="images")
        else:
            progress = None
        
        # 並列処理での画像前処理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # すべてのタスクを提出
            future_to_path = {
                executor.submit(self.process_single_image, path, output_split): path
                for path in image_paths
            }
            
            # 完了したタスクから結果を収集
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 統計更新
                    if result['status'] == 'success':
                        self.stats['processed'] += 1
                    elif result['status'] == 'failed':
                        self.stats['failed'] += 1
                    elif result['status'] == 'skipped':
                        self.stats['skipped'] += 1
                    
                    if progress:
                        progress.update(1)
                        progress.set_postfix({
                            'processed': self.stats['processed'],
                            'failed': self.stats['failed'],
                            'skipped': self.stats['skipped']
                        })
                
                except Exception as e:
                    self.logger.error(f"Unexpected error processing {path}: {str(e)}")
                    results.append({
                        'status': 'failed',
                        'input_path': str(path),
                        'error': f"Unexpected error: {str(e)}"
                    })
                    self.stats['failed'] += 1
        
        if progress:
            progress.close()
        
        return results
    
    def process_dataset(self, 
                       dataset_path: Optional[Path] = None,
                       splits: Optional[List[str]] = None,
                       batch_size: int = 32,
                       num_workers: int = 4) -> Dict:
        """
        データセット全体の前処理
        
        Args:
            dataset_path: BSDS500データセットパス（Noneの場合は環境変数から取得）
            splits: 処理するsplitのリスト（Noneの場合は全split）
            batch_size: バッチサイズ
            num_workers: 並列処理のワーカー数
            
        Returns:
            Dict: 処理結果サマリー
        """
        self.stats['start_time'] = time.time()
        
        if splits is None:
            splits = ['train', 'val', 'test']
        
        # データローダーの初期化
        try:
            loader = BSDS500ImageLoader(dataset_path)
        except Exception as e:
            self.logger.error(f"Failed to initialize dataset loader: {str(e)}")
            return {'error': str(e)}
        
        all_results = {}
        
        # 各splitの処理
        for split in splits:
            self.logger.info(f"Processing {split} split...")
            
            try:
                # 画像パス取得
                image_paths = loader.get_image_paths(split)
                self.logger.info(f"Found {len(image_paths)} images in {split}")
                
                # バッチ処理実行
                split_results = self.process_batch(
                    image_paths, 
                    split, 
                    batch_size=batch_size,
                    num_workers=num_workers
                )
                
                all_results[split] = split_results
                
            except Exception as e:
                self.logger.error(f"Failed to process {split} split: {str(e)}")
                all_results[split] = {'error': str(e)}
        
        self.stats['end_time'] = time.time()
        
        # 統計情報保存
        self._save_processing_stats(all_results)
        
        return {
            'stats': self.stats,
            'results': all_results,
            'summary': self._generate_summary()
        }
    
    def _save_processing_stats(self, results: Dict):
        """処理統計を保存"""
        stats_file = self.output_dir / 'stats' / 'processing_log.json'
        
        stats_data = {
            'processing_config': {
                'target_size': self.target_size,
                'normalize_range': self.normalize_range,
                'save_metadata': self.save_metadata,
                'overwrite': self.overwrite
            },
            'stats': self.stats,
            'detailed_results': results
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        self.logger.info(f"Processing statistics saved to {stats_file}")
    
    def _generate_summary(self) -> Dict:
        """処理サマリーを生成"""
        duration = (self.stats['end_time'] - self.stats['start_time']) if self.stats['end_time'] else 0
        total_images = self.stats['processed'] + self.stats['failed'] + self.stats['skipped']
        
        return {
            'total_images': total_images,
            'processed': self.stats['processed'],
            'failed': self.stats['failed'],
            'skipped': self.stats['skipped'],
            'success_rate': self.stats['processed'] / total_images if total_images > 0 else 0,
            'processing_time': duration,
            'images_per_second': self.stats['processed'] / duration if duration > 0 else 0
        }
    
    def validate_output(self, split: str, sample_size: int = 5) -> Dict:
        """
        出力結果の検証
        
        Args:
            split: 検証するsplit
            sample_size: 検証するサンプル数
            
        Returns:
            Dict: 検証結果
        """
        images_dir = self.output_dir / split / 'images'
        
        if not images_dir.exists():
            return {'error': f"Output directory not found: {images_dir}"}
        
        npz_files = list(images_dir.glob("*.npz"))
        
        if len(npz_files) == 0:
            return {'error': f"No processed files found in {images_dir}"}
        
        # サンプルファイルを選択
        sample_files = npz_files[:sample_size]
        validation_results = []
        
        for npz_file in sample_files:
            try:
                # NPZファイル読み込み
                data = np.load(npz_file)
                image_array = data['image']
                metadata_json = data['metadata'].item()
                metadata = json.loads(metadata_json)
                
                # 基本検証
                result = {
                    'file': npz_file.name,
                    'shape': image_array.shape,
                    'dtype': str(image_array.dtype),
                    'range': [float(image_array.min()), float(image_array.max())],
                    'metadata_keys': list(metadata.keys()),
                    'valid': True
                }
                
                # 形状検証
                if image_array.shape != (self.target_size, self.target_size, 3):
                    result['valid'] = False
                    result['error'] = f"Invalid shape: {image_array.shape}"
                
                # 範囲検証
                min_val, max_val = self.normalize_range
                if image_array.min() < min_val - 0.1 or image_array.max() > max_val + 0.1:
                    result['valid'] = False
                    result['error'] = f"Values out of range: [{image_array.min():.3f}, {image_array.max():.3f}]"
                
                validation_results.append(result)
                
            except Exception as e:
                validation_results.append({
                    'file': npz_file.name,
                    'valid': False,
                    'error': str(e)
                })
        
        # 検証サマリー
        valid_count = sum(1 for r in validation_results if r['valid'])
        
        return {
            'total_files': len(npz_files),
            'sampled_files': len(sample_files),
            'valid_samples': valid_count,
            'validation_rate': valid_count / len(sample_files) if sample_files else 0,
            'details': validation_results
        }