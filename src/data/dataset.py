"""
BSDS500データセットクラス

元BSDS500データからvae-toolkitを使用して直接読み込み・前処理を行うデータセットクラスです。
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import logging

try:
    from vae_toolkit import load_and_preprocess_image
    VAE_TOOLKIT_AVAILABLE = True
except ImportError:
    VAE_TOOLKIT_AVAILABLE = False
    load_and_preprocess_image = None


class BSDS500Dataset:
    """BSDS500データセットクラス - 元画像から直接読み込み"""
    
    def __init__(self, 
                 bsds500_path: Optional[Union[str, Path]] = None,
                 split: str = 'train',
                 target_size: int = 256,
                 cache_in_memory: bool = False):
        """
        Args:
            bsds500_path: BSDS500データセットのパス。Noneの場合は$BSDS500_PATH環境変数を使用
            split: データセットの分割（'train', 'val', 'test'）
            target_size: リサイズ先のサイズ (default: 256)
            cache_in_memory: データをメモリにキャッシュするかどうか
        """
        if not VAE_TOOLKIT_AVAILABLE:
            raise ImportError("vae-toolkit is required for dataset loading. Install with: pip install vae-toolkit")
        
        # BSDS500パスの設定
        if bsds500_path is None:
            bsds500_path = os.environ.get('BSDS500_PATH')
            if bsds500_path is None:
                raise ValueError("BSDS500 path not provided. Set BSDS500_PATH environment variable or pass bsds500_path argument.")
        
        self.bsds500_path = Path(bsds500_path)
        self.split = split
        self.target_size = target_size
        self.cache_in_memory = cache_in_memory
        
        # パス設定
        self.images_dir = self.bsds500_path / split
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        # データファイル一覧を取得
        self._load_file_list()
        
        # キャッシュ初期化
        self.data_cache = {} if cache_in_memory else None
        
        self.logger.info(f"BSDS500Dataset initialized: {len(self.file_list)} images in {split} (target size: {target_size}x{target_size})")
    
    def _load_file_list(self):
        """元BSDS500画像ファイルリストを読み込み"""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # JPGファイルを取得
        jpg_files = list(self.images_dir.glob("*.jpg"))
        jpg_files.sort()
        
        self.file_list = []
        for jpg_file in jpg_files:
            file_info = {
                'image_path': jpg_file,
                'name': jpg_file.stem
            }
            self.file_list.append(file_info)
        
        if len(self.file_list) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
    
    def __len__(self) -> int:
        """データセットサイズを返す"""
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", Dict]:
        """
        指定されたインデックスのデータを取得
        
        Args:
            idx: データインデックス
            
        Returns:
            Tuple[torch.Tensor, Dict]: (image_tensor, metadata)
                image_tensor: torch.Size([1, 3, target_size, target_size]), [-1, 1] normalized
                metadata: ファイル名等のメタデータ
        """
        if idx >= len(self.file_list):
            raise IndexError(f"Index {idx} out of range (dataset size: {len(self.file_list)})")
        
        file_info = self.file_list[idx]
        
        # キャッシュから取得を試行
        if self.cache_in_memory and idx in self.data_cache:
            image_tensor, metadata = self.data_cache[idx]
        else:
            # vae-toolkitで画像を読み込み・前処理
            image_tensor, pil_image = load_and_preprocess_image(
                str(file_info['image_path']), 
                target_size=self.target_size
            )
            
            # メタデータを作成
            metadata = {
                'name': file_info['name'],
                'original_path': str(file_info['image_path']),
                'target_size': self.target_size,
                'split': self.split
            }
            
            # キャッシュに保存
            if self.cache_in_memory:
                self.data_cache[idx] = (image_tensor, metadata)
        
        return image_tensor, metadata
    
    def get_image_path(self, idx: int) -> str:
        """指定されたインデックスの画像パスを取得"""
        if idx >= len(self.file_list):
            raise IndexError(f"Index {idx} out of range (dataset size: {len(self.file_list)})")
        return str(self.file_list[idx]['image_path'])
        
    def get_image_name(self, idx: int) -> str:
        """指定されたインデックスの画像名を取得"""
        if idx >= len(self.file_list):
            raise IndexError(f"Index {idx} out of range (dataset size: {len(self.file_list)})")
        return self.file_list[idx]['name']
    
    def get_batch(self, indices: List[int]) -> Tuple["torch.Tensor", List[Dict]]:
        """
        指定されたインデックスのバッチデータを取得
        
        Args:
            indices: データインデックスのリスト
            
        Returns:
            Tuple[torch.Tensor, List[Dict]]: 
                バッチデータ（形状: [batch_size, 3, target_size, target_size]）
        """
        import torch
        
        batch_images = []
        batch_metadata = []
        
        for idx in indices:
            image_tensor, metadata = self[idx]
            batch_images.append(image_tensor.squeeze(0))  # [1, 3, H, W] -> [3, H, W]
            batch_metadata.append(metadata)
        
        # torch.Tensorに変換
        batch_tensor = torch.stack(batch_images, dim=0)  # [batch_size, 3, H, W]
        
        return batch_tensor, batch_metadata
    
    def clear_cache(self):
        """メモリキャッシュをクリア"""
        if self.data_cache is not None:
            self.data_cache.clear()
            self.logger.info("Data cache cleared")


class BSDS500DataLoader:
    """BSDS500データセット用のシンプルなデータローダー"""
    
    def __init__(self, 
                 dataset: BSDS500Dataset,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        Args:
            dataset: BSDS500Datasetインスタンス
            batch_size: バッチサイズ
            shuffle: データをシャッフルするかどうか
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # インデックスリストを作成
        self.indices = list(range(len(dataset)))
        if shuffle:
            import random
            random.shuffle(self.indices)
        
        self.current_idx = 0
    
    def __iter__(self):
        """イテレータの初期化"""
        self.current_idx = 0
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        return self
    
    def __next__(self):
        """次のバッチを取得"""
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        # バッチインデックスを取得
        end_idx = min(self.current_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.current_idx:end_idx]
        
        # バッチデータを取得
        batch_tensor, batch_metadata = self.dataset.get_batch(batch_indices)
        
        self.current_idx = end_idx
        return batch_tensor, batch_metadata
    
    def __len__(self):
        """バッチ数を返す"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size