"""
BSDS500前処理済みデータセットクラス

前処理済みのBSDS500データを効率的に読み込むためのデータセットクラスです。
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    import numpy as np

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class BSDS500Dataset:
    """前処理済みBSDS500データセットクラス"""
    
    def __init__(self, 
                 processed_data_dir: Union[str, Path],
                 split: str = 'train',
                 load_metadata: bool = True,
                 cache_in_memory: bool = False):
        """
        Args:
            processed_data_dir: 前処理済みデータのディレクトリ
            split: データセットの分割（'train', 'val', 'test'）
            load_metadata: メタデータを読み込むかどうか
            cache_in_memory: データをメモリにキャッシュするかどうか
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy is required for dataset loading")
        
        self.processed_data_dir = Path(processed_data_dir)
        self.split = split
        self.load_metadata = load_metadata
        self.cache_in_memory = cache_in_memory
        
        # パス設定
        self.images_dir = self.processed_data_dir / split / 'images'
        self.metadata_dir = self.processed_data_dir / split / 'metadata'
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        # データファイル一覧を取得
        self._load_file_list()
        
        # キャッシュ初期化
        self.data_cache = {} if cache_in_memory else None
        self.metadata_cache = {} if cache_in_memory and load_metadata else None
        
        self.logger.info(f"BSDS500Dataset initialized: {len(self.file_list)} images in {split}")
    
    def _load_file_list(self):
        """処理済みファイルリストを読み込み"""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # NPZファイルを取得
        npz_files = list(self.images_dir.glob("*.npz"))
        npz_files.sort()
        
        self.file_list = []
        for npz_file in npz_files:
            file_info = {
                'image_path': npz_file,
                'name': npz_file.stem
            }
            
            # メタデータファイルパスも追加
            if self.load_metadata and self.metadata_dir.exists():
                metadata_path = self.metadata_dir / (npz_file.stem + '.json')
                if metadata_path.exists():
                    file_info['metadata_path'] = metadata_path
            
            self.file_list.append(file_info)
        
        if len(self.file_list) == 0:
            raise ValueError(f"No processed images found in {self.images_dir}")
    
    def __len__(self) -> int:
        """データセットサイズを返す"""
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Union[Tuple["np.ndarray", Dict], "np.ndarray"]:
        """
        指定されたインデックスのデータを取得
        
        Args:
            idx: データインデックス
            
        Returns:
            Union[Tuple[np.ndarray, Dict], np.ndarray]: 
                load_metadata=Trueの場合は(image, metadata)、Falseの場合はimage
        """
        if idx >= len(self.file_list):
            raise IndexError(f"Index {idx} out of range (dataset size: {len(self.file_list)})")
        
        file_info = self.file_list[idx]
        
        # キャッシュから取得を試行
        if self.cache_in_memory and idx in self.data_cache:
            image = self.data_cache[idx]
        else:
            # NPZファイルから画像データを読み込み
            image = self._load_image(file_info['image_path'])
            
            # キャッシュに保存
            if self.cache_in_memory:
                self.data_cache[idx] = image
        
        # メタデータが必要な場合
        if self.load_metadata:
            # キャッシュから取得を試行
            if self.cache_in_memory and idx in self.metadata_cache:
                metadata = self.metadata_cache[idx]
            else:
                metadata = self._load_metadata(file_info)
                
                # キャッシュに保存
                if self.cache_in_memory:
                    self.metadata_cache[idx] = metadata
            
            return image, metadata
        else:
            return image
    
    def _load_image(self, image_path: Path) -> "np.ndarray":
        """NPZファイルから画像データを読み込み"""
        try:
            data = np.load(image_path)
            image = data['image']
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to load image from {image_path}: {str(e)}")
            raise
    
    def _load_metadata(self, file_info: Dict) -> Dict:
        """メタデータを読み込み"""
        metadata = {'name': file_info['name']}
        
        # NPZファイル内のメタデータを読み込み
        try:
            data = np.load(file_info['image_path'])
            if 'metadata' in data:
                npz_metadata = json.loads(data['metadata'].item())
                metadata.update(npz_metadata)
        except Exception as e:
            self.logger.warning(f"Failed to load NPZ metadata: {str(e)}")
        
        # 外部メタデータファイルを読み込み
        if 'metadata_path' in file_info:
            try:
                with open(file_info['metadata_path'], 'r') as f:
                    external_metadata = json.load(f)
                    metadata.update(external_metadata)
            except Exception as e:
                self.logger.warning(f"Failed to load external metadata: {str(e)}")
        
        return metadata
    
    def get_batch(self, indices: List[int]) -> Union[Tuple["np.ndarray", List[Dict]], "np.ndarray"]:
        """
        指定されたインデックスのバッチデータを取得
        
        Args:
            indices: データインデックスのリスト
            
        Returns:
            Union[Tuple[np.ndarray, List[Dict]], np.ndarray]: 
                バッチデータ（形状: [batch_size, H, W, C]）
        """
        batch_images = []
        batch_metadata = [] if self.load_metadata else None
        
        for idx in indices:
            if self.load_metadata:
                image, metadata = self[idx]
                batch_metadata.append(metadata)
            else:
                image = self[idx]
            
            batch_images.append(image)
        
        # numpy配列に変換
        batch_array = np.stack(batch_images, axis=0)
        
        if self.load_metadata:
            return batch_array, batch_metadata
        else:
            return batch_array
    
    def clear_cache(self):
        """メモリキャッシュをクリア"""
        if self.data_cache is not None:
            self.data_cache.clear()
            self.logger.info("Data cache cleared")
        
        if self.metadata_cache is not None:
            self.metadata_cache.clear()
            self.logger.info("Metadata cache cleared")


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
            np.random.shuffle(self.indices)
        
        self.current_idx = 0
    
    def __iter__(self):
        """イテレータの初期化"""
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        """次のバッチを取得"""
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        # バッチインデックスを取得
        end_idx = min(self.current_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.current_idx:end_idx]
        
        # バッチデータを取得
        batch_data = self.dataset.get_batch(batch_indices)
        
        self.current_idx = end_idx
        return batch_data
    
    def __len__(self):
        """バッチ数を返す"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size