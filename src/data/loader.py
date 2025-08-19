"""
画像読み込み機能

BSDS500データセットの画像を読み込み、基本的な検証を行います。
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from PIL import Image

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    logging.warning("PIL not available. Image loading will not work.")


class ImageLoader:
    """画像読み込みクラス"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def __init__(self, convert_to_rgb: bool = True):
        """
        Args:
            convert_to_rgb: 画像をRGB形式に変換するかどうか
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for image loading")
        
        self.convert_to_rgb = convert_to_rgb
        self.logger = logging.getLogger(__name__)
    
    def load_image(self, image_path: Path) -> Optional["Image.Image"]:
        """
        単一画像の読み込み
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            PIL.Image.Image: 読み込んだ画像（失敗時はNone）
        """
        try:
            # ファイル存在確認
            if not image_path.exists():
                self.logger.error(f"Image file not found: {image_path}")
                return None
            
            # ファイル拡張子確認
            if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                self.logger.warning(f"Unsupported format: {image_path.suffix}")
                return None
            
            # 画像読み込み
            with Image.open(image_path) as img:
                # RGB形式に変換
                if self.convert_to_rgb and img.mode != 'RGB':
                    img = img.convert('RGB')
                    self.logger.debug(f"Converted {image_path.name} to RGB")
                
                # 画像をコピーして返す（withブロック外でも使用可能にする）
                return img.copy()
                
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None
    
    def validate_image(self, image: "Image.Image") -> bool:
        """
        画像の妥当性を検証
        
        Args:
            image: PIL画像オブジェクト
            
        Returns:
            bool: 画像が有効かどうか
        """
        try:
            # 基本的な属性確認
            if image.size[0] <= 0 or image.size[1] <= 0:
                self.logger.error("Invalid image size")
                return False
            
            # 最小サイズチェック（あまりに小さい画像は除外）
            min_size = 32
            if image.size[0] < min_size or image.size[1] < min_size:
                self.logger.warning(f"Image too small: {image.size}")
                return False
            
            # モードチェック
            if self.convert_to_rgb and image.mode != 'RGB':
                self.logger.warning(f"Unexpected image mode: {image.mode}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Image validation failed: {str(e)}")
            return False
    
    def get_image_info(self, image: "Image.Image") -> dict:
        """
        画像の情報を取得
        
        Args:
            image: PIL画像オブジェクト
            
        Returns:
            dict: 画像情報
        """
        return {
            'size': image.size,
            'mode': image.mode,
            'format': getattr(image, 'format', None),
            'width': image.size[0],
            'height': image.size[1]
        }


class BSDS500ImageLoader(ImageLoader):
    """BSDS500データセット専用の画像読み込みクラス"""
    
    def __init__(self, dataset_path: Optional[Path] = None):
        """
        Args:
            dataset_path: BSDS500データセットのパス（Noneの場合は環境変数から取得）
        """
        super().__init__(convert_to_rgb=True)
        
        # データセットパスの取得
        if dataset_path is None:
            bsds_path = os.environ.get('BSDS500_PATH')
            if not bsds_path:
                raise ValueError("BSDS500_PATH environment variable not set")
            self.dataset_path = Path(bsds_path) / "BSDS500" / "data" / "images"
        else:
            self.dataset_path = dataset_path
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        self.logger.info(f"BSDS500 dataset path: {self.dataset_path}")
    
    def get_image_paths(self, split: str = 'train') -> List[Path]:
        """
        指定されたsplitの画像パス一覧を取得
        
        Args:
            split: データセットの分割（'train', 'val', 'test'）
            
        Returns:
            List[Path]: 画像パスのリスト
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        split_dir = self.dataset_path / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # JPEGファイルを取得
        image_paths = list(split_dir.glob("*.jpg"))
        image_paths.sort()  # ファイル名でソート
        
        self.logger.info(f"Found {len(image_paths)} images in {split} split")
        return image_paths
    
    def load_split_images(self, split: str = 'train') -> List[Tuple[Path, "Image.Image"]]:
        """
        指定されたsplitの全画像を読み込み
        
        Args:
            split: データセットの分割
            
        Returns:
            List[Tuple[Path, "Image.Image"]]: (パス, 画像)のタプルリスト
        """
        image_paths = self.get_image_paths(split)
        loaded_images = []
        
        failed_count = 0
        for path in image_paths:
            image = self.load_image(path)
            if image is not None and self.validate_image(image):
                loaded_images.append((path, image))
            else:
                failed_count += 1
                self.logger.warning(f"Failed to load or validate: {path.name}")
        
        self.logger.info(f"Successfully loaded {len(loaded_images)} images, "
                        f"failed: {failed_count}")
        
        return loaded_images
    
    def get_dataset_statistics(self) -> dict:
        """
        データセット全体の統計情報を取得
        
        Returns:
            dict: 統計情報
        """
        stats = {
            'splits': {},
            'total_images': 0,
            'image_sizes': [],
            'failed_loads': 0
        }
        
        for split in ['train', 'val', 'test']:
            try:
                image_paths = self.get_image_paths(split)
                split_stats = {
                    'count': len(image_paths),
                    'sizes': [],
                    'failed': 0
                }
                
                # サンプル画像から統計を取得
                sample_size = min(10, len(image_paths))
                for path in image_paths[:sample_size]:
                    image = self.load_image(path)
                    if image is not None:
                        split_stats['sizes'].append(image.size)
                        stats['image_sizes'].append(image.size)
                    else:
                        split_stats['failed'] += 1
                        stats['failed_loads'] += 1
                
                stats['splits'][split] = split_stats
                stats['total_images'] += split_stats['count']
                
            except Exception as e:
                self.logger.error(f"Failed to get statistics for {split}: {str(e)}")
                stats['splits'][split] = {'error': str(e)}
        
        return stats