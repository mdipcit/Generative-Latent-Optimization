"""
画像変換関数

リサイズ、クロップ、正規化などの画像変換処理を提供します。
"""

import math
from typing import Tuple, Union, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from PIL import Image
    import numpy as np

try:
    from PIL import Image, ImageOps
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    np = None

logger = logging.getLogger(__name__)


def resize_short_edge(image: "Image.Image", target_size: int, 
                     resample: Optional[int] = None) -> "Image.Image":
    """
    短辺基準でリサイズ
    
    短い方の辺を指定されたサイズに合わせ、アスペクト比を保持してリサイズします。
    
    Args:
        image: 入力画像
        target_size: 短辺の目標サイズ
        resample: リサンプリング方法
        
    Returns:
        Image.Image: リサイズされた画像
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for image transforms")
    
    if resample is None:
        resample = Image.LANCZOS
    
    width, height = image.size
    
    # 短辺を特定
    short_edge = min(width, height)
    
    # リサイズ倍率を計算
    scale_factor = target_size / short_edge
    
    # 新しいサイズを計算
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    logger.debug(f"Resizing from {width}x{height} to {new_width}x{new_height} "
                f"(scale: {scale_factor:.3f})")
    
    return image.resize((new_width, new_height), resample)


def center_crop(image: "Image.Image", crop_size: Union[int, Tuple[int, int]]) -> "Image.Image":
    """
    センタークロップ
    
    画像の中央部分を指定されたサイズで切り出します。
    
    Args:
        image: 入力画像
        crop_size: クロップサイズ（intの場合は正方形、tupleの場合は(width, height)）
        
    Returns:
        Image.Image: クロップされた画像
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for image transforms")
    
    # クロップサイズの正規化
    if isinstance(crop_size, int):
        crop_width = crop_height = crop_size
    else:
        crop_width, crop_height = crop_size
    
    width, height = image.size
    
    # クロップサイズが元画像より大きい場合のチェック
    if crop_width > width or crop_height > height:
        raise ValueError(f"Crop size ({crop_width}x{crop_height}) is larger than "
                        f"image size ({width}x{height})")
    
    # 中央座標を計算
    center_x = width // 2
    center_y = height // 2
    
    # クロップ領域を計算
    left = center_x - crop_width // 2
    top = center_y - crop_height // 2
    right = left + crop_width
    bottom = top + crop_height
    
    logger.debug(f"Center cropping {width}x{height} to {crop_width}x{crop_height} "
                f"at ({left}, {top}, {right}, {bottom})")
    
    return image.crop((left, top, right, bottom))


def resize_and_crop(image: "Image.Image", target_size: int, 
                   resample: Optional[int] = None) -> "Image.Image":
    """
    リサイズとセンタークロップを組み合わせた変換
    
    短辺基準でリサイズした後、センタークロップで正方形にします。
    
    Args:
        image: 入力画像
        target_size: 最終的な画像サイズ（正方形）
        resample: リサンプリング方法
        
    Returns:
        Image.Image: 変換された画像
    """
    # デフォルトのリサンプリング方法を設定
    if resample is None:
        resample = Image.LANCZOS
    
    # Step 1: 短辺基準でリサイズ
    resized_image = resize_short_edge(image, target_size, resample)
    
    # Step 2: センタークロップで正方形にする
    cropped_image = center_crop(resized_image, target_size)
    
    return cropped_image


def normalize_to_range(image: "Image.Image", 
                      target_range: Tuple[float, float] = (-1.0, 1.0),
                      dtype = None) -> "np.ndarray":
    """
    画像を指定された範囲に正規化
    
    PIL画像をnumpy配列に変換し、指定された範囲に正規化します。
    
    Args:
        image: 入力画像
        target_range: 正規化の目標範囲 (min, max)
        dtype: 出力配列のデータ型
        
    Returns:
        np.ndarray: 正規化された画像配列（形状: [H, W, C]）
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL and numpy are required for normalization")
    
    if dtype is None:
        dtype = np.float32
    
    # PIL画像をnumpy配列に変換
    img_array = np.array(image, dtype=np.float32)
    
    # [0, 255] → [0, 1]
    img_array = img_array / 255.0
    
    # [0, 1] → [target_min, target_max]
    target_min, target_max = target_range
    img_array = img_array * (target_max - target_min) + target_min
    
    # データ型を変換
    if dtype != np.float32:
        img_array = img_array.astype(dtype)
    
    logger.debug(f"Normalized image to range {target_range}, "
                f"shape: {img_array.shape}, dtype: {img_array.dtype}")
    
    return img_array


def denormalize_from_range(img_array: "np.ndarray", 
                          source_range: Tuple[float, float] = (-1.0, 1.0)) -> "Image.Image":
    """
    正規化された配列を画像に逆変換
    
    Args:
        img_array: 正規化された画像配列
        source_range: 元の正規化範囲
        
    Returns:
        Image.Image: PIL画像
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL and numpy are required for denormalization")
    
    # 配列をコピー
    array = img_array.copy()
    
    # [source_min, source_max] → [0, 1]
    source_min, source_max = source_range
    array = (array - source_min) / (source_max - source_min)
    
    # [0, 1] → [0, 255]
    array = array * 255.0
    
    # クリッピングと型変換
    array = np.clip(array, 0, 255).astype(np.uint8)
    
    # PIL画像に変換
    if len(array.shape) == 3:
        return Image.fromarray(array, 'RGB')
    elif len(array.shape) == 2:
        return Image.fromarray(array, 'L')
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")


def validate_image_properties(image: "Image.Image", 
                            min_size: Optional[int] = None,
                            max_size: Optional[int] = None,
                            required_mode: Optional[str] = None) -> bool:
    """
    画像プロパティの検証
    
    Args:
        image: 検証する画像
        min_size: 最小サイズ（短辺基準）
        max_size: 最大サイズ（長辺基準）
        required_mode: 必要な画像モード
        
    Returns:
        bool: 検証に通ったかどうか
    """
    try:
        width, height = image.size
        
        # サイズ検証
        if min_size is not None:
            if min(width, height) < min_size:
                logger.warning(f"Image too small: {width}x{height} < {min_size}")
                return False
        
        if max_size is not None:
            if max(width, height) > max_size:
                logger.warning(f"Image too large: {width}x{height} > {max_size}")
                return False
        
        # モード検証
        if required_mode is not None:
            if image.mode != required_mode:
                logger.warning(f"Wrong image mode: {image.mode} != {required_mode}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        return False


def compute_resize_params(original_size: Tuple[int, int], 
                         target_size: int) -> dict:
    """
    リサイズパラメータの計算
    
    Args:
        original_size: 元画像サイズ (width, height)
        target_size: 目標サイズ（短辺基準）
        
    Returns:
        dict: リサイズパラメータ
    """
    width, height = original_size
    short_edge = min(width, height)
    scale_factor = target_size / short_edge
    
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # センタークロップ座標
    center_x = new_width // 2
    center_y = new_height // 2
    crop_left = center_x - target_size // 2
    crop_top = center_y - target_size // 2
    crop_right = crop_left + target_size
    crop_bottom = crop_top + target_size
    
    return {
        'original_size': original_size,
        'target_size': target_size,
        'scale_factor': scale_factor,
        'resized_size': (new_width, new_height),
        'crop_coordinates': (crop_left, crop_top, crop_right, crop_bottom),
        'final_size': (target_size, target_size)
    }


class ImageTransformPipeline:
    """画像変換パイプライン"""
    
    def __init__(self, target_size: int = 512, 
                 normalize_range: Tuple[float, float] = (-1.0, 1.0),
                 resample: Optional[int] = None):
        """
        Args:
            target_size: 最終画像サイズ
            normalize_range: 正規化範囲
            resample: リサンプリング方法
        """
        self.target_size = target_size
        self.normalize_range = normalize_range
        self.resample = resample if resample is not None else (Image.LANCZOS if PIL_AVAILABLE else None)
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, image: "Image.Image") -> Tuple["np.ndarray", dict]:
        """
        画像変換パイプラインの実行
        
        Args:
            image: 入力画像
            
        Returns:
            Tuple[np.ndarray, dict]: (変換済み画像配列, メタデータ)
        """
        # 元画像情報を記録
        original_size = image.size
        
        # 変換パラメータを計算
        params = compute_resize_params(original_size, self.target_size)
        
        # Step 1: リサイズ + クロップ
        transformed_image = resize_and_crop(image, self.target_size, self.resample)
        
        # Step 2: 正規化
        normalized_array = normalize_to_range(transformed_image, self.normalize_range)
        
        # メタデータ作成
        metadata = {
            **params,
            'normalize_range': self.normalize_range,
            'resample_method': self.resample
        }
        
        self.logger.debug(f"Transformed image from {original_size} to "
                         f"{normalized_array.shape}")
        
        return normalized_array, metadata