# 画像前処理パイプライン実装計画

## 概要

Stable Diffusion VAEの要求に合わせて、BSDS500データセットを512×512ピクセルに前処理し、正規化を行うパイプラインを実装します。

## 技術仕様

### 入力要件
- **入力画像**: BSDS500データセット (JPEGフォーマット、様々なサイズ)
- **出力サイズ**: 512×512ピクセル (Stable Diffusion VAE標準)
- **出力形式**: RGB画像、正規化済み [-1, 1] 範囲
- **データ型**: float32 (PyTorch tensor互換)

### 前処理ステップ

#### 1. 画像読み込み
```python
# PIL/OpenCVを使用してJPEG画像を読み込み
# RGB形式で統一 (BGRからの変換が必要な場合)
```

#### 2. リサイズ処理
```python
# 短辺基準リサイズ: 短い方の辺を512ピクセルに合わせる
# アスペクト比を保持しながらリサイズ
# 例: 481×321 → 768×512 (短辺321を512に拡大)
```

#### 3. センタークロップ
```python
# リサイズ後の画像から中央部分512×512を切り出し
# 例: 768×512 → 512×512 (左右から128ピクセルずつクロップ)
```

#### 4. 正規化
```python
# [0, 255] → [0, 1]: 255で除算
# [0, 1] → [-1, 1]: 2倍して1を減算
# normalized = (pixel / 255.0) * 2.0 - 1.0
```

## 実装アーキテクチャ

### ディレクトリ構成
```
src/
├── data/
│   ├── __init__.py
│   ├── loader.py          # データセット読み込み
│   ├── preprocessor.py    # 前処理クラス
│   ├── transforms.py      # 変換関数群
│   └── utils.py          # ユーティリティ関数
├── config/
│   └── preprocess_config.py  # 設定管理
└── scripts/
    ├── preprocess_bsds.py    # メイン前処理スクリプト
    └── validate_preprocessing.py  # 検証スクリプト
```

### クラス設計

#### 1. `ImagePreprocessor`クラス
```python
class ImagePreprocessor:
    def __init__(self, target_size=512, normalize_range=(-1, 1)):
        self.target_size = target_size
        self.normalize_range = normalize_range
    
    def process_image(self, image_path):
        """単一画像の前処理"""
        pass
    
    def process_batch(self, image_paths, batch_size=32):
        """バッチ処理"""
        pass
```

#### 2. `BSDS500Dataset`クラス
```python
class BSDS500Dataset:
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.split = split
        self.image_paths = self._load_image_paths()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """前処理済み画像とメタデータを返す"""
        pass
```

### 変換関数

#### `transforms.py`の実装
```python
def resize_short_edge(image, target_size):
    """短辺基準リサイズ"""
    pass

def center_crop(image, crop_size):
    """センタークロップ"""
    pass

def normalize_to_range(image, target_range=(-1, 1)):
    """正規化"""
    pass

def validate_image_properties(image):
    """画像プロパティの検証"""
    pass
```

## データ管理

### 出力ディレクトリ構成
```
processed_data/
├── train/
│   ├── images/           # 前処理済み画像 (.png)
│   └── metadata/         # メタデータ (.json)
├── val/
│   ├── images/
│   └── metadata/
├── test/
│   ├── images/
│   └── metadata/
└── stats/
    ├── dataset_stats.json   # データセット統計情報
    └── preprocessing_log.json  # 前処理ログ
```

### メタデータ形式
```json
{
  "original_path": "/path/to/original/image.jpg",
  "original_size": [481, 321],
  "processed_size": [512, 512],
  "resize_factor": 1.596,
  "crop_coordinates": [128, 0, 640, 512],
  "processing_timestamp": "2024-08-19T14:30:00Z",
  "checksum": "sha256_hash"
}
```

## 実装段階

### Phase 1: 基本機能実装
1. **画像読み込み機能**
   - PIL/OpenCVでの画像読み込み
   - エラーハンドリング (破損ファイル、サポート外形式)
   - RGB形式への統一

2. **変換関数の実装**
   - 短辺基準リサイズ
   - センタークロップ
   - 正規化処理

3. **単体テスト**
   - 各変換関数の動作確認
   - エッジケースのテスト

### Phase 2: パイプライン統合
1. **`ImagePreprocessor`クラス**
   - 変換関数の組み合わせ
   - バッチ処理機能
   - プログレスバー表示

2. **`BSDS500Dataset`クラス**
   - データセット読み込み
   - メタデータ管理
   - キャッシュ機能

### Phase 3: 検証と最適化
1. **品質検証**
   - 前処理前後の画像比較
   - 統計情報の収集
   - 異常値検出

2. **パフォーマンス最適化**
   - 並列処理の実装
   - メモリ使用量の最適化
   - I/O効率化

## コマンドラインインターフェース

### メイン前処理スクリプト
```bash
python scripts/preprocess_bsds.py \
    --input_path $BSDS500_PATH/BSDS500/data/images \
    --output_path ./processed_data \
    --target_size 512 \
    --normalize_range -1 1 \
    --batch_size 32 \
    --num_workers 4 \
    --save_metadata
```

### 検証スクリプト
```bash
python scripts/validate_preprocessing.py \
    --processed_path ./processed_data \
    --original_path $BSDS500_PATH/BSDS500/data/images \
    --sample_size 10 \
    --save_comparison
```

## パフォーマンス目標

- **処理速度**: 500枚を5分以内で処理
- **メモリ使用量**: 4GB以内
- **品質**: PSNR > 30dB (リサイズ・クロップによる劣化)
- **データ整合性**: 100% (全画像の正常処理)

## 実装スケジュール

1. **Day 3 前半**: Phase 1実装 (基本機能)
2. **Day 3 後半**: Phase 2実装 (パイプライン統合)
3. **Day 4 前半**: Phase 3実装 (検証・最適化)
4. **Day 4 後半**: 統合テストと文書化

## 次のステップへの準備

前処理完了後、以下の準備が整います：
- VAE エンコーダ/デコーダのロード
- 潜在表現の最適化実装
- バッチ処理パイプラインの構築