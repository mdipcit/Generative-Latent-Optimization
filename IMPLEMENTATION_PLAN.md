# 実装計画

## 🎯 プロジェクト現在状況

### ✅ フェーズ1完了: PyPIパッケージ配布 (100%完了)
`vae-toolkit` v0.1.0 パッケージの公開と、元BSDS500から直接アクセスできる効率的なシステムが完了。

### 💾 利用可能リソース
```
$BSDS500_PATH/
├── train/               # 200枚 (512×512 png)
├── val/                 # 100枚 (512×512 png)
└── test/                # 200枚 (512×512 png)

vae-toolkitパッケージで直接512×512、[-1,1]正規化へ変換
```

### 📁 現在の構造
```
src/
├── data/
│   └── dataset.py       # BSDS500直接アクセス用クラス
└── config/              # 設定管理
```

### 🔧 データアクセス
```python
import os
from vae_toolkit import load_and_preprocess_image

# 元BSDS500から直接利用
bsds500_path = os.environ["BSDS500_PATH"]
image_path = f"{bsds500_path}/train/12003.png"
image_tensor, pil_img = load_and_preprocess_image(image_path, target_size=512)
# 結果: torch.Size([1, 3, 512, 512]), [-1,1]正規化済み
```

## 🚀 フェーズ2: VAE + 潜在表現最適化

### 📋 開発計画

#### 2.1 VAEモジュール統合
- Stable Diffusion VAE (HuggingFace Diffusers)
- エンコーダ/デコーダ分離実装
- 潜在空間: 512×512 → 64×64×4

#### 2.2 最適化エンジン
- Adam最適化器 (学習率: 1e-1)
- L1/L2再構成損失
- 収束判定・履歴追跡

#### 2.3 統合パイプライン
- 元BSDS500→vae-toolkit前処理→VAE→最適化→保存
- バッチ処理・進捗監視
- HDF5/NPZ形式で結果保存

### 🔧 開発環境
```bash
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure
```

### 📁 予定構造
```
src/
├── data/dataset.py
├── vae/                 # VAEモジュール
├── optimization/       # 最適化エンジン
└── config/             # 設定管理
```

### ⏱️ スケジュール予定
- VAE統合: 3-4日
- 最適化エンジン: 4-5日
- 統合・テスト: 2-3日
