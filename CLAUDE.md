# 背景
Stable Diffusionに基づく画像補完タスクにおいて、VAE (Variational Autoencoder) による入力画像の潜在表現へのエンコードは不可欠な前処理です。しかし、既存のVAEエンコーダは入力画像の詳細な特徴を十分に保持できず、生成される潜在表現の質の低さが、最終的な補完画像の品質に悪影響を及ぼしている可能性が指摘されています

# 目的
本研究の目的は、VAEエンコーダの性能問題に起因する画像補完の精度限界を克服することです。そのために、入力画像の情報を最大限に保持した理想的な潜在表現を生成する手法を確立し、そのデータセットを構築します。

# 方法
本研究では、VAEエンコーダによって一度生成された潜在表現に対し、事後的に最適化を行います。具体的には、潜在表現からデコーダを用いて画像を再構成し、元の入力画像との再構成誤差を最小化するように、潜在表現を繰り返し更新します。この最適化プロセスを通じて、補完タスクに最適な潜在表現を獲得します。

# プロジェクト現在状況

## 🎯 フェーズ1完了: 前処理システム + PyPIパッケージ配布
前処理フェーズが完了し、BSDS500データセット全体が処理済みです。さらに、研究で開発したimage_utils.pyとvae_loader.pyを汎用的なPyPIパッケージとして世界に公開しました。

### ✅ 完了済み作業
- **データセット前処理**: BSDS500の200枚 (train) を256×256、[-1,1]正規化形式で処理完了
- **リファクタリング**: 前処理関連の不要なコードを削除し、構造を簡潔化
- **データアクセス**: `src/data/dataset.py`による効率的な処理済みデータ読み込み
- **🆕 PyPIパッケージ配布**: `vae-toolkit` v0.1.0 を本番PyPIで公開完了

## 📁 現在のプロジェクト構造
```
Generative-Latent-Optimization/
├── processed_data/        # 前処理済みデータセット
│   ├── train/            # 200枚処理済み (NPZ形式)
│   ├── val/              # 準備完了
│   └── test/             # 準備完了
├── src/
│   ├── data/
│   │   └── dataset.py    # データセット読み込みクラス
│   ├── config/           # 設定管理
│   └── generative_latent_optimization/
│       └── models/       # モデル関連コード
├── flake.nix             # Nix開発環境
├── CLAUDE.md             # プロジェクト概要
└── IMPLEMENTATION_PLAN.md # 実装計画
```

## 🚀 次期開発: VAEと潜在表現最適化

### システムアーキテクチャ（計画）
```
前処理済み画像 → VAEエンコーダ → 初期潜在表現 → 反復最適化 → 最適化済み潜在表現 → データセット
```

### 実装予定コンポーネント

#### 1. VAEモジュール (✅ PyPI配布済み)
- **エンコーダ**: 入力画像を潜在空間へマッピング (256×256 → 32×32×4)
- **デコーダ**: 潜在表現から画像を再構成
- **モデル**: ✅ Stable Diffusion v1.4/v1.5対応済み (`vae-toolkit==0.1.0` パッケージとして依存関係に追加済み)
- **画像前処理**: ✅ VAE最適化済み前処理パイプライン (`vae-toolkit` パッケージ)

#### 2. 最適化エンジン (未実装)
- **最適化アルゴリズム**: Adam (学習率: 1e-1)
- **損失関数**: L1/L2再構成損失 (ピクセルレベル)
- **収束判定**: 損失値の変化率による自動停止

#### 3. データ管理層 (一部実装済み)
- **データ読み込み**: ✅ 実装済み (`src/data/dataset.py`)
- **画像処理ユーティリティ**: ✅ PyPI配布済み (`vae-toolkit==0.1.0` パッケージから利用)
- **最適化結果保存**: 未実装 (HDF5/NPZ形式予定)
- **メタデータ管理**: 未実装 (最適化ステップ数、損失値等)

# データアクセス

## 前処理済みデータセットの利用方法

### 開発環境の起動
```bash
# unfreeライセンス許可が必要 (CUDA関連)
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure
```

### 前処理済みデータへのアクセス
前処理済みデータは`processed_data/`ディレクトリに保存されています：

```bash
# データ構造の確認
ls -la processed_data/
# train/  val/  test/  stats/

# 処理済み画像数確認
find processed_data/train/images -name "*.npz" | wc -l  # 200
```

### Pythonでの利用例
```python
import numpy as np
from src.data.dataset import BSDS500Dataset
from vae_toolkit import VAELoader, load_and_preprocess_image

# データセットの読み込み
dataset = BSDS500Dataset(
    processed_data_dir="processed_data",
    split="train",
    cache_in_memory=True
)

# 画像の取得
image, metadata = dataset[0]
print(f"Image shape: {image.shape}")  # (256, 256, 3)
print(f"Value range: [{image.min():.2f}, {image.max():.2f}]")  # [-1.00, 1.00]

# VAE関連機能の利用（vae-toolkitパッケージから）
vae, device = VAELoader.load_sd_vae_simple("sd14", "auto")
image_tensor, pil_img = load_and_preprocess_image("image.png", 256)
```

### 元のBSDS500データセット
開発環境内では `BSDS500_PATH` 環境変数で元の512×512データセットにもアクセス可能です。

# プロジェクト文書

## 実装計画
詳細な実装計画は `IMPLEMENTATION_PLAN.md` に記載されています。このファイルには以下の内容が含まれます：
- フェーズごとの開発スケジュール
- 技術スタックと依存関係
- マイルストーンとリスク管理
