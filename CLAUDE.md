# 背景
Stable Diffusionに基づく画像補完タスクにおいて、VAE (Variational Autoencoder) による入力画像の潜在表現へのエンコードは不可欠な前処理です。しかし、既存のVAEエンコーダは入力画像の詳細な特徴を十分に保持できず、生成される潜在表現の質の低さが、最終的な補完画像の品質に悪影響を及ぼしている可能性が指摘されています

# 目的
本研究の目的は、VAEエンコーダの性能問題に起因する画像補完の精度限界を克服することです。そのために、入力画像の情報を最大限に保持した理想的な潜在表現を生成する手法を確立し、そのデータセットを構築します。

# 方法
本研究では、VAEエンコーダによって一度生成された潜在表現に対し、事後的に最適化を行います。具体的には、潜在表現からデコーダを用いて画像を再構成し、元の入力画像との再構成誤差を最小化するように、潜在表現を繰り返し更新します。この最適化プロセスを通じて、補完タスクに最適な潜在表現を獲得します。

# プロジェクト現在状況

## 🎯 フェーズ1完了: PyPIパッケージ配布
研究で開発したimage_utils.pyとvae_loader.pyを汎用的なPyPIパッケージとして世界に公開しました。

### ✅ 完了済み作業
- **PyPIパッケージ配布**: `vae-toolkit` v0.1.0 を本番PyPIで公開完了
- **画像前処理パイプライン**: 元画像から直接VAE最適化済み前処理を実行
- **データアクセス**: 元BSDS500データから`vae-toolkit`経由で直接読み込み

## 📁 現在のプロジェクト構造
```
Generative-Latent-Optimization/
├── src/
│   ├── data/
│   │   └── dataset.py    # BSDS500直接アクセス用クラス
│   ├── config/           # 設定管理
│   └── generative_latent_optimization/
│       └── models/       # モデル関連コード
├── flake.nix             # Nix開発環境 ($BSDS500_PATH提供)
├── CLAUDE.md             # プロジェクト概要
└── IMPLEMENTATION_PLAN.md # 実装計画
```

## 🚀 次期開発: VAEと潜在表現最適化

### システムアーキテクチャ（計画）
```
元BSDS500画像 → vae-toolkit前処理 → VAEエンコーダ → 初期潜在表現 → 反復最適化 → 最適化済み潜在表現 → データセット
```

### 実装予定コンポーネント

#### 1. VAEモジュール (✅ PyPI配布済み)
- **エンコーダ**: 入力画像を潜在空間へマッピング (512×512 → 64×64×4)
- **デコーダ**: 潜在表現から画像を再構成
- **モデル**: ✅ Stable Diffusion v1.4/v1.5対応済み (`vae-toolkit==0.1.0` パッケージとして依存関係に追加済み)
- **画像前処理**: ✅ VAE最適化済み前処理パイプライン (`vae-toolkit` パッケージ)

#### 2. 最適化エンジン (未実装)
- **最適化アルゴリズム**: Adam (学習率: 1e-1)
- **損失関数**: L1/L2再構成損失 (ピクセルレベル)
- **収束判定**: 損失値の変化率による自動停止

#### 3. データ管理層 (実装済み)
- **データ読み込み**: ✅ 実装済み (元BSDS500から`vae-toolkit`経由)
- **画像処理ユーティリティ**: ✅ PyPI配布済み (`vae-toolkit==0.1.0` パッケージから利用)
- **最適化結果保存**: 未実装 (HDF5/NPZ形式予定)
- **メタデータ管理**: 未実装 (最適化ステップ数、損失値等)

# データアクセス

## BSDS500データセットの利用方法

### 開発環境の起動
```bash
# unfreeライセンス許可が必要 (CUDA関連)
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure
```

### 元BSDS500データへの直接アクセス
開発環境内では `BSDS500_PATH` 環境変数で元の512×512データセットにアクセス可能です：

```bash
# データ構造の確認
ls -la $BSDS500_PATH/
# train/  val/  test/

# 画像数確認
find $BSDS500_PATH/train -name "*.png" | wc -l  # 200
```

### Pythonでの利用例
```python
import os
from vae_toolkit import VAELoader, load_and_preprocess_image

# 元画像から直接前処理とVAE処理
bsds500_path = os.environ["BSDS500_PATH"]
image_path = f"{bsds500_path}/train/12003.png"

# vae-toolkitで直接前処理（512×512、[-1,1]正規化）
image_tensor, pil_img = load_and_preprocess_image(image_path, target_size=256)
print(f"Image shape: {image_tensor.shape}")  # torch.Size([1, 3, 512, 512])
print(f"Value range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")  # [-1.00, 1.00]

# VAE関連機能の利用
vae, device = VAELoader.load_sd_vae_simple("sd14", "auto")

# 直接エンコード・デコードが可能
with torch.no_grad():
    latents = vae.encode(image_tensor.to(device)).latent_dist.mode()
    decoded = vae.decode(latents).sample
```

# プロジェクト文書

## 実装計画
詳細な実装計画は `IMPLEMENTATION_PLAN.md` に記載されています。このファイルには以下の内容が含まれます：
- フェーズごとの開発スケジュール
- 技術スタックと依存関係
- マイルストーンとリスク管理
