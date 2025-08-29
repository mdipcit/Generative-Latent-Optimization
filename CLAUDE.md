# 背景
Stable Diffusionに基づく画像補完タスクにおいて、VAE (Variational Autoencoder) による入力画像の潜在表現へのエンコードは不可欠な前処理です。しかし、既存のVAEエンコーダは入力画像の詳細な特徴を十分に保持できず、生成される潜在表現の質の低さが、最終的な補完画像の品質に悪影響を及ぼしている可能性が指摘されています

# 目的
本研究の目的は、VAEエンコーダの性能問題に起因する画像補完の精度限界を克服することです。そのために、入力画像の情報を最大限に保持した理想的な潜在表現を生成する手法を確立し、そのデータセットを構築します。

# 方法
本研究では、VAEエンコーダによって一度生成された潜在表現に対し、事後的に最適化を行います。具体的には、潜在表現からデコーダを用いて画像を再構成し、元の入力画像との再構成誤差を最小化するように、潜在表現を繰り返し更新します。この最適化プロセスを通じて、補完タスクに最適な潜在表現を獲得します。

# プロジェクト現在状況

## 🎯 フェーズ1完了: PyPIパッケージ配布
研究で開発したimage_utils.pyとvae_loader.pyを汎用的なPyPIパッケージとして世界に公開しました。

## 🚀 フェーズ2完了: モジュラー最適化システム
元の単一スクリプトを完全にモジュール化し、再利用可能な最適化システムを構築しました。

### ✅ 完了済み作業
- **PyPIパッケージ配布**: `vae-toolkit` v0.1.0 を本番PyPIで公開完了
- **画像前処理パイプライン**: 元画像から直接VAE最適化済み前処理を実行
- **データアクセス**: 元BSDS500データから`vae-toolkit`経由で直接読み込み
- **Phase 2A完了**: コア最適化モジュール（最適化エンジン、メトリクス、I/O）
- **Phase 2B完了**: バッチ処理システム（ディレクトリ単位処理、チェックポイント）
- **Phase 2C完了**: デュアルデータセット作成（PyTorch + PNG同時生成）

## 📁 現在のプロジェクト構造（Phase 2完了後）
```
Generative-Latent-Optimization/
├── src/
│   ├── generative_latent_optimization/     # メインパッケージ
│   │   ├── optimization/                   # 最適化エンジン
│   │   │   ├── latent_optimizer.py         # ✅ VAE潜在最適化
│   │   │   └── __init__.py
│   │   ├── metrics/                        # 評価指標
│   │   │   ├── image_metrics.py            # ✅ PSNR/SSIM計算
│   │   │   └── __init__.py
│   │   ├── dataset/                        # データセット処理
│   │   │   ├── batch_processor.py          # ✅ バッチ処理
│   │   │   ├── pytorch_dataset.py          # ✅ PyTorchデータセット
│   │   │   ├── png_dataset.py              # ✅ PNGデータセット
│   │   │   └── __init__.py
│   │   ├── workflows/                      # ワークフロー
│   │   │   ├── batch_processing.py         # ✅ 高レベルAPI
│   │   │   └── __init__.py
│   │   ├── utils/                          # ユーティリティ
│   │   │   ├── io_utils.py                 # ✅ ファイルI/O
│   │   │   └── __init__.py
│   │   ├── visualization/                  # 可視化（実装済み）
│   │   │   ├── image_viz.py                # ✅ 画像比較表示
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── data/
│   │   └── dataset.py                      # BSDS500直接アクセス
│   └── config/                             # 設定管理
├── experiments/                            # 元実験スクリプト
│   └── single_image_optimization.py       # 元の単一スクリプト
├── test_dual_datasets.py                  # ✅ 動作確認スクリプト
├── flake.nix                               # Nix開発環境
├── CLAUDE.md                               # プロジェクト概要
└── IMPLEMENTATION_PLAN.md                  # 実装計画
```

## ✅ 実装完了システム: VAE潜在表現最適化

### システムアーキテクチャ（実装済み）
```
元BSDS500画像 → vae-toolkit前処理 → VAEエンコーダ → 初期潜在表現 → 反復最適化 → 最適化済み潜在表現 → デュアルデータセット
                                                      ↓
                                              PyTorchデータセット (.pt)
                                                      +
                                              PNGデータセット (ディレクトリ)
```

### 実装完了コンポーネント

#### 1. VAEモジュール (✅ PyPI配布済み)
- **エンコーダ**: 入力画像を潜在空間へマッピング (512×512 → 64×64×4)
- **デコーダ**: 潜在表現から画像を再構成
- **モデル**: ✅ Stable Diffusion v1.4/v1.5対応済み (`vae-toolkit==0.1.0`)
- **画像前処理**: ✅ VAE最適化済み前処理パイプライン

#### 2. 最適化エンジン (✅ 実装完了)
- **最適化アルゴリズム**: ✅ Adam (学習率設定可能、デフォルト: 0.4)
- **損失関数**: ✅ MSE/L1再構成損失選択可能
- **収束判定**: ✅ 損失変化率による自動停止機能
- **チェックポイント**: ✅ 途中経過保存・再開機能
- **進捗追跡**: ✅ tqdmによるリアルタイム表示

#### 3. データ管理層 (✅ 実装完了)
- **データ読み込み**: ✅ BSDS500全splits対応
- **バッチ処理**: ✅ ディレクトリレベル処理
- **最適化結果保存**: ✅ 複数形式対応 (.pt, PNG, JSON)
- **メタデータ管理**: ✅ 最適化統計・設定情報保存
- **PyTorchデータセット**: ✅ DataLoader対応データセット
- **PNGデータセット**: ✅ 可視化・README付きディレクトリ

## 🚀 利用可能な機能（Phase 2完了）

### 基本機能
- **単一画像最適化**: 個別画像のVAE潜在表現最適化
- **バッチ処理**: ディレクトリ内全画像の一括処理
- **BSDS500処理**: 全splits（train/val/test）の完全処理
- **チェックポイント**: 処理中断・再開機能

### データセット作成機能
- **PyTorchデータセット**: `.pt`形式でlatents、メトリクス、メタデータを保存
- **PNGデータセット**: 組織化ディレクトリに画像、比較表、統計を保存
- **デュアル作成**: 両形式同時生成
- **柔軟選択**: 必要な形式のみ選択生成可能

### 簡単使用例
```python
from src.generative_latent_optimization.workflows import optimize_bsds500_test

# BSDS500の小規模テスト（両形式データセット作成）
datasets = optimize_bsds500_test(
    output_path='./my_dataset',
    max_images=10,
    create_pytorch=True,
    create_png=True
)
print(f"PyTorch: {datasets['pytorch']}")
print(f"PNG: {datasets['png']}")
```

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

#### 基本的なVAE処理
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

#### VAE潜在表現最適化（Phase 2完了済み）
```python
from src.generative_latent_optimization.workflows import (
    optimize_bsds500_test,
    optimize_bsds500_full
)

# 小規模テスト（推奨：最初の動作確認用）
datasets = optimize_bsds500_test(
    output_path='./test_dataset',
    max_images=5,
    create_pytorch=True,
    create_png=True
)

# 本格的なBSDS500全体処理
full_datasets = optimize_bsds500_full(
    output_path='./full_bsds500_optimized',
    iterations=150,
    learning_rate=0.4,
    create_pytorch=True,
    create_png=True
)

print(f"PyTorchデータセット: {datasets['pytorch']}")
print(f"PNGデータセット: {datasets['png']}")
```

#### 作成されたデータセットの利用
```python
from src.generative_latent_optimization.dataset import load_optimized_dataset

# PyTorchデータセットの読み込み
dataset = load_optimized_dataset('./test_dataset.pt')
print(f"総サンプル数: {len(dataset)}")

# DataLoaderでの利用
dataloader = dataset.create_dataloader(batch_size=4, shuffle=True)
for batch in dataloader:
    original_latents = batch['initial_latents']
    optimized_latents = batch['optimized_latents']
    metrics = batch['metrics']
    # 訓練・評価処理
```

## 🔄 次期開発計画（Phase 2D/3）

### Phase 2D: 統合と可視化強化（予定）
- **統合テストスイート**: 全機能の包括的テスト
- **可視化ダッシュボード**: Web UI による結果閲覧
- **パフォーマンス最適化**: GPU並列処理、メモリ効率化
- **ドキュメント完成**: API仕様書、チュートリアル

### Phase 3: 研究応用展開（構想）
- **カスタムデータセット対応**: BSDS500以外のデータセット
- **最適化アルゴリズム拡張**: 異なる最適化手法の実装
- **損失関数バリエーション**: 知覚損失、敵対的損失等
- **論文・発表準備**: 研究成果の学術発表

# プロジェクト文書

## 実装計画
詳細な実装計画は `IMPLEMENTATION_PLAN.md` に記載されています。このファイルには以下の内容が含まれます：
- **Phase 2完了状況**: ✅ 2A, 2B, 2C完了
- フェーズごとの開発スケジュール
- 技術スタックと依存関係
- マイルストーンとリスク管理

## 使用方法とAPI
基本的な使用方法やAPIの詳細については、今後 `README.md` で提供予定です。

## テストとサンプル
`test_dual_datasets.py` で全機能の動作確認が可能です：
```bash
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python test_dual_datasets.py
```
