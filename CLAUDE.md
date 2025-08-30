# 背景
Stable Diffusionの画像補完において、VAEエンコーダによる潜在表現の質の低さが最終的な補完画像の品質を制限しています。

# 目的
VAEエンコーダの性能限界を克服し、入力画像の情報を最大限に保持した理想的な潜在表現を生成する手法を確立します。

# 方法
VAEエンコーダによる初期潜在表現に対し、デコーダからの再構成誤差を最小化する事後最適化を実行し、補完タスクに最適な潜在表現を獲得します。

# プロジェクト状況

## ✅ 全フェーズ完了済み
- **フェーズ1**: PyPIパッケージ配布（`vae-toolkit` v0.1.0）
- **フェーズ2**: モジュラー最適化システム（バッチ処理、デュアルデータセット）
- **フェーズ3**: 高度品質評価システム（LPIPS/改良SSIM/FID/統合評価API）

## 📁 プロジェクト構造
```
src/generative_latent_optimization/
├── optimization/latent_optimizer.py    # VAE最適化エンジン
├── metrics/                            # 評価システム
│   ├── image_metrics.py               # PSNR/SSIM/MSE/MAE
│   ├── individual_metrics.py          # LPIPS/改良SSIM
│   ├── dataset_metrics.py             # FID
│   └── metrics_integration.py         # 統合計算
├── evaluation/                         # 評価API
│   ├── dataset_evaluator.py           # 包括的評価
│   └── simple_evaluator.py            # 簡潔API
├── dataset/                            # データセット処理
│   ├── bsds500_dataset.py             # BSDS500データセット
│   ├── batch_processor.py             # バッチ処理
│   ├── png_dataset.py                 # PNG形式データセット
│   └── pytorch_dataset.py             # PyTorch形式データセット
├── config/                             # 設定管理
│   └── model_config.py                # モデル設定
├── workflows/batch_processing.py       # 高レベルAPI
├── utils/                              # ユーティリティ
│   └── io_utils.py                    # I/O処理
└── visualization/                      # 可視化
    └── image_viz.py                   # 画像可視化

experiments/                            # 実験データ
├── data/                              # 実験用データセット
└── results/                           # 実験結果

tests/                                  # テストスイート
├── test_vae_basic.py                  # 基本VAEテスト
├── test_vae_fixed.py                  # 修正VAEテスト
├── test_document_encode_decode.py     # エンコード・デコードテスト
└── compare_implementations.py         # 実装比較
```

## ✅ システム概要
```
BSDS500画像 → VAE前処理 → 最適化 → デュアルデータセット → 品質評価
```

## 完成機能
- **VAE最適化**: Adam最適化器、収束判定、チェックポイント
- **データセット**: PyTorch/PNG形式、バッチ処理
- **品質評価**: PSNR/SSIM/LPIPS/FID、統計分析、美しいレポート

## 🚀 利用例

### データセット作成
```python
from generative_latent_optimization.workflows import optimize_bsds500_test

# デュアルデータセット作成
datasets = optimize_bsds500_test(
    output_path='./my_dataset',
    max_images=10,
    create_pytorch=True,
    create_png=True
)
```

### 品質評価
```python
from generative_latent_optimization import SimpleAllMetricsEvaluator

# ワンコマンド全メトリクス評価
evaluator = SimpleAllMetricsEvaluator(device='cuda')
results = evaluator.evaluate_dataset_all_metrics('./created', './original')
evaluator.print_summary(results)
# 📊 All Metrics Evaluation Summary
# 🎯 Dataset-level FID Score: 12.34
# 🏆 Overall Quality: Excellent ✨
```

### 単体最適化
```python
from generative_latent_optimization import LatentOptimizer, OptimizationConfig
from vae_toolkit import VAELoader, load_and_preprocess_image

# VAEモデル読み込み
vae_loader = VAELoader()
vae = vae_loader.load_vae('sd15', device='cuda')

# 画像読み込み
image_tensor, _ = load_and_preprocess_image('document.png', target_size=512)

# 最適化設定
config = OptimizationConfig(
    iterations=100,
    learning_rate=0.1,
    device='cuda'
)

# 最適化実行
optimizer = LatentOptimizer(vae, config)
result = optimizer.optimize(image_tensor)

print(f"PSNR improvement: {result.metrics['final_psnr'] - result.metrics['initial_psnr']:.2f} dB")
```

# 環境・データアクセス

## 開発環境セットアップ

### 1. Nix環境の起動
```bash
# 開発環境に入る（CUDA対応の不自由パッケージを含む）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure
```

### 2. 依存関係のインストール
```bash
# Python依存関係の同期（初回セットアップ時）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv sync

# 新しい依存関係を追加する場合
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv add [パッケージ名]
```

### 3. コマンド実行パターン

#### テストスイート実行
```bash
# 基本VAE動作テスト
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python tests/unit/test_vae_basic.py

# 修正版VAEテスト（推奨）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python tests/unit/test_vae_fixed.py

# 最適化統合テスト
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python tests/integration/test_optimization_integration.py

# ドキュメント画像のエンコード・デコード例
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python scripts/examples/document_encode_decode_example.py

# 実装比較分析
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python scripts/analysis/implementation_comparison.py
```

#### 実験スクリプト実行
```bash
# 高速最適化テスト（20回最適化）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/optimization/quick_optimization_test.py

# 単一画像最適化実験
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/optimization/single_image_optimization.py

# メトリクス評価デモ
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/evaluation/metrics_evaluation_demo.py

# 包括的評価デモ
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/evaluation/comprehensive_evaluation_demo.py

# バッチ処理デモ（時間がかかります）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/datasets/batch_processing_demo.py

# デュアルデータセット作成デモ
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/datasets/dual_datasets_demo.py

# 損失可視化
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/visualization/loss_visualization.py
```

#### インタラクティブ実行
```bash
# Pythonシェル起動
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python

# IPython起動（利用可能な場合）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run ipython
```

### 4. 環境変数の設定

#### 必須環境変数
```bash
# Hugging Face認証トークン（必須）
export HF_TOKEN="your_huggingface_token_here"

# BSDS500データセット（バッチ処理で必要）
export BSDS500_PATH="/path/to/bsds500/dataset"
```

#### 実行例（環境変数込み）
```bash
# 環境変数を設定してテスト実行
HF_TOKEN="your_token" NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python tests/test_vae_fixed.py
```

### 5. 共通エラーと対処法

#### UNFREE パッケージエラー
```bash
# エラー：unfree packageが利用できない
# 対処：NIXPKGS_ALLOW_UNFREE=1 を必ず付ける
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c [コマンド]
```

#### CUDA関連エラー  
```bash
# CUDAが利用できない場合、CPUで実行される
# ログで確認：CUDA available: False
# 正常：CUDA available: True
```

#### 依存関係エラー
```bash
# ModuleNotFoundError が発生した場合
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv sync

# 特定パッケージ追加が必要な場合
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv add [パッケージ名]
```

## BSDS500データセット
```bash
$BSDS500_PATH/train/  # 200枚
$BSDS500_PATH/val/    # 100枚
$BSDS500_PATH/test/   # 200枚
```

## データセット利用
```python
from generative_latent_optimization.dataset import load_optimized_dataset

# PyTorchデータセット読み込み
dataset = load_optimized_dataset('./dataset.pt')
dataloader = dataset.create_dataloader(batch_size=4, shuffle=True)
```

# 🔄 次期開発構想

## Phase 4以降
- **実用化**: パフォーマンス最適化、Webダッシュボード
- **研究応用**: カスタムデータセット対応、論文準備
- **オープンソース**: 包括的ドキュメント、コミュニティ構築

## 💻 クイックスタート

### 基本テスト（推奨）
```bash
# 環境セットアップ
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv sync

# VAE基本機能テスト
HF_TOKEN="your_token" NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python tests/unit/test_vae_fixed.py

# 統合評価システムテスト
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/evaluation/metrics_evaluation_demo.py

# 高速最適化テスト（約1分）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/optimization/quick_optimization_test.py
```

### 結果確認
```bash
# 実験結果を確認
ls -la experiments/results/

# 最適化テスト結果を確認
ls -la experiments/results/quick_test/

# 可視化結果を確認
ls -la experiments/results/visualization/
```

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
