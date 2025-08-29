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
├── workflows/batch_processing.py       # 高レベルAPI
└── utils/visualization/                # I/O・可視化
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
from src.generative_latent_optimization.workflows import optimize_bsds500_test

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
from src.generative_latent_optimization import SimpleAllMetricsEvaluator

# ワンコマンド全メトリクス評価
evaluator = SimpleAllMetricsEvaluator(device='cuda')
results = evaluator.evaluate_dataset_all_metrics('./created', './original')
evaluator.print_summary(results)
# 📊 All Metrics Evaluation Summary
# 🎯 Dataset-level FID Score: 12.34
# 🏆 Overall Quality: Excellent ✨
```

# 環境・データアクセス

## 開発環境
```bash
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure
```

## BSDS500データセット
```bash
$BSDS500_PATH/train/  # 200枚
$BSDS500_PATH/val/    # 100枚
$BSDS500_PATH/test/   # 200枚
```

## データセット利用
```python
from src.generative_latent_optimization.dataset import load_optimized_dataset

# PyTorchデータセット読み込み
dataset = load_optimized_dataset('./dataset.pt')
dataloader = dataset.create_dataloader(batch_size=4, shuffle=True)
```

# 🔄 次期開発構想

## Phase 4以降
- **実用化**: パフォーマンス最適化、Webダッシュボード
- **研究応用**: カスタムデータセット対応、論文準備
- **オープンソース**: 包括的ドキュメント、コミュニティ構築

## テスト実行
```bash
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python test_simple_evaluator.py
```

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
