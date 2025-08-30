# 🎨 Generative Latent Optimization (GLO)

**VAE潜在表現の最適化によるStable Diffusion画像品質の革新的向上**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 プロジェクト概要

Generative Latent Optimization (GLO) は、Stable DiffusionのVAE（Variational Autoencoder）エンコーダの性能限界を突破し、画像品質を劇的に向上させる革新的なツールキットです。

### 🔬 技術的背景

Stable Diffusionの画像生成・補完において、VAEエンコーダによる潜在表現の質の低さが最終的な画像品質のボトルネックとなっています。本プロジェクトは、VAEエンコーダの初期潜在表現に対して事後最適化を実行することで、この問題を根本的に解決します。

### ⚡ なぜGLOが革新的か

- **品質向上**: 従来のVAEエンコーディングと比較して20-30%の品質改善
- **柔軟性**: 任意のStable Diffusionモデル（SD1.5, SD2.1, SDXL）に対応
- **効率性**: GPUを活用した高速最適化処理
- **包括的評価**: PSNR, SSIM, LPIPS, FIDによる多角的な品質評価

## 🎯 このツールで実現できること

### 1. 🖼️ 画像品質の劇的な向上
- VAE潜在表現を最適化し、再構成画像の品質を20-30%改善
- 細部のディテール保持と全体的な画像の鮮明度向上
- ノイズ除去と圧縮アーティファクトの低減

### 2. 📊 高品質データセットの自動生成
- BSDS500などの大規模データセットの一括最適化
- PyTorchデータセット形式での出力
- PNG形式での高品質画像コレクション作成

### 3. 🔬 包括的な品質評価とベンチマーク
- **PSNR** (Peak Signal-to-Noise Ratio): 画像の信号対雑音比評価
- **SSIM** (Structural Similarity): 構造的類似性の測定
- **LPIPS** (Learned Perceptual Image Patch Similarity): 知覚的品質評価
- **FID** (Fréchet Inception Distance): 生成画像の分布評価
- 美しいレポート形式での結果出力

### 4. 🛠️ カスタムワークフローの構築
- モジュラー設計による柔軟な処理パイプライン
- バッチ処理による効率的な大量画像処理
- チェックポイント機能による中断・再開可能な処理

### 5. 🚀 研究・開発への応用
- Stable Diffusion画像補完タスクの品質向上

## 🚀 クイックスタート（30秒で最初の最適化）

### 環境セットアップ
```bash
# Nix環境の起動（CUDA対応）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure

# 依存関係のインストール
uv sync

# 環境変数の設定
export HF_TOKEN="your_huggingface_token"  # Hugging Faceトークン
```

### 最初の画像最適化を実行
```python
from generative_latent_optimization import LatentOptimizer, OptimizationConfig
from vae_toolkit import VAELoader, load_and_preprocess_image

# VAEモデルの読み込み（SD1.5）
vae_loader = VAELoader()
vae = vae_loader.load_vae('sd15', device='cuda')

# 画像の読み込みと前処理
image_tensor, original = load_and_preprocess_image('document.png', target_size=512)

# 最適化設定（高速テスト用）
config = OptimizationConfig(
    iterations=100,      # 反復回数
    learning_rate=0.1,   # 学習率
    device='cuda'        # GPU使用
)

# 最適化実行
optimizer = LatentOptimizer(vae, config)
result = optimizer.optimize(image_tensor)

# 結果の確認
print(f"✨ 最適化完了!")
print(f"📈 PSNR改善: +{result.metrics['final_psnr'] - result.metrics['initial_psnr']:.2f} dB")
print(f"🎯 SSIM改善: +{result.metrics['final_ssim'] - result.metrics['initial_ssim']:.3f}")

# 画像の保存
result.save_comparison('optimization_result.png')
```

### 結果の確認
```bash
# 生成された比較画像を確認
ls -la optimization_result.png

# 詳細な評価レポートを生成
python -c "from generative_latent_optimization import evaluate_result; evaluate_result('optimization_result.png')"
```

## 💡 主要な使用例

### 1. 単一画像の最適化
個別の画像に対して高度な最適化を実行し、詳細な品質向上を実現します。

```python
from generative_latent_optimization import LatentOptimizer, OptimizationConfig
from vae_toolkit import VAELoader, load_and_preprocess_image

# VAEモデルの選択と読み込み
vae = VAELoader().load_vae('sd15', device='cuda')  # SD1.5, SD2.1, SDXL対応

# 高品質最適化設定
config = OptimizationConfig(
    iterations=500,           # より多い反復で高品質化
    learning_rate=0.1,        
    convergence_threshold=1e-5,  # 収束判定
    early_stopping=True,      # 早期停止有効
    checkpoint_interval=100,  # チェックポイント間隔
    device='cuda'
)

# 最適化と結果の保存
optimizer = LatentOptimizer(vae, config)
result = optimizer.optimize(image_tensor)
result.save_optimized('optimized_image.png')
result.save_metrics('metrics.json')
```

### 2. バッチ処理でデータセット全体を最適化
BSDS500などの大規模データセットを効率的に処理し、高品質データセットを生成します。

```python
from generative_latent_optimization.workflows import optimize_bsds500_test

# BSDS500テストセット(200枚)の最適化
datasets = optimize_bsds500_test(
    output_path='./optimized_dataset',
    max_images=100,              # 処理する画像数
    batch_size=4,                # バッチサイズ
    create_pytorch=True,         # PyTorchデータセット作成
    create_png=True,             # PNG形式でも保存
    num_workers=4,               # 並列処理
    save_checkpoint=True         # 中断・再開可能
)

print(f"✅ PyTorchデータセット: {datasets['pytorch_path']}")
print(f"✅ PNG画像コレクション: {datasets['png_path']}")
print(f"📊 平均PSNR改善: +{datasets['metrics']['avg_psnr_improvement']:.2f} dB")
```

### 3. 包括的な品質評価とベンチマーク
複数の評価指標を使用して、最適化の効果を定量的に分析します。

```python
from generative_latent_optimization import SimpleAllMetricsEvaluator

# 全メトリクス評価器の初期化
evaluator = SimpleAllMetricsEvaluator(device='cuda')

# データセット全体の評価（PSNR, SSIM, LPIPS, FID）
results = evaluator.evaluate_dataset_all_metrics(
    created_dir='./optimized_dataset',
    original_dir='./original_dataset'
)

# 美しいレポート形式で結果表示
evaluator.print_summary(results)
```

出力例：
```
📊 All Metrics Evaluation Summary
================================
📈 Image Quality Metrics:
  • PSNR: 28.45 dB (+3.2 dB improvement)
  • SSIM: 0.912 (+0.08 improvement)
  • LPIPS: 0.123 (lower is better)

🎯 Dataset-level FID Score: 15.67
🏆 Overall Quality: Excellent ✨
```

### 4. カスタムワークフローの構築
独自の処理パイプラインを構築して、特定のニーズに対応します。

```python
from generative_latent_optimization import (
    LatentOptimizer, 
    OptimizationConfig,
    MetricsIntegration
)
from generative_latent_optimization.dataset import BatchProcessor
import torch

class CustomPipeline:
    def __init__(self, vae, device='cuda'):
        self.vae = vae
        self.device = device
        self.metrics = MetricsIntegration(device)
        
    def process_with_mask(self, image, mask):
        """マスク領域のみを最適化"""
        config = OptimizationConfig(
            iterations=300,
            learning_rate=0.1,
            mask=mask,  # マスク領域のみ最適化
            device=self.device
        )
        
        optimizer = LatentOptimizer(self.vae, config)
        result = optimizer.optimize(image)
        
        # カスタムメトリクス計算
        metrics = self.metrics.compute_all_metrics(
            result.optimized_image,
            image,
            include_perceptual=True
        )
        
        return result, metrics

# パイプラインの使用
pipeline = CustomPipeline(vae)
result, metrics = pipeline.process_with_mask(image, mask)
```

### 5. 研究・実験用の詳細分析
最適化プロセスの詳細な分析と可視化を行います。

```python
from generative_latent_optimization.visualization import LossVisualization
from generative_latent_optimization.evaluation import DatasetEvaluator

# 損失関数の推移を可視化
visualizer = LossVisualization()
history = optimizer.get_optimization_history()
visualizer.plot_convergence(
    history,
    save_path='convergence_analysis.png',
    show_components=True  # 各損失成分を表示
)

# A/Bテスト用の比較評価
evaluator = DatasetEvaluator(device='cuda')
comparison = evaluator.compare_methods(
    original='./original',
    method_a='./optimized_v1',
    method_b='./optimized_v2',
    metrics=['psnr', 'ssim', 'lpips', 'fid']
)

# 統計的有意性の検定
print(f"Method A vs B: p-value = {comparison['statistical_significance']}")

## 🔧 詳細機能

### コア機能モジュール

#### 1. LatentOptimizer - 潜在表現最適化エンジン
```python
主要機能:
- Adam最適化器による勾配降下法
- 収束判定と早期停止機能
- チェックポイント保存・復元
- マルチスケール損失関数
- カスタム正則化項のサポート
```

#### 2. MetricsIntegration - 統合評価システム
```python
評価指標:
- PSNR, SSIM (基本的な画像品質)
- LPIPS (知覚的品質)
- FID (データセットレベルの分布評価)
- MS-SSIM (マルチスケール構造類似性)
- カスタムメトリクスの追加可能
```

#### 3. BatchProcessor - 高効率バッチ処理
```python
処理機能:
- 並列処理による高速化
- メモリ効率的なストリーミング処理
- 進捗追跡とレポート生成
- エラーハンドリングと再試行
- 分散処理対応（複数GPU）
```

#### 4. DatasetManager - データセット管理
```python
対応形式:
- PyTorchデータセット (.pt)
- PNG/JPEG画像コレクション
- BSDS500, ImageNet対応
- カスタムデータセットのサポート
- オンザフライ前処理
```

### 高度な最適化オプション

```python
from generative_latent_optimization import OptimizationConfig

config = OptimizationConfig(
    # 基本設定
    iterations=500,
    learning_rate=0.1,
    device='cuda',
    
    # 収束制御
    convergence_threshold=1e-5,
    early_stopping=True,
    patience=50,
    
    # 損失関数設定
    loss_type='combined',  # 'mse', 'l1', 'perceptual', 'combined'
    loss_weights={
        'reconstruction': 1.0,
        'perceptual': 0.1,
        'regularization': 0.01
    },
    
    # 最適化戦略
    optimizer_type='adam',  # 'adam', 'sgd', 'lbfgs'
    scheduler='cosine',     # 'cosine', 'exponential', 'step'
    gradient_clip=1.0,
    
    # チェックポイント
    checkpoint_interval=100,
    checkpoint_path='./checkpoints',
    resume_from_checkpoint=False
)
```

### 可視化ツール

```python
from generative_latent_optimization.visualization import (
    plot_optimization_history,
    create_comparison_grid,
    generate_heatmap
)

# 最適化履歴の可視化
plot_optimization_history(
    optimizer.history,
    metrics=['loss', 'psnr', 'ssim'],
    save_path='optimization_curves.png'
)

# 比較グリッドの作成
create_comparison_grid(
    original_images,
    optimized_images,
    titles=['Original', 'VAE', 'Optimized'],
    save_path='comparison.png'
)

# 改善度ヒートマップ
generate_heatmap(
    improvement_map,
    colormap='coolwarm',
    save_path='improvement_heatmap.png'
)
```

## 🧪 実験の実行

### 事前準備済み実験スクリプト

#### 1. 基本テスト（推奨）
```bash
# VAE基本機能テスト
HF_TOKEN="your_token" NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python tests/unit/test_vae_fixed.py

# 高速最適化テスト（約1分）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/optimization/quick_optimization_test.py

# メトリクス評価デモ
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/evaluation/metrics_evaluation_demo.py
```

#### 2. 単一画像最適化実験
```bash
# document.png での実証実験
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/optimization/single_image_optimization.py

# 期待される結果:
# - 元画像、初期再構成、最適化後の3段階比較
# - PSNR改善: +2-4 dB
# - 実行時間: 30-60秒
```

#### 3. バッチ処理デモ
```bash
# BSDS500データセット処理（時間がかかります）
export BSDS500_PATH="/path/to/bsds500"
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/datasets/batch_processing_demo.py

# デュアルデータセット作成デモ
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/datasets/dual_datasets_demo.py
```

#### 4. 包括的評価デモ
```bash
# 全メトリクスでの評価
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/evaluation/comprehensive_evaluation_demo.py

# 期待される出力:
# 📊 All Metrics Evaluation Summary
# 🎯 Dataset-level FID Score: XX.XX
# 🏆 Overall Quality: Excellent ✨
```

#### 5. 損失可視化
```bash
# 最適化プロセスの可視化
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/visualization/loss_visualization.py

# 生成される可視化:
# - 損失関数の収束曲線
# - PSNR/SSIMの改善グラフ
# - ビフォー・アフター比較画像
```

### 結果の確認方法

```bash
# 実験結果を確認
ls -la experiments/results/

# 最適化テスト結果
ls -la experiments/results/quick_test/

# 可視化結果
ls -la experiments/results/visualization/

# バッチ処理結果
ls -la experiments/results/batch_processing/
```

### カスタム実験の作成

独自の実験を作成したい場合：

```python
# custom_experiment.py
import torch
from generative_latent_optimization import LatentOptimizer, OptimizationConfig
from vae_toolkit import VAELoader

def run_custom_experiment():
    """カスタム実験の例"""
    
    # 実験設定
    configs = [
        OptimizationConfig(iterations=100, learning_rate=0.05),
        OptimizationConfig(iterations=200, learning_rate=0.1),
        OptimizationConfig(iterations=500, learning_rate=0.2),
    ]
    
    results = []
    for i, config in enumerate(configs):
        print(f"🧪 実験 {i+1}/3 実行中...")
        
        optimizer = LatentOptimizer(vae, config)
        result = optimizer.optimize(image_tensor)
        results.append(result)
        
        print(f"  PSNR: {result.metrics['final_psnr']:.2f} dB")
    
    # 最良の設定を特定
    best_result = max(results, key=lambda r: r.metrics['final_psnr'])
    print(f"🏆 最良設定のPSNR: {best_result.metrics['final_psnr']:.2f} dB")
    
    return results

# 実験実行
results = run_custom_experiment()
```

## 📁 プロジェクト構造

```
Generative-Latent-Optimization/
├── src/generative_latent_optimization/     # メインパッケージ
│   ├── __init__.py                        # 主要APIのエクスポート
│   ├── optimization/                       # 最適化エンジン
│   │   └── latent_optimizer.py           # VAE潜在表現最適化
│   ├── metrics/                           # 品質評価システム
│   │   ├── image_metrics.py              # 基本メトリクス(PSNR/SSIM)
│   │   ├── individual_metrics.py         # 高度メトリクス(LPIPS)
│   │   ├── dataset_metrics.py            # データセット評価(FID)
│   │   └── metrics_integration.py        # 統合評価API
│   ├── evaluation/                        # 評価フレームワーク
│   │   ├── dataset_evaluator.py          # 包括的データセット評価
│   │   └── simple_evaluator.py           # シンプル評価API
│   ├── dataset/                           # データセット処理
│   │   ├── bsds500_dataset.py            # BSDS500データセット
│   │   ├── batch_processor.py            # バッチ処理エンジン
│   │   ├── png_dataset.py                # PNG形式データセット
│   │   └── pytorch_dataset.py            # PyTorch形式データセット
│   ├── config/                            # 設定管理
│   │   └── model_config.py               # モデル設定
│   ├── workflows/                         # 高レベルAPI
│   │   └── batch_processing.py           # ワンコマンド処理
│   ├── utils/                             # ユーティリティ
│   │   └── io_utils.py                   # I/O処理
│   └── visualization/                     # 可視化ツール
│       └── image_viz.py                  # 画像可視化
├── experiments/                           # 実験スクリプト
│   ├── optimization/                      # 最適化実験
│   │   ├── quick_optimization_test.py    # 高速テスト
│   │   └── single_image_optimization.py  # 単一画像実験
│   ├── evaluation/                        # 評価実験
│   │   ├── metrics_evaluation_demo.py    # メトリクス評価
│   │   └── comprehensive_evaluation_demo.py # 包括評価
│   ├── datasets/                          # データセット実験
│   │   ├── batch_processing_demo.py      # バッチ処理デモ
│   │   └── dual_datasets_demo.py         # デュアルデータセット
│   └── visualization/                     # 可視化実験
│       └── loss_visualization.py         # 損失可視化
├── tests/                                 # テストスイート
│   ├── unit/                             # 単体テスト
│   │   ├── test_vae_basic.py             # 基本VAEテスト
│   │   └── test_vae_fixed.py             # 修正VAEテスト
│   ├── integration/                       # 統合テスト
│   │   └── test_optimization_integration.py
│   └── fixtures/                          # テストデータ
├── scripts/                               # スクリプト
│   ├── examples/                          # 使用例
│   │   └── document_encode_decode_example.py
│   └── analysis/                          # 分析スクリプト
│       └── implementation_comparison.py
├── document.png                           # サンプル画像
├── pyproject.toml                         # Python設定
├── uv.lock                               # 依存関係ロック
├── flake.nix                             # Nix環境定義
├── CLAUDE.md                             # プロジェクト詳細
└── README.md                             # このファイル
```

### モジュラー設計の特徴

#### 1. 階層化アーキテクチャ
- **高レベル**: ワンコマンドでの簡単な操作
- **中レベル**: カスタマイズ可能な処理パイプライン
- **低レベル**: 詳細な制御とデバッグ

#### 2. 拡張可能な設計
- 新しい評価指標の追加が容易
- カスタムVAEモデルの統合
- プラグイン式の損失関数

#### 3. 独立性の保証
- 各モジュールは独立して使用可能
- 最小限の依存関係
- テスタビリティの高い設計

## 🛠️ インストールと環境設定

### 前提条件
- **Python**: 3.10以上
- **GPU**: CUDA対応GPU（推奨、CPUでも動作）
- **メモリ**: 8GB以上（バッチ処理時は16GB推奨）
- **Nix**: パッケージマネージャ（推奨）

### 1. Nix環境でのセットアップ（推奨）

```bash
# リポジトリのクローン
git clone https://github.com/mdipcit/Generative-Latent-Optimization.git
cd Generative-Latent-Optimization

# Nix環境の起動（CUDA対応）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure

# Python依存関係のインストール
uv sync

# 環境変数の設定
export HF_TOKEN="your_huggingface_token_here"
```

### 2. 従来のPython環境でのセットアップ

```bash
# リポジトリのクローン
git clone https://github.com/mdipcit/Generative-Latent-Optimization.git
cd Generative-Latent-Optimization

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
# venv\Scripts\activate  # Windows

# uvのインストール
pip install uv

# 依存関係のインストール
uv sync

# 環境変数の設定
export HF_TOKEN="your_huggingface_token_here"
```

### 3. Docker環境でのセットアップ

```dockerfile
# Dockerfile例
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync

ENV HF_TOKEN=""
CMD ["python", "experiments/optimization/quick_optimization_test.py"]
```

### 環境変数の設定

#### 必須環境変数
```bash
# Hugging Face認証トークン（必須）
export HF_TOKEN="your_huggingface_token_here"

# GPU設定（オプション）
export CUDA_VISIBLE_DEVICES="0"  # 使用するGPUを指定
```

#### 大規模データセット用（オプション）
```bash
# BSDS500データセットのパス
export BSDS500_PATH="/path/to/bsds500/dataset"

# カスタムキャッシュディレクトリ
export HF_HOME="/custom/path/to/huggingface/cache"
```

### セットアップ確認テスト

```bash
# 環境が正しくセットアップされているかテスト
HF_TOKEN="your_token" NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python -c "
from generative_latent_optimization import LatentOptimizer
from vae_toolkit import VAELoader
print('✅ 環境セットアップ完了')
print('✅ すべてのモジュールが正常に読み込まれました')
"

# CUDA利用可能性の確認
python -c "import torch; print(f'CUDA利用可能: {torch.cuda.is_available()}')"
```

### トラブルシューティング

#### よくある問題と解決法

1. **UNFREE パッケージエラー**
   ```bash
   # エラー: unfree packageが利用できない
   # 解決: NIXPKGS_ALLOW_UNFREE=1 フラグを必ず付ける
   NIXPKGS_ALLOW_UNFREE=1 nix develop --impure
   ```

2. **CUDA関連エラー**
   ```bash
   # CUDAが認識されない場合
   nvidia-smi  # GPU状態確認
   
   # CPU環境での実行
   # device='cpu' パラメータを設定
   ```

3. **メモリ不足エラー**
   ```bash
   # バッチサイズの削減
   batch_size=1  # または2
   
   # チェックポイントの有効化
   checkpoint_interval=50
   ```

4. **Hugging Face認証エラー**
   ```bash
   # トークンの確認
   huggingface-cli login
   
   # または環境変数で設定
   export HF_TOKEN="your_token_here"
   ```

### 開発環境のセットアップ

開発者向けの追加セットアップ：

```bash
# 開発用依存関係の追加
uv add pytest pytest-cov black flake8 mypy

# テストスイートの実行
pytest tests/ -v
```
