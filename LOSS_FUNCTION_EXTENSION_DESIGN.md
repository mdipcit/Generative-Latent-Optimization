# 損失関数拡張設計: LPIPS・SSIM・Improved SSIM・PSNR対応

## 🎯 拡張目的

現在MSEとL1のみに対応しているLatentOptimizerの損失関数を、LPIPS、SSIM、Improved SSIM、PSNRに拡張し、FID実験での各指標による最適化を可能にする。

## 📋 現在の実装状況

### 既存の _calculate_batch_loss メソッド
```python
# src/generative_latent_optimization/optimization/latent_optimizer.py:322-346

def _calculate_batch_loss(self, targets: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    if self.config.loss_function == 'mse':
        losses = torch.nn.functional.mse_loss(
            targets, reconstructed, reduction='none'
        ).mean(dim=(1, 2, 3))
    elif self.config.loss_function == 'l1':
        losses = torch.nn.functional.l1_loss(
            targets, reconstructed, reduction='none'
        ).mean(dim=(1, 2, 3))
    else:
        raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
    
    return losses
```

## 🔧 拡張実装設計

### 1. LPIPS損失実装

#### 設計方針
- LPIPSMetricを利用したバッチ対応損失計算
- 勾配計算可能な実装
- メモリ効率的なバッチ処理

#### 実装案
```python
def _calculate_lpips_batch_loss(self, targets: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """LPIPS損失をバッチで計算（勾配計算対応）"""
    
    # LPIPSメトリクが未初期化の場合は初期化
    if not hasattr(self, '_lpips_metric'):
        from ..metrics.individual_metrics import LPIPSMetric
        self._lpips_metric = LPIPSMetric(device=self.device)
    
    # バッチ処理: 各画像ペアに対してLPIPS計算
    batch_size = targets.shape[0]
    lpips_losses = []
    
    for i in range(batch_size):
        # 個別画像のLPIPS計算
        target_single = targets[i:i+1]  # [1, C, H, W]
        recon_single = reconstructed[i:i+1]  # [1, C, H, W]
        
        # LPIPS計算（勾配有効）
        with torch.enable_grad():
            # 範囲調整: [0,1] → [-1,1] (LPIPS要求)
            target_normalized = target_single * 2.0 - 1.0
            recon_normalized = recon_single * 2.0 - 1.0
            
            # LPIPSメトリックの内部計算を直接利用
            lpips_value = self._lpips_metric.loss_fn(target_normalized, recon_normalized)
            lpips_losses.append(lpips_value.squeeze())
    
    return torch.stack(lpips_losses)
```

### 2. SSIM損失実装

#### 設計方針
- SSIMを損失に変換: `loss = 1.0 - ssim`
- 勾配計算可能なSSIM実装
- バッチ効率的な計算

#### 実装案
```python
def _calculate_ssim_batch_loss(self, targets: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """SSIM損失をバッチで計算（1 - SSIMで損失に変換）"""
    
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    
    # SSIMメトリクが未初期化の場合は初期化
    if not hasattr(self, '_ssim_metric'):
        self._ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=1.0,  # [0, 1]範囲
            gaussian_kernel=True,
            kernel_size=11,
            sigma=1.5,
            reduction='none'  # バッチ毎の結果を取得
        ).to(self.device)
    
    # SSIM計算（勾配有効）
    with torch.enable_grad():
        ssim_values = self._ssim_metric(targets, reconstructed)
        # SSIM損失 = 1 - SSIM（高いSSIMで低い損失）
        ssim_losses = 1.0 - ssim_values
    
    return ssim_losses
```

### 3. Improved SSIM損失実装

#### 設計方針
- ImprovedSSIMクラスの内部実装を直接利用
- 勾配計算対応の設計
- 標準SSIMとの差別化

#### 実装案
```python
def _calculate_improved_ssim_batch_loss(self, targets: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Improved SSIM損失をバッチで計算"""
    
    # Improved SSIMメトリクが未初期化の場合は初期化
    if not hasattr(self, '_improved_ssim_metric'):
        from ..metrics.individual_metrics import ImprovedSSIM
        self._improved_ssim_metric = ImprovedSSIM(device=self.device)
    
    # バッチ処理での計算
    batch_size = targets.shape[0]
    ssim_losses = []
    
    for i in range(batch_size):
        target_single = targets[i:i+1]
        recon_single = reconstructed[i:i+1]
        
        # Improved SSIM計算（勾配有効）
        with torch.enable_grad():
            # ImprovedSSIMの内部SSIMメトリクを直接利用
            ssim_value = self._improved_ssim_metric.ssim(target_single, recon_single)
            ssim_loss = 1.0 - ssim_value
            ssim_losses.append(ssim_loss)
    
    return torch.stack(ssim_losses)
```

### 4. PSNR損失実装

#### 設計方針
- PSNR最大化 = MSE最小化の等価性利用
- 対数演算による勾配計算の安定性確保
- 数値安定性のためのクランプ処理

#### 実装案
```python
def _calculate_psnr_batch_loss(self, targets: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """PSNR損失をバッチで計算（-PSNRで損失に変換）"""
    
    # MSE計算
    mse_values = torch.nn.functional.mse_loss(
        targets, reconstructed, reduction='none'
    ).mean(dim=(1, 2, 3))  # [B]
    
    # PSNRベース損失計算
    with torch.enable_grad():
        # 数値安定性のためのクランプ
        mse_clamped = torch.clamp(mse_values, min=1e-10)
        
        # PSNR = 20 * log10(MAX_VAL) - 10 * log10(MSE)
        # MAX_VAL = 1.0 (画像が[0,1]範囲の場合)
        psnr_values = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse_clamped)
        
        # PSNR損失 = -PSNR（高いPSNRで低い損失）
        psnr_losses = -psnr_values
    
    return psnr_losses
```

## 🔄 統合実装: 拡張された _calculate_batch_loss

### 完全な実装
```python
def _calculate_batch_loss(self, targets: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Calculate loss for batch with extended metrics support
    
    Args:
        targets: Target images [B, C, H, W] in [0, 1] range
        reconstructed: Reconstructed images [B, C, H, W] in [0, 1] range
        
    Returns:
        Per-sample losses [B]
    """
    loss_function = self.config.loss_function
    
    if loss_function == 'mse':
        losses = torch.nn.functional.mse_loss(
            targets, reconstructed, reduction='none'
        ).mean(dim=(1, 2, 3))
        
    elif loss_function == 'l1':
        losses = torch.nn.functional.l1_loss(
            targets, reconstructed, reduction='none'
        ).mean(dim=(1, 2, 3))
        
    elif loss_function == 'lpips':
        losses = self._calculate_lpips_batch_loss(targets, reconstructed)
        
    elif loss_function == 'ssim':
        losses = self._calculate_ssim_batch_loss(targets, reconstructed)
        
    elif loss_function == 'improved_ssim':
        losses = self._calculate_improved_ssim_batch_loss(targets, reconstructed)
        
    elif loss_function == 'psnr':
        losses = self._calculate_psnr_batch_loss(targets, reconstructed)
        
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")
    
    return losses
```

## 🧪 テスト実装計画

### 1. 単体テスト
```python
# tests/unit/test_optimization/test_extended_loss_functions.py

class TestExtendedLossFunctions:
    def test_lpips_loss_calculation(self):
        """LPIPS損失計算の正確性テスト"""
        # 同一画像のLPIPS損失は0に近い
        # 異なる画像のLPIPS損失は正の値
        
    def test_ssim_loss_calculation(self):
        """SSIM損失計算の正確性テスト"""
        # 同一画像のSSIM損失は0に近い
        # 白画像と黒画像のSSIM損失は1に近い
        
    def test_gradient_computation(self):
        """各損失関数での勾配計算テスト"""
        # requires_grad=Trueでの勾配計算確認
        
    def test_batch_consistency(self):
        """バッチ処理と個別処理の一貫性テスト"""
        # バッチ処理結果 = 個別処理結果の組み合わせ
```

### 2. 統合テスト
```python
# tests/integration/test_optimization_extended.py

class TestExtendedOptimizationIntegration:
    def test_end_to_end_optimization_all_metrics(self):
        """全指標での最適化エンドツーエンドテスト"""
        for metric in ['mse', 'l1', 'lpips', 'ssim', 'improved_ssim', 'psnr']:
            config = OptimizationConfig(loss_function=metric)
            optimizer = LatentOptimizer(config)
            # 小さなテスト画像で最適化実行
            
    def test_convergence_behavior(self):
        """各指標での収束挙動テスト"""
        # 収束速度と最終品質の比較
```

## ⚡ パフォーマンス最適化

### 1. 計算効率化

#### LPIPS効率化
```python
class LPIPSBatchOptimizer:
    """LPIPS計算の効率化クラス"""
    
    def __init__(self, device, batch_size=8):
        self.device = device
        self.batch_size = batch_size
        import lpips
        self.loss_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
    
    def calculate_batch_loss(self, targets, reconstructed):
        """効率的なバッチLPIPS計算"""
        # 小バッチに分割してメモリ使用量を制御
        total_batch_size = targets.shape[0]
        all_losses = []
        
        for i in range(0, total_batch_size, self.batch_size):
            end_idx = min(i + self.batch_size, total_batch_size)
            batch_targets = targets[i:end_idx]
            batch_recons = reconstructed[i:end_idx]
            
            # 範囲変換
            batch_targets_norm = batch_targets * 2.0 - 1.0
            batch_recons_norm = batch_recons * 2.0 - 1.0
            
            # LPIPS計算
            batch_losses = self.loss_fn(batch_targets_norm, batch_recons_norm)
            all_losses.append(batch_losses.squeeze(-1).squeeze(-1))
        
        return torch.cat(all_losses)
```

### 2. メモリ効率化

#### グラデーション蓄積戦略
```python
def _calculate_memory_efficient_loss(self, targets, reconstructed, accumulation_steps=4):
    """メモリ効率的な損失計算（グラデーション蓄積）"""
    
    batch_size = targets.shape[0]
    micro_batch_size = max(1, batch_size // accumulation_steps)
    
    total_loss = 0
    for i in range(0, batch_size, micro_batch_size):
        end_idx = min(i + micro_batch_size, batch_size)
        
        micro_targets = targets[i:end_idx]
        micro_recons = reconstructed[i:end_idx]
        
        # マイクロバッチでの損失計算
        micro_losses = self._calculate_batch_loss(micro_targets, micro_recons)
        micro_loss = micro_losses.mean() / accumulation_steps
        
        total_loss += micro_loss
    
    return total_loss
```

### 3. 数値安定性

#### 安定化実装
```python
def _calculate_numerically_stable_psnr_loss(self, targets, reconstructed):
    """数値的に安定なPSNR損失計算"""
    
    # MSE計算
    mse_values = torch.nn.functional.mse_loss(
        targets, reconstructed, reduction='none'
    ).mean(dim=(1, 2, 3))
    
    # 数値安定性のための処理
    epsilon = 1e-10
    mse_stable = torch.clamp(mse_values, min=epsilon)
    
    # PSNR計算（log10の安定化）
    with torch.enable_grad():
        # log10の代わりにlogを使用して数値安定性向上
        log_mse = torch.log(mse_stable)
        psnr_values = 20 * torch.log10(torch.tensor(1.0, device=self.device)) - 10 * log_mse / torch.log(torch.tensor(10.0, device=self.device))
        
        # PSNR損失（負値で損失に変換）
        psnr_losses = -psnr_values
    
    return psnr_losses
```

## 📊 実装段階計画

### Stage 1: 基本実装
1. **LPIPS損失**: 個別画像ベースの実装
2. **SSIM損失**: TorchMetrics使用の実装
3. **基本テスト**: 動作確認レベルのテスト

### Stage 2: 効率化実装
1. **バッチ効率化**: メモリ使用量最適化
2. **並列処理**: マルチGPU対応
3. **数値安定性**: 極端ケースでの安定性確保

### Stage 3: 高度実装  
1. **ハイブリッド損失**: 複数指標の加重平均
2. **適応的重み**: 最適化進行に応じた重み調整
3. **カスタム指標**: ユーザー定義損失関数サポート

## 🔍 詳細実装仕様

### OptimizationConfig の拡張

#### 新しい設定オプション
```python
@dataclass
class OptimizationConfig:
    iterations: int = 150
    learning_rate: float = 0.4
    loss_function: str = 'mse'  # 'mse', 'l1', 'lpips', 'ssim', 'improved_ssim', 'psnr'
    convergence_threshold: float = 1e-6
    checkpoint_interval: int = 20
    device: str = "cuda"
    
    # 新規追加: 高度な設定
    lpips_network: str = 'alex'  # LPIPS用ネットワーク ('alex', 'vgg', 'squeeze')
    ssim_kernel_size: int = 11   # SSIM用カーネルサイズ
    ssim_sigma: float = 1.5      # SSIMガウシアンσ
    numerical_stability: bool = True  # 数値安定性機能
    memory_efficient: bool = True     # メモリ効率化モード
    gradient_accumulation_steps: int = 1  # グラデーション蓄積ステップ
```

### エラーハンドリング強化

#### 堅牢なエラー処理
```python
def _safe_calculate_batch_loss(self, targets, reconstructed):
    """安全な損失計算（エラー回復機能付き）"""
    
    try:
        return self._calculate_batch_loss(targets, reconstructed)
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            # メモリ不足時の自動回復
            torch.cuda.empty_cache()
            return self._calculate_memory_efficient_loss(targets, reconstructed)
        else:
            raise
            
    except ImportError as e:
        if "lpips" in str(e) and self.config.loss_function == 'lpips':
            # LPIPSが利用できない場合はSSIMにフォールバック
            logger.warning("LPIPS not available, falling back to SSIM")
            old_loss_function = self.config.loss_function
            self.config.loss_function = 'ssim'
            result = self._calculate_batch_loss(targets, reconstructed)
            self.config.loss_function = old_loss_function  # 復元
            return result
        else:
            raise
```

### バリデーション機能

#### 損失関数の妥当性検証
```python
def _validate_loss_function_setup(self):
    """損失関数のセットアップ検証"""
    
    loss_function = self.config.loss_function
    
    # 依存関係チェック
    if loss_function == 'lpips':
        try:
            import lpips
        except ImportError:
            raise ImportError("LPIPS loss requires lpips package: pip install lpips")
    
    elif loss_function in ['ssim', 'improved_ssim']:
        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
        except ImportError:
            raise ImportError("SSIM loss requires torchmetrics package: pip install torchmetrics")
    
    # デバイス互換性チェック
    if not torch.cuda.is_available() and self.device == 'cuda':
        logger.warning("CUDA not available, some metrics may run slower on CPU")
    
    # 数値範囲チェック用のテストテンソル
    test_targets = torch.rand(1, 3, 64, 64, device=self.device)
    test_recons = torch.rand(1, 3, 64, 64, device=self.device)
    
    try:
        test_loss = self._calculate_batch_loss(test_targets, test_recons)
        logger.info(f"Loss function '{loss_function}' validated successfully")
        
    except Exception as e:
        raise RuntimeError(f"Loss function '{loss_function}' validation failed: {e}")
```

## 🎯 FID実験特化機能

### 実験制御クラス設計

#### FIDExperimentController
```python
# experiments/fid_comparison/experiment_controller.py

class FIDExperimentController:
    """FID最適化実験の制御クラス"""
    
    def __init__(self, bsds500_path, output_base_path):
        self.bsds500_path = bsds500_path
        self.output_base_path = Path(output_base_path)
        
        # 実験対象指標
        self.loss_functions = [
            'mse', 'l1', 'lpips', 'ssim', 'improved_ssim', 'psnr'
        ]
        
        # FID評価器
        self.fid_evaluator = DatasetFIDEvaluator(device='cuda')
    
    def run_complete_experiment(self, max_images_per_split=None):
        """完全なFID比較実験を実行"""
        
        experiment_results = {}
        
        for loss_func in self.loss_functions:
            logger.info(f"🔄 Starting optimization with {loss_func} loss")
            
            # 最適化設定
            config = OptimizationConfig(
                iterations=150,
                learning_rate=0.4,
                loss_function=loss_func,
                device='cuda'
            )
            
            # データセット最適化実行
            result = self._optimize_and_evaluate_single_metric(
                config, max_images_per_split
            )
            
            experiment_results[loss_func] = result
            logger.info(f"✅ {loss_func} optimization completed: FID = {result['fid_score']:.2f}")
        
        # 結果比較・分析
        comparison_results = self._analyze_experiment_results(experiment_results)
        
        return {
            'individual_results': experiment_results,
            'comparison_analysis': comparison_results
        }
    
    def _optimize_and_evaluate_single_metric(self, config, max_images):
        """単一指標での最適化とFID評価"""
        
        # 出力パス設定
        output_dir = self.output_base_path / f"{config.loss_function}_optimized"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # バッチ処理実行
        start_time = time.time()
        datasets = process_bsds500_dataset(
            self.bsds500_path,
            output_dir / "dataset",
            config,
            max_images_per_split=max_images,
            create_pytorch_dataset=True,
            create_png_dataset=True
        )
        processing_time = time.time() - start_time
        
        # FID評価実行
        fid_score = self.fid_evaluator.evaluate_created_dataset_vs_original(
            datasets['png'], 
            self.bsds500_path
        ).fid_score
        
        return {
            'fid_score': fid_score,
            'dataset_paths': datasets,
            'processing_time_seconds': processing_time,
            'optimization_config': config
        }
```

### 結果分析システム

#### 統計分析機能
```python
class FIDComparisonAnalyzer:
    """FID実験結果の分析クラス"""
    
    def analyze_results(self, experiment_results):
        """実験結果の包括的分析"""
        
        # FIDスコア抽出
        fid_scores = {
            metric: result['fid_score'] 
            for metric, result in experiment_results.items()
        }
        
        # ランキング作成
        fid_ranking = sorted(fid_scores.items(), key=lambda x: x[1])
        
        # 統計分析
        analysis = {
            'fid_scores': fid_scores,
            'ranking': fid_ranking,
            'best_metric': fid_ranking[0][0],
            'worst_metric': fid_ranking[-1][0],
            'score_range': fid_ranking[-1][1] - fid_ranking[0][1],
            'relative_improvements': self._calculate_relative_improvements(fid_scores)
        }
        
        return analysis
    
    def _calculate_relative_improvements(self, fid_scores):
        """相対的改善率計算"""
        baseline_score = fid_scores.get('mse', max(fid_scores.values()))
        
        improvements = {}
        for metric, score in fid_scores.items():
            if metric != 'mse':
                improvement = ((baseline_score - score) / baseline_score) * 100
                improvements[metric] = improvement
        
        return improvements
    
    def generate_visualization(self, analysis_results, output_path):
        """結果可視化の生成"""
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # FIDスコア比較棒グラフ
        metrics = list(analysis_results['fid_scores'].keys())
        scores = list(analysis_results['fid_scores'].values())
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(metrics, scores, color=sns.color_palette("husl", len(metrics)))
        plt.title('FID Score Comparison Across Optimization Metrics', fontsize=16)
        plt.ylabel('FID Score (Lower is Better)', fontsize=12)
        plt.xlabel('Optimization Metric', fontsize=12)
        
        # 数値表示
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{score:.1f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'fid_comparison_chart.png', dpi=300)
        plt.close()
```

## 🚀 実行可能な実装ファイル

### experiments/fid_comparison/pilot_experiment.py
```python
#!/usr/bin/env python3
"""FID最適化実験: パイロット版（小規模テスト）"""

import torch
import time
from pathlib import Path
from generative_latent_optimization import OptimizationConfig
from generative_latent_optimization.workflows import process_bsds500_dataset
from generative_latent_optimization.metrics import DatasetFIDEvaluator

def run_pilot_experiment(max_images=10):
    """パイロット実験実行"""
    
    # 基本設定
    bsds500_path = os.environ.get('BSDS500_PATH')
    output_base = Path('./experiments/fid_comparison/results/pilot')
    
    # 実験対象指標（パイロットでは基本指標のみ）
    loss_functions = ['mse', 'l1']  # 'lpips', 'ssim'は実装後追加
    
    results = {}
    
    for loss_func in loss_functions:
        print(f"🔄 Pilot experiment: {loss_func} optimization")
        
        config = OptimizationConfig(
            iterations=50,  # パイロットでは短縮
            learning_rate=0.4,
            loss_function=loss_func
        )
        
        # 最適化実行
        start_time = time.time()
        datasets = process_bsds500_dataset(
            bsds500_path,
            output_base / f"{loss_func}_dataset",
            config,
            max_images_per_split=max_images,
            create_png_dataset=True
        )
        processing_time = time.time() - start_time
        
        # FID評価
        fid_evaluator = DatasetFIDEvaluator()
        fid_result = fid_evaluator.evaluate_created_dataset_vs_original(
            datasets['png'], bsds500_path
        )
        
        results[loss_func] = {
            'fid_score': fid_result.fid_score,
            'processing_time': processing_time,
            'total_images': fid_result.total_images
        }
        
        print(f"✅ {loss_func}: FID = {fid_result.fid_score:.2f} ({processing_time/60:.1f}min)")
    
    return results

if __name__ == "__main__":
    results = run_pilot_experiment()
    print("\n📊 Pilot Experiment Results:")
    for metric, result in results.items():
        print(f"  {metric}: FID = {result['fid_score']:.2f}")
```

## 🧮 推定リソース要件

### 計算資源
- **GPU**: NVIDIA RTX 3080/4080以上推奨
- **VRAM**: 最低12GB（LPIPS使用時は16GB推奨）
- **RAM**: 32GB以上（データセットキャッシュ用）
- **ストレージ**: 100GB以上（実験結果保存用）

### 処理時間推定

#### パイロット実験（30枚）
- MSE/L1: 各約20分
- LPIPS: 各約40分（計算コスト高）
- SSIM系: 各約30分
- **合計**: 約3時間

#### 全データセット実験（500枚）
- MSE/L1: 各約6時間
- LPIPS: 各約12時間
- SSIM系: 各約8時間
- **合計**: 約48時間（並列実行で24時間に短縮可能）

### 並列処理戦略
```python
# 2GPU環境での最適割り当て
gpu_assignments = {
    'cuda:0': ['mse', 'l1', 'psnr'],        # 高速指標
    'cuda:1': ['lpips', 'ssim', 'improved_ssim']  # 重い指標
}
```

## ✅ 品質保証チェックリスト

### 実装品質
- [ ] すべての損失関数での勾配計算確認
- [ ] バッチ処理での結果一貫性確認
- [ ] メモリリーク防止確認
- [ ] 数値安定性の極値テスト
- [ ] エラー処理の網羅的テスト

### 実験品質
- [ ] 同一条件での再現性確認（±1% FIDスコア）
- [ ] FID計算の妥当性確認（ベースライン ≈ 0）
- [ ] 統計的有意性の確保（n≥30）
- [ ] 外れ値処理の適切性確認

### 結果品質
- [ ] FIDスコアの妥当な範囲確認（0-200程度）
- [ ] 指標間の論理的整合性確認
- [ ] 予想結果との整合性確認
- [ ] 統計検定による有意差確認

---

この拡張設計により、BSDS500データセットでの包括的なFID最適化実験が実現可能となります。実装は段階的に進め、各段階での品質確認を徹底することで、信頼性の高い実験結果を獲得できます。