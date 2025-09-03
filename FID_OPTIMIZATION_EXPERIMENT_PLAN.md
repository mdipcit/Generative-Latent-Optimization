# FID最適化実験計画: 評価指標別のデータセット品質比較

## 🎯 実験目的

BSDS500データセット（500枚）を用いて、異なる評価指標で最適化を行った場合のデータセットレベルでのFIDスコアを比較し、どの最適化指標がFIDを悪化させるかを分析する。

## 📊 背景と動機

### 仮説
- 個別画像の品質向上に最適化された指標（PSNR、SSIM等）は、データセット全体の多様性やリアリズムを犠牲にする可能性がある
- 知覚的品質を重視する指標（LPIPS）は、統計的品質指標とは異なるFID傾向を示す可能性がある
- 各最適化指標の特性により、FIDへの影響が大きく異なることが予想される

### 期待される成果
1. 最適化指標とFIDスコアの相関関係の定量化
2. データセット品質劣化を招く最適化手法の特定
3. 将来の最適化戦略選択のための指針獲得

## 🔬 実験設計

### 実験条件
- **データセット**: BSDS500全体（train: 200枚、val: 100枚、test: 200枚）
- **モデル**: Stable Diffusion 1.5 VAE
- **最適化**: 150イテレーション、学習率0.4
- **評価**: FID（Fréchet Inception Distance）によるデータセット品質評価

### 実験対象指標

#### 現在実装済み指標
1. **MSE** (Mean Squared Error)
   - 最も基本的な画素レベル損失
   - 高周波数詳細の過度な強調傾向

2. **L1/MAE** (Mean Absolute Error)
   - MSEより外れ値に頑健
   - エッジ保持特性に優れる

#### 実装が必要な指標
3. **LPIPS** (Learned Perceptual Image Patch Similarity)
   - 人間の知覚に基づく損失
   - より自然な画像生成が期待される

4. **SSIM** (Structural Similarity Index)
   - 構造情報を重視
   - 明度・コントラスト・構造の統合評価

5. **Improved SSIM** (TorchMetrics Implementation)
   - 標準SSIMの改良版
   - より正確なガウシアンカーネル実装

6. **PSNR-based Loss**
   - PSNRを直接最適化する損失関数
   - 信号対雑音比最大化

### 実験設計の詳細

#### Phase 1: 実装拡張（準備フェーズ）
```python
# src/generative_latent_optimization/optimization/latent_optimizer.py
# _calculate_batch_loss()メソッドを拡張して以下を追加:

def _calculate_batch_loss(self, targets, reconstructed):
    if self.config.loss_function == 'mse':
        # 既存実装
    elif self.config.loss_function == 'l1':
        # 既存実装
    elif self.config.loss_function == 'lpips':
        # LPIPS実装を追加
    elif self.config.loss_function == 'ssim':
        # SSIM損失実装を追加
    elif self.config.loss_function == 'improved_ssim':
        # Improved SSIM損失実装を追加
    elif self.config.loss_function == 'psnr':
        # PSNR損失実装を追加
```

#### Phase 2: パイロット実験（小規模テスト）
- **対象**: 各分割から10枚ずつ（計30枚）
- **目的**: 実装検証と処理時間推定
- **予想処理時間**: 約2-3時間（6指標 × 30分/指標）

#### Phase 3: 中規模実験（統計的有意性確保）
- **対象**: 各分割から50枚ずつ（計150枚）
- **目的**: 統計的に有意なFID傾向の検出
- **予想処理時間**: 約10-15時間

#### Phase 4: 全データセット実験（最終分析）
- **対象**: BSDS500全体（500枚）
- **目的**: 確定的な結論の獲得
- **予想処理時間**: 約50-75時間（分散処理推奨）

## 📋 実験実行計画

### 実行順序
1. **実装拡張**: 新しい損失関数の追加
2. **パイロットテスト**: 小規模実験での動作確認
3. **中規模実験**: 統計的傾向の確認
4. **全データセット実験**: 最終結論

### データフロー
```
BSDS500画像
    ↓
VAEエンコーダ（初期潜在表現）
    ↓
指標別最適化（6種類）
    ↓
最適化後潜在表現
    ↓
VAEデコーダ（再構成画像）
    ↓
PNGデータセット保存
    ↓
FID評価（vs 原画像データセット）
    ↓
指標別FIDスコア比較
```

### 結果データ構造
```python
experiment_results = {
    'mse': {
        'fid_score': 45.2,
        'dataset_path': './experiments/results/mse_optimized_dataset',
        'processing_time_hours': 8.5,
        'individual_metrics': {...}
    },
    'l1': {
        'fid_score': 42.8,
        'dataset_path': './experiments/results/l1_optimized_dataset',
        'processing_time_hours': 8.2,
        'individual_metrics': {...}
    },
    'lpips': {
        'fid_score': 38.5,  # 予想: より良いFID
        'dataset_path': './experiments/results/lpips_optimized_dataset',
        'processing_time_hours': 12.3,
        'individual_metrics': {...}
    },
    # 他の指標...
}
```

## 🔧 技術的実装要件

### 必要な実装変更

#### 1. LatentOptimizerの拡張
```python
# src/generative_latent_optimization/optimization/latent_optimizer.py

class LatentOptimizer:
    def _calculate_batch_loss(self, targets, reconstructed):
        if self.config.loss_function == 'lpips':
            if not hasattr(self, '_lpips_metric'):
                from ..metrics.individual_metrics import LPIPSMetric
                self._lpips_metric = LPIPSMetric(device=self.device)
            
            # LPIPS計算（バッチ対応）
            batch_size = targets.shape[0]
            lpips_losses = []
            for i in range(batch_size):
                lpips_val = self._lpips_metric.calculate(
                    targets[i:i+1], reconstructed[i:i+1]
                )
                lpips_losses.append(lpips_val)
            return torch.tensor(lpips_losses, device=self.device)
        
        elif self.config.loss_function == 'ssim':
            # SSIM損失 = 1 - SSIM（SSIMを損失に変換）
            ssim_values = []
            for i in range(targets.shape[0]):
                ssim_val = self.metrics.calculate_ssim(
                    targets[i:i+1], reconstructed[i:i+1]
                )
                ssim_loss = 1.0 - ssim_val  # SSIMを損失に変換
                ssim_values.append(ssim_loss)
            return torch.tensor(ssim_values, device=self.device)
        
        # 他の指標も同様に実装...
```

#### 2. 実験制御スクリプト
```python
# experiments/fid_comparison/fid_optimization_experiment.py

class FIDOptimizationExperiment:
    def __init__(self, bsds500_path, output_base_path):
        self.bsds500_path = bsds500_path
        self.output_base_path = output_base_path
        self.optimization_metrics = [
            'mse', 'l1', 'lpips', 'ssim', 'improved_ssim', 'psnr'
        ]
    
    def run_full_experiment(self, max_images_per_split=None):
        results = {}
        
        for metric in self.optimization_metrics:
            print(f"🔄 実験開始: {metric}最適化")
            
            # 最適化設定
            config = OptimizationConfig(
                iterations=150,
                learning_rate=0.4,
                loss_function=metric
            )
            
            # データセット最適化
            dataset_path = self._optimize_dataset(config, metric, max_images_per_split)
            
            # FID評価
            fid_score = self._evaluate_fid(dataset_path)
            
            results[metric] = {
                'fid_score': fid_score,
                'dataset_path': dataset_path,
                'config': config
            }
            
            print(f"✅ {metric}最適化完了: FID = {fid_score:.2f}")
        
        return results
```

### 実装優先順位

#### Priority 1: 損失関数拡張
1. **LPIPS損失**: 知覚的品質重視の実装
2. **SSIM損失**: 構造保持重視の実装

#### Priority 2: 実験フレームワーク
1. **実験制御スクリプト**: 指標別自動実験実行
2. **結果比較システム**: FIDスコア集計・可視化

#### Priority 3: 最適化
1. **並列処理**: 複数GPUでの指標別並列実行
2. **チェックポイント**: 長時間実験の中断・再開

## 📈 予想される実験結果

### FIDスコア予想（低い方が良い品質）

#### 仮説1: 知覚的指標の優位性
```
LPIPS < Improved SSIM < SSIM < L1 < PSNR < MSE
```
- **根拠**: LPIPSは人間の知覚に基づくため、より自然な画像生成が期待される

#### 仮説2: 統計的指標の劣位
```
MSE, PSNR → 高いFID（悪い品質）
```
- **根拠**: 画素レベル最適化は不自然なアーティファクトを生成する可能性

#### 仮説3: 構造保持指標の中間性能
```
SSIM系指標 → 中程度のFID
```
- **根拣**: 構造情報は保持するが、知覚的自然さは限定的

### 分析指標

#### 主要評価指標
1. **FIDスコア**: データセット品質の主要指標
2. **FIDランキング**: 指標間の相対的順位
3. **FID分散**: 指標内での品質ばらつき

#### 補助分析指標
1. **個別メトリクス平均**: PSNR、SSIM等の平均値
2. **最適化効率**: 収束速度と処理時間
3. **メモリ使用量**: 各指標での計算コスト

## 🚀 実験実行手順

### 事前準備
```bash
# 環境変数設定
export BSDS500_PATH="/path/to/bsds500"
export HF_TOKEN="your_huggingface_token"

# 実験ディレクトリ作成
mkdir -p experiments/fid_comparison/{results,logs,checkpoints}
```

### Phase 1: パイロット実験（推奨）
```bash
# 小規模テスト（30枚）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python \
  experiments/fid_comparison/pilot_experiment.py \
  --max_images 10 \
  --metrics mse,l1 \
  --output ./experiments/fid_comparison/results/pilot
```

### Phase 2: 中規模実験
```bash
# 中規模テスト（150枚）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python \
  experiments/fid_comparison/medium_scale_experiment.py \
  --max_images 50 \
  --metrics mse,l1,lpips,ssim \
  --output ./experiments/fid_comparison/results/medium
```

### Phase 3: 全データセット実験
```bash
# 全データセット（500枚）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python \
  experiments/fid_comparison/full_scale_experiment.py \
  --max_images None \
  --metrics all \
  --output ./experiments/fid_comparison/results/full \
  --parallel_processing true
```

### Phase 4: 結果分析
```bash
# 結果比較とレポート生成
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python \
  experiments/fid_comparison/analyze_results.py \
  --results_dir ./experiments/fid_comparison/results \
  --generate_report true
```

## 📁 出力ファイル構造

```
experiments/fid_comparison/
├── results/
│   ├── mse_optimized/
│   │   ├── dataset.pt                    # PyTorch形式データセット
│   │   ├── png/                          # PNG画像ディレクトリ
│   │   └── evaluation_results.json       # 個別メトリクス結果
│   ├── l1_optimized/
│   ├── lpips_optimized/
│   ├── ssim_optimized/
│   ├── improved_ssim_optimized/
│   ├── psnr_optimized/
│   └── comparison_report.json            # 全指標比較結果
├── logs/
│   ├── mse_optimization.log
│   ├── l1_optimization.log
│   └── ...
├── checkpoints/                          # 長時間実験用チェックポイント
└── analysis/
    ├── fid_comparison_chart.png          # FIDスコア比較グラフ
    ├── correlation_analysis.png          # 相関分析結果
    └── statistical_report.pdf            # 統計分析レポート
```

## 📊 期待される分析結果

### メインアウトプット: FID比較表
| 最適化指標 | FIDスコア | ランキング | 95%信頼区間 | 処理時間(h) |
|------------|-----------|------------|-------------|-------------|
| LPIPS      | 35.2      | 1位        | [33.1, 37.3] | 15.2       |
| Improved SSIM | 38.7   | 2位        | [36.4, 41.0] | 12.8       |
| SSIM       | 41.3      | 3位        | [39.1, 43.5] | 11.5       |
| L1         | 44.8      | 4位        | [42.2, 47.4] | 10.1       |
| PSNR       | 48.2      | 5位        | [45.6, 50.8] | 11.3       |
| MSE        | 52.7      | 6位        | [49.8, 55.6] | 10.5       |

### 副次分析

#### 相関分析
- FIDスコア vs 個別PSNR平均
- FIDスコア vs 個別SSIM平均  
- FIDスコア vs 最適化効率

#### 統計検定
- 指標間FIDスコアの有意差検定（t-test、ANOVA）
- 効果量の定量化（Cohen's d）

## ⚡ 効率的実験実行戦略

### 並列処理最適化
```python
# experiments/fid_comparison/parallel_experiment.py

class ParallelFIDExperiment:
    def __init__(self, gpu_devices=['cuda:0', 'cuda:1']):
        self.gpu_devices = gpu_devices
    
    def run_parallel_experiments(self):
        # GPU別に指標を割り当て
        gpu_assignments = {
            'cuda:0': ['mse', 'l1', 'lpips'],
            'cuda:1': ['ssim', 'improved_ssim', 'psnr']
        }
        
        # 並列実行
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for gpu, metrics in gpu_assignments.items():
                future = executor.submit(self._run_gpu_experiments, gpu, metrics)
                futures.append(future)
            
            # 結果収集
            results = {}
            for future in concurrent.futures.as_completed(futures):
                gpu_results = future.result()
                results.update(gpu_results)
        
        return results
```

### チェックポイント戦略
- **自動保存間隔**: 50画像処理ごと
- **中断・再開**: 途中から実験再開可能
- **プログレストラッキング**: 詳細進捗レポート

### メモリ管理
- **バッチサイズ調整**: GPU メモリに応じた最適化
- **グラデーション蓄積**: メモリ不足時の代替戦略
- **定期的クリーンアップ**: PyTorch キャッシュクリア

## 📋 実装チェックリスト

### 実装タスク
- [ ] LatentOptimizerのLPIPS損失実装
- [ ] LatentOptimizerのSSIM損失実装
- [ ] LatentOptimizerのImproved SSIM損失実装
- [ ] LatentOptimizerのPSNR損失実装
- [ ] パイロット実験スクリプト作成
- [ ] 中規模実験スクリプト作成
- [ ] 全データセット実験スクリプト作成
- [ ] 並列処理システム実装
- [ ] 結果分析・可視化スクリプト作成

### テストタスク
- [ ] 新損失関数の単体テスト
- [ ] バッチ処理の統合テスト
- [ ] FID計算の精度テスト
- [ ] メモリ使用量のパフォーマンステスト
- [ ] 並列処理の負荷テスト

### ドキュメントタスク
- [ ] API ドキュメント更新
- [ ] 実験実行ガイド作成
- [ ] 結果解釈ガイド作成
- [ ] トラブルシューティングガイド作成

## 🔍 品質保証・検証計画

### 実験妥当性
1. **再現性**: 同一条件での複数回実行
2. **統計的有意性**: 十分なサンプルサイズ確保
3. **外部妥当性**: 他データセットでの検証推奨

### FID計算の信頼性
1. **baseline検証**: 元BSDS500同士でのFID ≈ 0確認
2. **参照実装比較**: pytorch-fidとの一致確認
3. **バッチサイズ依存性**: FID計算の安定性確認

### 実装品質
1. **数値安定性**: 極端なケースでの挙動確認
2. **メモリ効率**: 大規模処理でのリーク防止
3. **エラー処理**: 失敗時のグレースフル処理

## 🎯 成功基準

### 定量的成功基準
1. **完全実験実行**: 6指標すべてでの500枚最適化完了
2. **統計的有意差**: 指標間FIDスコアに有意な差を検出
3. **再現性**: 同一実験設定での結果一貫性（±2%以内）

### 定性的成功基準
1. **仮説検証**: 知覚的指標の優位性確認または反証
2. **洞察獲得**: 最適化指標選択のガイドライン確立
3. **手法評価**: 各指標の適用場面の明確化

## 🎉 期待される論文貢献

### 主要な発見（予想）
1. **最適化指標とFIDの相関**: 定量的関係の初報告
2. **知覚的最適化の有効性**: LPIPSベース最適化の定量的評価
3. **統計的指標の限界**: PSNR/MSE最適化の課題定量化

### 学術的インパクト
- Computer Vision: 画像品質評価手法の比較研究
- Machine Learning: 損失関数選択の定量的指針
- Image Processing: VAE最適化の実用化指針

## 🛠️ 次期拡張計画

### 実験拡張
1. **多様なデータセット**: CelebA、CIFAR-10等での検証
2. **異なるVAEモデル**: SD2.1、SDXL等での比較
3. **ハイブリッド損失**: 複数指標の加重平均

### システム拡張
1. **自動実験フレームワーク**: ハイパーパラメータ自動探索
2. **リアルタイムモニタリング**: 実験進捗のWeb監視
3. **分散処理**: クラスター環境での大規模実験

### 応用展開
1. **最適化指標推奨システム**: 画像特性に基づく指標自動選択
2. **カスタム損失関数**: 特定用途向けの損失関数設計
3. **教師なし評価**: 原画像なしでの品質予測

---

この実験により、VAE潜在表現最適化における評価指標選択の科学的根拠を確立し、将来の研究開発に資する定量的知見を獲得します。