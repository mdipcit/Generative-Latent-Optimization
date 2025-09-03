# 🚀 BSDS500全データセット比較実験実行ガイド

## 📋 実験概要

**目的**: BSDS500全500枚を対象とした3つの損失関数（PSNR, SSIM, LPIPS）の決定的性能比較  
**予想実行時間**: 約42-50時間（GPU使用時）  
**必要リソース**: CUDA対応GPU、約100GB空き容量  

## 🛠️ 事前準備

### 1. 環境変数設定
```bash
# 必須：Hugging Face認証トークン
export HF_TOKEN="your_huggingface_token_here"

# 必須：BSDS500データセットパス
export BSDS500_PATH="/path/to/bsds500/dataset"
```

### 2. 環境確認
```bash
# Nix環境に入る
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure

# 依存関係同期
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv sync

# GPU確認
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. ディスク容量確認
```bash
# 空き容量確認（約100GB必要）
df -h .

# 出力ディレクトリ作成
mkdir -p experiments/full_comparison
```

## 🎯 実行方法

### オプション1: 全自動実行（推奨）

```bash
# 全実験を自動実行（42-50時間）
cd experiments/full_comparison
HF_TOKEN="$HF_TOKEN" BSDS500_PATH="$BSDS500_PATH" \
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python run_full_bsds500_comparison.py
```

**特徴**:
- 全3データセット作成 + 評価 + レポート生成
- 自動的な中間保存・エラー回復
- 進捗監視・残り時間推定
- 完全無人実行可能

### オプション2: 段階的実行（制御重視）

#### Step 1: 個別データセット作成

```bash
cd experiments/full_comparison

# PSNR最適化データセット（推奨最高性能）
HF_TOKEN="$HF_TOKEN" BSDS500_PATH="$BSDS500_PATH" \
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python run_individual_optimization.py psnr

# SSIM最適化データセット（構造保持重視）
HF_TOKEN="$HF_TOKEN" BSDS500_PATH="$BSDS500_PATH" \
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python run_individual_optimization.py improved_ssim

# LPIPS最適化データセット（知覚品質重視）
HF_TOKEN="$HF_TOKEN" BSDS500_PATH="$BSDS500_PATH" \
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python run_individual_optimization.py lpips
```

#### Step 2: クロス評価実行

```bash
# 作成されたデータセットの包括的評価
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python run_cross_evaluation.py
```

#### Step 3: 最終レポート生成

```bash
# 評価結果に基づく包括的レポート作成
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python generate_final_report.py
```

### オプション3: 高速テスト実行

```bash
# 設定確認のみ（実際の最適化は実行しない）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python run_individual_optimization.py psnr --dry-run
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python run_individual_optimization.py improved_ssim --dry-run
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python run_individual_optimization.py lpips --dry-run
```

## 📊 実行進捗の監視

### ログファイル確認
```bash
# リアルタイムログ監視
tail -f experiments/full_comparison/full_comparison_experiment.log

# 最適化進捗確認
cat experiments/full_comparison/intermediate_results.json | jq '.psnr.processing_time_hours'
```

### 中間結果確認
```bash
# データセット作成状況
ls -la experiments/full_comparison/

# 個別最適化結果
cat experiments/full_comparison/psnr_dataset/optimization_result.json | jq '.processing_time_hours'
cat experiments/full_comparison/improved_ssim_dataset/optimization_result.json | jq '.processing_time_hours'
cat experiments/full_comparison/lpips_dataset/optimization_result.json | jq '.processing_time_hours'
```

### 実時間性能監視
```bash
# GPU使用率監視
nvidia-smi -l 1

# CPU・メモリ使用率
htop

# ディスク使用量監視  
watch df -h .
```

## 📈 予想される結果

### 既存60枚実験に基づく予測

| 損失関数 | 予想FIDスコア | 処理時間予測 | 信頼度 |
|----------|---------------|--------------|--------|
| **PSNR** | 15-25 | 8-10時間 | 高（既存最高性能） |
| **Improved SSIM** | 35-45 | 8-10時間 | 中（構造特化） |
| **LPIPS** | 20-30 | 20-25時間 | 中（知覚特化） |

### スケーリング効果予測
- **統計的安定性**: サンプル数8倍増による信頼性向上
- **処理効率**: 並列化・最適化による実時間短縮可能性
- **品質向上**: 大規模データによる更なる品質改善期待

## ⚠️ トラブルシューティング

### よくあるエラーと対処法

#### 1. 環境変数エラー
```bash
# エラー: BSDS500_PATH not set
export BSDS500_PATH="/path/to/your/bsds500"

# エラー: HF_TOKEN not set  
export HF_TOKEN="your_huggingface_token"
```

#### 2. GPU/CUDA関連エラー
```bash
# CUDA OOM エラー
# 対処: 他のGPUプロセスを終了
pkill -f python

# GPU使用状況確認
nvidia-smi
```

#### 3. ディスク容量不足
```bash
# 容量確認
df -h .

# 古い実験データ削除
rm -rf experiments/old_results/

# 部分的実行（画像数制限）
# run_individual_optimization.py を編集してmax_images_per_splitを調整
```

#### 4. 処理中断・再開
```bash
# 実験ログで最後に処理された画像確認
tail experiments/full_comparison/full_comparison_experiment.log

# 中間結果から状況確認
cat experiments/full_comparison/intermediate_results.json

# 個別に再実行
HF_TOKEN="$HF_TOKEN" BSDS500_PATH="$BSDS500_PATH" \
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python run_individual_optimization.py [failed_loss_function]
```

#### 5. メモリ不足エラー
```python
# スクリプト内でバッチサイズ調整
# run_individual_optimization.py の process_bsds500_dataset 呼び出し時に
# max_images_per_split パラメータを追加

datasets = process_bsds500_dataset(
    bsds500_path=bsds500_path,
    output_path=output_path,
    config=config,
    max_images_per_split=50,  # デフォルトの全画像から50枚に制限
    create_pytorch_dataset=True,
    create_png_dataset=True
)
```

## 🔄 実行バリエーション

### 高速プロトタイプ実行
```bash
# 各損失関数で10枚のみテスト
# run_individual_optimization.py を編集
# max_images_per_split=10 に設定して実行
```

### 特定損失関数のみ
```bash
# PSNR最適化のみ実行（最高性能期待）
HF_TOKEN="$HF_TOKEN" BSDS500_PATH="$BSDS500_PATH" \
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python run_individual_optimization.py psnr

# 結果確認
cat experiments/full_comparison/psnr_dataset/optimization_result.json | jq '.processing_time_hours'
```

### 評価のみ実行
```bash
# 既存データセットの再評価
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python run_cross_evaluation.py

# レポート再生成
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python generate_final_report.py
```

## 📊 期待される成果物

### 1. データセット（約60GB）
```
experiments/full_comparison/
├── psnr_dataset/
│   ├── pytorch/bsds500_optimized_psnr.pt
│   └── png/ (500枚)
├── improved_ssim_dataset/
│   ├── pytorch/bsds500_optimized_ssim.pt  
│   └── png/ (500枚)
└── lpips_dataset/
    ├── pytorch/bsds500_optimized_lpips.pt
    └── png/ (500枚)
```

### 2. 評価結果
```
experiments/full_comparison/
├── cross_evaluation_results.json
├── intermediate_evaluation.json
└── final_experiment_results.json
```

### 3. 包括的レポート
```
experiments/full_comparison/
├── FULL_BSDS500_COMPARISON_FINAL_REPORT.md
└── full_comparison_experiment.log
```

## 🎯 成功確認チェックリスト

### データセット作成成功
- [ ] `experiments/full_comparison/psnr_dataset/png/` に約500画像存在
- [ ] `experiments/full_comparison/improved_ssim_dataset/png/` に約500画像存在  
- [ ] `experiments/full_comparison/lpips_dataset/png/` に約500画像存在
- [ ] 各データセットの `optimization_result.json` で status='success'

### 評価実行成功
- [ ] `cross_evaluation_results.json` ファイル存在
- [ ] 各損失関数の評価で status='success'
- [ ] FIDスコアが妥当な範囲（10-80）

### レポート生成成功
- [ ] `FULL_BSDS500_COMPARISON_FINAL_REPORT.md` ファイル存在
- [ ] レポート内にFIDスコアランキング表示
- [ ] 実用的推奨事項セクション存在

## 🚨 重要注意事項

### 処理時間管理
- **LPIPS**: 他の3倍の時間（約25時間）
- **推奨実行順**: PSNR → Improved SSIM → LPIPS
- **中断リスク**: 長時間処理のため電源・ネットワーク安定性確保

### リソース管理
- **GPU メモリ**: 定期的な `torch.cuda.empty_cache()` 実行
- **ディスク容量**: 処理中は約120GB必要（最終的に100GB）
- **CPU負荷**: 単一プロセスで高負荷継続

### データ整合性
- **チェックポイント**: 10画像ごとに自動保存
- **エラー回復**: 失敗画像をスキップして継続
- **結果検証**: 各段階で画像数・品質チェック

## 🔧 高度な設定

### 並列実行（複数GPU環境）
```python
# run_individual_optimization.py を編集
# デバイス指定を動的に変更
device_mapping = {
    'psnr': 'cuda:0',
    'improved_ssim': 'cuda:1', 
    'lpips': 'cuda:2'
}
```

### パフォーマンスチューニング
```python
# より高速な設定（品質妥協）
quick_configs = {
    'psnr': {'iterations': 30, 'learning_rate': 0.1},
    'improved_ssim': {'iterations': 30, 'learning_rate': 0.2},
    'lpips': {'iterations': 100, 'learning_rate': 0.15}
}
```

### メモリ最適化
```python
# 低メモリ環境用設定
memory_efficient_config = {
    'checkpoint_interval': 5,    # より頻繁な保存
    'max_images_per_split': 25,  # バッチサイズ削減
    'enable_mixed_precision': True  # メモリ効率向上
}
```

## 📈 結果の活用

### 最適手法の決定
```python
# 結果に基づく推奨設定の使用
from generative_latent_optimization import OptimizationConfig

# 実験結果から最高性能設定を抽出
with open('experiments/full_comparison/cross_evaluation_results.json') as f:
    results = json.load(f)

best_method = results['summary']['best_method']['loss_function']
print(f"Recommended loss function: {best_method}")
```

### 本格運用への移行
```python
# 実験結果を元にした本格的なデータセット作成
from generative_latent_optimization.workflows import optimize_bsds500_full

# 最高性能設定での完全データセット作成
datasets = optimize_bsds500_full(
    output_path="./production_dataset",
    iterations=50,
    learning_rate=0.05,  # 実験結果に基づく最適値
    create_pytorch=True,
    create_png=True
)
```

### 学術利用
```markdown
実験結果は学術論文・会議発表で以下のように引用可能:

"我々はBerkeley Segmentation Dataset 500の全500枚を対象に、
PSNR、SSIM、LPIPSの3つの損失関数によるVAE潜在表現最適化を実行し、
FIDスコアによる包括的性能比較を行った。その結果、PSNR最適化が
最も優秀な性能（FID: XX.XX）を示し、従来手法より最大XX%の改善を
達成した。"
```

## 🎯 次期展開

### Phase 2計画: 拡張実験
1. **異なるVAEモデル**: SD2.0, SD2.1, SDXL
2. **追加データセット**: CelebA-HQ, ImageNet-1K subset
3. **新損失関数**: Multi-scale SSIM, Feature matching

### Phase 3計画: 実用化
1. **ワンクリック実行環境**: Webダッシュボード
2. **リアルタイム監視**: 進捗・品質・リソース
3. **自動パラメータ調整**: 動的最適化

---

**実行前最終チェック**:
- [ ] `HF_TOKEN` 設定済み
- [ ] `BSDS500_PATH` 正しく設定
- [ ] 100GB以上の空き容量確認
- [ ] CUDA GPU利用可能
- [ ] 長時間実行のための安定環境確保

**予想実行時間**: PSNR (8h) + Improved SSIM (8h) + LPIPS (25h) + 評価 (2h) = **合計約43時間**