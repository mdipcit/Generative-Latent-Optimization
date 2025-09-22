# BSDS500 FID Optimization Datasets v1.0.0

## 🎯 概要

Berkeley Segmentation Dataset 500全体（500画像）を用いたVAE潜在表現最適化実験の結果データセットです。異なる損失関数による最適化がFIDスコアに与える影響を包括的に検証した研究成果を公開します。

## 📊 実験結果

| 順位 | 最適化手法 | FIDスコア | ファイル名 | 特徴 |
|------|------------|-----------|------------|------|
| 🥇 | **LPIPS** | **13.10** | `bsds500_lpips_dataset.pt` | 最優秀知覚品質 |
| 🥈 | **PSNR** | **22.19** | `bsds500_psnr_dataset.pt` | 高効率・高品質 |
| 🥉 | **Improved SSIM** | **27.71** | `bsds500_improved_ssim_dataset.pt` | 構造保持特化 |

## 🔬 実験設定詳細

### データセット構成
- **総画像数**: 500枚
- **分割**: Train (200枚), Val (100枚), Test (200枚)
- **ソース**: Berkeley Segmentation Dataset 500

### 技術仕様
- **VAEモデル**: Stable Diffusion 1.5 VAE
- **最適化器**: Adam
- **デバイス**: CUDA対応GPU
- **総計算時間**: 約7時間

### 最適化パラメータ
- **LPIPS**: 150回反復, 学習率0.1 (4.0時間)
- **PSNR**: 50回反復, 学習率0.05 (1.45時間)  
- **Improved SSIM**: 50回反復, 学習率0.1 (1.45時間)

## 🎯 研究価値

### VAE最適化研究への貢献
- **ベンチマークデータセット**: 標準的な比較基準を提供
- **損失関数評価**: 異なる最適化目標の実証的比較
- **FID評価研究**: 大規模データセットでの定量的評価

### 主要発見
1. **LPIPS最適化が最優秀**: FID 13.10で知覚品質が最も高い
2. **PSNR最適化が高効率**: 短時間でFID 22.19の良好な結果
3. **構造保持とFIDは別指標**: Improved SSIMはFID 27.71だが構造保持に優秀

## 💻 利用方法

### 基本的な読み込み

```python
import torch

# データセット読み込み
dataset = torch.load('bsds500_lpips_dataset.pt')

# データ構造の確認
print(f"Keys: {dataset.keys()}")
print(f"Train images: {len(dataset['train'])}")
print(f"Val images: {len(dataset['val'])}")
print(f"Test images: {len(dataset['test'])}")
```

### PyTorchデータローダーでの利用

```python
from torch.utils.data import DataLoader

# データローダー作成
train_loader = DataLoader(
    dataset['train'], 
    batch_size=16, 
    shuffle=True
)

# バッチ処理
for batch in train_loader:
    # バッチ処理ロジック
    pass
```

## 📈 評価指標詳細

各データセットには以下の評価指標が含まれています：

- **FID (Fréchet Inception Distance)**: メイン評価指標
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error

## 📝 引用

この データセットを研究で使用される場合は、以下のように引用してください：

```
BSDS500 FID Optimization Datasets v1.0.0
Generative Latent Optimization Project
GitHub: https://github.com/mdipcit/Generative-Latent-Optimization
Release: v1.0.0-datasets (2025-09-22)
```

## ⚠️ 利用条件

- **データセット**: Berkeley Segmentation Dataset 500の利用条件に準拠
- **ライセンス**: 研究・教育目的での利用を推奨
- **商用利用**: 元データセットの利用条件を確認してください

## 🔗 関連リソース

- **プロジェクトリポジトリ**: [Generative-Latent-Optimization](https://github.com/mdipcit/Generative-Latent-Optimization)
- **元データセット**: [Berkeley Segmentation Dataset 500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- **技術詳細**: プロジェクトREADMEを参照

## 📞 サポート

質問や問題がある場合は、GitHubのIssuesでお知らせください。

---

**リリース日**: 2025年9月22日  
**バージョン**: v1.0.0  
**総ファイルサイズ**: 285MB