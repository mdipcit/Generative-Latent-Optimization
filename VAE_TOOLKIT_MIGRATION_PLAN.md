# VAE-Toolkit v0.2.0 移行計画書

## 📋 概要

本プロジェクトの汎用性の高いモジュール（DeviceManager、ModelConfig）をvae-toolkit v0.2.0に統合し、他のプロジェクトでも活用可能な包括的VAE基盤パッケージを構築する。

## 🎯 移行目的

### 主要目標
1. **機能重複の解消**: 同一機能の重複実装を排除
2. **汎用性向上**: 高機能モジュールの他プロジェクト活用
3. **保守性改善**: 基盤機能の一元管理
4. **エコシステム強化**: vae-toolkitの機能拡張

### 期待効果
- ✅ 他プロジェクトでの高度なデバイス管理機能利用
- ✅ VAE関連設定の一元化・統一
- ✅ コード重複削減・保守コスト低減
- ✅ vae-toolkitエコシステムの価値向上

## 🔍 移行対象モジュール分析

### 1. core/device_manager.py → vae-toolkit統合

**移行理由**:
- 既存vae-toolkit.get_optimal_device()の大幅拡張版
- VAE操作で頻繁に使用されるデバイス管理機能
- プロジェクト固有性なし（完全汎用）

**既存機能vs我々の機能**:
```python
# 既存vae-toolkit (基本)
VAELoader.get_optimal_device()  # 単純なデバイス選択

# 我々のDeviceManager (高機能)
DeviceManager.auto_select_device()     # メモリ考慮した最適選択  
DeviceManager.get_memory_summary()     # GPU メモリ監視
DeviceManager.ensure_same_device()     # 一括デバイス移動
DeviceManager.synchronize()            # CUDA同期管理
```

**統合メリット**: 既存機能を完全包含し、大幅な機能向上を提供

### 2. config/model_config.py → vae-toolkit統合

**移行理由**:
- 既存vae-toolkit/model_config.pyと**100%同一内容**
- 完全な重複実装（無意味な重複）
- VAE設定管理の自然な統合先

**重複状況**:
```python
# 両方とも同一の設定
MODEL_CONFIGS = {
    "sd14": {"repo_id": "CompVis/stable-diffusion-v1-4", ...},
    "sd15": {"repo_id": "runwayml/stable-diffusion-v1-5", ...}
}
```

**統合メリット**: 重複排除、設定の一元管理

## 🏗️ 技術的統合設計

### vae-toolkit v0.2.0 新構造

```
vae_toolkit/
├── __init__.py                 # 拡張API定義
├── image_utils.py             # 既存維持
├── vae_loader.py              # DeviceManager統合拡張
├── model_config.py            # 既存維持（重複排除）
└── device_manager.py          # NEW - 高機能デバイス管理
```

### API設計詳細

#### 1. **完全後方互換API**
```python
# v0.1.0 ユーザーのコードは変更不要
from vae_toolkit import VAELoader, load_and_preprocess_image, get_model_config

loader = VAELoader()
device = VAELoader.get_optimal_device()  # 既存staticmethod維持
vae, device = loader.load_sd_vae('sd15')  # 既存API維持
```

#### 2. **新機能API (v0.2.0)**
```python
# 新機能利用
from vae_toolkit import DeviceManager, auto_select_device

# 高機能デバイス管理
dm = DeviceManager()
print(dm.get_memory_summary())  # GPU メモリ監視
device = auto_select_device()   # メモリ考慮した最適選択

# VAELoaderとDeviceManager統合
loader = VAELoader(device_manager=dm)  # 拡張機能
```

#### 3. **統合されたvae_loader.py設計**
```python
class VAELoader:
    def __init__(self, device_manager: Optional[DeviceManager] = None):
        """
        Args:
            device_manager: オプショナルなDeviceManager（新機能）
        """
        self._model_cache = {}
        self.device_manager = device_manager  # NEW: 内部で高機能活用
        
    @staticmethod
    def get_optimal_device(preferred="auto") -> torch.device:
        """既存API維持（完全後方互換）"""
        return torch.device(DeviceManager.auto_select_device())
        
    def load_sd_vae(self, model_name="sd14", device="auto", **kwargs):
        """既存API + 内部DeviceManager活用で機能向上"""
        if self.device_manager:
            target_device = self.device_manager.get_optimal_device(device)
            # メモリ状況も考慮したロード
        else:
            target_device = self.get_optimal_device(device)  # fallback
```

## 📅 14日間詳細実装スケジュール

### Week 1: 準備・実装・基本検証

#### **Day 1-2: 準備段階**
```bash
# 環境セットアップ
git clone https://github.com/your-username/vae-toolkit.git
cd vae-toolkit
git checkout -b feature/v0.2.0-enhanced

# 現状確認
python -m pytest tests/ -v
uv sync

# 影響範囲調査
grep -r "DeviceManager\|device_manager\|model_config" ../Generative-Latent-Optimization/src/
```

#### **Day 3-4: DeviceManager統合実装**
```bash
# ファイル移行
cp ../Generative-Latent-Optimization/src/generative_latent_optimization/core/device_manager.py ./vae_toolkit/

# 統合実装
# - device_manager.py のimport調整
# - vae_loader.py にDeviceManager統合
# - __init__.py に新API追加
```

#### **Day 5-6: テスト作成・基本検証**
```bash
# テスト実装
# tests/test_device_manager.py
# tests/test_backward_compatibility.py  
# tests/test_vae_loader_enhanced.py

# 基本動作確認
python -m pytest tests/test_backward_compatibility.py -v
python -m pytest tests/test_device_manager.py -v
```

#### **Day 7: Week 1 総合検証**
```bash
# 全テスト実行
python -m pytest tests/ -v --tb=short

# パフォーマンステスト
python tests/test_performance_benchmarks.py

# Week 1 完了判定
- ✅ 全既存テスト通過
- ✅ 新機能テスト通過  
- ✅ パフォーマンス回帰なし
```

---

### Week 2: 統合検証・リリース・本プロジェクト更新

#### **Day 8-9: 本プロジェクト統合テスト**
```bash
# 本プロジェクトでの動作確認
cd ../Generative-Latent-Optimization

# pyproject.toml 更新（ローカルパス使用）
dependencies = ["vae-toolkit @ file:///path/to/vae-toolkit"]

# 既存テスト全実行
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv sync
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python tests/unit/test_vae_fixed.py
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python tests/integration/test_optimization_integration.py
```

#### **Day 10-11: import文更新・クリーンアップ**
```bash
# import文一括更新
find src/ -name "*.py" -exec sed -i 's/from \.\.core\.device_manager import DeviceManager/from vae_toolkit import DeviceManager/g' {} \;

# 移行完了後のクリーンアップ
rm -rf src/generative_latent_optimization/core/
rm -rf src/generative_latent_optimization/config/model_config.py

# __init__.py 更新（import削除）
# 依存関係整理
```

#### **Day 12: ドキュメント・リリース準備**
```bash
# vae-toolkit文書更新
cd vae-toolkit

# README.md更新（v0.2.0新機能説明）
# CHANGELOG.md作成
# pyproject.toml最終確認
# VERSION確認

# リリースファイル準備
uv build
```

#### **Day 13-14: リリース・検証完了**
```bash
# vae-toolkit v0.2.0 リリース
git tag v0.2.0
git push origin v0.2.0
twine upload dist/vae_toolkit-0.2.0*

# 本プロジェクト依存関係をPyPI版に更新
cd ../Generative-Latent-Optimization
# pyproject.toml更新: "vae-toolkit>=0.2.0"
uv sync

# 最終統合テスト
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python -c "
from vae_toolkit import DeviceManager, VAELoader
from generative_latent_optimization import LatentOptimizer
print('✅ 統合完了')
"
```

## 🧪 包括的テスト戦略

### 1. **互換性保証テストスイート**

#### A. 既存API完全互換性テスト
```python
# tests/test_backward_compatibility.py
class TestBackwardCompatibility:
    def test_v01_imports_unchanged(self):
        """v0.1.0のimport文が全て動作"""
        from vae_toolkit import (
            VAELoader, load_and_preprocess_image, tensor_to_pil,
            ImageProcessor, get_model_config, add_model_config
        )
        
    def test_v01_api_signatures(self):
        """v0.1.0のAPI署名が変更されていない"""
        from vae_toolkit import VAELoader
        
        # 既存staticmethod維持
        device = VAELoader.get_optimal_device()
        assert isinstance(device, torch.device)
        
        # 既存constructor維持
        loader = VAELoader()  # 引数なしで動作
        assert loader is not None
        
    def test_v01_workflows_unchanged(self):
        """v0.1.0のワークフローが変更なく動作"""
        # 実際のユースケース模倣
        from vae_toolkit import VAELoader, load_and_preprocess_image
        
        loader = VAELoader()
        tensor, pil = load_and_preprocess_image('test_image.png')
        assert tensor.shape[0] == 1  # バッチ次元
```

#### B. 新機能動作テスト
```python
# tests/test_device_manager.py  
class TestDeviceManager:
    def test_device_manager_initialization(self):
        """DeviceManager基本動作"""
        from vae_toolkit import DeviceManager
        
        dm = DeviceManager()
        assert dm.device is not None
        
    def test_advanced_device_features(self):
        """高機能デバイス管理"""
        from vae_toolkit import DeviceManager
        
        dm = DeviceManager()
        
        # メモリサマリー（新機能）
        memory_info = dm.get_memory_summary()
        # GPU使用時のみ None以外を返す
        
        # 自動デバイス選択（新機能）
        device = dm.auto_select_device()
        assert device in ['cpu', 'cuda:0', 'cuda:1']  # 有効なデバイス
        
    def test_vae_loader_device_manager_integration(self):
        """VAELoaderとDeviceManager統合"""
        from vae_toolkit import VAELoader, DeviceManager
        
        dm = DeviceManager()
        loader = VAELoader(device_manager=dm)  # 新機能
        assert loader.device_manager is dm
```

### 2. **パフォーマンス回帰テスト**
```python
# tests/test_performance_regression.py
class TestPerformanceRegression:
    def test_load_time_regression(self):
        """VAE読み込み時間の回帰なし"""
        import time
        from vae_toolkit import VAELoader
        
        start = time.time()
        loader = VAELoader()
        init_time = time.time() - start
        
        assert init_time < 0.1  # 100ms以内（基準値）
        
    def test_memory_usage_regression(self):
        """メモリ使用量の回帰なし"""
        import psutil
        from vae_toolkit import DeviceManager
        
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        dm = DeviceManager()
        _ = dm.get_device_info()
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        assert memory_increase < 10 * 1024 * 1024  # 10MB以内
```

### 3. **統合テスト**
```python
# tests/test_integration.py
class TestIntegration:
    def test_end_to_end_workflow(self):
        """エンドツーエンドワークフロー"""
        from vae_toolkit import VAELoader, DeviceManager, load_and_preprocess_image
        
        # 1. デバイス管理
        dm = DeviceManager()
        optimal_device = dm.auto_select_device()
        
        # 2. VAEローダー（統合）
        loader = VAELoader(device_manager=dm)
        
        # 3. 画像処理
        tensor, pil = load_and_preprocess_image('test.png')
        
        # 4. VAEロード（実際のテストではスキップ可能）
        # vae, device = loader.load_sd_vae('sd15')
        
        assert tensor is not None
        assert dm.device == optimal_device
```

## 🛠️ 詳細実装ガイド

### Phase 1: vae-toolkit準備・実装

#### Step 1.1: 環境準備
```bash
# 作業ディレクトリセットアップ
mkdir -p ~/vae_toolkit_migration
cd ~/vae_toolkit_migration

# vae-toolkit取得
git clone https://github.com/your-username/vae-toolkit.git
cd vae-toolkit
git checkout -b feature/v0.2.0-enhanced

# 開発環境確認
uv sync
python -c "from vae_toolkit import VAELoader; print('✅ 基本動作OK')"
```

#### Step 1.2: DeviceManager統合実装
```bash
# ファイルコピー
cp ../../Generative-Latent-Optimization/src/generative_latent_optimization/core/device_manager.py ./vae_toolkit/

# device_manager.py 調整実装
# - パッケージ内import調整
# - vae-toolkit専用最適化
# - 不要依存関係削除
```

**device_manager.py実装調整**:
```python
"""Enhanced device management for VAE operations."""
import torch
from typing import Dict, Any, Optional, List

# 既存実装をほぼそのまま利用
# 変更点: 内部import削除、torch基本機能のみ使用

class DeviceManager:
    def __init__(self, device: str = 'cuda'):
        """Initialize device manager"""
        self.device = self._detect_optimal_device(device)
        self._log_device_info()
    
    # ... 既存実装維持（400行程度）

# 後方互換性関数追加
def auto_select_device() -> str:
    """Convenience function for auto device selection"""
    return DeviceManager.auto_select_device()
```

#### Step 1.3: VAELoader拡張
```python
# vae_toolkit/vae_loader.py 拡張
from typing import Optional
from .device_manager import DeviceManager  # NEW import

class VAELoader:
    def __init__(self, device_manager: Optional[DeviceManager] = None):
        """
        Initialize VAE loader
        
        Args:
            device_manager: オプショナルなDeviceManager（v0.2.0新機能）
                           Noneの場合は既存動作維持
        """
        self._model_cache = {}
        self.device_manager = device_manager  # NEW
        
    @staticmethod
    def get_optimal_device(preferred="auto") -> torch.device:
        """既存staticmethod維持（完全後方互換）"""
        # 内部実装をDeviceManagerに委譲（透明な改善）
        return torch.device(DeviceManager().auto_select_device())
        
    def load_sd_vae(self, model_name="sd14", device="auto", **kwargs):
        """既存API + DeviceManager統合による機能向上"""
        
        # デバイス選択（拡張）
        if self.device_manager:
            target_device = self.device_manager.get_optimal_device(device)
            # メモリサマリーも取得可能
            if self.device_manager.get_memory_summary():
                logger.info(f"GPU Memory: {self.device_manager.get_memory_summary()}")
        else:
            target_device = self.get_optimal_device(device)  # 既存fallback
            
        # 以下は既存実装維持
        config = get_model_config(model_name)
        # ... VAEロード処理
```

#### Step 1.4: __init__.py更新
```python
# vae_toolkit/__init__.py v0.2.0
"""
VAE Toolkit v0.2.0 - Enhanced with advanced device management

This version maintains 100% backward compatibility while adding
powerful device management capabilities.
"""

__version__ = "0.2.0"
__author__ = "Yus314"
__email__ = "shizhaoyoujie@gmail.com"

# 既存API（完全維持）
from .image_utils import (
    load_and_preprocess_image, tensor_to_pil, pil_to_tensor,
    ImageProcessor, ImageProcessingError, 
    DEFAULT_PROCESSOR, SD_PROCESSOR
)
from .vae_loader import VAELoader
from .model_config import (
    get_model_config, get_all_model_configs, list_available_models,
    add_model_config, get_default_token
)

# NEW v0.2.0 API
from .device_manager import DeviceManager, auto_select_device

__all__ = [
    # Package metadata
    "__version__", "__author__", "__email__",
    
    # v0.1.0 既存API（完全互換）
    "load_and_preprocess_image", "tensor_to_pil", "pil_to_tensor",
    "ImageProcessor", "ImageProcessingError", "DEFAULT_PROCESSOR", "SD_PROCESSOR",
    "VAELoader",
    "get_model_config", "get_all_model_configs", "list_available_models", 
    "add_model_config", "get_default_token",
    
    # v0.2.0 新機能
    "DeviceManager", "auto_select_device"
]
```

### Phase 2: 検証・統合・リリース

#### Step 2.1: 包括的テスト実行
```bash
# vae-toolkit単体テスト
cd vae-toolkit
python -m pytest tests/ -v -x  # 失敗時即停止

# 本プロジェクト統合テスト  
cd ../Generative-Latent-Optimization
uv sync  # 新vae-toolkit反映
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python tests/unit/test_vae_fixed.py
```

#### Step 2.2: 本プロジェクト更新実装
```bash
# import文の一括更新
find src/ -name "*.py" -print0 | xargs -0 sed -i 's/from \.\.core\.device_manager import DeviceManager/from vae_toolkit import DeviceManager/g'
find src/ -name "*.py" -print0 | xargs -0 sed -i 's/from \.\.config\.model_config import/from vae_toolkit import/g'

# __init__.py から削除されるimport削除
# core/, config/model_config.py 削除
rm -rf src/generative_latent_optimization/core/
```

#### Step 2.3: リリース実行
```bash
# vae-toolkit v0.2.0 リリース
cd vae-toolkit
git add .
git commit -m "feat: add DeviceManager and enhanced device management for v0.2.0"
git tag v0.2.0
git push origin feature/v0.2.0-enhanced
git push origin v0.2.0

# PyPI配布
uv build
twine upload dist/vae_toolkit-0.2.0*
```

## ⚠️ リスク管理とロールバック戦略

### 🚨 識別されたリスク

#### **リスク 1: 既存ユーザーの互換性破壊**
- **影響度**: 高
- **対策**: 100%後方互換性維持、段階的移行サポート
- **検証**: 既存APIの全機能テスト
- **回避**: 破壊的変更の完全禁止

#### **リスク 2: パフォーマンス回帰**  
- **影響度**: 中
- **対策**: 詳細ベンチマーク、最適化実装
- **検証**: 処理時間・メモリ使用量測定
- **回避**: 新機能のオーバーヘッド最小化

#### **リスク 3: 依存関係の複雑化**
- **影響度**: 低
- **対策**: torch以外の新依存関係なし
- **検証**: 依存関係ツリー確認
- **回避**: 軽量実装優先

### 🔄 ロールバック手順

#### **Level 1: 緊急ロールバック** (重大問題発生時)
```bash
# PyPI上でv0.1.0を再推奨
pip install vae-toolkit==0.1.0

# 本プロジェクトの緊急復旧
git revert [移行commit]
# または
git checkout [移行前commit]
```

#### **Level 2: 段階的ロールバック** (部分的問題)
```python
# 新機能のみ無効化
from vae_toolkit import VAELoader  # v0.1.0互換API使用
# DeviceManager使用停止
```

#### **Level 3: 機能別無効化** (特定機能問題)
```python
# 問題のある機能のみ迂回
loader = VAELoader()  # device_manager=None（デフォルト）
device = VAELoader.get_optimal_device()  # 既存機能使用
```

## 📈 成功指標と検証基準

### 🎯 **完了判定基準**

#### **Week 1 完了基準**
- ✅ vae-toolkit全既存テスト通過（100%）
- ✅ 新機能テスト通過（100%）
- ✅ パフォーマンス回帰 <5%
- ✅ メモリ使用量増加 <10MB
- ✅ API互換性確認完了

#### **Week 2 完了基準**  
- ✅ 本プロジェクト統合成功
- ✅ 全既存テストスイート通過
- ✅ PyPI配布成功
- ✅ ドキュメント更新完了

#### **最終成功指標**
- ✅ 他プロジェクトでの利用確認
- ✅ 既存ユーザーからの問題報告なし
- ✅ 新機能の有効活用事例作成

### 📊 **品質ゲート**

| Phase | 必須条件 | 進行判定 |
|-------|----------|----------|
| 実装完了 | 全テスト通過 | ✅ → Phase 2 |
| 統合完了 | 本プロジェクト動作確認 | ✅ → リリース |
| リリース | PyPI配布成功 | ✅ → 完了 |

## 🔧 実装チェックリスト

### **Phase 1: vae-toolkit実装**
- [ ] DeviceManager移行・統合
- [ ] VAELoader拡張実装
- [ ] __init__.py API更新
- [ ] 互換性テスト作成
- [ ] 新機能テスト作成
- [ ] パフォーマンステスト作成

### **Phase 2: 統合・検証**
- [ ] 本プロジェクト統合テスト
- [ ] import文一括更新実装
- [ ] core/, config/削除・クリーンアップ
- [ ] ドキュメント更新
- [ ] PyPI配布準備

### **Phase 3: リリース・完了**
- [ ] vae-toolkit v0.2.0 リリース
- [ ] 本プロジェクト依存関係更新
- [ ] 最終統合検証
- [ ] 成功指標達成確認

## 💡 期待される最終状態

### **vae-toolkit v0.2.0 利用例**

**基本利用（既存互換）**:
```python
# v0.1.0ユーザーのコードは無変更で動作
from vae_toolkit import VAELoader, load_and_preprocess_image

loader = VAELoader()
vae, device = loader.load_sd_vae('sd15')
```

**拡張利用（新機能）**:
```python
# 高機能デバイス管理活用
from vae_toolkit import DeviceManager, VAELoader

dm = DeviceManager()
print(f"GPU状況: {dm.get_memory_summary()}")
optimal_device = dm.auto_select_device()  # メモリ考慮

loader = VAELoader(device_manager=dm)  # 統合利用
vae, device = loader.load_sd_vae('sd15', device=optimal_device)
```

**他プロジェクトでの活用**:
```python
# 新規VAEプロジェクト
pip install vae-toolkit>=0.2.0

from vae_toolkit import DeviceManager, VAELoader
# 高度なデバイス管理がすぐに利用可能
```

### **本プロジェクトでの変更**

**Before**:
```python
from .core.device_manager import DeviceManager
from .config.model_config import get_model_config
```

**After**:
```python
from vae_toolkit import DeviceManager, get_model_config
```

---

## 🎉 移行完了による価値創出

1. **✅ 機能統合**: 重複排除、一元管理
2. **✅ 汎用化**: 他プロジェクトでの高機能活用
3. **✅ エコシステム**: vae-toolkitの価値向上
4. **✅ 保守性**: 基盤機能の中央集権管理

この移行により、vae-toolkitが「VAE操作の包括的基盤パッケージ」として完成し、プロジェクト間での効率的な機能共有が実現されます。