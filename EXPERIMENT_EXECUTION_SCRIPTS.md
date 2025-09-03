# FID最適化実験 実行スクリプト設計

## 🎯 スクリプト構成

### 実行段階別スクリプト
1. **pilot_experiment.py**: 小規模テスト（30枚）
2. **medium_scale_experiment.py**: 中規模実験（150枚）
3. **full_scale_experiment.py**: 全データセット実験（500枚）
4. **analyze_results.py**: 結果分析・レポート生成

## 📁 推奨ディレクトリ構造

```
experiments/
├── fid_comparison/
│   ├── pilot_experiment.py
│   ├── medium_scale_experiment.py
│   ├── full_scale_experiment.py
│   ├── analyze_results.py
│   ├── utils/
│   │   ├── experiment_utils.py
│   │   └── visualization_utils.py
│   └── results/
│       ├── pilot/
│       ├── medium/
│       ├── full/
│       └── analysis/
```

## 🚀 実行スクリプト詳細設計

### 1. experiments/fid_comparison/pilot_experiment.py
```python
#!/usr/bin/env python3
"""
FID最適化実験: パイロット版
小規模テストでシステム動作確認と処理時間推定
"""

import os
import sys
import time
import json
from pathlib import Path
import logging
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from generative_latent_optimization import OptimizationConfig
from generative_latent_optimization.workflows import process_bsds500_dataset
from generative_latent_optimization.metrics import DatasetFIDEvaluator

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PilotExperiment:
    """パイロット実験制御クラス"""
    
    def __init__(self, max_images_per_split=10):
        self.max_images = max_images_per_split
        self.output_base = Path(__file__).parent / "results" / "pilot"
        self.bsds500_path = os.environ.get('BSDS500_PATH')
        
        # 現在実装済みの指標のみテスト
        self.test_metrics = ['mse', 'l1']
        
        # 実験環境検証
        self._validate_environment()
    
    def _validate_environment(self):
        """実験環境の妥当性チェック"""
        if not self.bsds500_path:
            raise ValueError("BSDS500_PATH environment variable not set")
        
        if not Path(self.bsds500_path).exists():
            raise FileNotFoundError(f"BSDS500 dataset not found: {self.bsds500_path}")
        
        # HF_TOKEN確認
        if not os.environ.get('HF_TOKEN'):
            logger.warning("HF_TOKEN not set - VAE model loading may fail")
    
    def run_experiment(self):
        """パイロット実験実行"""
        
        logger.info("🚀 Starting FID Pilot Experiment")
        logger.info(f"   Max images per split: {self.max_images}")
        logger.info(f"   Output directory: {self.output_base}")
        logger.info(f"   Test metrics: {self.test_metrics}")
        
        experiment_results = {}
        total_start_time = time.time()
        
        for metric in self.test_metrics:
            logger.info(f"\n🔄 Testing {metric} optimization")
            metric_start_time = time.time()
            
            try:
                result = self._run_single_metric_experiment(metric)
                experiment_results[metric] = result
                
                metric_duration = time.time() - metric_start_time
                logger.info(f"✅ {metric} completed: FID = {result['fid_score']:.2f} ({metric_duration/60:.1f}min)")
                
            except Exception as e:
                logger.error(f"❌ {metric} experiment failed: {e}")
                experiment_results[metric] = {'error': str(e)}
        
        total_duration = time.time() - total_start_time
        
        # 結果保存
        results_summary = {
            'experiment_type': 'pilot',
            'max_images_per_split': self.max_images,
            'total_duration_minutes': total_duration / 60,
            'timestamp': datetime.now().isoformat(),
            'results': experiment_results
        }
        
        self._save_results(results_summary)
        self._print_summary(experiment_results, total_duration)
        
        return experiment_results
    
    def _run_single_metric_experiment(self, metric):
        """単一指標での実験実行"""
        
        # 最適化設定
        config = OptimizationConfig(
            iterations=50,  # パイロットでは短縮
            learning_rate=0.4,
            loss_function=metric,
            device='cuda'
        )
        
        # 出力パス
        metric_output_dir = self.output_base / metric
        metric_output_dir.mkdir(parents=True, exist_ok=True)
        
        # データセット最適化
        start_time = time.time()
        datasets = process_bsds500_dataset(
            self.bsds500_path,
            metric_output_dir / "dataset",
            config,
            max_images_per_split=self.max_images,
            create_pytorch_dataset=False,  # パイロットでは軽量化
            create_png_dataset=True
        )
        processing_time = time.time() - start_time
        
        # FID評価
        fid_evaluator = DatasetFIDEvaluator(device='cuda')
        fid_result = fid_evaluator.evaluate_created_dataset_vs_original(
            datasets['png'], 
            self.bsds500_path
        )
        
        return {
            'fid_score': fid_result.fid_score,
            'processing_time_seconds': processing_time,
            'total_images': fid_result.total_images,
            'dataset_path': str(datasets['png']),
            'optimization_config': {
                'iterations': config.iterations,
                'learning_rate': config.learning_rate,
                'loss_function': config.loss_function
            }
        }
    
    def _save_results(self, results):
        """実験結果の保存"""
        output_file = self.output_base / "pilot_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
    
    def _print_summary(self, results, total_duration):
        """結果サマリーの表示"""
        print("\n" + "="*60)
        print("📊 FID Pilot Experiment Results")
        print("="*60)
        
        for metric, result in results.items():
            if 'error' in result:
                print(f"❌ {metric.upper()}: FAILED - {result['error']}")
            else:
                fid = result['fid_score']
                time_min = result['processing_time_seconds'] / 60
                print(f"✅ {metric.upper()}: FID = {fid:.2f} ({time_min:.1f}min)")
        
        print(f"\n⏱️  Total experiment time: {total_duration/60:.1f} minutes")
        print("="*60)

def main():
    """メイン実行関数"""
    experiment = PilotExperiment(max_images_per_split=10)
    results = experiment.run_experiment()
    
    # 成功判定
    successful_count = len([r for r in results.values() if 'error' not in r])
    total_count = len(results)
    
    print(f"\n🎯 Experiment completed: {successful_count}/{total_count} metrics tested successfully")
    
    if successful_count == total_count:
        print("✅ Ready for medium-scale experiment")
    else:
        print("⚠️ Some metrics failed - check logs and fix issues before proceeding")

if __name__ == "__main__":
    main()
```

### 2. experiments/fid_comparison/utils/experiment_utils.py
```python
#!/usr/bin/env python3
"""FID実験用ユーティリティ関数"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import torch

logger = logging.getLogger(__name__)

class ExperimentValidator:
    """実験環境・設定の検証クラス"""
    
    @staticmethod
    def validate_environment():
        """実験実行環境の検証"""
        issues = []
        
        # BSDS500パス確認
        bsds500_path = os.environ.get('BSDS500_PATH')
        if not bsds500_path:
            issues.append("BSDS500_PATH environment variable not set")
        elif not Path(bsds500_path).exists():
            issues.append(f"BSDS500 dataset not found: {bsds500_path}")
        
        # HF_TOKEN確認
        if not os.environ.get('HF_TOKEN'):
            issues.append("HF_TOKEN environment variable not set")
        
        # GPU確認
        if not torch.cuda.is_available():
            issues.append("CUDA not available - experiments will run slowly on CPU")
        
        # 必要パッケージ確認
        try:
            import lpips
        except ImportError:
            issues.append("lpips package not available - LPIPS experiments will fail")
        
        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
        except ImportError:
            issues.append("torchmetrics package not available - SSIM experiments will fail")
        
        return issues
    
    @staticmethod
    def estimate_experiment_time(max_images_per_split, metrics_count):
        """実験時間の推定"""
        
        # 基本処理時間（分/画像/指標）
        time_per_image_per_metric = {
            'mse': 0.5,
            'l1': 0.5,
            'lpips': 1.2,
            'ssim': 0.8,
            'improved_ssim': 0.8,
            'psnr': 0.6
        }
        
        # 総画像数（train + val + test）
        total_images = max_images_per_split * 3 if max_images_per_split else 500
        
        # 推定時間計算
        estimated_minutes = total_images * metrics_count * 0.8  # 平均処理時間
        
        return {
            'total_images': total_images,
            'estimated_minutes': estimated_minutes,
            'estimated_hours': estimated_minutes / 60
        }

class ExperimentLogger:
    """実験専用ログシステム"""
    
    def __init__(self, experiment_name, output_dir):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ログファイル設定
        log_file = self.output_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # ロガー設定
        self.logger = logging.getLogger(experiment_name)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_experiment_start(self, config):
        """実験開始ログ"""
        self.logger.info(f"Experiment {self.experiment_name} started")
        self.logger.info(f"Configuration: {config}")
    
    def log_metric_start(self, metric, config):
        """指標別実験開始ログ"""
        self.logger.info(f"Starting {metric} optimization")
        self.logger.info(f"  Iterations: {config.iterations}")
        self.logger.info(f"  Learning rate: {config.learning_rate}")
    
    def log_metric_complete(self, metric, fid_score, processing_time):
        """指標別実験完了ログ"""
        self.logger.info(f"{metric} optimization completed")
        self.logger.info(f"  FID score: {fid_score:.2f}")
        self.logger.info(f"  Processing time: {processing_time/60:.1f} minutes")
    
    def log_experiment_complete(self, total_time, results_summary):
        """実験完了ログ"""
        self.logger.info(f"Experiment {self.experiment_name} completed")
        self.logger.info(f"Total time: {total_time/3600:.1f} hours")
        self.logger.info(f"Results summary: {results_summary}")

class ResultsManager:
    """実験結果管理クラス"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_experiment_results(self, experiment_name, results):
        """実験結果の保存"""
        
        # タイムスタンプ付きファイル名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f"{experiment_name}_results_{timestamp}.json"
        
        # メタデータ追加
        full_results = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'python_version': sys.version,
                'torch_version': torch.__version__
            },
            'results': results
        }
        
        # JSON保存
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        return str(results_file)
    
    def load_previous_results(self, experiment_name):
        """過去の実験結果読み込み"""
        
        pattern = f"{experiment_name}_results_*.json"
        result_files = list(self.output_dir.glob(pattern))
        
        if not result_files:
            return None
        
        # 最新ファイルを取得
        latest_file = max(result_files, key=os.path.getctime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)

class ExperimentRunner:
    """実験実行の基底クラス"""
    
    def __init__(self, experiment_name, output_dir, max_images_per_split=None):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.max_images = max_images_per_split
        
        # 環境設定
        self.bsds500_path = os.environ.get('BSDS500_PATH')
        
        # ユーティリティ初期化
        self.logger = ExperimentLogger(experiment_name, self.output_dir / "logs")
        self.results_manager = ResultsManager(self.output_dir / "results")
        self.fid_evaluator = DatasetFIDEvaluator(device='cuda')
    
    def run_metric_experiment(self, metric, config):
        """単一指標での実験実行"""
        
        self.logger.log_metric_start(metric, config)
        metric_start_time = time.time()
        
        try:
            # 出力ディレクトリ作成
            metric_output_dir = self.output_dir / "datasets" / metric
            metric_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 最適化実行
            datasets = process_bsds500_dataset(
                self.bsds500_path,
                metric_output_dir / "optimized",
                config,
                max_images_per_split=self.max_images,
                create_pytorch_dataset=True,
                create_png_dataset=True
            )
            
            # FID評価
            fid_result = self.fid_evaluator.evaluate_created_dataset_vs_original(
                datasets['png'], 
                self.bsds500_path
            )
            
            processing_time = time.time() - metric_start_time
            
            # 結果構造化
            result = {
                'fid_score': fid_result.fid_score,
                'processing_time_seconds': processing_time,
                'total_images': fid_result.total_images,
                'dataset_paths': datasets,
                'optimization_config': {
                    'iterations': config.iterations,
                    'learning_rate': config.learning_rate,
                    'loss_function': config.loss_function,
                    'device': config.device
                }
            }
            
            self.logger.log_metric_complete(metric, fid_result.fid_score, processing_time)
            return result
            
        except Exception as e:
            self.logger.logger.error(f"{metric} experiment failed: {e}")
            raise

def main():
    """パイロット実験メイン関数"""
    
    # 環境検証
    issues = ExperimentValidator.validate_environment()
    if issues:
        print("⚠️ Environment issues detected:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease resolve these issues before running the experiment")
        return
    
    # 処理時間推定
    time_estimate = ExperimentValidator.estimate_experiment_time(
        max_images_per_split=10, 
        metrics_count=2
    )
    
    print(f"📊 Experiment Estimate:")
    print(f"   Total images: {time_estimate['total_images']}")
    print(f"   Estimated time: {time_estimate['estimated_minutes']:.1f} minutes")
    
    # 実験実行
    experiment = PilotExperiment(max_images_per_split=10)
    results = experiment.run_experiment()
    
    # 次ステップ推奨
    successful_metrics = len([r for r in results.values() if 'error' not in r])
    
    if successful_metrics == len(results):
        print("\n🎉 Pilot experiment successful!")
        print("💡 Next step: Run medium-scale experiment with:")
        print("   NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/fid_comparison/medium_scale_experiment.py")
    else:
        print(f"\n⚠️ {len(results) - successful_metrics} metrics failed")
        print("🔧 Please check logs and fix issues before proceeding")

if __name__ == "__main__":
    main()
```

### 3. experiments/fid_comparison/medium_scale_experiment.py
```python
#!/usr/bin/env python3
"""
FID最適化実験: 中規模版
統計的有意性を確保するための中規模実験
"""

import sys
from pathlib import Path

# プロジェクトルート追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.experiment_utils import ExperimentRunner, ExperimentValidator
from generative_latent_optimization import OptimizationConfig

class MediumScaleExperiment(ExperimentRunner):
    """中規模FID実験クラス"""
    
    def __init__(self, max_images_per_split=50):
        super().__init__(
            experiment_name="medium_scale_fid",
            output_dir=Path(__file__).parent / "results" / "medium",
            max_images_per_split=max_images_per_split
        )
        
        # 実装済み指標（拡張後は全指標追加）
        self.test_metrics = ['mse', 'l1']  # TODO: 拡張後は ['mse', 'l1', 'lpips', 'ssim', 'improved_ssim', 'psnr']
    
    def run_experiment(self):
        """中規模実験実行"""
        
        self.logger.log_experiment_start({
            'experiment_type': 'medium_scale',
            'max_images_per_split': self.max_images,
            'metrics': self.test_metrics
        })
        
        experiment_results = {}
        
        for metric in self.test_metrics:
            # より詳細な最適化設定
            config = OptimizationConfig(
                iterations=100,  # 中規模では少し長め
                learning_rate=0.4,
                loss_function=metric,
                device='cuda',
                checkpoint_interval=20
            )
            
            result = self.run_metric_experiment(metric, config)
            experiment_results[metric] = result
        
        # 結果保存
        results_file = self.results_manager.save_experiment_results(
            "medium_scale", experiment_results
        )
        
        # 統計分析
        self._perform_statistical_analysis(experiment_results)
        
        return experiment_results
    
    def _perform_statistical_analysis(self, results):
        """中規模実験での統計分析"""
        
        # FIDスコア抽出
        fid_scores = {
            metric: result['fid_score'] 
            for metric, result in results.items()
            if 'fid_score' in result
        }
        
        if len(fid_scores) < 2:
            logger.warning("Insufficient results for statistical analysis")
            return
        
        # 基本統計
        best_metric = min(fid_scores.keys(), key=lambda k: fid_scores[k])
        worst_metric = max(fid_scores.keys(), key=lambda k: fid_scores[k])
        
        improvement = fid_scores[worst_metric] - fid_scores[best_metric]
        improvement_percent = (improvement / fid_scores[worst_metric]) * 100
        
        analysis = {
            'best_metric': best_metric,
            'best_fid': fid_scores[best_metric],
            'worst_metric': worst_metric,
            'worst_fid': fid_scores[worst_metric],
            'absolute_improvement': improvement,
            'relative_improvement_percent': improvement_percent
        }
        
        # 分析結果保存
        analysis_file = self.output_dir / "results" / "statistical_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Statistical analysis saved to: {analysis_file}")
        
        # コンソール出力
        print(f"\n📈 Medium-Scale Statistical Analysis:")
        print(f"   🥇 Best metric: {best_metric} (FID: {fid_scores[best_metric]:.2f})")
        print(f"   🔴 Worst metric: {worst_metric} (FID: {fid_scores[worst_metric]:.2f})")
        print(f"   📊 Improvement: {improvement:.2f} ({improvement_percent:.1f}%)")

def main():
    """中規模実験メイン実行"""
    
    # 環境検証
    issues = ExperimentValidator.validate_environment()
    if issues:
        print("⚠️ Environment issues:")
        for issue in issues:
            print(f"   - {issue}")
        return
    
    # 実験実行
    experiment = MediumScaleExperiment(max_images_per_split=50)
    
    # 時間推定表示
    time_estimate = ExperimentValidator.estimate_experiment_time(50, 2)
    print(f"📊 Estimated experiment time: {time_estimate['estimated_hours']:.1f} hours")
    
    # 実行確認
    response = input("Continue with medium-scale experiment? (y/N): ")
    if response.lower() != 'y':
        print("Experiment cancelled")
        return
    
    # 実験実行
    results = experiment.run_experiment()
    
    print(f"\n✅ Medium-scale experiment completed!")
    print(f"💡 Next: Run full-scale experiment for definitive results")

if __name__ == "__main__":
    main()
```

### 4. experiments/fid_comparison/analyze_results.py
```python
#!/usr/bin/env python3
"""
FID実験結果分析・レポート生成スクリプト
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class FIDExperimentAnalyzer:
    """FID実験結果分析クラス"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_all_experiments(self):
        """全実験結果の統合分析"""
        
        # 結果ファイル検索
        pilot_results = self._load_latest_results("pilot")
        medium_results = self._load_latest_results("medium_scale")
        full_results = self._load_latest_results("full_scale")
        
        # 統合分析
        analysis = {
            'pilot': pilot_results,
            'medium': medium_results,
            'full': full_results,
            'summary': self._create_comprehensive_summary(pilot_results, medium_results, full_results)
        }
        
        # 可視化生成
        self._generate_visualizations(analysis)
        
        # レポート生成
        self._generate_report(analysis)
        
        return analysis
    
    def _load_latest_results(self, experiment_type):
        """最新の実験結果読み込み"""
        
        pattern = f"*{experiment_type}_results_*.json"
        result_files = list(self.results_dir.glob(pattern))
        
        if not result_files:
            return None
        
        latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def _create_comprehensive_summary(self, pilot, medium, full):
        """包括的サマリー作成"""
        
        summary = {
            'experiment_progression': {},
            'metric_rankings': {},
            'statistical_significance': {}
        }
        
        # 各段階での結果比較
        for stage, results in [('pilot', pilot), ('medium', medium), ('full', full)]:
            if results and 'results' in results:
                fid_scores = {
                    metric: data['fid_score'] 
                    for metric, data in results['results'].items()
                    if 'fid_score' in data
                }
                
                summary['experiment_progression'][stage] = {
                    'fid_scores': fid_scores,
                    'best_metric': min(fid_scores.keys(), key=lambda k: fid_scores[k]) if fid_scores else None,
                    'worst_metric': max(fid_scores.keys(), key=lambda k: fid_scores[k]) if fid_scores else None
                }
        
        return summary
    
    def _generate_visualizations(self, analysis):
        """分析結果の可視化"""
        
        # 実験段階別FID比較
        self._plot_experiment_progression(analysis)
        
        # 最終ランキング（全データセット結果）
        if analysis['full']:
            self._plot_final_ranking(analysis['full'])
    
    def _plot_experiment_progression(self, analysis):
        """実験段階別の進捗可視化"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        stages = ['pilot', 'medium', 'full']
        
        for i, stage in enumerate(stages):
            ax = axes[i]
            
            if analysis[stage] and 'experiment_progression' in analysis['summary']:
                stage_data = analysis['summary']['experiment_progression'].get(stage)
                if stage_data and 'fid_scores' in stage_data:
                    fid_scores = stage_data['fid_scores']
                    
                    metrics = list(fid_scores.keys())
                    scores = list(fid_scores.values())
                    
                    bars = ax.bar(metrics, scores, color=sns.color_palette("husl", len(metrics)))
                    ax.set_title(f'{stage.title()} Experiment')
                    ax.set_ylabel('FID Score')
                    
                    # 数値表示
                    for bar, score in zip(bars, scores):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{score:.1f}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{stage.title()} Experiment (No Data)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'experiment_progression.png', dpi=300)
        plt.close()
    
    def _plot_final_ranking(self, full_results):
        """最終ランキングの可視化"""
        
        if 'results' not in full_results:
            return
        
        fid_scores = {
            metric: data['fid_score']
            for metric, data in full_results['results'].items()
            if 'fid_score' in data
        }
        
        # FID順でソート
        sorted_items = sorted(fid_scores.items(), key=lambda x: x[1])
        
        metrics = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(metrics)))
        bars = plt.bar(metrics, scores, color=colors)
        
        plt.title('Final FID Score Ranking (Full Dataset)', fontsize=16, fontweight='bold')
        plt.ylabel('FID Score (Lower = Better Quality)', fontsize=12)
        plt.xlabel('Optimization Metric', fontsize=12)
        
        # 数値表示とランキング
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}\\n(#{i+1})', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'final_fid_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_report(self, analysis):
        """実験レポート生成"""
        
        report_content = f"""
# FID最適化実験 分析レポート

**生成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 📊 実験概要

本実験では、BSDS500データセットを用いて異なる最適化指標でのVAE潜在表現最適化を行い、
各指標がデータセットレベルのFIDスコアに与える影響を定量的に分析した。

## 🎯 実験段階別結果

"""
        
        # 各段階の結果サマリー
        stages = ['pilot', 'medium', 'full']
        for stage in stages:
            if analysis[stage]:
                results = analysis[stage].get('results', {})
                if results:
                    report_content += f"### {stage.title()} Experiment\\n"
                    
                    for metric, data in results.items():
                        if 'fid_score' in data:
                            fid = data['fid_score']
                            time_min = data.get('processing_time_seconds', 0) / 60
                            report_content += f"- **{metric.upper()}**: FID = {fid:.2f} (処理時間: {time_min:.1f}分)\\n"
                    
                    report_content += "\\n"
        
        # 結論
        if analysis['full']:
            full_fid = analysis['full'].get('results', {})
            if full_fid:
                best_metric = min(full_fid.keys(), key=lambda k: full_fid[k].get('fid_score', float('inf')))
                worst_metric = max(full_fid.keys(), key=lambda k: full_fid[k].get('fid_score', 0))
                
                report_content += f"""
## 🏆 最終結論

### 最優秀指標
- **{best_metric.upper()}**: FID = {full_fid[best_metric]['fid_score']:.2f}
- データセット品質が最も良好に保たれる最適化手法

### 最劣位指標  
- **{worst_metric.upper()}**: FID = {full_fid[worst_metric]['fid_score']:.2f}
- データセット品質の劣化が最も顕著な最適化手法

### 推奨事項
1. 高品質データセット生成には **{best_metric}** 最適化を推奨
2. **{worst_metric}** 最適化はFID観点から推奨しない
3. 用途に応じた最適化指標の選択が重要

## 📈 詳細分析

詳細な分析結果は以下のファイルを参照：
- `final_fid_ranking.png`: 最終ランキング可視化
- `experiment_progression.png`: 実験段階別進捗
"""
        
        # レポート保存
        report_file = self.output_dir / "experiment_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 Analysis report generated: {report_file}")

def main():
    """分析メイン実行"""
    
    analyzer = FIDExperimentAnalyzer(Path(__file__).parent / "results")
    analysis = analyzer.analyze_all_experiments()
    
    print("📊 FID Experiment Analysis Completed!")
    print(f"📁 Results saved in: {analyzer.output_dir}")
    print("\nFiles generated:")
    print("   - experiment_report.md")
    print("   - final_fid_ranking.png")
    print("   - experiment_progression.png")

if __name__ == "__main__":
    main()
```

## ⚡ 実行コマンド一覧

### 段階的実験実行
```bash
# 1. パイロット実験（約1時間）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/fid_comparison/pilot_experiment.py

# 2. 中規模実験（約8時間）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/fid_comparison/medium_scale_experiment.py

# 3. 全データセット実験（約48時間）
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/fid_comparison/full_scale_experiment.py

# 4. 結果分析
NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python experiments/fid_comparison/analyze_results.py
```

### 環境変数設定例
```bash
# 必須環境変数
export BSDS500_PATH="/path/to/bsds500/dataset"
export HF_TOKEN="your_huggingface_token"

# オプション環境変数
export CUDA_VISIBLE_DEVICES=0,1  # マルチGPU使用時
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # メモリ効率化
```

### 並列実験実行（2GPU環境）
```bash
# GPU0: 軽量指標
CUDA_VISIBLE_DEVICES=0 NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python \
  experiments/fid_comparison/parallel_light_metrics.py &

# GPU1: 重い指標
CUDA_VISIBLE_DEVICES=1 NIXPKGS_ALLOW_UNFREE=1 nix develop --impure -c uv run python \
  experiments/fid_comparison/parallel_heavy_metrics.py &

# 両方の完了を待機
wait
```

## 📋 実行前チェックリスト

### 環境準備
- [ ] BSDS500データセットのダウンロード・配置
- [ ] HF_TOKENの設定
- [ ] 必要パッケージの確認（lpips、torchmetricsなど）
- [ ] ストレージ容量確認（100GB以上推奨）

### 実装準備
- [ ] LatentOptimizerの損失関数拡張実装
- [ ] 新損失関数の単体テスト実行
- [ ] メモリ効率化機能のテスト
- [ ] エラーハンドリングのテスト

### 実行準備
- [ ] パイロット実験の成功確認
- [ ] 処理時間の妥当性確認
- [ ] 中規模実験での統計的傾向確認
- [ ] 全実験実行の最終決定

---

このスクリプト設計により、段階的かつ効率的なFID最適化実験の実行が可能となります。各段階で結果を検証しながら、最終的に科学的に妥当な結論を獲得できる実験フレームワークです。