#!/usr/bin/env python3
"""
BSDS500データセット前処理メインスクリプト

BSDS500データセットをStable Diffusion VAE用に前処理します。
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import ImagePreprocessor


def setup_logging(verbose: bool = False):
    """ログ設定"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='BSDS500データセットの前処理を実行します',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 入力・出力設定
    parser.add_argument(
        '--input_path', 
        type=str,
        default=None,
        help='BSDS500データセットパス（未指定時は環境変数BSDS500_PATHを使用）'
    )
    parser.add_argument(
        '--output_path', 
        type=str,
        default='./processed_data',
        help='前処理済みデータの出力ディレクトリ'
    )
    
    # 前処理設定
    parser.add_argument(
        '--target_size',
        type=int,
        default=512,
        help='前処理後の画像サイズ（正方形）'
    )
    parser.add_argument(
        '--normalize_min',
        type=float,
        default=-1.0,
        help='正規化範囲の最小値'
    )
    parser.add_argument(
        '--normalize_max',
        type=float,
        default=1.0,
        help='正規化範囲の最大値'
    )
    
    # 処理設定
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        choices=['train', 'val', 'test'],
        help='処理するデータセット分割'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='バッチサイズ'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='並列処理のワーカー数'
    )
    
    # 動作設定
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='既存の出力ファイルを上書きする'
    )
    parser.add_argument(
        '--no_metadata',
        action='store_true',
        help='メタデータファイルを保存しない'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='前処理後に出力を検証する'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='詳細なログを出力する'
    )
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # 設定表示
    logger.info("BSDS500データセット前処理を開始します")
    logger.info(f"入力パス: {args.input_path or '環境変数BSDS500_PATH'}")
    logger.info(f"出力パス: {args.output_path}")
    logger.info(f"目標サイズ: {args.target_size}x{args.target_size}")
    logger.info(f"正規化範囲: [{args.normalize_min}, {args.normalize_max}]")
    logger.info(f"処理split: {args.splits}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"ワーカー数: {args.num_workers}")
    
    try:
        # 前処理器初期化
        preprocessor = ImagePreprocessor(
            target_size=args.target_size,
            normalize_range=(args.normalize_min, args.normalize_max),
            output_dir=args.output_path,
            save_metadata=not args.no_metadata,
            overwrite=args.overwrite
        )
        
        # 入力パス設定
        input_path = Path(args.input_path) if args.input_path else None
        
        # データセット前処理実行
        logger.info("前処理を開始します...")
        results = preprocessor.process_dataset(
            dataset_path=input_path,
            splits=args.splits,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # エラーチェック
        if 'error' in results:
            logger.error(f"前処理中にエラーが発生しました: {results['error']}")
            return 1
        
        # 結果サマリー表示
        summary = results['summary']
        logger.info("=== 前処理完了 ===")
        logger.info(f"総画像数: {summary['total_images']}")
        logger.info(f"処理成功: {summary['processed']}")
        logger.info(f"処理失敗: {summary['failed']}")
        logger.info(f"スキップ: {summary['skipped']}")
        logger.info(f"成功率: {summary['success_rate']:.1%}")
        logger.info(f"処理時間: {summary['processing_time']:.1f}秒")
        logger.info(f"処理速度: {summary['images_per_second']:.1f}画像/秒")
        
        # 出力検証
        if args.validate:
            logger.info("出力データの検証を実行します...")
            for split in args.splits:
                try:
                    validation_result = preprocessor.validate_output(split, sample_size=5)
                    
                    if 'error' in validation_result:
                        logger.warning(f"{split} split検証エラー: {validation_result['error']}")
                        continue
                    
                    logger.info(f"{split} split検証結果:")
                    logger.info(f"  - 総ファイル数: {validation_result['total_files']}")
                    logger.info(f"  - 検証率: {validation_result['validation_rate']:.1%}")
                    
                    # 問題があるファイルの詳細表示
                    invalid_files = [d for d in validation_result['details'] if not d['valid']]
                    if invalid_files:
                        logger.warning(f"  - 問題のあるファイル: {len(invalid_files)}個")
                        for detail in invalid_files[:3]:  # 最初の3つのみ表示
                            logger.warning(f"    * {detail['file']}: {detail.get('error', 'unknown error')}")
                
                except Exception as e:
                    logger.error(f"{split} split検証中にエラー: {str(e)}")
        
        logger.info(f"前処理済みデータは {args.output_path} に保存されました")
        
        # 使用例の表示
        logger.info("\n=== 使用例 ===")
        logger.info("前処理済みデータの読み込み:")
        logger.info(f"from src.data.dataset import BSDS500Dataset")
        logger.info(f"dataset = BSDS500Dataset('{args.output_path}', split='train')")
        logger.info(f"image, metadata = dataset[0]")
        
        return 0
        
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())