#!/usr/bin/env python3
"""
BSDS500データセットアクセステスト

このスクリプトは、nix flake環境で設定されたBSDS500_PATH環境変数を使用して
データセットにアクセスできるかテストします。
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def test_bsds500_access():
    """BSDS500データセットへのアクセステスト"""
    
    # 環境変数からパスを取得
    bsds500_path = os.environ.get('BSDS500_PATH')
    if not bsds500_path:
        print("❌ BSDS500_PATH環境変数が設定されていません")
        return False
    
    print(f"✓ BSDS500_PATH: {bsds500_path}")
    
    # パスの存在確認
    dataset_path = Path(bsds500_path)
    if not dataset_path.exists():
        print(f"❌ パスが存在しません: {dataset_path}")
        return False
    
    print(f"✓ データセットパスが存在します")
    
    # BSDS500/data/images/構造の確認
    images_dir = dataset_path / "BSDS500" / "data" / "images"
    if not images_dir.exists():
        print(f"❌ 画像ディレクトリが存在しません: {images_dir}")
        return False
    
    print(f"✓ 画像ディレクトリが存在します: {images_dir}")
    
    # train/test/valディレクトリの確認
    splits = ['train', 'test', 'val']
    split_counts = {}
    
    for split in splits:
        split_dir = images_dir / split
        if split_dir.exists():
            jpg_files = list(split_dir.glob("*.jpg"))
            split_counts[split] = len(jpg_files)
            print(f"✓ {split}: {len(jpg_files)}枚の画像")
        else:
            print(f"❌ {split}ディレクトリが存在しません")
            return False
    
    # 画像読み込みテスト
    train_dir = images_dir / "train"
    jpg_files = list(train_dir.glob("*.jpg"))
    
    if len(jpg_files) == 0:
        print("❌ 訓練用画像が見つかりません")
        return False
    
    # ファイル情報表示
    test_image_path = jpg_files[0]
    print(f"✓ テスト画像: {test_image_path.name}")
    print(f"  - パス: {test_image_path}")
    print(f"  - ファイルサイズ: {test_image_path.stat().st_size / 1024:.1f} KB")
    
    # PIL利用可能なら画像読み込みテスト
    if PIL_AVAILABLE:
        try:
            with Image.open(test_image_path) as img:
                img_array = np.array(img)
                print(f"✓ 画像読み込み成功")
                print(f"  - サイズ: {img.size}")
                print(f"  - モード: {img.mode}")
                print(f"  - 配列形状: {img_array.shape}")
                
        except Exception as e:
            print(f"❌ 画像読み込みエラー: {e}")
            return False
    else:
        print("ℹ️  PIL未インストール - ファイル存在確認のみ実行")
    
    # 統計情報の表示
    total_images = sum(split_counts.values())
    print(f"\n📊 データセット統計:")
    print(f"  - 総画像数: {total_images}枚")
    for split, count in split_counts.items():
        print(f"  - {split}: {count}枚")
    
    return True


def main():
    """メイン関数"""
    print("🔍 BSDS500データセットアクセステスト開始\n")
    
    success = test_bsds500_access()
    
    if success:
        print("\n✅ すべてのテストが成功しました！")
        print("BSDS500データセットが正常にアクセス可能です。")
        return 0
    else:
        print("\n❌ テストに失敗しました。")
        print("nix develop環境でスクリプトを実行してください。")
        return 1


if __name__ == "__main__":
    sys.exit(main())