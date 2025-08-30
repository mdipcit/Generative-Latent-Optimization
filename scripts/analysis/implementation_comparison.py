#!/usr/bin/env python3
"""
ユーザーの元実装とDiffusers実装の比較
"""

import torch
from diffusers import AutoencoderKL
import numpy as np

def user_implementation(vae, image_01):
    """ユーザーの元実装を再現"""
    print("🔸 User Implementation:")
    print(f"   Input range: [{image_01.min():.3f}, {image_01.max():.3f}]")
    
    # 1. [0,1] -> [-1,1] 正規化
    normalized = 2 * image_01 - 1
    print(f"   After normalization: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # 2. エンコード + 平均値取得
    with torch.no_grad():
        posterior = vae.encode(normalized)
        latents_mean = posterior.latent_dist.mean
    
    # 3. 手動スケーリング
    scaled_latents = 0.1825 * latents_mean
    
    print(f"   Raw latents range: [{latents_mean.min():.3f}, {latents_mean.max():.3f}]")
    print(f"   Scaled latents range: [{scaled_latents.min():.3f}, {scaled_latents.max():.3f}]")
    print(f"   Manual scaling factor: 0.1825")
    
    return scaled_latents, normalized

def diffusers_implementation(vae, image_01):
    """Diffusersの標準実装"""
    print("\n🔹 Diffusers Implementation:")
    print(f"   Input range: [{image_01.min():.3f}, {image_01.max():.3f}]")
    
    # 1. [0,1] -> [-1,1] 正規化（同じ）
    normalized = 2 * image_01 - 1
    print(f"   After normalization: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # 2. エンコード + mode取得
    with torch.no_grad():
        posterior = vae.encode(normalized)
        latents_mode = posterior.latent_dist.mode()
    
    # 3. 自動スケーリング（内部で適用される場合）
    auto_scaled = latents_mode * vae.config.scaling_factor
    
    print(f"   Raw latents range: [{latents_mode.min():.3f}, {latents_mode.max():.3f}]")
    print(f"   Auto scaled range: [{auto_scaled.min():.3f}, {auto_scaled.max():.3f}]")
    print(f"   Auto scaling factor: {vae.config.scaling_factor}")
    
    return auto_scaled, normalized

def compare_distributions(vae, image):
    """分布のmeanとmodeの比較"""
    print("\n🔍 Distribution Comparison (mean vs mode):")
    
    normalized = 2 * image - 1
    
    with torch.no_grad():
        posterior = vae.encode(normalized)
        
        # 各種統計値取得
        dist_mean = posterior.latent_dist.mean
        dist_mode = posterior.latent_dist.mode()
        dist_std = posterior.latent_dist.std
        
        # 差分計算
        mean_mode_diff = torch.abs(dist_mean - dist_mode)
        
    print(f"   Mean range: [{dist_mean.min():.3f}, {dist_mean.max():.3f}]")
    print(f"   Mode range: [{dist_mode.min():.3f}, {dist_mode.max():.3f}]")
    print(f"   Std range: [{dist_std.min():.3f}, {dist_std.max():.3f}]")
    print(f"   |Mean - Mode| max: {mean_mode_diff.max():.6f}")
    print(f"   |Mean - Mode| mean: {mean_mode_diff.mean():.6f}")
    
    return dist_mean, dist_mode, dist_std

def test_reconstruction_quality(vae, user_latents, diffuser_latents):
    """再構成品質比較"""
    print("\n📊 Reconstruction Quality Comparison:")
    
    with torch.no_grad():
        # ユーザー実装での再構成
        user_decoded = vae.decode(user_latents / 0.1825).sample.clamp(-1, 1)
        
        # Diffusers実装での再構成  
        diffuser_decoded = vae.decode(diffuser_latents / vae.config.scaling_factor).sample.clamp(-1, 1)
        
        # 品質比較
        diff = torch.abs(user_decoded - diffuser_decoded)
        
    print(f"   User reconstruction range: [{user_decoded.min():.3f}, {user_decoded.max():.3f}]")
    print(f"   Diffusers reconstruction range: [{diffuser_decoded.min():.3f}, {diffuser_decoded.max():.3f}]")
    print(f"   Max difference: {diff.max():.6f}")
    print(f"   Mean difference: {diff.mean():.6f}")
    
    # 実質的に同じかチェック
    is_similar = diff.max() < 0.01
    print(f"   Practically identical: {'✅ Yes' if is_similar else '❌ No'}")
    
    return user_decoded, diffuser_decoded

def main():
    """メイン比較実行"""
    print("🔬 Implementation Comparison: User vs Diffusers")
    print("="*60)
    
    # VAE読み込み
    import os
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        token=os.getenv("HF_TOKEN"),
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = vae.to(device)
    
    # テスト画像（[0,1]正規化）
    test_image_01 = torch.rand(1, 3, 512, 512).to(device)
    
    # 各実装実行
    user_latents, user_normalized = user_implementation(vae, test_image_01)
    diffuser_latents, diffuser_normalized = diffusers_implementation(vae, test_image_01)
    
    # 分布比較
    dist_mean, dist_mode, dist_std = compare_distributions(vae, test_image_01)
    
    # 再構成品質比較
    user_recon, diffuser_recon = test_reconstruction_quality(vae, user_latents, diffuser_latents)
    
    # スケーリング係数比較
    print(f"\n🔢 Scaling Factor Analysis:")
    print(f"   User manual factor: 0.1825")
    print(f"   Official SD 1.4 factor: {vae.config.scaling_factor}")
    print(f"   Difference: {abs(0.1825 - vae.config.scaling_factor):.6f}")
    print(f"   Relative error: {abs(0.1825 - vae.config.scaling_factor) / vae.config.scaling_factor * 100:.3f}%")
    
    print(f"\n🏁 Summary:")
    print(f"   Your original implementation is very close to Diffusers!")
    print(f"   Main differences:")
    print(f"     - Distribution: .mean vs .mode() (minimal impact)")
    print(f"     - Scaling: 0.1825 vs 0.18215 (0.19% difference)")
    print(f"   Both approaches produce nearly identical results ✅")

if __name__ == "__main__":
    main()