#!/usr/bin/env python3
"""
document.pngをSD 1.4 VAEでエンコード・デコードするスクリプト
"""

import torch
from PIL import Image
import numpy as np
import sys
import os

# vae-toolkit パッケージからインポート
from vae_toolkit import load_and_preprocess_image, tensor_to_pil, VAELoader

def encode_decode_image(vae, image_tensor, device):
    """VAEでエンコード・デコード実行"""
    print("\n" + "="*50)
    print("ENCODE-DECODE PROCESS")
    print("="*50)
    
    # デバイスに移動
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        print("🔄 Encoding...")
        # エンコード
        posterior = vae.encode(image_tensor)
        latents = posterior.latent_dist.mode()
        
        print(f"✅ Encoded to latent space")
        print(f"   Input shape: {image_tensor.shape}")
        print(f"   Latent shape: {latents.shape}")
        print(f"   Latent range: [{latents.min():.3f}, {latents.max():.3f}]")
        print(f"   Compression ratio: {image_tensor.numel() / latents.numel():.1f}x")
        
        print("\n🔄 Decoding...")
        # デコード
        decoded = vae.decode(latents).sample
        decoded_clamped = decoded.clamp(-1, 1)
        
        print(f"✅ Decoded from latent space")
        print(f"   Decoded shape: {decoded.shape}")
        print(f"   Raw decoded range: [{decoded.min():.3f}, {decoded.max():.3f}]")
        print(f"   Clamped range: [{decoded_clamped.min():.3f}, {decoded_clamped.max():.3f}]")
        
        # 再構成誤差計算
        mse_loss = torch.nn.functional.mse_loss(image_tensor, decoded_clamped)
        mae_loss = torch.nn.functional.l1_loss(image_tensor, decoded_clamped)
        
        print(f"\n📊 Reconstruction Quality:")
        print(f"   MSE Loss: {mse_loss.item():.6f}")
        print(f"   MAE Loss: {mae_loss.item():.6f}")
        
        if mse_loss > 0:
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse_loss))
            print(f"   PSNR: {psnr.item():.2f} dB")
    
    return decoded_clamped

def main():
    """メイン実行"""
    print("📄 Document PNG Encode-Decode Test")
    print("Using SD 1.4 VAE for reconstruction")
    
    # モデル読み込み
    print("\n🔧 Loading VAE model...")
    try:
        vae, device = VAELoader.load_sd_vae_simple(
            model_name="sd14",
            device="auto"
        )
        print(f"✅ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load VAE model: {e}")
        return
    
    # 画像読み込み・前処理
    image_path = "document.png"
    try:
        image_tensor, original_pil = load_and_preprocess_image(image_path)
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return
    
    # エンコード・デコード実行
    try:
        decoded_tensor = encode_decode_image(vae, image_tensor, device)
    except Exception as e:
        print(f"❌ Encode-decode failed: {e}")
        return
    
    # 結果をPIL画像に変換
    print("\n💾 Converting results to images...")
    
    # 元画像（前処理後）
    original_processed = tensor_to_pil(image_tensor.squeeze(0))
    
    # 再構成画像
    reconstructed = tensor_to_pil(decoded_tensor.squeeze(0))
    
    # 保存
    output_dir = "outputs"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    original_processed.save(f"{output_dir}/document_original_processed.png")
    reconstructed.save(f"{output_dir}/document_reconstructed.png")
    
    print(f"✅ Results saved:")
    print(f"   Original (processed): {output_dir}/document_original_processed.png")
    print(f"   Reconstructed: {output_dir}/document_reconstructed.png")
    
    # サイド・バイ・サイド比較画像作成
    comparison = Image.new('RGB', (original_processed.width * 2, original_processed.height))
    comparison.paste(original_processed, (0, 0))
    comparison.paste(reconstructed, (original_processed.width, 0))
    comparison.save(f"{output_dir}/document_comparison.png")
    print(f"   Comparison: {output_dir}/document_comparison.png")
    
    print("\n🏁 Encode-decode test completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)