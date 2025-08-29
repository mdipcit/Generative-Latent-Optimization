#!/usr/bin/env python3
"""
SD 1.4 VAE基本動作確認スクリプト
目的：Diffusersライブラリが正しく動作することを確認
"""

import torch
from diffusers import AutoencoderKL
import numpy as np
import time

def test_model_loading():
    """SD 1.4 VAEモデルが正しく読み込めるかテスト"""
    print("=" * 50)
    print("TEST 1: Model Loading")
    print("=" * 50)
    
    try:
        vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            token="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",
        )
        print("✅ Model loaded successfully")
        print(f"   Model config scaling factor: {vae.config.scaling_factor}")
        print(f"   Model device: {vae.device}")
        print(f"   Model dtype: {next(vae.parameters()).dtype}")
        return vae
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_encode_decode_shapes(vae):
    """エンコード・デコードの形状が正しいかテスト"""
    print("\n" + "=" * 50)
    print("TEST 2: Shape Validation")
    print("=" * 50)
    
    # テスト用ランダム画像 (512x512x3, [-1,1] range)
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 512, 512) * 2 - 1
    test_images = test_images.clamp(-1, 1)
    
    print(f"Input shape: {test_images.shape}")
    print(f"Input range: [{test_images.min():.3f}, {test_images.max():.3f}]")
    
    try:
        # エンコード
        with torch.no_grad():
            posterior = vae.encode(test_images)
            latents = posterior.latent_dist.mode()
            scaled_latents = latents * vae.config.scaling_factor
            
        print(f"✅ Encode successful")
        print(f"   Latent shape: {latents.shape}")
        print(f"   Expected shape: ({batch_size}, 4, 64, 64)")
        print(f"   Latent range (raw): [{latents.min():.3f}, {latents.max():.3f}]")
        print(f"   Scaled latent range: [{scaled_latents.min():.3f}, {scaled_latents.max():.3f}]")
        
        # 形状確認
        expected_shape = (batch_size, 4, 64, 64)
        if latents.shape == expected_shape:
            print("✅ Latent shape correct")
        else:
            print(f"❌ Latent shape incorrect. Got {latents.shape}, expected {expected_shape}")
            
    except Exception as e:
        print(f"❌ Encode failed: {e}")
        return None, None
    
    try:
        # デコード
        with torch.no_grad():
            decoded_images = vae.decode(latents).sample
            
        print(f"✅ Decode successful")
        print(f"   Decoded shape: {decoded_images.shape}")
        print(f"   Decoded range: [{decoded_images.min():.3f}, {decoded_images.max():.3f}]")
        
        # 形状確認
        if decoded_images.shape == test_images.shape:
            print("✅ Decoded shape correct")
        else:
            print(f"❌ Decoded shape incorrect")
            
        return test_images, decoded_images
        
    except Exception as e:
        print(f"❌ Decode failed: {e}")
        return test_images, None

def test_reconstruction_quality(original, reconstructed):
    """再構成品質の基本チェック"""
    print("\n" + "=" * 50)
    print("TEST 3: Reconstruction Quality")
    print("=" * 50)
    
    if reconstructed is None:
        print("❌ No reconstructed image to test")
        return
    
    # MSE損失計算
    mse_loss = torch.nn.functional.mse_loss(original, reconstructed)
    print(f"MSE Loss: {mse_loss.item():.6f}")
    
    # MAE損失計算
    mae_loss = torch.nn.functional.l1_loss(original, reconstructed)
    print(f"MAE Loss: {mae_loss.item():.6f}")
    
    # 値範囲チェック
    if torch.all(reconstructed >= -1.2) and torch.all(reconstructed <= 1.2):
        print("✅ Output range within expected bounds [-1.2, 1.2]")
    else:
        print(f"❌ Output range issue: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # 基本的な品質指標
    if mse_loss < 1.0:  # 経験的な閾値
        print("✅ Reconstruction quality acceptable (MSE < 1.0)")
    else:
        print("⚠️  High reconstruction error - check model/input")
    
    # PSNR計算（参考値）
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse_loss))
    print(f"PSNR: {psnr.item():.2f} dB")

def test_different_input_sizes(vae):
    """異なる入力サイズでの動作確認"""
    print("\n" + "=" * 50)
    print("TEST 4: Input Size Validation")
    print("=" * 50)
    
    test_sizes = [
        (1, 3, 512, 512),   # 正常サイズ
        (1, 3, 256, 256),   # 小さいサイズ
        (4, 3, 512, 512),   # バッチサイズ4
    ]
    
    for size in test_sizes:
        try:
            test_input = torch.randn(*size).clamp(-1, 1)
            with torch.no_grad():
                latents = vae.encode(test_input).latent_dist.mode()
                decoded = vae.decode(latents).sample
            
            expected_latent_h = size[2] // 8  # VAEは8倍ダウンサンプル
            expected_latent_w = size[3] // 8
            
            if latents.shape == (size[0], 4, expected_latent_h, expected_latent_w):
                print(f"✅ Size {size}: Latent shape {latents.shape}")
            else:
                print(f"❌ Size {size}: Unexpected latent shape {latents.shape}")
                
        except Exception as e:
            print(f"❌ Size {size}: Failed - {e}")

def test_device_compatibility(vae):
    """CPU/CUDA互換性テスト"""
    print("\n" + "=" * 50)
    print("TEST 5: Device Compatibility")
    print("=" * 50)
    
    test_input = torch.randn(1, 3, 512, 512).clamp(-1, 1)
    
    # CPU テスト
    try:
        vae_cpu = vae.to('cpu')
        test_input_cpu = test_input.to('cpu')
        
        start_time = time.time()
        with torch.no_grad():
            latents = vae_cpu.encode(test_input_cpu).latent_dist.mode()
            decoded = vae_cpu.decode(latents).sample
        cpu_time = time.time() - start_time
            
        print(f"✅ CPU execution successful ({cpu_time:.3f}s)")
        
    except Exception as e:
        print(f"❌ CPU execution failed: {e}")
    
    # CUDA テスト（利用可能な場合）
    if torch.cuda.is_available():
        try:
            vae_cuda = vae.to('cuda')
            test_input_cuda = test_input.to('cuda')
            
            start_time = time.time()
            with torch.no_grad():
                latents = vae_cuda.encode(test_input_cuda).latent_dist.mode()
                decoded = vae_cuda.decode(latents).sample
            gpu_time = time.time() - start_time
                
            print(f"✅ CUDA execution successful ({gpu_time:.3f}s)")
            
            if cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"   GPU speedup: {speedup:.1f}x")
            
        except Exception as e:
            print(f"❌ CUDA execution failed: {e}")
    else:
        print("⚠️  CUDA not available, skipping GPU test")

def test_scaling_factor_validation(vae):
    """SD 1.4 スケーリングファクターの正確性テスト"""
    print("\n" + "=" * 50)
    print("TEST 6: Scaling Factor Validation")
    print("=" * 50)
    
    test_input = torch.randn(1, 3, 512, 512).clamp(-1, 1)
    
    try:
        with torch.no_grad():
            # Diffusersの標準エンコード
            posterior = vae.encode(test_input)
            latents_mode = posterior.latent_dist.mode()
            
            # スケーリングファクター確認
            expected_scaling = 0.18215
            actual_scaling = vae.config.scaling_factor
            
            print(f"Expected scaling factor: {expected_scaling}")
            print(f"Actual scaling factor: {actual_scaling}")
            
            if abs(actual_scaling - expected_scaling) < 1e-5:
                print("✅ Scaling factor matches SD 1.4 specification")
            else:
                print("⚠️  Scaling factor differs from expected")
            
            # スケーリング適用テスト
            scaled_latents = latents_mode * actual_scaling
            print(f"Raw latent range: [{latents_mode.min():.3f}, {latents_mode.max():.3f}]")
            print(f"Scaled latent range: [{scaled_latents.min():.3f}, {scaled_latents.max():.3f}]")
            
    except Exception as e:
        print(f"❌ Scaling factor test failed: {e}")

def main():
    """メイン検証実行"""
    print("🧪 SD 1.4 VAE Library Verification")
    print("Purpose: Verify Diffusers library works correctly")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # 1. モデル読み込み
    vae = test_model_loading()
    if vae is None:
        print("\n❌ Critical failure: Cannot proceed without model")
        return False
    
    # 2. 基本的なエンコード・デコード
    original, reconstructed = test_encode_decode_shapes(vae)
    
    # 3. 再構成品質チェック  
    test_reconstruction_quality(original, reconstructed)
    
    # 4. 異なる入力サイズテスト
    test_different_input_sizes(vae)
    
    # 5. デバイス互換性
    test_device_compatibility(vae)
    
    # 6. スケーリングファクター検証
    test_scaling_factor_validation(vae)
    
    print("\n" + "=" * 50)
    print("🏁 VERIFICATION COMPLETE")
    print("=" * 50)
    print("If all tests show ✅, the library is working correctly.")
    print("If any test shows ❌, there may be an issue with:")
    print("  - Installation")
    print("  - Authentication token") 
    print("  - System configuration")
    print("  - GPU drivers (for CUDA tests)")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)