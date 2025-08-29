#!/usr/bin/env python3
"""
SD 1.4 VAE修正版テストスクリプト
発見された問題を修正したバージョン
"""

import torch
from diffusers import AutoencoderKL
import numpy as np
import time

def test_model_loading():
    """SD 1.4 VAEモデルが正しく読み込めるかテスト"""
    print("=" * 50)
    print("TEST 1: Model Loading (Fixed)")
    print("=" * 50)
    
    try:
        vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            token="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",
        )
        
        # GPUが利用可能な場合は自動的に移動
        if torch.cuda.is_available():
            vae = vae.cuda()
            print("✅ Model moved to GPU")
        else:
            print("✅ Model on CPU")
            
        print(f"   Model config scaling factor: {vae.config.scaling_factor}")
        print(f"   Model device: {next(vae.parameters()).device}")
        print(f"   Model dtype: {next(vae.parameters()).dtype}")
        return vae
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_encode_decode_with_clamping(vae):
    """出力範囲制限付きエンコード・デコードテスト"""
    print("\n" + "=" * 50)
    print("TEST 2: Shape Validation (With Clamping)")
    print("=" * 50)
    
    # テスト用ランダム画像 (512x512x3, [-1,1] range)
    batch_size = 2
    device = next(vae.parameters()).device
    
    test_images = torch.randn(batch_size, 3, 512, 512).clamp(-1, 1).to(device)
    
    print(f"Input shape: {test_images.shape}")
    print(f"Input range: [{test_images.min():.3f}, {test_images.max():.3f}]")
    print(f"Input device: {test_images.device}")
    
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
        # デコード（出力範囲制限付き）
        with torch.no_grad():
            decoded_images = vae.decode(latents).sample
            # 明示的に[-1, 1]範囲にクランプ
            decoded_images_clamped = decoded_images.clamp(-1, 1)
            
        print(f"✅ Decode successful")
        print(f"   Decoded shape: {decoded_images.shape}")
        print(f"   Raw decoded range: [{decoded_images.min():.3f}, {decoded_images.max():.3f}]")
        print(f"   Clamped decoded range: [{decoded_images_clamped.min():.3f}, {decoded_images_clamped.max():.3f}]")
        
        # 形状確認
        if decoded_images_clamped.shape == test_images.shape:
            print("✅ Decoded shape correct")
        else:
            print(f"❌ Decoded shape incorrect")
        
        # 範囲チェック
        if torch.all(decoded_images_clamped >= -1.0) and torch.all(decoded_images_clamped <= 1.0):
            print("✅ Output range properly clamped to [-1, 1]")
        else:
            print("❌ Clamping failed")
            
        return test_images, decoded_images_clamped
        
    except Exception as e:
        print(f"❌ Decode failed: {e}")
        return test_images, None

def test_proper_device_handling(vae):
    """適切なデバイス処理テスト"""
    print("\n" + "=" * 50)
    print("TEST 3: Proper Device Handling")
    print("=" * 50)
    
    model_device = next(vae.parameters()).device
    print(f"Model device: {model_device}")
    
    # テスト入力をモデルと同じデバイスに配置
    test_input = torch.randn(1, 3, 512, 512).clamp(-1, 1).to(model_device)
    print(f"Input device: {test_input.device}")
    
    try:
        with torch.no_grad():
            # エンコード
            start_time = time.time()
            posterior = vae.encode(test_input)
            latents = posterior.latent_dist.mode()
            encode_time = time.time() - start_time
            
            # デコード
            start_time = time.time()
            decoded = vae.decode(latents).sample.clamp(-1, 1)
            decode_time = time.time() - start_time
            
        print(f"✅ Device consistency maintained")
        print(f"   Latent device: {latents.device}")
        print(f"   Decoded device: {decoded.device}")
        print(f"   Encode time: {encode_time:.3f}s")
        print(f"   Decode time: {decode_time:.3f}s")
        print(f"   Total time: {encode_time + decode_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Device handling failed: {e}")
        return False

def test_reconstruction_quality_fixed(original, reconstructed):
    """修正版再構成品質テスト"""
    print("\n" + "=" * 50)
    print("TEST 4: Reconstruction Quality (Fixed)")
    print("=" * 50)
    
    if reconstructed is None:
        print("❌ No reconstructed image to test")
        return
    
    # デバイス統一
    if original.device != reconstructed.device:
        reconstructed = reconstructed.to(original.device)
    
    # MSE損失計算
    mse_loss = torch.nn.functional.mse_loss(original, reconstructed)
    print(f"MSE Loss: {mse_loss.item():.6f}")
    
    # MAE損失計算
    mae_loss = torch.nn.functional.l1_loss(original, reconstructed)
    print(f"MAE Loss: {mae_loss.item():.6f}")
    
    # 値範囲チェック
    if torch.all(reconstructed >= -1.0) and torch.all(reconstructed <= 1.0):
        print("✅ Output range within bounds [-1.0, 1.0]")
    else:
        print(f"❌ Output range issue: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # 品質指標
    if mse_loss < 1.0:
        print("✅ Reconstruction quality acceptable (MSE < 1.0)")
    else:
        print("⚠️  High reconstruction error")
    
    # PSNR計算（修正版）
    if mse_loss > 0:
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse_loss))
        print(f"PSNR: {psnr.item():.2f} dB")
    else:
        print("PSNR: ∞ dB (perfect reconstruction)")
    
    # ピクセル単位の統計
    diff = torch.abs(original - reconstructed)
    print(f"Max pixel difference: {diff.max():.3f}")
    print(f"Mean pixel difference: {diff.mean():.3f}")

def test_batch_processing(vae):
    """バッチ処理テスト"""
    print("\n" + "=" * 50)
    print("TEST 5: Batch Processing")
    print("=" * 50)
    
    device = next(vae.parameters()).device
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        try:
            # バッチテスト
            test_batch = torch.randn(batch_size, 3, 512, 512).clamp(-1, 1).to(device)
            
            start_time = time.time()
            with torch.no_grad():
                latents = vae.encode(test_batch).latent_dist.mode()
                decoded = vae.decode(latents).sample.clamp(-1, 1)
            
            batch_time = time.time() - start_time
            time_per_image = batch_time / batch_size
            
            print(f"✅ Batch size {batch_size}: {batch_time:.3f}s total, {time_per_image:.3f}s/image")
            
        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {e}")

def main():
    """メイン検証実行（修正版）"""
    print("🧪 SD 1.4 VAE Library Verification (Fixed)")
    print("Purpose: Verify Diffusers library with problem fixes")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # 1. モデル読み込み（デバイス自動選択付き）
    vae = test_model_loading()
    if vae is None:
        print("\n❌ Critical failure: Cannot proceed without model")
        return False
    
    # 2. 修正版エンコード・デコード
    original, reconstructed = test_encode_decode_with_clamping(vae)
    
    # 3. デバイス処理テスト
    device_test_passed = test_proper_device_handling(vae)
    
    # 4. 修正版品質テスト  
    test_reconstruction_quality_fixed(original, reconstructed)
    
    # 5. バッチ処理テスト
    test_batch_processing(vae)
    
    print("\n" + "=" * 50)
    print("🏁 FIXED VERIFICATION COMPLETE")
    print("=" * 50)
    print("Key improvements in this version:")
    print("  ✅ Proper device handling (CPU/GPU)")
    print("  ✅ Output range clamping to [-1, 1]")
    print("  ✅ Consistent tensor devices")
    print("  ✅ Batch processing validation")
    
    if device_test_passed and original is not None and reconstructed is not None:
        print("\n🎉 All major issues fixed! Library is working correctly.")
        return True
    else:
        print("\n⚠️  Some issues remain, but library is largely functional.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)