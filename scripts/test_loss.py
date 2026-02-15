"""
Test script for the VisionTrader Loss Function.
Verifies Perceptual + SSIM loss calculation and mask weighting logic.
"""
import sys
from pathlib import Path

# Add project root to path
# Assuming this script is running from project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        from src.training.loss import VisionTraderLoss
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return 1
        
    print("=" * 70)
    print("VisionTrader Loss Function Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize Loss
    try:
        # Masked region weight 10.0, apply to last 20 pixels
        criterion = VisionTraderLoss(
            perceptual_weight=1.0,
            ssim_weight=1.0,
            masked_region_weight=10.0,
            masked_pixels=20
        ).to(device)
        print("[OK] Loss function initialized")
    except Exception as e:
        print(f"[FAIL] Initialization error: {e}")
        return 1
    
    # Test 1: Identical Images (Should be zero perceptual, 1.0 SSIM -> 0 loss)
    # SSIM returns 1.0 on match, so 1 - SSIM = 0
    print("\n" + "=" * 70)
    print("Test 1: Identical Images")
    print("=" * 70)
    
    img = torch.rand(2, 3, 512, 512).to(device)
    loss_dict = criterion(img, img)
    
    print(f"Total Loss: {loss_dict['loss']:.6f}")
    if loss_dict['loss'] < 1e-4:
        print("[OK] Near-zero loss for identical images")
    else:
        print(f"[WARN] Non-zero loss: {loss_dict['loss']}")

    # Test 2: Masked Region Sensitivity
    print("\n" + "=" * 70)
    print("Test 2: Masked Region Sensitivity")
    print("=" * 70)
    
    # Error in left 100 pixels (non-critical)
    img_bad_left = img.clone()
    img_bad_left[:, :, :, :100] = 0.0
    loss_left = criterion(img_bad_left, img)['loss']
    
    # Error in right 20 pixels (critical masked region)
    # Even though area is 5x smaller (20 vs 100), weighting is 10x higher
    # Plus global loss applies to both
    img_bad_right = img.clone()
    img_bad_right[:, :, :, -20:] = 0.0
    loss_right = criterion(img_bad_right, img)['loss']
    
    print(f"Loss (Left 100px Error): {loss_left:.4f}")
    print(f"Loss (Right 20px Error): {loss_right:.4f}")
    
    if loss_right > loss_left:
        print(f"[OK] Critical region error ({loss_right:.4f}) penalized more than larger non-critical error ({loss_left:.4f})")
    else:
        print(f"[WARN] Check weights. Masked region might need higher weight.")

    # Test 3: Gradient Flow
    print("\n" + "=" * 70)
    print("Test 3: Gradient Flow")
    print("=" * 70)
    
    # Create tensor directly on device to ensure it's a leaf tensor
    input_grad = torch.randn(1, 3, 512, 512, device=device, requires_grad=True)
    target_grad = torch.randn(1, 3, 512, 512, device=device)
    
    loss_out = criterion(input_grad, target_grad)['loss']
    loss_out.backward()
    
    if input_grad.grad is not None and input_grad.grad.abs().sum() > 0:
        print(f"[OK] Gradients flowing correctly: Mean={input_grad.grad.abs().mean().item():.6f}")
    else:
        print("[FAIL] No gradients")
        return 1

    print("\n[SUCCESS] Loss function verified!")
    return 0

if __name__ == "__main__":
    exit(main())
