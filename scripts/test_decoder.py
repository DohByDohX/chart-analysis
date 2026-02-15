"""
Test script for the CNN Decoder.
Validates output shapes, parameter counts, and gradient flow.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    from src.model.decoder import ChartDecoder
    from config import DECODER_CHANNELS, ENCODER_EMBED_DIM, IMAGE_SIZE
    
    print("=" * 70)
    print("Chart Decoder Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test 1: Initialization
    print("\n" + "=" * 70)
    print("Test 1: Initialization")
    print("=" * 70)
    
    decoder = ChartDecoder(
        embed_dim=ENCODER_EMBED_DIM,
        channels=DECODER_CHANNELS
    ).to(device)
    
    print("[OK] Decoder initialized")
    
    # Test 2: Parameter Count
    print("\n" + "=" * 70)
    print("Test 2: Parameters")
    print("=" * 70)
    
    params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {params:,}")
    # Simple check to ensure it's not massive
    if params < 10_000_000:
        print("[OK] Parameter count reasonable (<10M)")
    else:
        print("[WARN] Parameter count seems high")
        
    # Test 3: Forward Pass
    print("\n" + "=" * 70)
    print("Test 3: Forward Pass")
    print("=" * 70)
    
    # Input from encoder is (B, 768, 16, 16)
    # 512 / 32 = 16
    grid_size = IMAGE_SIZE[0] // 32
    dummy_input = torch.randn(2, ENCODER_EMBED_DIM, grid_size, grid_size).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    output = decoder(dummy_input)
    print(f"Output shape: {output.shape}")
    
    expected_shape = (2, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    if output.shape == expected_shape:
        print(f"[OK] Output shape correct: {expected_shape}")
    else:
        print(f"[FAIL] Expected {expected_shape}, got {output.shape}")
        return 1
        
    # Test 4: Output Range
    print("\n" + "=" * 70)
    print("Test 4: Output Range")
    print("=" * 70)
    
    min_val = output.min().item()
    max_val = output.max().item()
    print(f"Min value: {min_val:.4f}")
    print(f"Max value: {max_val:.4f}")
    
    if 0.0 <= min_val and max_val <= 1.0:
        print("[OK] Output values in [0, 1] range (Sigmoid working)")
    else:
        print("[FAIL] Output values out of range")
        return 1

    # Test 5: Gradient Flow
    print("\n" + "=" * 70)
    print("Test 5: Gradient Flow")
    print("=" * 70)
    
    loss = output.mean()
    loss.backward()
    
    has_grad = False
    for param in decoder.parameters():
        if param.grad is not None:
            has_grad = True
            break
            
    if has_grad:
        print("[OK] Gradients flowing")
    else:
        print("[FAIL] No gradients")
        return 1
        
    print("\n[SUCCESS] All decoder tests passed!")
    return 0

if __name__ == "__main__":
    exit(main())
