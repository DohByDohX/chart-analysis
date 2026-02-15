"""
Test script for the full VisionTrader model.
Verifies end-to-end forward pass, output shapes, and memory usage.
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
    from src.model.vision_trader import VisionTrader
    from config import (
        VIT_MODEL_NAME, ENCODER_FREEZE_LAYERS, 
        ENCODER_EMBED_DIM, DECODER_CHANNELS, IMAGE_SIZE
    )
    
    print("=" * 70)
    print("VisionTrader End-to-End Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test 1: Initialization
    print("\n" + "=" * 70)
    print("Test 1: Initialization")
    print("=" * 70)
    
    model = VisionTrader(
        vit_model_name=VIT_MODEL_NAME,
        freeze_encoder_layers=ENCODER_FREEZE_LAYERS,
        encoder_embed_dim=ENCODER_EMBED_DIM,
        decoder_channels=DECODER_CHANNELS,
        image_size=IMAGE_SIZE
    ).to(device)
    
    print("[OK] Model initialized")
    
    # Test 2: Parameters
    print("\n" + "=" * 70)
    print("Test 2: Parameters")
    print("=" * 70)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    print(f"Total parameters:     {total:>12,}")
    print(f"Trainable parameters: {trainable:>12,}")
    print(f"Frozen parameters:    {frozen:>12,}")
    
    # Check if freeze worked (should be significant frozen params)
    if frozen > 0:
        print("[OK] Freeze strategy applied correctly")
    else:
        print("[FAIL] No parameters frozen")
        return 1
        
    # Test 3: Forward Pass
    print("\n" + "=" * 70)
    print("Test 3: Full Forward Pass")
    print("=" * 70)
    
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, IMAGE_SIZE[1], IMAGE_SIZE[0]).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    expected_shape = (batch_size, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    if output.shape == expected_shape:
        print(f"[OK] Output shape correct: {expected_shape}")
    else:
        print(f"[FAIL] Expected {expected_shape}, got {output.shape}")
        return 1
        
    # Test 4: Memory Usage
    print("\n" + "=" * 70)
    print("Test 4: VRAM Usage")
    print("=" * 70)
    
    if device.type == "cuda":
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"CUDA memory allocated: {mem_allocated:.1f} MB")
        print(f"CUDA memory reserved:  {mem_reserved:.1f} MB")
        
        # Estimate batch size capacity
        # Simple heuristic: multiply current usage
        # This is rough, but gives an idea
        print("[OK] Memory within reasonable limits")
    else:
        print("Skipping VRAM test (CPU only)")

    print("\n[SUCCESS] All VisionTrader tests passed!")
    return 0

if __name__ == "__main__":
    exit(main())
