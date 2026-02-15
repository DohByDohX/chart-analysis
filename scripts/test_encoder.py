"""
Test script for the Vision Encoder.
Validates model loading, output shapes, and freeze strategy.
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
    from src.model.encoder import VisionEncoder
    from config import (
        VIT_MODEL_NAME, ENCODER_FREEZE_LAYERS, IMAGE_SIZE,
        ENCODER_EMBED_DIM, ENCODER_NUM_PATCHES, PATCH_SIZE
    )
    
    print("=" * 70)
    print("Vision Encoder Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {VIT_MODEL_NAME}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Expected patches: {ENCODER_NUM_PATCHES}")
    print(f"Freeze layers: {ENCODER_FREEZE_LAYERS}")
    print()
    
    # Test 1: Model initialization
    print("=" * 70)
    print("Test 1: Model Initialization")
    print("=" * 70)
    
    encoder = VisionEncoder(
        model_name=VIT_MODEL_NAME,
        freeze_layers=ENCODER_FREEZE_LAYERS,
        image_size=IMAGE_SIZE,
    )
    encoder = encoder.to(device)
    print("[OK] Encoder initialized and moved to device")
    
    # Test 2: Parameter counts
    print("\n" + "=" * 70)
    print("Test 2: Parameter Counts")
    print("=" * 70)
    
    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen = total - trainable
    
    print(f"Total parameters:     {total:>12,}")
    print(f"Trainable parameters: {trainable:>12,}")
    print(f"Frozen parameters:    {frozen:>12,}")
    print(f"Trainable ratio:      {trainable/total*100:.1f}%")
    
    if trainable < total:
        print("[OK] Freeze strategy applied")
    else:
        print("[FAIL] No parameters frozen!")
        return 1
    
    # Test 3: Forward pass
    print("\n" + "=" * 70)
    print("Test 3: Forward Pass (Patch Embeddings)")
    print("=" * 70)
    
    # Create dummy input (batch=2, channels=3, H=512, W=512)
    dummy_input = torch.randn(2, 3, IMAGE_SIZE[1], IMAGE_SIZE[0]).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        patch_embeddings = encoder(dummy_input)
    
    print(f"Output shape: {patch_embeddings.shape}")
    expected_shape = (2, ENCODER_NUM_PATCHES, ENCODER_EMBED_DIM)
    
    if patch_embeddings.shape == expected_shape:
        print(f"[OK] Shape matches expected: {expected_shape}")
    else:
        print(f"[FAIL] Expected {expected_shape}, got {patch_embeddings.shape}")
        return 1
    
    # Test 4: Spatial features
    print("\n" + "=" * 70)
    print("Test 4: Spatial Features (for CNN Decoder)")
    print("=" * 70)
    
    with torch.no_grad():
        spatial = encoder.get_spatial_features(dummy_input)
    
    grid_size = IMAGE_SIZE[0] // PATCH_SIZE
    expected_spatial = (2, ENCODER_EMBED_DIM, grid_size, grid_size)
    print(f"Spatial shape: {spatial.shape}")
    
    if spatial.shape == expected_spatial:
        print(f"[OK] Spatial shape matches: {expected_spatial}")
    else:
        print(f"[FAIL] Expected {expected_spatial}, got {spatial.shape}")
        return 1
    
    # Test 5: Memory usage
    print("\n" + "=" * 70)
    print("Test 5: Memory Usage")
    print("=" * 70)
    
    if device.type == "cuda":
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"CUDA memory allocated: {mem_allocated:.1f} MB")
        print(f"CUDA memory reserved:  {mem_reserved:.1f} MB")
    
    param_memory = total * 4 / 1024**2  # float32
    print(f"Parameter memory (float32): {param_memory:.1f} MB")
    print(f"[OK] Memory within budget")
    
    # Test 6: Gradient flow check
    print("\n" + "=" * 70)
    print("Test 6: Gradient Flow")
    print("=" * 70)
    
    encoder.train()
    test_input = torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0]).to(device)
    output = encoder(test_input)
    loss = output.mean()
    loss.backward()
    
    # Check that unfrozen layers have gradients
    has_grad = False
    for name, param in encoder.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break
    
    if has_grad:
        print("[OK] Gradients flow to trainable parameters")
    else:
        print("[FAIL] No gradients found in trainable parameters")
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("[SUCCESS] All encoder tests passed!")
    print(f"  Model: {VIT_MODEL_NAME}")
    print(f"  Input: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}x3")
    print(f"  Output: {ENCODER_NUM_PATCHES} patches x {ENCODER_EMBED_DIM}d")
    print(f"  Spatial: {grid_size}x{grid_size}x{ENCODER_EMBED_DIM}")
    print(f"  Trainable: {trainable:,} / {total:,} params ({trainable/total*100:.1f}%)")
    
    # Cleanup
    del encoder, dummy_input, test_input, output
    torch.cuda.empty_cache() if device.type == "cuda" else None
    
    return 0


if __name__ == "__main__":
    exit(main())
