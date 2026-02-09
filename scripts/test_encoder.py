import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.vision_encoder import VisionEncoder
from config import IMAGE_SIZE, PATCH_SIZE, ENCODER_EMBED_DIM, SEQUENCE_LENGTH

def test_vision_encoder():
    print("=" * 70)
    print("Testing Vision Encoder (ViT)")
    print("=" * 70)
    
    # Initialize Encoder
    try:
        print("Initializing VisionEncoder...")
        encoder = VisionEncoder()
        print("[OK] Encoder initialized")
    except Exception as e:
        print(f"[FAIL] Could not initialize encoder: {e}")
        return 1
    
    # Create dummy input: (Batch=2, Channels=3, Height=512, Width=512)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    print(f"Input shape: {dummy_input.shape}")
    
    # Run forward pass
    try:
        print("Running forward pass...")
        output = encoder(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # Verify shape
        expected_seq_len = SEQUENCE_LENGTH # (512/32)^2 + 1 = 257
        expected_dim = ENCODER_EMBED_DIM # 768
        
        expected_shape = (batch_size, expected_seq_len, expected_dim)
        
        if output.shape == expected_shape:
            print(f"[OK] Output shape matches expected: {expected_shape}")
        else:
            print(f"[FAIL] Expected {expected_shape}, got {output.shape}")
            return 1
            
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return 1
        
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("[SUCCESS] Vision Encoder verified successfully!")
    print(f"Model ID: google/vit-base-patch32-384")
    print(f"Resolution: {IMAGE_SIZE}")
    print(f"Patch Size: {PATCH_SIZE}")
    print(f"Interpolation: Standard (224/384 -> 512)")
    
    return 0

if __name__ == "__main__":
    exit(test_vision_encoder())
