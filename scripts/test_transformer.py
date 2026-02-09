import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.transformer import VisionTrader
from config import (
    VIT_MODEL_NAME, VOCABULARY_SIZE, ENCODER_EMBED_DIM,
    DECODER_NUM_LAYERS, DECODER_NUM_HEADS, DECODER_DROPOUT,
    MAX_TGT_SEQ_LEN, IMAGE_SIZE
)

def test_vision_trader():
    print("=" * 70)
    print("Testing VisionTrader (Full Model)")
    print("=" * 70)
    
    # Initialize Model
    try:
        print("Initializing VisionTrader...")
        model = VisionTrader(
            vit_model_name=VIT_MODEL_NAME,
            vocab_size=VOCABULARY_SIZE,
            embed_dim=ENCODER_EMBED_DIM,
            num_heads=DECODER_NUM_HEADS,
            num_layers=DECODER_NUM_LAYERS,
            dropout=DECODER_DROPOUT,
            max_seq_len=MAX_TGT_SEQ_LEN
        )
        print("[OK] Model initialized")
    except Exception as e:
        print(f"[FAIL] Could not initialize model: {e}")
        return 1
    
    # Test 1: Training Forward Pass
    print("\n" + "=" * 70)
    print("Test 1: Training Forward Pass")
    print("=" * 70)
    
    batch_size = 2
    seq_len = 5
    
    # Create dummy inputs
    images = torch.randn(batch_size, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    tgt_tokens = torch.randint(0, VOCABULARY_SIZE, (batch_size, seq_len))
    
    print(f"Images shape: {images.shape}")
    print(f"Target tokens shape: {tgt_tokens.shape}")
    
    try:
        print("Running forward pass...")
        logits = model(images, tgt_tokens)
        print(f"Logits shape: {logits.shape}")
        
        expected_shape = (batch_size, seq_len, VOCABULARY_SIZE)
        if logits.shape == expected_shape:
            print(f"[OK] Output shape correct: {expected_shape}")
        else:
            print(f"[FAIL] Expected {expected_shape}, got {logits.shape}")
            return 1
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return 1
    
    # Test 2: Generation
    print("\n" + "=" * 70)
    print("Test 2: Autoregressive Generation")
    print("=" * 70)
    
    start_token = 0  # Using 0 as start token
    max_len = 5
    
    try:
        print(f"Generating {max_len} tokens...")
        generated = model.generate_greedy(images, start_token, max_len)
        print(f"Generated tokens shape: {generated.shape}")
        
        expected_shape = (batch_size, max_len)
        if generated.shape == expected_shape:
            print(f"[OK] Generation shape correct: {expected_shape}")
            print(f"Sample generated sequence: {generated[0].tolist()}")
        else:
            print(f"[FAIL] Expected {expected_shape}, got {generated.shape}")
            return 1
            
        # Verify tokens are in vocab range
        if (generated >= 0).all() and (generated < VOCABULARY_SIZE).all():
            print(f"[OK] All tokens in valid range [0, {VOCABULARY_SIZE})")
        else:
            print("[FAIL] Generated tokens out of vocabulary range")
            return 1
            
    except Exception as e:
        print(f"[FAIL] Generation failed: {e}")
        return 1
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("[SUCCESS] VisionTrader verified successfully!")
    print(f"Encoder: {VIT_MODEL_NAME}")
    print(f"Decoder: {DECODER_NUM_LAYERS} layers, {DECODER_NUM_HEADS} heads")
    print(f"Vocab Size: {VOCABULARY_SIZE}")
    
    return 0

if __name__ == "__main__":
    exit(test_vision_trader())
