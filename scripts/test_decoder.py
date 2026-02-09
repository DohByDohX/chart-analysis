import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.token_decoder import TokenDecoder
from config import (
    VOCABULARY_SIZE, ENCODER_EMBED_DIM,
    DECODER_NUM_LAYERS, DECODER_NUM_HEADS, DECODER_DROPOUT, MAX_TGT_SEQ_LEN
)

def test_token_decoder():
    print("=" * 70)
    print("Testing Token Decoder")
    print("=" * 70)
    
    # Initialize Decoder
    try:
        print("Initializing TokenDecoder...")
        decoder = TokenDecoder(
            vocab_size=VOCABULARY_SIZE,
            embed_dim=ENCODER_EMBED_DIM,
            num_heads=DECODER_NUM_HEADS,
            num_layers=DECODER_NUM_LAYERS,
            dropout=DECODER_DROPOUT,
            max_seq_len=MAX_TGT_SEQ_LEN
        )
        print("[OK] Decoder initialized")
    except Exception as e:
        print(f"[FAIL] Could not initialize decoder: {e}")
        return 1
    
    # Create dummy inputs
    batch_size = 2
    src_seq_len = 257 # ViT output length
    tgt_seq_len = 5 # Prediction horizon
    
    # Memory from Encoder: (Batch, SrcSeqLen, EmbedDim)
    memory = torch.randn(batch_size, src_seq_len, ENCODER_EMBED_DIM)
    
    # Target Tokens: (Batch, TgtSeqLen)
    # Using random integers in vocab range
    tgt_tokens = torch.randint(0, VOCABULARY_SIZE, (batch_size, tgt_seq_len))
    
    print(f"Memory shape: {memory.shape}")
    print(f"Target tokens shape: {tgt_tokens.shape}")
    
    # Run forward pass
    try:
        print("Running forward pass...")
        logits = decoder(tgt_tokens, memory)
        print(f"Logits shape: {logits.shape}")
        
        # Verify shape
        expected_shape = (batch_size, tgt_seq_len, VOCABULARY_SIZE)
        
        if logits.shape == expected_shape:
            print(f"[OK] Output shape matches expected: {expected_shape}")
        else:
            print(f"[FAIL] Expected {expected_shape}, got {logits.shape}")
            return 1
            
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return 1
        
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("[SUCCESS] Token Decoder verified successfully!")
    print(f"Layers: {DECODER_NUM_LAYERS}")
    print(f"Heads: {DECODER_NUM_HEADS}")
    print(f"Vocab Size: {VOCABULARY_SIZE}")
    
    return 0

if __name__ == "__main__":
    exit(test_token_decoder())
