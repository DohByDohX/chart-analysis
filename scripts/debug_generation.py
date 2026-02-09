"""
Debug script to investigate why the model generates only token 0.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.transformer import VisionTrader
from src.training.dataset import ChartDataset
from src.data.tokenizer import CandleTokenizer
from config import (
    VIT_MODEL_NAME, VOCABULARY_SIZE, ENCODER_EMBED_DIM,
    DECODER_NUM_LAYERS, DECODER_NUM_HEADS, DECODER_DROPOUT,
    MAX_TGT_SEQ_LEN, DATA_DIR, CHECKPOINT_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("Step 1: Debugging Generation Function")
    logger.info("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load tokenizer and dataset
    tokenizer = CandleTokenizer()
    tokenizer_path = DATA_DIR / "processed" / "tokenizer" / "vocabulary.json"
    tokenizer.load_vocabulary(str(tokenizer_path))
    
    windows_dir = DATA_DIR / "processed" / "windows"
    images_dir = DATA_DIR / "processed" / "images"
    dataset = ChartDataset(windows_dir=windows_dir, images_dir=images_dir, tokenizer=tokenizer)
    
    # Load model
    model = VisionTrader(
        vit_model_name=VIT_MODEL_NAME,
        vocab_size=VOCABULARY_SIZE,
        embed_dim=ENCODER_EMBED_DIM,
        num_heads=DECODER_NUM_HEADS,
        num_layers=DECODER_NUM_LAYERS,
        dropout=DECODER_DROPOUT,
        max_seq_len=MAX_TGT_SEQ_LEN
    ).to(device)
    
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Get a sample
    idx = 0
    image, target_tokens = dataset[idx]
    image_batch = image.unsqueeze(0).to(device)
    
    logger.info(f"\nTarget tokens: {target_tokens.numpy()}")
    
    # Step 1: Check encoder output
    logger.info("\n--- Checking Encoder ---")
    with torch.no_grad():
        memory = model.encoder(image_batch)
    logger.info(f"Encoder output shape: {memory.shape}")
    logger.info(f"Encoder output stats: min={memory.min():.4f}, max={memory.max():.4f}, mean={memory.mean():.4f}")
    
    # Step 2: Check decoder with ground truth tokens
    logger.info("\n--- Checking Decoder with Ground Truth ---")
    target_batch = target_tokens.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model.decoder(target_batch, memory)
    logger.info(f"Decoder logits shape: {logits.shape}")
    logger.info(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
    
    # Check what tokens the logits predict
    predicted_from_logits = logits.argmax(dim=-1)[0].cpu().numpy()
    logger.info(f"Predicted tokens from ground truth input: {predicted_from_logits}")
    logger.info(f"Accuracy on ground truth: {(predicted_from_logits == target_tokens.numpy()).sum()}/{len(target_tokens)}")
    
    # Step 3: Check generation step by step
    logger.info("\n--- Checking Greedy Generation Step-by-Step ---")
    start_token = 0
    generated = torch.tensor([[start_token]], dtype=torch.long, device=device)
    
    logger.info(f"Start token: {start_token}")
    
    for step in range(5):
        with torch.no_grad():
            logits = model.decoder(generated, memory)
        
        # Get logits for the last position
        last_logits = logits[0, -1, :]  # (vocab_size,)
        
        # Check top 5 predictions
        top5_values, top5_indices = torch.topk(last_logits, 5)
        logger.info(f"\nStep {step + 1}:")
        logger.info(f"  Current sequence: {generated[0].cpu().numpy()}")
        logger.info(f"  Top 5 predictions:")
        for i, (idx, val) in enumerate(zip(top5_indices, top5_values)):
            logger.info(f"    {i+1}. Token {idx.item():3d}: logit={val.item():.4f}")
        
        next_token = last_logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
        logger.info(f"  Next token (argmax): {next_token.item()}")
        
        generated = torch.cat([generated, next_token], dim=1)
    
    final_generated = generated[0].cpu().numpy()
    logger.info(f"\nFinal generated sequence: {final_generated}")
    logger.info(f"Target sequence:          {target_tokens.numpy()}")
    
    # Step 4: Check if all logits are similar (indicating no learning)
    logger.info("\n--- Checking Logit Distribution ---")
    with torch.no_grad():
        # Generate logits for a few different start tokens
        for start_tok in [0, 1, 50, 100, 200]:
            gen = torch.tensor([[start_tok]], dtype=torch.long, device=device)
            logits = model.decoder(gen, memory)
            last_logits = logits[0, -1, :]
            top_token = last_logits.argmax().item()
            logger.info(f"Start token {start_tok:3d} -> predicts token {top_token:3d} (logit={last_logits[top_token]:.4f})")

if __name__ == "__main__":
    main()
