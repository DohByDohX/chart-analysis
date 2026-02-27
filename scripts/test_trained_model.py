import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detokenize_sequence(token_ids, tokenizer):
    """Convert token IDs back to candle characteristics."""
    candles = []
    for token_id in token_ids:
        # Get the characteristics from vocabulary
        vocab_item = tokenizer.vocabulary.get(token_id)
        if vocab_item:
            candles.append(vocab_item)
        else:
            candles.append({'token_id': token_id, 'error': 'Unknown token'})
    return candles

def main():
    logger.info("=" * 70)
    logger.info("Testing Trained VisionTrader Model")
    logger.info("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = CandleTokenizer()
    tokenizer_path = DATA_DIR / "processed" / "tokenizer" / "vocabulary.json"
    tokenizer.load_vocabulary(str(tokenizer_path))
    
    # Load dataset
    logger.info("Loading dataset...")
    windows_dir = DATA_DIR / "processed" / "windows"
    images_dir = DATA_DIR / "processed" / "images"
    
    dataset = ChartDataset(
        windows_dir=windows_dir,
        images_dir=images_dir,
        tokenizer=tokenizer
    )
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = VisionTrader(
        vit_model_name=VIT_MODEL_NAME,
        vocab_size=VOCABULARY_SIZE,
        embed_dim=ENCODER_EMBED_DIM,
        num_heads=DECODER_NUM_HEADS,
        num_layers=DECODER_NUM_LAYERS,
        dropout=DECODER_DROPOUT,
        max_seq_len=MAX_TGT_SEQ_LEN
    ).to(device)
    
    # Load trained checkpoint
    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
    
    # Test on a few samples
    num_test_samples = 5
    logger.info(f"\nTesting on {num_test_samples} random samples...")
    
    test_indices = np.random.choice(len(dataset), num_test_samples, replace=False)
    
    for i, idx in enumerate(test_indices):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Sample {i+1}/{num_test_samples} (Dataset Index: {idx})")
        logger.info(f"{'=' * 70}")
        
        # Get sample
        image, target_tokens = dataset[idx]
        window_data = dataset.get_window_data(idx)
        
        logger.info(f"Symbol: {window_data['symbol']}")
        logger.info(f"Date Range: {window_data['start_date']} to {window_data['end_date']}")
        logger.info(f"Target sequence length: {len(target_tokens)}")
        
        # Prepare input
        image_batch = image.unsqueeze(0).to(device)  # (1, 3, H, W)
        
        # Generate prediction
        with torch.no_grad():
            # Use START token for generation
            from config import START_TOKEN
            predicted_tokens = model.generate_greedy(
                image_batch,
                start_token=START_TOKEN,
                max_len=len(target_tokens)
            )
        
        # Convert to numpy for comparison
        predicted_tokens_np = predicted_tokens[0].cpu().numpy()
        target_tokens_np = target_tokens.numpy()
        
        logger.info(f"\nGround Truth Tokens: {target_tokens_np}")
        logger.info(f"Predicted Tokens:    {predicted_tokens_np}")
        
        # Calculate accuracy
        correct = (predicted_tokens_np == target_tokens_np).sum()
        accuracy = correct / len(target_tokens_np) * 100
        logger.info(f"Token Accuracy: {correct}/{len(target_tokens_np)} ({accuracy:.1f}%)")
        
        # Show token comparison
        logger.info("\n--- Token Comparison ---")
        for j in range(len(target_tokens_np)):
            match = "✓" if target_tokens_np[j] == predicted_tokens_np[j] else "✗"
            logger.info(f"Candle {j+1}: GT={target_tokens_np[j]:3d}, Pred={predicted_tokens_np[j]:3d} {match}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Testing completed!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
