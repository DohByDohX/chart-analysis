"""
Predict future candles from chart images using the trained VisionTrader model.
"""
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
    MAX_TGT_SEQ_LEN, DATA_DIR, CHECKPOINT_DIR, START_TOKEN
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("Future Candle Prediction with OHLCV Output")
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
    logger.info(f"\nPredicting future candles for {num_test_samples} samples...\n")
    
    test_indices = [0, 100, 200, 300, 400]  # Sample from different parts of dataset
    
    for i, idx in enumerate(test_indices):
        logger.info(f"{'=' * 70}")
        logger.info(f"Sample {i+1}/{num_test_samples} (Dataset Index: {idx})")
        logger.info(f"{'=' * 70}")
        
        # Get sample
        image, target_tokens = dataset[idx]
        window_data = dataset.get_window_data(idx)
        
        logger.info(f"Symbol: {window_data['symbol']}")
        logger.info(f"Date Range: {window_data['start_date']} to {window_data['end_date']}")
        
        # Get last candle from input window
        input_window_df = pd.DataFrame(window_data['input_window'])
        last_input_candle = input_window_df.iloc[-1]
        
        last_candle = {
            'Close': last_input_candle['Close'],
            'Open': last_input_candle['Open'],
            'High': last_input_candle['High'],
            'Low': last_input_candle['Low'],
            'Volume': last_input_candle['Volume']
        }
        
        logger.info(f"Last input candle Close: ${last_candle['Close']:.2f}")
        
        # Prepare input
        image_batch = image.unsqueeze(0).to(device)  # (1, 3, H, W)
        
        # Generate prediction
        with torch.no_grad():
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
        
        # Convert predicted tokens to OHLCV
        logger.info("\n--- Predicted OHLCV ---")
        predicted_ohlcv = tokenizer.tokens_to_ohlcv(predicted_tokens_np.tolist(), last_candle)
        logger.info(predicted_ohlcv.to_string(index=False))
        
        # Calculate statistics
        total_return = ((predicted_ohlcv.iloc[-1]['Close'] / last_candle['Close']) - 1) * 100
        logger.info(f"\nPredicted Return: {total_return:+.2f}%")
        logger.info(f"Price Range: ${predicted_ohlcv['Low'].min():.2f} - ${predicted_ohlcv['High'].max():.2f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Prediction completed!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
