"""
Visualize model predictions by generating comparison charts.

Creates side-by-side and overlay charts comparing actual vs predicted candles.
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
from src.visualization.future_renderer import FutureRenderer
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
    logger.info("Generating Prediction Visualizations")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir = DATA_DIR / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
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
    dataset = ChartDataset(
        windows_dir=DATA_DIR / "processed" / "windows",
        images_dir=DATA_DIR / "processed" / "images",
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Initialize renderer
    renderer = FutureRenderer(output_size=(1200, 600), dpi=100)
    
    # Generate visualizations for multiple samples
    num_samples = 10
    test_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    
    logger.info(f"\nGenerating visualizations for {num_samples} samples...\n")
    
    summary_lines = []
    
    for i, idx in enumerate(test_indices[:num_samples]):
        logger.info(f"{'=' * 70}")
        logger.info(f"Sample {i+1}/{num_samples} (Dataset Index: {idx})")
        logger.info(f"{'=' * 70}")
        
        # Get sample
        image, target_tokens = dataset[idx]
        window_data = dataset.get_window_data(idx)
        
        symbol = window_data['symbol']
        date_range = f"{window_data['start_date']} to {window_data['end_date']}"
        
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Date Range: {date_range}")
        
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
        
        # Generate predictions
        with torch.no_grad():
            predicted_tokens = model.generate_greedy(
                image.unsqueeze(0).to(device),
                start_token=START_TOKEN,
                max_len=len(target_tokens)
            )
        
        predicted_tokens_np = predicted_tokens[0].cpu().numpy()
        target_tokens_np = target_tokens.numpy()
        
        # Calculate accuracy
        correct = (predicted_tokens_np == target_tokens_np).sum()
        accuracy = correct / len(target_tokens_np) * 100
        
        logger.info(f"Token Accuracy: {correct}/{len(target_tokens_np)} ({accuracy:.1f}%)")
        
        # Convert tokens to OHLCV
        predicted_ohlcv = tokenizer.tokens_to_ohlcv(predicted_tokens_np.tolist(), last_candle)
        
        # Get actual target OHLCV
        target_window_df = pd.DataFrame(window_data['target_window'])
        actual_ohlcv = target_window_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Create sample directory
        sample_dir = output_dir / f"sample_{i:03d}_{symbol}"
        sample_dir.mkdir(exist_ok=True)
        
        # Generate visualizations
        # 1. Side-by-side comparison
        side_by_side_path = sample_dir / f"{symbol}_side_by_side.png"
        title = f"{symbol} - Accuracy: {accuracy:.1f}%"
        renderer.render_comparison(
            actual_ohlcv, predicted_ohlcv,
            side_by_side_path, title, mode="side_by_side"
        )
        
        # 2. Overlay comparison
        overlay_path = sample_dir / f"{symbol}_overlay.png"
        renderer.render_comparison(
            actual_ohlcv, predicted_ohlcv,
            overlay_path, title, mode="overlay"
        )
        
        # 3. Predicted only
        predicted_path = sample_dir / f"{symbol}_predicted.png"
        renderer.render_prediction_only(
            predicted_ohlcv, predicted_path,
            title=f"{symbol} - Predicted Candles"
        )
        
        # Calculate statistics
        actual_return = ((actual_ohlcv.iloc[-1]['Close'] / last_candle['Close']) - 1) * 100
        predicted_return = ((predicted_ohlcv.iloc[-1]['Close'] / last_candle['Close']) - 1) * 100
        return_error = abs(predicted_return - actual_return)
        
        logger.info(f"Actual Return: {actual_return:+.2f}%")
        logger.info(f"Predicted Return: {predicted_return:+.2f}%")
        logger.info(f"Return Error: {return_error:.2f}%")
        logger.info(f"Saved visualizations to {sample_dir}")
        
        # Add to summary
        summary_lines.append(
            f"Sample {i+1:2d} | {symbol:6s} | Accuracy: {accuracy:5.1f}% | "
            f"Actual: {actual_return:+6.2f}% | Predicted: {predicted_return:+6.2f}% | "
            f"Error: {return_error:5.2f}%"
        )
    
    # Save summary report
    summary_path = output_dir / "summary_report.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("PREDICTION VISUALIZATION SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\n" + "=" * 100 + "\n")
        
        # Calculate aggregate statistics
        avg_accuracy = np.mean([float(line.split("Accuracy:")[1].split("%")[0]) for line in summary_lines])
        avg_error = np.mean([float(line.split("Error:")[1].split("%")[0]) for line in summary_lines])
        
        f.write(f"\nAverage Token Accuracy: {avg_accuracy:.1f}%\n")
        f.write(f"Average Return Error: {avg_error:.2f}%\n")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Summary report saved to {summary_path}")
    logger.info("Visualization generation completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
