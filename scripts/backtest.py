"""
Comprehensive backtesting script for evaluating VisionTrader model.

Runs inference on test set and generates detailed performance report.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import logging
from typing import List, Dict
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.transformer import VisionTrader
from src.training.dataset import ChartDataset
from src.data.tokenizer import CandleTokenizer
from src.evaluation.metrics import PredictionMetrics
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
    logger.info("VisionTrader Model Backtesting")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir = DATA_DIR / "evaluation"
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Run backtest on samples
    num_samples = 50  # Test on 50 samples for comprehensive evaluation
    test_indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    logger.info(f"\nRunning backtest on {num_samples} samples...\n")
    
    results = []
    
    for i, idx in enumerate(test_indices):
        if (i + 1) % 10 == 0:
            logger.info(f"Processing sample {i+1}/{num_samples}...")
        
        # Get sample
        image, target_tokens = dataset[idx]
        window_data = dataset.get_window_data(idx)
        
        symbol = window_data['symbol']
        
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
        
        # Convert tokens to OHLCV
        predicted_ohlcv = tokenizer.tokens_to_ohlcv(predicted_tokens_np.tolist(), last_candle)
        
        # Get actual target OHLCV
        target_window_df = pd.DataFrame(window_data['target_window'])
        actual_ohlcv = target_window_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Calculate metrics
        metrics = PredictionMetrics(
            actual_ohlcv=actual_ohlcv,
            predicted_ohlcv=predicted_ohlcv,
            actual_tokens=target_tokens_np,
            predicted_tokens=predicted_tokens_np,
            last_close=last_candle['Close']
        )
        
        summary = metrics.summary()
        
        # Store result
        result = {
            'sample_idx': int(idx),
            'symbol': symbol,
            'date_range': f"{window_data['start_date']} to {window_data['end_date']}",
            'token_accuracy': summary['token_accuracy'],
            'directional_accuracy': summary['directional_accuracy'],
            'mae_close': summary['mae']['Close'],
            'rmse_close': summary['rmse']['Close'],
            'mape_close': summary['mape']['Close'],
            'actual_return': summary['return']['actual_return'],
            'predicted_return': summary['return']['predicted_return'],
            'return_error': summary['return']['return_error'],
            'return_direction_correct': summary['return']['return_direction_correct'],
            'sequence_length': summary['sequence_length']
        }
        
        results.append(result)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = output_dir / "backtest_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved detailed results to {csv_path}")
    
    # Calculate aggregate statistics
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATE PERFORMANCE METRICS")
    logger.info("=" * 70)
    
    aggregate_stats = {
        'token_accuracy': {
            'mean': results_df['token_accuracy'].mean(),
            'median': results_df['token_accuracy'].median(),
            'std': results_df['token_accuracy'].std(),
            'min': results_df['token_accuracy'].min(),
            'max': results_df['token_accuracy'].max()
        },
        'directional_accuracy': {
            'mean': results_df['directional_accuracy'].mean(),
            'median': results_df['directional_accuracy'].median(),
            'std': results_df['directional_accuracy'].std(),
            'min': results_df['directional_accuracy'].min(),
            'max': results_df['directional_accuracy'].max()
        },
        'mae_close': {
            'mean': results_df['mae_close'].mean(),
            'median': results_df['mae_close'].median(),
            'std': results_df['mae_close'].std(),
            'min': results_df['mae_close'].min(),
            'max': results_df['mae_close'].max()
        },
        'rmse_close': {
            'mean': results_df['rmse_close'].mean(),
            'median': results_df['rmse_close'].median(),
            'std': results_df['rmse_close'].std(),
            'min': results_df['rmse_close'].min(),
            'max': results_df['rmse_close'].max()
        },
        'return_error': {
            'mean': results_df['return_error'].mean(),
            'median': results_df['return_error'].median(),
            'std': results_df['return_error'].std(),
            'min': results_df['return_error'].min(),
            'max': results_df['return_error'].max()
        },
        'return_direction_accuracy': {
            'percentage': (results_df['return_direction_correct'].sum() / len(results_df)) * 100
        }
    }
    
    # Print aggregate stats
    logger.info(f"\nToken Accuracy:")
    logger.info(f"  Mean:   {aggregate_stats['token_accuracy']['mean']:.1f}%")
    logger.info(f"  Median: {aggregate_stats['token_accuracy']['median']:.1f}%")
    logger.info(f"  Std:    {aggregate_stats['token_accuracy']['std']:.1f}%")
    logger.info(f"  Range:  {aggregate_stats['token_accuracy']['min']:.1f}% - {aggregate_stats['token_accuracy']['max']:.1f}%")
    
    logger.info(f"\nDirectional Accuracy:")
    logger.info(f"  Mean:   {aggregate_stats['directional_accuracy']['mean']:.1f}%")
    logger.info(f"  Median: {aggregate_stats['directional_accuracy']['median']:.1f}%")
    logger.info(f"  Std:    {aggregate_stats['directional_accuracy']['std']:.1f}%")
    logger.info(f"  Range:  {aggregate_stats['directional_accuracy']['min']:.1f}% - {aggregate_stats['directional_accuracy']['max']:.1f}%")
    
    logger.info(f"\nClose Price MAE:")
    logger.info(f"  Mean:   ${aggregate_stats['mae_close']['mean']:.2f}")
    logger.info(f"  Median: ${aggregate_stats['mae_close']['median']:.2f}")
    logger.info(f"  Std:    ${aggregate_stats['mae_close']['std']:.2f}")
    
    logger.info(f"\nClose Price RMSE:")
    logger.info(f"  Mean:   ${aggregate_stats['rmse_close']['mean']:.2f}")
    logger.info(f"  Median: ${aggregate_stats['rmse_close']['median']:.2f}")
    
    logger.info(f"\nReturn Error:")
    logger.info(f"  Mean:   {aggregate_stats['return_error']['mean']:.2f}%")
    logger.info(f"  Median: {aggregate_stats['return_error']['median']:.2f}%")
    logger.info(f"  Std:    {aggregate_stats['return_error']['std']:.2f}%")
    
    logger.info(f"\nReturn Direction Accuracy: {aggregate_stats['return_direction_accuracy']['percentage']:.1f}%")
    
    # Performance by symbol
    logger.info("\n" + "=" * 70)
    logger.info("PERFORMANCE BY SYMBOL")
    logger.info("=" * 70)
    
    symbol_stats = results_df.groupby('symbol').agg({
        'token_accuracy': 'mean',
        'directional_accuracy': 'mean',
        'mae_close': 'mean',
        'return_error': 'mean',
        'return_direction_correct': 'mean'
    }).round(2)
    
    for symbol, row in symbol_stats.iterrows():
        logger.info(f"\n{symbol}:")
        logger.info(f"  Token Accuracy: {row['token_accuracy']:.1f}%")
        logger.info(f"  Directional Accuracy: {row['directional_accuracy']:.1f}%")
        logger.info(f"  MAE (Close): ${row['mae_close']:.2f}")
        logger.info(f"  Return Error: {row['return_error']:.2f}%")
        logger.info(f"  Return Direction Accuracy: {row['return_direction_correct']*100:.1f}%")
    
    # Save aggregate stats
    stats_path = output_dir / "aggregate_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(aggregate_stats, f, indent=2)
    logger.info(f"\nSaved aggregate stats to {stats_path}")
    
    # Generate markdown report
    generate_report(results_df, aggregate_stats, symbol_stats, output_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info("Backtesting completed!")
    logger.info("=" * 70)


def generate_report(
    results_df: pd.DataFrame,
    aggregate_stats: Dict,
    symbol_stats: pd.DataFrame,
    output_dir: Path
):
    """Generate markdown performance report."""
    report_path = output_dir / "performance_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# VisionTrader Model - Performance Report\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"**Test Samples**: {len(results_df)}\n\n")
        f.write("**Key Metrics**:\n")
        f.write(f"- **Token Accuracy**: {aggregate_stats['token_accuracy']['mean']:.1f}%\n")
        f.write(f"- **Directional Accuracy**: {aggregate_stats['directional_accuracy']['mean']:.1f}%\n")
        f.write(f"- **MAE (Close)**: ${aggregate_stats['mae_close']['mean']:.2f}\n")
        f.write(f"- **Return Error**: {aggregate_stats['return_error']['mean']:.2f}%\n")
        f.write(f"- **Return Direction Accuracy**: {aggregate_stats['return_direction_accuracy']['percentage']:.1f}%\n\n")
        
        # Detailed Metrics
        f.write("## Detailed Performance Metrics\n\n")
        
        f.write("### Token Accuracy\n")
        f.write(f"- Mean: {aggregate_stats['token_accuracy']['mean']:.1f}%\n")
        f.write(f"- Median: {aggregate_stats['token_accuracy']['median']:.1f}%\n")
        f.write(f"- Std Dev: {aggregate_stats['token_accuracy']['std']:.1f}%\n")
        f.write(f"- Range: {aggregate_stats['token_accuracy']['min']:.1f}% - {aggregate_stats['token_accuracy']['max']:.1f}%\n\n")
        
        f.write("### Directional Accuracy\n")
        f.write(f"- Mean: {aggregate_stats['directional_accuracy']['mean']:.1f}%\n")
        f.write(f"- Median: {aggregate_stats['directional_accuracy']['median']:.1f}%\n")
        f.write(f"- Std Dev: {aggregate_stats['directional_accuracy']['std']:.1f}%\n")
        f.write(f"- Range: {aggregate_stats['directional_accuracy']['min']:.1f}% - {aggregate_stats['directional_accuracy']['max']:.1f}%\n\n")
        
        f.write("### Price Error Metrics\n")
        f.write(f"**MAE (Close)**:\n")
        f.write(f"- Mean: ${aggregate_stats['mae_close']['mean']:.2f}\n")
        f.write(f"- Median: ${aggregate_stats['mae_close']['median']:.2f}\n\n")
        f.write(f"**RMSE (Close)**:\n")
        f.write(f"- Mean: ${aggregate_stats['rmse_close']['mean']:.2f}\n")
        f.write(f"- Median: ${aggregate_stats['rmse_close']['median']:.2f}\n\n")
        
        f.write("### Return Forecasting\n")
        f.write(f"- Mean Return Error: {aggregate_stats['return_error']['mean']:.2f}%\n")
        f.write(f"- Median Return Error: {aggregate_stats['return_error']['median']:.2f}%\n")
        f.write(f"- Return Direction Accuracy: {aggregate_stats['return_direction_accuracy']['percentage']:.1f}%\n\n")
        
        # Performance by Symbol
        f.write("## Performance by Symbol\n\n")
        f.write("| Symbol | Token Acc | Dir Acc | MAE (Close) | Return Error | Return Dir Acc |\n")
        f.write("|--------|-----------|---------|-------------|--------------|----------------|\n")
        
        for symbol, row in symbol_stats.iterrows():
            f.write(f"| {symbol} | {row['token_accuracy']:.1f}% | "
                   f"{row['directional_accuracy']:.1f}% | "
                   f"${row['mae_close']:.2f} | "
                   f"{row['return_error']:.2f}% | "
                   f"{row['return_direction_correct']*100:.1f}% |\n")
        
        f.write("\n")
        
        # Top and Bottom Performers
        f.write("## Top 5 Predictions\n\n")
        top_5 = results_df.nsmallest(5, 'return_error')[['symbol', 'date_range', 'token_accuracy', 'return_error']]
        f.write("| Symbol | Date Range | Token Acc | Return Error |\n")
        f.write("|--------|------------|-----------|-------------|\n")
        for _, row in top_5.iterrows():
            f.write(f"| {row['symbol']} | {row['date_range']} | {row['token_accuracy']:.1f}% | {row['return_error']:.2f}% |\n")
        
        f.write("\n## Bottom 5 Predictions\n\n")
        bottom_5 = results_df.nlargest(5, 'return_error')[['symbol', 'date_range', 'token_accuracy', 'return_error']]
        f.write("| Symbol | Date Range | Token Acc | Return Error |\n")
        f.write("|--------|------------|-----------|-------------|\n")
        for _, row in bottom_5.iterrows():
            f.write(f"| {row['symbol']} | {row['date_range']} | {row['token_accuracy']:.1f}% | {row['return_error']:.2f}% |\n")
        
        f.write("\n")
    
    logger.info(f"Generated performance report: {report_path}")


if __name__ == "__main__":
    main()
