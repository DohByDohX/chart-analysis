"""
Analyze token distribution in the training dataset.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.dataset import ChartDataset
from src.data.tokenizer import CandleTokenizer
from config import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("Step 2: Analyzing Token Distribution")
    logger.info("=" * 70)
    
    # Load dataset
    tokenizer = CandleTokenizer()
    tokenizer_path = DATA_DIR / "processed" / "tokenizer" / "vocabulary.json"
    tokenizer.load_vocabulary(str(tokenizer_path))
    
    windows_dir = DATA_DIR / "processed" / "windows"
    images_dir = DATA_DIR / "processed" / "images"
    dataset = ChartDataset(windows_dir=windows_dir, images_dir=images_dir, tokenizer=tokenizer)
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Collect all tokens
    all_tokens = []
    for i in range(len(dataset)):
        _, target_tokens = dataset[i]
        all_tokens.extend(target_tokens.numpy().tolist())
    
    logger.info(f"Total tokens: {len(all_tokens)}")
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    logger.info(f"Unique tokens: {len(token_counts)}")
    
    # Show top 20 most common tokens
    logger.info("\n--- Top 20 Most Common Tokens ---")
    for i, (token, count) in enumerate(token_counts.most_common(20)):
        percentage = (count / len(all_tokens)) * 100
        logger.info(f"{i+1:2d}. Token {token:3d}: {count:5d} times ({percentage:5.2f}%)")
    
    # Check token 0 specifically
    token_0_count = token_counts.get(0, 0)
    token_0_percentage = (token_0_count / len(all_tokens)) * 100
    logger.info(f"\n--- Token 0 Analysis ---")
    logger.info(f"Token 0 appears: {token_0_count} times ({token_0_percentage:.2f}%)")
    logger.info(f"This is rank #{list(token_counts.keys()).index(0) + 1 if 0 in token_counts else 'N/A'} most common")
    
    # Distribution statistics
    counts = list(token_counts.values())
    logger.info(f"\n--- Distribution Statistics ---")
    logger.info(f"Mean tokens per unique ID: {np.mean(counts):.2f}")
    logger.info(f"Median tokens per unique ID: {np.median(counts):.2f}")
    logger.info(f"Std dev: {np.std(counts):.2f}")
    logger.info(f"Min: {np.min(counts)}, Max: {np.max(counts)}")
    
    # Check if distribution is roughly uniform
    expected_count = len(all_tokens) / len(token_counts)
    logger.info(f"Expected count if uniform: {expected_count:.2f}")
    logger.info(f"Actual token 0 count: {token_0_count}")
    logger.info(f"Ratio (actual/expected): {token_0_count / expected_count:.2f}x")
    
    # Save plot
    plt.figure(figsize=(12, 6))
    sorted_tokens = sorted(token_counts.items())
    tokens, counts = zip(*sorted_tokens)
    plt.bar(range(len(tokens)), counts, alpha=0.7)
    plt.xlabel('Token ID (sorted)')
    plt.ylabel('Frequency')
    plt.title('Token Distribution in Training Data')
    plt.axhline(y=expected_count, color='r', linestyle='--', label=f'Expected if uniform ({expected_count:.0f})')
    # Highlight token 0
    if 0 in token_counts:
        token_0_idx = list(tokens).index(0)
        plt.bar(token_0_idx, token_0_count, color='red', alpha=0.8, label=f'Token 0 ({token_0_count})')
    plt.legend()
    plt.tight_layout()
    
    plot_path = DATA_DIR / "token_distribution.png"
    plt.savefig(plot_path)
    logger.info(f"\nSaved distribution plot to {plot_path}")
    
    logger.info("\n" + "=" * 70)

if __name__ == "__main__":
    main()
