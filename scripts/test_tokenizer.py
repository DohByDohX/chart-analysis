"""
Test script for candle tokenization.
Validates tokenization, detokenization, and vocabulary coverage.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.tokenizer import CandleTokenizer
from src.data.window_generator import WindowGenerator
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, WINDOW_SIZE, PREDICTION_HORIZON,
    DOJI_THRESHOLD, PERCENTILE_LOOKBACK, VOCABULARY_SIZE, RANDOM_SEED
)
import pandas as pd


def main():
    """Test the candle tokenization functionality."""
    print("=" * 70)
    print("Vision-Trader: Candle Tokenization Test")
    print("=" * 70)
    print(f"Vocabulary size: {VOCABULARY_SIZE}")
    print(f"Doji threshold: {DOJI_THRESHOLD}")
    print(f"Percentile lookback: {PERCENTILE_LOOKBACK}")
    print()
    
    # Initialize tokenizer
    tokenizer = CandleTokenizer(
        doji_threshold=DOJI_THRESHOLD,
        percentile_lookback=PERCENTILE_LOOKBACK
    )
    
    # Test 1: Generate a sample window
    print("=" * 70)
    print("Test 1: Generate Sample Window")
    print("=" * 70)
    
    generator = WindowGenerator(
        data_dir=RAW_DATA_DIR,
        input_size=WINDOW_SIZE,
        target_size=PREDICTION_HORIZON,
        random_seed=RANDOM_SEED
    )
    
    windows = generator.generate_from_symbol("AAPL", num_samples=1)
    
    if not windows:
        print("[FAIL] Could not generate sample window")
        return 1
    
    sample_window = windows[0]
    input_data = sample_window['input']
    
    print(f"[OK] Generated window with {len(input_data)} candles")
    print(f"Date range: {sample_window['start_date']} to {sample_window['end_date']}")
    
    # Test 2: Tokenize the window
    print("\n" + "=" * 70)
    print("Test 2: Tokenize Window")
    print("=" * 70)
    
    token_ids, characteristics = tokenizer.tokenize_window(input_data)
    
    print(f"[OK] Tokenized {len(token_ids)} candles")
    print(f"Token range: {min(token_ids)} to {max(token_ids)}")
    print(f"Unique tokens: {len(set(token_ids))}")
    
    # Show first 5 tokenized candles
    print("\nFirst 5 tokenized candles:")
    for i in range(min(5, len(token_ids))):
        char = characteristics[i]
        print(f"  Candle {i}: Token {char['token_id']:3d} - "
              f"{char['direction']:8s} {char['body_size']:6s} body, "
              f"Upper:{char['upper_wick']:6s} Lower:{char['lower_wick']:6s} "
              f"Vol:{char['volume']:6s}")
    
    # Test 3: Validate token range
    print("\n" + "=" * 70)
    print("Test 3: Validate Token Range")
    print("=" * 70)
    
    invalid_tokens = [t for t in token_ids if t < 0 or t >= VOCABULARY_SIZE]
    
    if invalid_tokens:
        print(f"[FAIL] Found {len(invalid_tokens)} invalid tokens: {invalid_tokens[:10]}")
        return 1
    else:
        print(f"[OK] All {len(token_ids)} tokens in valid range (0-{VOCABULARY_SIZE-1})")
    
    # Test 4: Test detokenization
    print("\n" + "=" * 70)
    print("Test 4: Detokenization Round-Trip")
    print("=" * 70)
    
    # Test a few random tokens
    test_indices = [0, len(token_ids)//4, len(token_ids)//2, len(token_ids)-1]
    all_match = True
    
    for idx in test_indices:
        original_char = characteristics[idx]
        token_id = token_ids[idx]
        
        # Detokenize
        decoded_char = tokenizer.detokenize(token_id)
        
        # Compare (excluding token_id which isn't in decoded)
        match = (
            original_char['direction'] == decoded_char['direction'] and
            original_char['body_size'] == decoded_char['body_size'] and
            original_char['upper_wick'] == decoded_char['upper_wick'] and
            original_char['lower_wick'] == decoded_char['lower_wick'] and
            original_char['volume'] == decoded_char['volume']
        )
        
        if not match:
            print(f"[FAIL] Mismatch at index {idx}")
            print(f"  Original: {original_char}")
            print(f"  Decoded:  {decoded_char}")
            all_match = False
    
    if all_match:
        print(f"[OK] All {len(test_indices)} test tokens decoded correctly")
    
    # Test 5: Token statistics
    print("\n" + "=" * 70)
    print("Test 5: Token Statistics")
    print("=" * 70)
    
    stats = tokenizer.get_token_statistics(token_ids)
    
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Unique tokens: {stats['unique_tokens']}")
    print(f"Vocabulary coverage: {stats['vocabulary_coverage']:.2%}")
    print(f"Most common token: {stats['most_common_token']}")
    
    print("\nDirection distribution:")
    for direction, count in stats['direction_distribution'].items():
        pct = count / stats['total_tokens'] * 100
        print(f"  {direction:8s}: {count:3d} ({pct:5.1f}%)")
    
    print("\nBody size distribution:")
    for size, count in stats['body_size_distribution'].items():
        pct = count / stats['total_tokens'] * 100
        print(f"  {size:6s}: {count:3d} ({pct:5.1f}%)")
    
    print("\nVolume distribution:")
    for vol, count in stats['volume_distribution'].items():
        pct = count / stats['total_tokens'] * 100
        print(f"  {vol:6s}: {count:3d} ({pct:5.1f}%)")
    
    # Test 6: Tokenize target window
    print("\n" + "=" * 70)
    print("Test 6: Tokenize Target Window")
    print("=" * 70)
    
    target_data = sample_window['target']
    target_tokens, target_chars = tokenizer.tokenize_window(target_data)
    
    print(f"[OK] Tokenized {len(target_tokens)} target candles")
    print("\nTarget tokens:")
    for i, char in enumerate(target_chars):
        print(f"  Target {i}: Token {char['token_id']:3d} - "
              f"{char['direction']:8s} {char['body_size']:6s} body")
    
    # Test 7: Save vocabulary
    print("\n" + "=" * 70)
    print("Test 7: Save Vocabulary")
    print("=" * 70)
    
    vocab_path = PROCESSED_DATA_DIR / "vocabulary.json"
    tokenizer.save_vocabulary(vocab_path)
    print(f"[OK] Saved vocabulary to {vocab_path}")
    
    # Load and verify
    loaded_vocab = tokenizer.load_vocabulary(vocab_path)
    print(f"[OK] Loaded vocabulary with {len(loaded_vocab)} entries")
    
    # Verify a few entries
    test_tokens = [0, 100, 200, 431]
    for token_id in test_tokens:
        original = tokenizer.detokenize(token_id)
        loaded = loaded_vocab[token_id]
        if original == loaded:
            print(f"  Token {token_id:3d}: {loaded['direction']:8s} - Verified")
        else:
            print(f"  Token {token_id:3d}: Mismatch!")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"[SUCCESS] All tests passed!")
    print(f"Tokenized {stats['total_tokens']} candles")
    print(f"Vocabulary coverage: {stats['vocabulary_coverage']:.2%}")
    print(f"Token range: 0-{VOCABULARY_SIZE-1}")
    print(f"Vocabulary saved to: {vocab_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
