"""
Test the tokens_to_ohlcv functionality.
"""
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.tokenizer import CandleTokenizer

def main():
    # Initialize tokenizer
    tokenizer = CandleTokenizer()
    
    print("=" * 70)
    print("Testing tokens_to_ohlcv()")
    print("=" * 70)
    
    # Example: Last known candle
    last_candle = {
        'Open': 100.0,
        'High': 102.0,
        'Low': 99.0,
        'Close': 101.0,
        'Volume': 1000000
    }
    
    print(f"\nLast Known Candle:")
    print(f"  Close: ${last_candle['Close']:.2f}")
    
    # Example tokens (from a hypothetical prediction)
    test_tokens = [260, 177, 113, 107, 209]  # These were 100% accurate in our test!
    
    print(f"\nPredicted Tokens: {test_tokens}")
    print("\nDetokenized Characteristics:")
    for i, token in enumerate(test_tokens):
        chars = tokenizer.detokenize(token)
        print(f"  Token {token}: {chars['direction']}, {chars['body_size']} body, "
              f"{chars['upper_wick']}/{chars['lower_wick']} wicks, {chars['volume']} volume")
    
    # Convert to OHLCV
    print("\n" + "=" * 70)
    print("Converting to OHLCV...")
    print("=" * 70)
    
    predicted_ohlcv = tokenizer.tokens_to_ohlcv(test_tokens, last_candle)
    
    print("\nPredicted OHLCV:")
    print(predicted_ohlcv.to_string(index=False))
    
    # Calculate some statistics
    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    
    returns = (predicted_ohlcv['Close'] - predicted_ohlcv['Open']) / predicted_ohlcv['Open'] * 100
    print(f"\nReturns per candle (%): {returns.values}")
    print(f"Total return: {((predicted_ohlcv.iloc[-1]['Close'] / last_candle['Close']) - 1) * 100:.2f}%")
    print(f"Highest price: ${predicted_ohlcv['High'].max():.2f}")
    print(f"Lowest price: ${predicted_ohlcv['Low'].min():.2f}")
    
    # Test with different tokens
    print("\n" + "=" * 70)
    print("Test 2: Mixed directions")
    print("=" * 70)
    
    # Create some test tokens manually
    # Let's test: Bullish Large, Bearish Medium, Doji Small, Bullish Small, Bearish Large
    mixed_tokens = [257, 113, 191, 45, 333]
    
    print(f"\nTokens: {mixed_tokens}")
    for i, token in enumerate(mixed_tokens):
        chars = tokenizer.detokenize(token)
        print(f"  Token {token}: {chars['direction']}, {chars['body_size']}")
    
    predicted_ohlcv2 = tokenizer.tokens_to_ohlcv(mixed_tokens, last_candle)
    print("\nPredicted OHLCV:")
    print(predicted_ohlcv2.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
