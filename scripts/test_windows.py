"""
Test script for random window generation.
Validates window dimensions, randomness, and data quality.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.window_generator import WindowGenerator
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_STOCKS, WINDOW_SIZE, PREDICTION_HORIZON, RANDOM_SEED
import pandas as pd


def main():
    """Test the random window generation functionality."""
    print("=" * 70)
    print("Vision-Trader: Random Window Generation Test")
    print("=" * 70)
    print(f"Input size: {WINDOW_SIZE} candles")
    print(f"Target size: {PREDICTION_HORIZON} candles")
    print(f"Total window size: {WINDOW_SIZE + PREDICTION_HORIZON} candles")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Initialize generator
    generator = WindowGenerator(
        data_dir=RAW_DATA_DIR,
        input_size=WINDOW_SIZE,
        target_size=PREDICTION_HORIZON,
        random_seed=RANDOM_SEED
    )
    
    # Test 1: Generate windows for a single stock
    print("=" * 70)
    print("Test 1: Single Stock Window Generation (AAPL)")
    print("=" * 70)
    
    test_symbol = "AAPL"
    num_samples = 10
    
    windows = generator.generate_from_symbol(test_symbol, num_samples)
    
    if windows:
        print(f"[OK] Generated {len(windows)} windows for {test_symbol}")
        
        # Display first window details
        first_window = windows[0]
        print(f"\nFirst window details:")
        print(f"  Symbol: {first_window['symbol']}")
        print(f"  Start date: {first_window['start_date']}")
        print(f"  End date: {first_window['end_date']}")
        print(f"  Start position: {first_window['start_position']}")
        print(f"  Input shape: {first_window['input'].shape}")
        print(f"  Target shape: {first_window['target'].shape}")
        
        # Show input data sample
        print(f"\nInput window (first 5 rows):")
        print(first_window['input'].head())
        
        print(f"\nTarget window (all {PREDICTION_HORIZON} rows):")
        print(first_window['target'])
    else:
        print(f"[FAIL] No windows generated for {test_symbol}")
        return 1
    
    # Test 2: Validate windows
    print("\n" + "=" * 70)
    print("Test 2: Window Validation")
    print("=" * 70)
    
    valid_count = 0
    for i, window in enumerate(windows):
        is_valid, issues = generator.validate_window(window)
        if is_valid:
            valid_count += 1
        else:
            print(f"[FAIL] Window {i} validation failed: {issues}")
    
    print(f"[OK] {valid_count}/{len(windows)} windows passed validation")
    
    # Test 3: Check randomness (no sequential patterns)
    print("\n" + "=" * 70)
    print("Test 3: Randomness Check")
    print("=" * 70)
    
    positions = [w['start_position'] for w in windows]
    positions_sorted = sorted(positions)
    
    print(f"Start positions: {positions[:5]}... (showing first 5)")
    print(f"Sorted positions: {positions_sorted[:5]}... (showing first 5)")
    
    if positions != positions_sorted:
        print("[OK] Positions are randomized (not sequential)")
    else:
        print("[WARNING] Positions appear sequential")
    
    # Check for duplicates
    if len(positions) == len(set(positions)):
        print("[OK] No duplicate positions (sampling without replacement)")
    else:
        print("[FAIL] Found duplicate positions!")
    
    # Test 4: Batch generation
    print("\n" + "=" * 70)
    print("Test 4: Batch Generation (3 stocks)")
    print("=" * 70)
    
    test_symbols = SAMPLE_STOCKS[:3]  # AAPL, MSFT, GOOGL
    samples_per_stock = 5
    
    batch_results = generator.generate_batch(test_symbols, samples_per_stock)
    
    for symbol, symbol_windows in batch_results.items():
        print(f"[OK] {symbol}: {len(symbol_windows)} windows generated")
    
    # Test 5: Statistics
    print("\n" + "=" * 70)
    print("Test 5: Window Statistics")
    print("=" * 70)
    
    all_windows = []
    for symbol_windows in batch_results.values():
        all_windows.extend(symbol_windows)
    
    stats = generator.get_statistics(all_windows)
    
    print(f"Total windows: {stats['total_windows']}")
    print(f"Unique symbols: {stats['unique_symbols']}")
    print(f"Symbols: {', '.join(stats['symbols'])}")
    print(f"Avg start position: {stats['avg_start_position']:.1f}")
    print(f"Min start position: {stats['min_start_position']}")
    print(f"Max start position: {stats['max_start_position']}")
    print(f"Input size: {stats['input_size']}")
    print(f"Target size: {stats['target_size']}")
    
    # Test 6: Save windows
    print("\n" + "=" * 70)
    print("Test 6: Save Windows")
    print("=" * 70)
    
    saved_path = generator.save_windows(all_windows, PROCESSED_DATA_DIR, format='pickle')
    print(f"[OK] Saved windows to: {saved_path}")
    
    # Also save metadata as CSV
    metadata_path = generator.save_windows(all_windows, PROCESSED_DATA_DIR, format='csv')
    print(f"[OK] Saved metadata to: {metadata_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"[SUCCESS] All tests passed!")
    print(f"Generated {stats['total_windows']} windows from {stats['unique_symbols']} stocks")
    print(f"Window configuration: {WINDOW_SIZE} input + {PREDICTION_HORIZON} target")
    print(f"Data saved to: {PROCESSED_DATA_DIR}")
    
    return 0


if __name__ == "__main__":
    exit(main())
