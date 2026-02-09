"""
Test script to validate data download functionality.
Downloads sample stocks and verifies data quality.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.downloader import StockDataDownloader
from config import RAW_DATA_DIR, SAMPLE_STOCKS, DEFAULT_START_DATE, DEFAULT_END_DATE


def main():
    """Test the data download functionality."""
    print("=" * 60)
    print("Vision-Trader: Data Download Test")
    print("=" * 60)
    
    # Initialize downloader
    downloader = StockDataDownloader(RAW_DATA_DIR)
    
    # Test 1: Download sample stocks
    print(f"\nTest 1: Downloading {len(SAMPLE_STOCKS)} sample stocks...")
    print(f"Date range: {DEFAULT_START_DATE} to {DEFAULT_END_DATE}")
    print(f"Symbols: {', '.join(SAMPLE_STOCKS)}")
    print()
    
    results = downloader.download_multiple(
        symbols=SAMPLE_STOCKS,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
        save=True
    )
    
    # Test 2: Validate downloaded data
    print("\n" + "=" * 60)
    print("Test 2: Validating downloaded data...")
    print("=" * 60)
    
    valid_count = 0
    for symbol, data in results.items():
        if downloader.validate_data(data, symbol):
            valid_count += 1
            print(f"[OK] {symbol}: {len(data)} records, "
                  f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Test 3: Test loading from CSV
    print("\n" + "=" * 60)
    print("Test 3: Testing CSV load functionality...")
    print("=" * 60)
    
    if SAMPLE_STOCKS:
        test_symbol = SAMPLE_STOCKS[0]
        loaded_data = downloader.load_from_csv(test_symbol)
        
        if loaded_data is not None:
            print(f"[OK] Successfully loaded {test_symbol} from CSV")
            print(f"  Records: {len(loaded_data)}")
            print(f"  Columns: {', '.join(loaded_data.columns)}")
        else:
            print(f"[FAIL] Failed to load {test_symbol} from CSV")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total stocks attempted: {len(SAMPLE_STOCKS)}")
    print(f"Successfully downloaded: {len(results)}")
    print(f"Validation passed: {valid_count}")
    print(f"Data saved to: {RAW_DATA_DIR}")
    
    if len(results) == len(SAMPLE_STOCKS) and valid_count == len(results):
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Check logs above.")
        return 1


if __name__ == "__main__":
    exit(main())
