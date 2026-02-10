"""
Download stock data with random date ranges for each ticker.

Per OSVariables.md:
- No I/O constraints for downloads (not writing images)
- Sequential downloads to avoid API rate limiting
- Random date ranges within past 20 years for diversity
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random
import logging
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import SAMPLE_STOCKS, DATA_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)

def generate_random_date_range():
    """
    Generate a random date range within the past 20 years.
    Ensures at least 2 years of data for sufficient windows.
    """
    # Reference date: today
    end_date = datetime.now()
    
    # Random end date: somewhere in the past 20 years
    days_back = random.randint(365, 20*365)  # 1-20 years ago
    random_end = end_date - timedelta(days=days_back)
    
    # Start date: 3-5 years before the random end date (ensures enough data)
    years_of_data = random.uniform(3.0, 5.0)
    random_start = random_end - timedelta(days=int(years_of_data * 365))
    
    return random_start.strftime('%Y-%m-%d'), random_end.strftime('%Y-%m-%d')


def main():
    logger.info("=" * 70)
    logger.info("Downloading Stock Data with Random Date Ranges")
    logger.info("=" * 70)
    
    # Output directory
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {raw_dir}")
    logger.info(f"Total stocks to download: {len(SAMPLE_STOCKS)}\n")
    
    successful = 0
    failed = []
    
    for idx, symbol in enumerate(SAMPLE_STOCKS, 1):
        logger.info(f"[{idx}/{len(SAMPLE_STOCKS)}] Processing {symbol}...")
        
        # Generate random date range for this ticker
        start_date, end_date = generate_random_date_range()
        logger.info(f"  Date range: {start_date} to {end_date}")
        
        try:
            # Check if already exists
            csv_path = raw_dir / f"{symbol}.csv"
            if csv_path.exists():
                logger.info(f"  ✓ Already exists, skipping download")
                successful += 1
                continue
            
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"  ✗ No data returned for {symbol}")
                failed.append(symbol)
                continue
            
            # Validate data
            if len(data) < 200:  # Need at least 200 days for meaningful windows
                logger.warning(f"  ✗ Insufficient data ({len(data)} days) for {symbol}")
                failed.append(symbol)
                continue
            
            # Save to CSV
            data.to_csv(csv_path)
            logger.info(f"  ✓ Downloaded {len(data)} days of data")
            successful += 1
            
        except Exception as e:
            logger.error(f"  ✗ Error downloading {symbol}: {e}")
            failed.append(symbol)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Download Summary")
    logger.info("=" * 70)
    logger.info(f"Successful: {successful}/{len(SAMPLE_STOCKS)}")
    if failed:
        logger.info(f"Failed: {len(failed)} - {', '.join(failed)}")
    logger.info("=" * 70)
    
    if successful < len(SAMPLE_STOCKS):
        logger.warning(f"\n⚠️  Only {successful} stocks downloaded successfully")
        logger.warning("You may want to retry the failed symbols or proceed with available data")


if __name__ == "__main__":
    main()
