"""
Stock data downloader using yfinance.
Downloads historical OHLCV data for stocks and indices.
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataDownloader:
    """
    Downloads and manages historical stock market data.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_stock(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data for a single stock.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with OHLCV data or None if download fails
        """
        try:
            logger.info(f"Downloading {symbol} from {start_date} to {end_date}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
                
            # Clean up the data
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.index.name = 'Date'
            
            logger.info(f"Successfully downloaded {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {str(e)}")
            return None
    
    def download_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        save: bool = True
    ) -> dict:
        """
        Download data for multiple stocks.
        
        Args:
            symbols: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (1d, 1h, etc.)
            save: Whether to save data to CSV files
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        failed = []
        
        for symbol in tqdm(symbols, desc="Downloading stocks"):
            data = self.download_stock(symbol, start_date, end_date, interval)
            
            if data is not None:
                results[symbol] = data
                if save:
                    self.save_to_csv(data, symbol)
            else:
                failed.append(symbol)
        
        if failed:
            logger.warning(f"Failed to download: {', '.join(failed)}")
        
        logger.info(f"Successfully downloaded {len(results)}/{len(symbols)} stocks")
        return results
    
    def save_to_csv(self, data: pd.DataFrame, symbol: str) -> Path:
        """
        Save stock data to CSV file.
        
        Args:
            data: DataFrame with stock data
            symbol: Stock ticker symbol
            
        Returns:
            Path to saved CSV file
        """
        filepath = self.output_dir / f"{symbol}.csv"
        data.to_csv(filepath)
        logger.info(f"Saved {symbol} data to {filepath}")
        return filepath
    
    def load_from_csv(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load stock data from CSV file.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with stock data or None if file doesn't exist
        """
        filepath = self.output_dir / f"{symbol}.csv"
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None
        
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        logger.info(f"Loaded {len(data)} records for {symbol}")
        return data
    
    def get_sp500_symbols(self) -> List[str]:
        """
        Get list of S&P 500 constituent symbols.
        
        Returns:
            List of stock ticker symbols
        """
        try:
            # Download S&P 500 constituents from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            # Clean symbols (remove dots, etc.)
            symbols = [s.replace('.', '-') for s in symbols]
            
            logger.info(f"Retrieved {len(symbols)} S&P 500 symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 symbols: {str(e)}")
            return []
    
    def validate_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Validate downloaded stock data.
        
        Args:
            data: DataFrame with stock data
            symbol: Stock ticker symbol
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            logger.error(f"{symbol}: Data is empty")
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"{symbol}: Missing columns: {missing_columns}")
            return False
        
        # Check for missing values
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"{symbol}: Found missing values:\n{null_counts[null_counts > 0]}")
        
        # Check for invalid OHLC relationships
        invalid_high = (data['High'] < data['Low']).sum()
        invalid_open = ((data['Open'] > data['High']) | (data['Open'] < data['Low'])).sum()
        invalid_close = ((data['Close'] > data['High']) | (data['Close'] < data['Low'])).sum()
        
        if invalid_high > 0 or invalid_open > 0 or invalid_close > 0:
            logger.warning(
                f"{symbol}: Found invalid OHLC relationships - "
                f"High<Low: {invalid_high}, Open out of range: {invalid_open}, "
                f"Close out of range: {invalid_close}"
            )
        
        logger.info(f"{symbol}: Data validation passed")
        return True
