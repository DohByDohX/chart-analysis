"""
Random window generator for creating training sequences.
Generates random 250-period input windows paired with 5-period target windows.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WindowGenerator:
    """
    Generates random windows from stock data for training.
    Creates 250-period input sequences paired with 5-period target sequences.
    """
    
    def __init__(
        self, 
        data_dir: Union[str, Path],
        input_size: int = 250,
        target_size: int = 5,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the window generator.
        
        Args:
            data_dir: Directory containing stock CSV files
            input_size: Number of candles in input window (default: 250)
            target_size: Number of candles in target window (default: 5)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.target_size = target_size
        self.window_size = input_size + target_size
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
            logger.info(f"Random seed set to {random_seed}")
    
    def generate_random_windows(
        self,
        data: pd.DataFrame,
        num_samples: Optional[int] = None,
        symbol: str = "UNKNOWN"
    ) -> List[Dict]:
        """
        Generate random windows from stock data.
        
        Args:
            data: DataFrame with OHLCV data
            num_samples: Number of random windows to generate (None = all possible)
            symbol: Stock symbol for metadata
            
        Returns:
            List of window dictionaries
        """
        if len(data) < self.window_size:
            logger.warning(
                f"{symbol}: Insufficient data. Need {self.window_size} candles, "
                f"got {len(data)}"
            )
            return []
        
        # Calculate valid start positions
        max_start = len(data) - self.window_size
        valid_positions = list(range(max_start + 1))
        
        # Determine number of samples
        if num_samples is None:
            num_samples = len(valid_positions)
        else:
            num_samples = min(num_samples, len(valid_positions))
        
        # Randomly sample positions without replacement
        selected_positions = np.random.choice(
            valid_positions,
            size=num_samples,
            replace=False
        )
        
        windows = []
        for idx, start_pos in enumerate(selected_positions):
            end_pos = start_pos + self.window_size
            
            # Extract input and target windows
            input_window = data.iloc[start_pos:start_pos + self.input_size].copy()
            target_window = data.iloc[start_pos + self.input_size:end_pos].copy()
            
            # Create window dictionary
            window = {
                'symbol': symbol,
                'input': input_window,
                'target': target_window,
                'start_date': str(input_window.index[0].date()),
                'end_date': str(target_window.index[-1].date()),
                'window_index': idx,
                'start_position': start_pos,
                'input_size': len(input_window),
                'target_size': len(target_window)
            }
            
            windows.append(window)
        
        logger.info(
            f"{symbol}: Generated {len(windows)} random windows "
            f"(max possible: {len(valid_positions)})"
        )
        return windows
    
    def generate_from_symbol(
        self,
        symbol: str,
        num_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Load stock data from CSV and generate random windows.
        
        Args:
            symbol: Stock ticker symbol
            num_samples: Number of random windows to generate
            
        Returns:
            List of window dictionaries
        """
        csv_path = self.data_dir / f"{symbol}.csv"
        
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return []
        
        try:
            data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
            logger.info(f"Loaded {len(data)} records for {symbol}")
            return self.generate_random_windows(data, num_samples, symbol)
        except Exception as e:
            logger.error(f"Error loading {symbol}: {str(e)}")
            return []
    
    def generate_batch(
        self,
        symbols: List[str],
        samples_per_symbol: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Generate random windows for multiple stocks.
        
        Args:
            symbols: List of stock ticker symbols
            samples_per_symbol: Number of windows per stock
            
        Returns:
            Dictionary mapping symbols to their windows
        """
        results = {}
        total_windows = 0
        
        for symbol in symbols:
            windows = self.generate_from_symbol(symbol, samples_per_symbol)
            if windows:
                results[symbol] = windows
                total_windows += len(windows)
        
        logger.info(
            f"Generated {total_windows} total windows across {len(results)} stocks"
        )
        return results
    
    def validate_window(self, window: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a window for data quality.
        
        Args:
            window: Window dictionary
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check sizes
        if window['input_size'] != self.input_size:
            issues.append(f"Input size mismatch: {window['input_size']} != {self.input_size}")
        
        if window['target_size'] != self.target_size:
            issues.append(f"Target size mismatch: {window['target_size']} != {self.target_size}")
        
        # Check for missing values
        input_nulls = window['input'].isnull().sum().sum()
        target_nulls = window['target'].isnull().sum().sum()
        
        if input_nulls > 0:
            issues.append(f"Input has {input_nulls} missing values")
        
        if target_nulls > 0:
            issues.append(f"Target has {target_nulls} missing values")
        
        # Check date continuity
        input_dates = window['input'].index
        target_dates = window['target'].index
        
        # Verify no overlap
        if input_dates[-1] >= target_dates[0]:
            issues.append("Input and target windows overlap")
        
        return len(issues) == 0, issues
    
    def save_windows(
        self,
        windows: List[Dict],
        output_dir: Union[str, Path],
        format: str = 'pickle'
    ) -> Path:
        """
        Save generated windows to disk.
        
        Args:
            windows: List of window dictionaries
            output_dir: Directory to save windows
            format: Save format ('pickle' or 'csv')
            
        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'pickle':
            filepath = output_dir / f"windows_{timestamp}.pkl"
            pd.to_pickle(windows, filepath)
        elif format == 'csv':
            # Save metadata only (not the full dataframes)
            metadata = []
            for w in windows:
                metadata.append({
                    'symbol': w['symbol'],
                    'start_date': w['start_date'],
                    'end_date': w['end_date'],
                    'window_index': w['window_index'],
                    'start_position': w['start_position'],
                    'input_size': w['input_size'],
                    'target_size': w['target_size']
                })
            filepath = output_dir / f"windows_metadata_{timestamp}.csv"
            pd.DataFrame(metadata).to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(windows)} windows to {filepath}")
        return filepath
    
    def get_statistics(self, windows: List[Dict]) -> Dict:
        """
        Get statistics about generated windows.
        
        Args:
            windows: List of window dictionaries
            
        Returns:
            Dictionary of statistics
        """
        if not windows:
            return {}
        
        symbols = [w['symbol'] for w in windows]
        start_positions = [w['start_position'] for w in windows]
        
        stats = {
            'total_windows': len(windows),
            'unique_symbols': len(set(symbols)),
            'symbols': list(set(symbols)),
            'avg_start_position': np.mean(start_positions),
            'min_start_position': np.min(start_positions),
            'max_start_position': np.max(start_positions),
            'input_size': self.input_size,
            'target_size': self.target_size,
            'window_size': self.window_size
        }
        
        return stats
