"""
Candle tokenizer for converting OHLCV data to categorical tokens.
Encodes candle characteristics into discrete tokens for model training.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandleTokenizer:
    """
    Tokenizes candlestick data into categorical tokens.
    
    Encodes each candle based on:
    - Body direction (Bullish/Bearish/Doji)
    - Body size (Small/Medium/Large)
    - Upper wick size (None/Small/Medium/Large)
    - Lower wick size (None/Small/Medium/Large)
    - Volume (Low/Medium/High)
    
    Total vocabulary: 3 * 3 * 4 * 4 * 3 * 5 = 2,160 tokens
    
    Gap encoding added to capture price discontinu ities:
    - None: |gap| < 0.2%
    - Small Gap Up/Down: 0.2% ≤ |gap| < 1.0%
    - Large Gap Up/Down: |gap| ≥ 1.0%
    """
    
    # Category definitions
    DIRECTION = ['Bullish', 'Bearish', 'Doji']
    BODY_SIZE = ['Small', 'Medium', 'Large']
    WICK_SIZE = ['None', 'Small', 'Medium', 'Large']
    VOLUME_SIZE = ['Low', 'Medium', 'High']
    GAP_TYPE = ['None', 'SmallUp', 'LargeUp', 'SmallDown', 'LargeDown']
    
    def __init__(
        self,
        doji_threshold: float = 0.001,
        percentile_lookback: int = 50,
        gap_small_threshold: float = 0.002,
        gap_large_threshold: float = 0.01
    ):
        """
        Initialize the tokenizer.
        
        Args:
            doji_threshold: Threshold for doji detection (as fraction of price)
            percentile_lookback: Number of candles to use for percentile calculations
            gap_small_threshold: Minimum gap size to register (0.2%)
            gap_large_threshold: Threshold for large gaps (1.0%)
        """
        self.doji_threshold = doji_threshold
        self.percentile_lookback = percentile_lookback
        self.gap_small_threshold = gap_small_threshold
        self.gap_large_threshold = gap_large_threshold
        
        self.vocabulary_size = len(self.DIRECTION) * len(self.BODY_SIZE) * \
                              len(self.WICK_SIZE) * len(self.WICK_SIZE) * \
                              len(self.VOLUME_SIZE) * len(self.GAP_TYPE)
        
        logger.info(f"Initialized CandleTokenizer with vocabulary size: {self.vocabulary_size}")
    
    def _get_direction(self, candle: pd.Series) -> int:
        """Get body direction category (0=Bullish, 1=Bearish, 2=Doji)."""
        open_price = candle['Open']
        close_price = candle['Close']
        
        # Calculate relative body size
        body_size = abs(close_price - open_price) / open_price
        
        if body_size < self.doji_threshold:
            return 2  # Doji
        elif close_price > open_price:
            return 0  # Bullish
        else:
            return 1  # Bearish
    
    def _get_body_size(self, candle: pd.Series, recent_candles: pd.DataFrame) -> int:
        """Get body size category (0=Small, 1=Medium, 2=Large)."""
        body_size = abs(candle['Close'] - candle['Open'])
        
        # Calculate body sizes for recent candles
        recent_bodies = abs(recent_candles['Close'] - recent_candles['Open'])
        
        # Get percentiles
        p33 = recent_bodies.quantile(0.33)
        p66 = recent_bodies.quantile(0.66)
        
        if body_size < p33:
            return 0  # Small
        elif body_size < p66:
            return 1  # Medium
        else:
            return 2  # Large
    
    def _get_wick_size(self, wick_length: float, body_size: float) -> int:
        """
        Get wick size category (0=None, 1=Small, 2=Medium, 3=Large).
        
        Wick size is relative to body size.
        """
        if body_size < 1e-8:  # Avoid division by zero for doji
            body_size = 1e-8
        
        ratio = wick_length / body_size
        
        if ratio < 0.1:
            return 0  # None
        elif ratio < 0.5:
            return 1  # Small
        elif ratio < 1.0:
            return 2  # Medium
        else:
            return 3  # Large
    
    def _get_volume_size(self, candle: pd.Series, recent_candles: pd.DataFrame) -> int:
        """Get volume category (0=Low, 1=Medium, 2=High)."""
        volume = candle['Volume']
        
        # Get percentiles from recent candles
        p33 = recent_candles['Volume'].quantile(0.33)
        p66 = recent_candles['Volume'].quantile(0.66)
        
        if volume < p33:
            return 0  # Low
        elif volume < p66:
            return 1  # Medium
        else:
            return 2  # High
    
    def _get_gap_type(self, candle: pd.Series, prev_candle: Optional[pd.Series]) -> int:
        """
        Get gap type category.
        
        Gap = (current_open - previous_close) / previous_close
        
        Categories:
        0 = None: |gap| < gap_small_threshold (0.2%)
        1 = SmallUp: gap_small ≤ gap < gap_large (0.2% to 1.0%)
        2 = LargeUp: gap ≥ gap_large (≥1.0%)
        3 = SmallDown: -gap_large < gap ≤ -gap_small (-1.0% to -0.2%)
        4 = LargeDown: gap ≤ -gap_large (≤-1.0%)
        """
        if prev_candle is None:
            return 0  # No gap for first candle
        
        prev_close = prev_candle['Close']
        current_open = candle['Open']
        
        # Calculate gap as percentage
        gap = (current_open - prev_close) / prev_close
        
        # Categorize gap
        if abs(gap) < self.gap_small_threshold:
            return 0  # None
        elif gap >= self.gap_large_threshold:
            return 2  # LargeUp
        elif gap >= self.gap_small_threshold:
            return 1  # SmallUp
        elif gap <= -self.gap_large_threshold:
            return 4  # LargeDown
        else:  # -gap_large < gap <= -gap_small
            return 3  # SmallDown
    
    def tokenize_candle(
        self,
        candle: pd.Series,
        recent_candles: pd.DataFrame,
        prev_candle: Optional[pd.Series] = None
    ) -> Tuple[int, Dict]:
        """
        Tokenize a single candle.
        
        Args:
            candle: Single candle data (OHLCV)
            recent_candles: Recent candles for context (percentile calculations)
            prev_candle: Previous candle for gap calculation (None for first candle)
            
        Returns:
            Tuple of (token_id, characteristics_dict)
        """
        # Extract candle characteristics
        direction = self._get_direction(candle)
        body_size_cat = self._get_body_size(candle, recent_candles)
        
        # Calculate wick lengths
        body_size = abs(candle['Close'] - candle['Open'])
        upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
        lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
        
        upper_wick_cat = self._get_wick_size(upper_wick, body_size)
        lower_wick_cat = self._get_wick_size(lower_wick, body_size)
        volume_cat = self._get_volume_size(candle, recent_candles)
        gap_cat = self._get_gap_type(candle, prev_candle)
        
        # Calculate token ID
        # New formula: direction*720 + body_size*240 + upper_wick*60 + lower_wick*15 + volume*5 + gap
        token_id = (
            direction * 720 +
            body_size_cat * 240 +
            upper_wick_cat * 60 +
            lower_wick_cat * 15 +
            volume_cat * 5 +
            gap_cat
        )
        
        # Create characteristics dictionary
        characteristics = {
            'direction': self.DIRECTION[direction],
            'body_size': self.BODY_SIZE[body_size_cat],
            'upper_wick': self.WICK_SIZE[upper_wick_cat],
            'lower_wick': self.WICK_SIZE[lower_wick_cat],
            'volume': self.VOLUME_SIZE[volume_cat],
            'gap': self.GAP_TYPE[gap_cat],
            'token_id': token_id
        }
        
        return token_id, characteristics
    
    def tokenize_window(self, window_df: pd.DataFrame) -> Tuple[List[int], List[Dict]]:
        """
        Tokenize an entire window of candles.
        
        Args:
            window_df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (token_ids, characteristics_list)
        """
        tokens = []
        characteristics_list = []
        
        for i in range(len(window_df)):
            candle = window_df.iloc[i]
            
            # Get recent candles for percentile calculations
            start_idx = max(0, i - self.percentile_lookback)
            recent_candles = window_df.iloc[start_idx:i+1]
            
            # Get previous candle for gap calculation (None for first candle)
            prev_candle = window_df.iloc[i-1] if i > 0 else None
            
            token, chars = self.tokenize_candle(candle, recent_candles, prev_candle)
            tokens.append(token)
            characteristics_list.append(chars)
        
        return tokens, characteristics_list
    
    def detokenize(self, token_id: int) -> Dict:
        """
        Convert token ID back to candle characteristics.
        
        Args:
            token_id: Token ID (0-2159)
            
        Returns:
            Dictionary of characteristics
        """
        if token_id < 0 or token_id >= self.vocabulary_size:
            raise ValueError(f"Token ID {token_id} out of range (0-{self.vocabulary_size-1})")
        
        # Reverse the encoding formula: direction*720 + body_size*240 + upper_wick*60 + lower_wick*15 + volume*5 + gap
        gap_cat = token_id % 5
        token_id //= 5
        
        volume_cat = token_id % 3
        token_id //= 3
        
        lower_wick_cat = token_id % 4
        token_id //= 4
        
        upper_wick_cat = token_id % 4
        token_id //= 4
        
        body_size_cat = token_id % 3
        token_id //= 3
        
        direction = token_id
        
        return {
            'direction': self.DIRECTION[direction],
            'body_size': self.BODY_SIZE[body_size_cat],
            'upper_wick': self.WICK_SIZE[upper_wick_cat],
            'lower_wick': self.WICK_SIZE[lower_wick_cat],
            'volume': self.VOLUME_SIZE[volume_cat],
            'gap': self.GAP_TYPE[gap_cat]
        }
    
    def get_vocabulary(self) -> Dict[int, Dict]:
        """
        Get the complete vocabulary mapping.
        
        Returns:
            Dictionary mapping token IDs to characteristics
        """
        vocabulary = {}
        for token_id in range(self.vocabulary_size):
            vocabulary[token_id] = self.detokenize(token_id)
        return vocabulary
    
    def tokens_to_ohlcv(
        self,
        token_ids: List[int],
        last_candle: Dict[str, float],
        avg_volatility: float = 0.015
    ) -> pd.DataFrame:
        """
        Convert token IDs to synthetic OHLCV data.
        
        Args:
            token_ids: List of token IDs to convert
            last_candle: Last known candle {'Open', 'High', 'Low', 'Close', 'Volume'}
            avg_volatility: Average volatility for scaling (default 1.5%)
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        ohlcv_data = []
        current_close = last_candle['Close']
        avg_volume = last_candle.get('Volume', 1000000)  # Default if not provided
        
        for token_id in token_ids:
            # Detokenize to get characteristics
            chars = self.detokenize(token_id)
            
            # Determine body size magnitude based on category
            body_magnitudes = {
                'Small': avg_volatility * 0.3,   # ~0.45%
                'Medium': avg_volatility * 1.0,  # ~1.5%
                'Large': avg_volatility * 2.0    # ~3.0%
            }
            body_size = body_magnitudes[chars['body_size']]
            
            # Calculate open and close based on direction
            open_price = current_close
            
            if chars['direction'] == 'Bullish':
                close_price = open_price * (1 + body_size)
            elif chars['direction'] == 'Bearish':
                close_price = open_price * (1 - body_size)
            else:  # Doji
                close_price = open_price * (1 + body_size * 0.1)  # Very small movement
            
            # Determine real body range
            body_top = max(open_price, close_price)
            body_bottom = min(open_price, close_price)
            body_height = abs(close_price - open_price)
            
            # Calculate wicks based on wick size categories
            wick_magnitudes = {
                'None': 0.0,
                'Small': body_height * 0.3 if body_height > 0 else open_price * 0.001,
                'Medium': body_height * 0.8 if body_height > 0 else open_price * 0.003,
                'Large': body_height * 1.5 if body_height > 0 else open_price * 0.005
            }
            
            upper_wick = wick_magnitudes[chars['upper_wick']]
            lower_wick = wick_magnitudes[chars['lower_wick']]
            
            # Calculate high and low
            high_price = body_top + upper_wick
            low_price = body_bottom - lower_wick
            
            # Estimate volume based on category
            volume_multipliers = {
                'Low': 0.7,
                'Medium': 1.0,
                'High': 1.5
            }
            volume = avg_volume * volume_multipliers[chars['volume']]
            
            # Store the candle
            ohlcv_data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
            
            # Update current close for next candle
            current_close = close_price
        
        return pd.DataFrame(ohlcv_data)
    
    def save_vocabulary(self, filepath: Union[str, Path]):
        """Save vocabulary to JSON file."""
        filepath = Path(filepath)
        vocabulary = self.get_vocabulary()
        
        # Convert int keys to strings for JSON
        vocab_str_keys = {str(k): v for k, v in vocabulary.items()}
        
        with open(filepath, 'w') as f:
            json.dump(vocab_str_keys, f, indent=2)
        
        logger.info(f"Saved vocabulary to {filepath}")
    
    def load_vocabulary(self, filepath: Union[str, Path]) -> Dict[int, Dict]:
        """Load vocabulary from JSON file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            vocab_str_keys = json.load(f)
        
        # Convert string keys back to int
        vocabulary = {int(k): v for k, v in vocab_str_keys.items()}
        
        logger.info(f"Loaded vocabulary from {filepath}")
        return vocabulary
    
    def get_token_statistics(
        self,
        token_ids: List[int]
    ) -> Dict:
        """
        Get statistics about token distribution.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Dictionary of statistics
        """
        token_counts = pd.Series(token_ids).value_counts()
        
        # Decode tokens to get characteristic distributions
        characteristics = [self.detokenize(tid) for tid in token_ids]
        
        direction_counts = pd.Series([c['direction'] for c in characteristics]).value_counts()
        body_size_counts = pd.Series([c['body_size'] for c in characteristics]).value_counts()
        volume_counts = pd.Series([c['volume'] for c in characteristics]).value_counts()
        
        stats = {
            'total_tokens': len(token_ids),
            'unique_tokens': len(token_counts),
            'vocabulary_coverage': len(token_counts) / self.vocabulary_size,
            'most_common_token': int(token_counts.index[0]),
            'direction_distribution': direction_counts.to_dict(),
            'body_size_distribution': body_size_counts.to_dict(),
            'volume_distribution': volume_counts.to_dict()
        }
        
        return stats
