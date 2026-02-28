"""
Comprehensive metrics for evaluating VisionTrader predictions.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionMetrics:
    """
    Calculate comprehensive evaluation metrics for VisionTrader predictions.
    
    Metrics include:
    - Token accuracy (exact matches)
    - Directional accuracy (up/down predictions)
    - Price error metrics (MAE, RMSE, MAPE)
    - Return forecasting accuracy
    """
    
    def __init__(
        self,
        actual_ohlcv: pd.DataFrame,
        predicted_ohlcv: pd.DataFrame,
        actual_tokens: np.ndarray,
        predicted_tokens: np.ndarray,
        last_close: float
    ):
        """
        Initialize metrics calculator.
        
        Args:
            actual_ohlcv: Ground truth OHLCV data
            predicted_ohlcv: Predicted OHLCV data
            actual_tokens: Ground truth token sequence
            predicted_tokens: Predicted token sequence
            last_close: Last known closing price (for return calculation)
        """
        self.actual_ohlcv = actual_ohlcv
        self.predicted_ohlcv = predicted_ohlcv
        self.actual_tokens = actual_tokens
        self.predicted_tokens = predicted_tokens
        self.last_close = last_close
        
        # Validate inputs
        if len(actual_ohlcv) != len(predicted_ohlcv):
            logger.warning(f"OHLCV length mismatch: {len(actual_ohlcv)} vs {len(predicted_ohlcv)}")
        if len(actual_tokens) != len(predicted_tokens):
            logger.warning(f"Token length mismatch: {len(actual_tokens)} vs {len(predicted_tokens)}")
    
    def token_accuracy(self) -> float:
        """
        Calculate exact token match accuracy.
        
        Returns:
            Percentage of tokens that match exactly
        """
        if len(self.actual_tokens) == 0:
            return 0.0
        
        matches = (self.actual_tokens == self.predicted_tokens).sum()
        accuracy = (matches / len(self.actual_tokens)) * 100
        return accuracy
    
    def directional_accuracy(self) -> float:
        """
        Calculate directional prediction accuracy (up/down).
        
        Returns:
            Percentage of candles with correct direction
        """
        min_len = min(len(self.actual_ohlcv), len(self.predicted_ohlcv))
        if min_len == 0:
            return 0.0
        
        actual_directions = (self.actual_ohlcv['Close'][:min_len] >= 
                           self.actual_ohlcv['Open'][:min_len]).astype(int)
        predicted_directions = (self.predicted_ohlcv['Close'][:min_len] >= 
                              self.predicted_ohlcv['Open'][:min_len]).astype(int)
        
        matches = (actual_directions.values == predicted_directions.values).sum()
        accuracy = (matches / min_len) * 100
        return accuracy
    
    def mae(self) -> Dict[str, float]:
        """
        Calculate Mean Absolute Error for OHLC prices.
        
        Returns:
            Dictionary with MAE for Open, High, Low, Close
        """
        min_len = min(len(self.actual_ohlcv), len(self.predicted_ohlcv))
        cols = ['Open', 'High', 'Low', 'Close']
        
        mae_dict = {}
        for col in cols:
            # Bolt optimization: Extract 1D numpy array directly, avoiding DataFrame multi-column slice overhead
            a = self.actual_ohlcv[col].values[:min_len]
            p = self.predicted_ohlcv[col].values[:min_len]
            mae_dict[col] = float(np.mean(np.abs(a - p)))

        return mae_dict
    
    def rmse(self) -> Dict[str, float]:
        """
        Calculate Root Mean Square Error for OHLC prices.
        
        Returns:
            Dictionary with RMSE for Open, High, Low, Close
        """
        min_len = min(len(self.actual_ohlcv), len(self.predicted_ohlcv))
        cols = ['Open', 'High', 'Low', 'Close']

        rmse_dict = {}
        for col in cols:
            # Bolt optimization: Extract 1D numpy array directly
            a = self.actual_ohlcv[col].values[:min_len]
            p = self.predicted_ohlcv[col].values[:min_len]
            rmse_dict[col] = float(np.sqrt(np.mean((a - p) ** 2)))

        return rmse_dict
    
    def mape(self) -> Dict[str, float]:
        """
        Calculate Mean Absolute Percentage Error for OHLC prices.
        
        Returns:
            Dictionary with MAPE for Open, High, Low, Close
        """
        min_len = min(len(self.actual_ohlcv), len(self.predicted_ohlcv))
        cols = ['Open', 'High', 'Low', 'Close']

        mape_dict = {}
        for col in cols:
            # Bolt optimization: Extract 1D numpy array directly
            a = self.actual_ohlcv[col].values[:min_len]
            p = self.predicted_ohlcv[col].values[:min_len]
            
            # Avoid division by zero
            mask = a != 0
            if not np.any(mask):
                mape_dict[col] = 0.0
            else:
                # Bolt optimization: Element-wise arithmetic using memory-efficient masked operations
                diff = np.abs(a - p)
                out_arr = np.zeros_like(diff, dtype=np.float64)
                pct_err = np.divide(diff, a, out=out_arr, where=mask)
                mape_dict[col] = float(np.mean(pct_err[mask])) * 100
        
        return mape_dict
    
    def return_metrics(self) -> Dict[str, Union[float, bool]]:
        """
        Calculate return prediction metrics.
        
        Returns:
            Dictionary with return error, direction accuracy, and values
        """
        if len(self.actual_ohlcv) == 0 or len(self.predicted_ohlcv) == 0:
            return {
                'actual_return': 0.0,
                'predicted_return': 0.0,
                'return_error': 0.0,
                'return_direction_correct': False
            }
        
        # Calculate actual return
        actual_final_close = self.actual_ohlcv.iloc[-1]['Close']
        actual_return = ((actual_final_close / self.last_close) - 1) * 100
        
        # Calculate predicted return
        predicted_final_close = self.predicted_ohlcv.iloc[-1]['Close']
        predicted_return = ((predicted_final_close / self.last_close) - 1) * 100
        
        # Return error
        return_error = abs(predicted_return - actual_return)
        
        # Direction correctness
        direction_correct = bool((actual_return >= 0) == (predicted_return >= 0))
        
        return {
            'actual_return': actual_return,
            'predicted_return': predicted_return,
            'return_error': return_error,
            'return_direction_correct': bool(direction_correct)
        }
    
    def price_range_accuracy(self) -> Dict[str, float]:
        """
        Calculate how well predicted high/low match actual range.
        
        Returns:
            Dictionary with high/low errors and range overlap
        """
        min_len = min(len(self.actual_ohlcv), len(self.predicted_ohlcv))
        
        actual_high = self.actual_ohlcv['High'][:min_len].max()
        actual_low = self.actual_ohlcv['Low'][:min_len].min()
        predicted_high = self.predicted_ohlcv['High'][:min_len].max()
        predicted_low = self.predicted_ohlcv['Low'][:min_len].min()
        
        high_error = abs(predicted_high - actual_high)
        low_error = abs(predicted_low - actual_low)
        
        # Calculate range overlap
        actual_range = actual_high - actual_low
        predicted_range = predicted_high - predicted_low
        range_error = abs(predicted_range - actual_range)
        
        return {
            'actual_high': actual_high,
            'actual_low': actual_low,
            'predicted_high': predicted_high,
            'predicted_low': predicted_low,
            'high_error': high_error,
            'low_error': low_error,
            'range_error': range_error
        }
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive metrics summary.
        
        Returns:
            Dictionary with all calculated metrics
        """
        mae_metrics = self.mae()
        rmse_metrics = self.rmse()
        mape_metrics = self.mape()
        return_metrics = self.return_metrics()
        range_metrics = self.price_range_accuracy()
        
        summary = {
            'token_accuracy': self.token_accuracy(),
            'directional_accuracy': self.directional_accuracy(),
            'mae': mae_metrics,
            'rmse': rmse_metrics,
            'mape': mape_metrics,
            'return': return_metrics,
            'price_range': range_metrics,
            'sequence_length': len(self.actual_tokens)
        }
        
        return summary
    
    def format_summary(self) -> str:
        """
        Format summary as readable string.
        
        Returns:
            Formatted summary string
        """
        summary = self.summary()
        
        lines = [
            "=" * 60,
            "PREDICTION METRICS SUMMARY",
            "=" * 60,
            f"\nToken Accuracy: {summary['token_accuracy']:.1f}%",
            f"Directional Accuracy: {summary['directional_accuracy']:.1f}%",
            f"\nPrice Errors (MAE):",
            f"  Open:  ${summary['mae']['Open']:.2f}",
            f"  High:  ${summary['mae']['High']:.2f}",
            f"  Low:   ${summary['mae']['Low']:.2f}",
            f"  Close: ${summary['mae']['Close']:.2f}",
            f"\nPrice Errors (RMSE):",
            f"  Open:  ${summary['rmse']['Open']:.2f}",
            f"  High:  ${summary['rmse']['High']:.2f}",
            f"  Low:   ${summary['rmse']['Low']:.2f}",
            f"  Close: ${summary['rmse']['Close']:.2f}",
            f"\nPrice Errors (MAPE):",
            f"  Open:  {summary['mape']['Open']:.2f}%",
            f"  High:  {summary['mape']['High']:.2f}%",
            f"  Low:   {summary['mape']['Low']:.2f}%",
            f"  Close: {summary['mape']['Close']:.2f}%",
            f"\nReturn Metrics:",
            f"  Actual Return: {summary['return']['actual_return']:+.2f}%",
            f"  Predicted Return: {summary['return']['predicted_return']:+.2f}%",
            f"  Return Error: {summary['return']['return_error']:.2f}%",
            f"  Direction Correct: {summary['return']['return_direction_correct']}",
            f"\nPrice Range:",
            f"  Actual: ${summary['price_range']['actual_low']:.2f} - ${summary['price_range']['actual_high']:.2f}",
            f"  Predicted: ${summary['price_range']['predicted_low']:.2f} - ${summary['price_range']['predicted_high']:.2f}",
            f"  Range Error: ${summary['price_range']['range_error']:.2f}",
            "=" * 60
        ]
        
        return "\n".join(lines)
