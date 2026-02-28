
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import PredictionMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rmse_original(actual_ohlcv, predicted_ohlcv):
    """
    Original RMSE implementation for verification.
    """
    min_len = min(len(actual_ohlcv), len(predicted_ohlcv))

    rmse_dict = {}
    for col in ['Open', 'High', 'Low', 'Close']:
        actual = actual_ohlcv[col][:min_len].values
        predicted = predicted_ohlcv[col][:min_len].values
        rmse_dict[col] = np.sqrt(np.mean((actual - predicted) ** 2))

    return rmse_dict

def verify_rmse_correctness(length=100):
    """Verifies that the current implementation matches the original logic."""
    logger.info("Verifying RMSE correctness...")

    dates = pd.date_range(start='2023-01-01', periods=length)
    actual_data = {
        'Open': np.random.rand(length) * 100,
        'High': np.random.rand(length) * 100,
        'Low': np.random.rand(length) * 100,
        'Close': np.random.rand(length) * 100,
        'Volume': np.random.randint(100, 1000, size=length)
    }
    predicted_data = {
        'Open': np.random.rand(length) * 100,
        'High': np.random.rand(length) * 100,
        'Low': np.random.rand(length) * 100,
        'Close': np.random.rand(length) * 100,
        'Volume': np.random.randint(100, 1000, size=length)
    }

    actual_ohlcv = pd.DataFrame(actual_data, index=dates)
    predicted_ohlcv = pd.DataFrame(predicted_data, index=dates)

    # Dummy tokens and last_close as they are not used in RMSE
    actual_tokens = np.zeros(length)
    predicted_tokens = np.zeros(length)
    last_close = 100.0

    metrics = PredictionMetrics(
        actual_ohlcv=actual_ohlcv,
        predicted_ohlcv=predicted_ohlcv,
        actual_tokens=actual_tokens,
        predicted_tokens=predicted_tokens,
        last_close=last_close
    )

    current_rmse = metrics.rmse()
    expected_rmse = rmse_original(actual_ohlcv, predicted_ohlcv)

    logger.info(f"Current RMSE: {current_rmse}")
    logger.info(f"Expected RMSE: {expected_rmse}")

    for col in ['Open', 'High', 'Low', 'Close']:
        np.testing.assert_allclose(current_rmse[col], expected_rmse[col], rtol=1e-5, err_msg=f"RMSE mismatch for {col}")

    logger.info("Verification passed: Current implementation matches original logic.")

if __name__ == "__main__":
    verify_rmse_correctness()
