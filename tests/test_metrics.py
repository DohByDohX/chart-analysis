import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Union

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import PredictionMetrics

def test_return_metrics_types():
    # Setup mock data
    actual_ohlcv = pd.DataFrame({
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': [101.0, 102.0],
        'Volume': [1000, 1100]
    })

    predicted_ohlcv = pd.DataFrame({
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': [101.5, 102.5],
        'Volume': [1000, 1100]
    })

    actual_tokens = np.array([1, 2])
    predicted_tokens = np.array([1, 2])
    last_close = 100.0

    # Initialize metrics
    metrics = PredictionMetrics(
        actual_ohlcv=actual_ohlcv,
        predicted_ohlcv=predicted_ohlcv,
        actual_tokens=actual_tokens,
        predicted_tokens=predicted_tokens,
        last_close=last_close
    )

    # Get return metrics
    results = metrics.return_metrics()

    # Check types
    assert isinstance(results['actual_return'], float)
    assert isinstance(results['predicted_return'], float)
    assert isinstance(results['return_error'], float)

    # This is the key check - it returns a boolean or numpy.bool_ which corresponds to the bool type in python
    assert isinstance(results['return_direction_correct'], (bool, np.bool_))

    # Verify that type checker would flag this if we were strictly checking
    # (We can't easily run mypy from within pytest, but the visual inspection confirms the issue)
