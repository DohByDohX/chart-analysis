import pytest
import numpy as np
import pandas as pd
from src.evaluation.metrics import PredictionMetrics

@pytest.fixture
def sample_data():
    """Create sample data for testing PredictionMetrics."""
    actual_ohlcv = pd.DataFrame({
        'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'High': [102.0, 103.0, 104.0, 105.0, 106.0],
        'Low': [98.0, 99.0, 100.0, 101.0, 102.0],
        'Close': [101.0, 102.0, 103.0, 104.0, 105.0]
    })

    # Predicted is slightly different
    predicted_ohlcv = pd.DataFrame({
        'Open': [100.5, 101.5, 102.5, 103.5, 104.5],
        'High': [102.5, 103.5, 104.5, 105.5, 106.5],
        'Low': [98.5, 99.5, 100.5, 101.5, 102.5],
        'Close': [101.5, 102.5, 103.5, 104.5, 105.5]
    })

    actual_tokens = np.array([10, 20, 30, 40, 50])
    predicted_tokens = np.array([10, 21, 30, 40, 51])  # 3/5 match

    last_close = 99.0

    return actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close

def test_init_mismatch():
    """Test initialization with length mismatch (should log warnings but not fail)."""
    actual_ohlcv = pd.DataFrame({'Open': [100], 'High': [101], 'Low': [99], 'Close': [100]})
    predicted_ohlcv = pd.DataFrame({'Open': [100], 'High': [101], 'Low': [99], 'Close': [100], 'Extra': [100]})
    actual_tokens = np.array([1, 2])
    predicted_tokens = np.array([1])

    # This should not raise an exception
    metrics = PredictionMetrics(actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, 100.0)
    assert metrics.last_close == 100.0

def test_token_accuracy(sample_data):
    """Test token accuracy calculation."""
    actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close = sample_data
    metrics = PredictionMetrics(actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close)

    # 3 out of 5 match = 60%
    assert metrics.token_accuracy() == 60.0

def test_token_accuracy_empty():
    """Test token accuracy with empty tokens."""
    metrics = PredictionMetrics(pd.DataFrame(), pd.DataFrame(), np.array([]), np.array([]), 100.0)
    assert metrics.token_accuracy() == 0.0

def test_directional_accuracy(sample_data):
    """Test directional accuracy calculation."""
    actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close = sample_data
    metrics = PredictionMetrics(actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close)

    # In sample_data, all actual and predicted Close >= Open, so all are 'Up' (1)
    # So directional accuracy should be 100%
    assert metrics.directional_accuracy() == 100.0

def test_directional_accuracy_mismatch():
    """Test directional accuracy with some mismatches."""
    actual_ohlcv = pd.DataFrame({
        'Open': [100, 100, 100],
        'Close': [101, 99, 101] # Up, Down, Up
    })
    predicted_ohlcv = pd.DataFrame({
        'Open': [100, 100, 100],
        'Close': [101, 101, 99] # Up, Up, Down
    })
    # Match at index 0 only. 1 out of 3 = 33.33%
    metrics = PredictionMetrics(actual_ohlcv, predicted_ohlcv, np.array([]), np.array([]), 100.0)
    assert metrics.directional_accuracy() == pytest.approx(33.33333333333333)

def test_mae_rmse(sample_data):
    """Test MAE and RMSE calculations."""
    actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close = sample_data
    metrics = PredictionMetrics(actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close)

    mae = metrics.mae()
    rmse = metrics.rmse()

    # Diff is 0.5 for all prices in all rows
    for col in ['Open', 'High', 'Low', 'Close']:
        assert mae[col] == pytest.approx(0.5)
        assert rmse[col] == pytest.approx(0.5)

def test_mape(sample_data):
    """Test MAPE calculation."""
    actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close = sample_data
    metrics = PredictionMetrics(actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close)

    mape = metrics.mape()

    # For 'Open' col: actuals are 100, 101, 102, 103, 104. Predictions are 100.5, 101.5...
    # Errors are 0.5. Percentage errors: 0.5/100, 0.5/101, 0.5/102, 0.5/103, 0.5/104
    errors = [0.5/100, 0.5/101, 0.5/102, 0.5/103, 0.5/104]
    expected_mape_open = (sum(errors) / len(errors)) * 100
    assert mape['Open'] == pytest.approx(expected_mape_open)

def test_mape_zero_division():
    """Test MAPE with zero prices to ensure it handles division by zero."""
    actual_ohlcv = pd.DataFrame({
        'Open': [0.0, 100.0],
        'High': [0.0, 100.0],
        'Low': [0.0, 100.0],
        'Close': [0.0, 100.0]
    })
    predicted_ohlcv = pd.DataFrame({
        'Open': [10.0, 110.0],
        'High': [10.0, 110.0],
        'Low': [10.0, 110.0],
        'Close': [10.0, 110.0]
    })
    metrics = PredictionMetrics(actual_ohlcv, predicted_ohlcv, np.array([]), np.array([]), 100.0)

    mape = metrics.mape()
    # The first row (0.0) should be masked out. Only the second row (100.0) counts.
    # Error is 10.0. MAPE = 10.0 / 100.0 * 100 = 10%
    assert mape['Close'] == 10.0

def test_return_metrics(sample_data):
    """Test return forecasting accuracy metrics."""
    actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close = sample_data
    metrics = PredictionMetrics(actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close)

    ret = metrics.return_metrics()

    # actual_final_close = 105.0. last_close = 99.0
    # actual_return = (105.0 / 99.0 - 1) * 100 = 6.0606...
    # predicted_final_close = 105.5.
    # predicted_return = (105.5 / 99.0 - 1) * 100 = 6.5656...

    expected_actual_return = (105.0 / 99.0 - 1) * 100
    expected_predicted_return = (105.5 / 99.0 - 1) * 100

    assert ret['actual_return'] == pytest.approx(expected_actual_return)
    assert ret['predicted_return'] == pytest.approx(expected_predicted_return)
    assert ret['return_error'] == pytest.approx(abs(expected_predicted_return - expected_actual_return))
    assert ret['return_direction_correct'] is True

def test_return_metrics_empty():
    """Test return metrics with empty data."""
    metrics = PredictionMetrics(pd.DataFrame(), pd.DataFrame(), np.array([]), np.array([]), 100.0)
    ret = metrics.return_metrics()
    assert ret['actual_return'] == 0.0
    assert ret['predicted_return'] == 0.0
    assert ret['return_direction_correct'] is False

def test_price_range_accuracy(sample_data):
    """Test price range accuracy calculation."""
    actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close = sample_data
    metrics = PredictionMetrics(actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close)

    range_metrics = metrics.price_range_accuracy()

    # actual_high = max([102, 103, 104, 105, 106]) = 106
    # actual_low = min([98, 99, 100, 101, 102]) = 98
    # predicted_high = max([102.5, 103.5, 104.5, 105.5, 106.5]) = 106.5
    # predicted_low = min([98.5, 99.5, 100.5, 101.5, 102.5]) = 98.5

    assert range_metrics['actual_high'] == 106.0
    assert range_metrics['actual_low'] == 98.0
    assert range_metrics['predicted_high'] == 106.5
    assert range_metrics['predicted_low'] == 98.5
    assert range_metrics['high_error'] == 0.5
    assert range_metrics['low_error'] == 0.5

    # actual_range = 106 - 98 = 8
    # predicted_range = 106.5 - 98.5 = 8
    # range_error = 0
    assert range_metrics['range_error'] == 0.0

def test_summary(sample_data):
    """Test the summary dictionary generation."""
    actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close = sample_data
    metrics = PredictionMetrics(actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close)

    summary = metrics.summary()
    assert isinstance(summary, dict)
    assert 'token_accuracy' in summary
    assert 'mae' in summary
    assert summary['sequence_length'] == 5

def test_format_summary(sample_data):
    """Test the formatted summary string."""
    actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close = sample_data
    metrics = PredictionMetrics(actual_ohlcv, predicted_ohlcv, actual_tokens, predicted_tokens, last_close)

    formatted = metrics.format_summary()
    assert isinstance(formatted, str)
    assert "PREDICTION METRICS SUMMARY" in formatted
    assert "Token Accuracy: 60.0%" in formatted
