# tests/test_data_processor.py

import pytest
import logging
import numpy as np
import pandas as pd
import warnings
from generalized_timeseries.data_processor import (
    MissingDataHandler,
    MissingDataHandlerFactory,
    test_stationarity,
    fill_data,
)
from unittest.mock import patch, MagicMock
from statsmodels.tsa.stattools import adfuller


# Add fixture for the test_stationarity function in the module
@pytest.fixture
def df():
    """Fixture to provide a DataFrame for testing stationarity."""
    # Create a test DataFrame with some values
    return pd.DataFrame(
        {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [5.0, 6.0, 7.0, 8.0, 9.0]}
    )


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    data = {
        "A": [1.0, 2.0, None, 4.0, 5.0],
        "B": [None, 2.0, 3.0, None, 5.0],
        "C": [1.0, None, None, 4.0, 5.0],
    }
    return pd.DataFrame(data)


def test_create_handler_drop():
    """Test the create_handler method of MissingDataHandlerFactory with 'drop' strategy."""
    handler_func = MissingDataHandlerFactory.create_handler("drop")
    assert callable(handler_func)
    assert handler_func.__name__ == "drop_na"


def test_create_handler_forward_fill():
    """Test the create_handler method of MissingDataHandlerFactory with 'forward_fill' strategy."""
    handler_func = MissingDataHandlerFactory.create_handler("forward_fill")
    assert callable(handler_func)
    assert handler_func.__name__ == "forward_fill"


def test_create_handler_invalid_strategy():
    """Test the create_handler method of MissingDataHandlerFactory with an invalid strategy."""
    with pytest.raises(ValueError):
        MissingDataHandlerFactory.create_handler("invalid_strategy")


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with NaN values for testing."""
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, np.nan, 4.0, 5.0],
            "B": [np.nan, 2.0, 3.0, np.nan, 5.0],
            "C": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_drop_na():
    """Test drop_na method removes rows with NaN values."""
    # Create test data
    df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0], "B": [5.0, np.nan, 7.0, 8.0]})

    # Create handler instance
    handler = MissingDataHandler()

    # Expected result
    expected = pd.DataFrame({"A": [1.0, 4.0], "B": [5.0, 8.0]}, index=[0, 3])

    # Test method
    result = handler.drop_na(df)

    # Check equality
    pd.testing.assert_frame_equal(result, expected)


def test_forward_fill(sample_df):
    """Test forward_fill method correctly propagates values forward."""
    # Create handler instance
    handler = MissingDataHandler()

    # Suppress the FutureWarning for the test
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        result = handler.forward_fill(sample_df)

    expected = pd.DataFrame(
        {
            "A": [1.0, 2.0, 2.0, 4.0, 5.0],
            "B": [np.nan, 2.0, 3.0, 3.0, 5.0],
            "C": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_fill_data(sample_df):
    """Test fill_data with various filling methods."""
    # Test forward fill (default)
    # Suppress the FutureWarning for the test
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        result_forward = fill_data(sample_df, strategy="forward_fill")

    expected_forward = pd.DataFrame(
        {
            "A": [1.0, 2.0, 2.0, 4.0, 5.0],
            "B": [np.nan, 2.0, 3.0, 3.0, 5.0],
            "C": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    pd.testing.assert_frame_equal(result_forward, expected_forward)

    # Test drop
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        result_drop = fill_data(sample_df, strategy="drop")

    # The dataset has NaNs in rows 0, 2, and 3, so only rows 1 and 4 remain
    expected_drop = pd.DataFrame(
        {"A": [2.0, 5.0], "B": [2.0, 5.0], "C": [2.0, 5.0]}, index=[1, 4]
    )

    pd.testing.assert_frame_equal(result_drop, expected_drop)

    # Test invalid method
    with pytest.raises(ValueError):
        fill_data(sample_df, strategy="invalid_strategy")


@patch("generalized_timeseries.data_processor.adfuller")
def test_ts_stationarity(mock_adfuller):
    """Test test_stationarity function properly uses adfuller test."""
    # Mock adfuller return value (test statistic, p-value, lags, nobs, critical values, icbest)
    mock_adfuller.return_value = (
        -3.5,
        0.01,
        1,
        100,
        {"1%": -3.5, "5%": -2.9, "10%": -2.6},
        1,
    )

    # Create test dataframe (not series)
    df = pd.DataFrame({"test_col": [1.0, 2.0, 3.0, 4.0, 5.0]})

    # Call function
    result = test_stationarity(df)

    # Check mock was called with right args
    mock_adfuller.assert_called_once()

    # Check expected structure based on implementation
    assert isinstance(result, dict)
    assert "test_col" in result
    assert "ADF Statistic" in result["test_col"]
    assert "p-value" in result["test_col"]
    assert result["test_col"]["ADF Statistic"] == -3.5
    assert result["test_col"]["p-value"] == 0.01


def test_stationarity_integration():
    """Integration test for test_stationarity with real adfuller function."""
    # Create stationary series
    np.random.seed(42)

    # Use more random data to avoid divide by zero in statsmodels
    stationary_series = np.random.randn(100) * 10 + 5  # scaled and shifted

    # Create non-stationary series (random walk)
    random_walk = np.cumsum(np.random.randn(100) * 2)  # scaled random walk

    # Create a DataFrame for testing (since test_stationarity requires DataFrame)
    df = pd.DataFrame({"stationary": stationary_series, "non_stationary": random_walk})

    # Suppress RuntimeWarning from statsmodels
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Call function with real adfuller
        adf_results = test_stationarity(df)

    # Check structure without returning
    assert isinstance(adf_results, dict)
    assert "stationary" in adf_results
    assert "non_stationary" in adf_results
    assert isinstance(adf_results["stationary"]["ADF Statistic"], float)
    assert isinstance(adf_results["stationary"]["p-value"], float)
    assert isinstance(adf_results["non_stationary"]["ADF Statistic"], float)
    assert isinstance(adf_results["non_stationary"]["p-value"], float)
