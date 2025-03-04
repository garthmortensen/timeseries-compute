# tests/test_stats_model.py

import pytest
import pandas as pd
from generalized_timeseries.stats_model import ModelARIMA

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 3, 4, 5, 6]
    }
    return pd.DataFrame(data)

def test_model_arima_initialization(sample_data):
    """Test the initialization of ModelARIMA."""
    model = ModelARIMA(data=sample_data, order=(1, 1, 1), steps=2)
    assert model.data.equals(sample_data)
    assert model.order == (1, 1, 1)
    assert model.steps == 2

def test_model_arima_fit(sample_data):
    """Test the fit method of ModelARIMA."""
    model = ModelARIMA(data=sample_data, order=(1, 1, 1), steps=2)
    fits = model.fit()
    assert isinstance(fits, dict)
    assert "A" in fits
    assert "B" in fits
    assert hasattr(fits["A"], "params")
    assert hasattr(fits["B"], "params")

def test_model_arima_summary(sample_data):
    """Test the summary method of ModelARIMA."""
    model = ModelARIMA(data=sample_data, order=(1, 1, 1), steps=2)
    model.fit()
    summaries = model.summary()
    assert isinstance(summaries, dict)
    assert "A" in summaries
    assert "B" in summaries
    assert isinstance(summaries["A"], str)
    assert isinstance(summaries["B"], str)

def test_model_arima_forecast(sample_data):
    """Test the forecast method of ModelARIMA."""
    model = ModelARIMA(data=sample_data, order=(1, 1, 1), steps=2)
    model.fit()
    forecasts = model.forecast()
    assert isinstance(forecasts, dict)
    assert "A" in forecasts
    assert "B" in forecasts
    assert isinstance(forecasts["A"], float)
    assert isinstance(forecasts["B"], float)
