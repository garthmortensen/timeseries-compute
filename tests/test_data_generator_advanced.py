# tests/test_data_generator_advanced.py

import pytest
import pandas as pd
import numpy as np
from generalized_timeseries.data_generator import generate_price_series, PriceSeriesGenerator

def test_generate_price_series_default_params():
    """Test the convenience wrapper function with default parameters."""
    price_dict, price_df = generate_price_series()
    
    # Test the returned types
    assert isinstance(price_dict, dict)
    assert isinstance(price_df, pd.DataFrame)
    
    # Test default tickers are present
    assert "GME" in price_dict
    assert "BYND" in price_dict
    
    # Test shape matches expected date range (workdays in 2023)
    # Approximately 252 trading days in a year
    expected_days = len(pd.date_range(start="2023-01-01", end="2023-12-31", freq="B"))
    assert len(price_df) == expected_days
    
    # Test initial values match anchor prices
    assert price_dict["GME"][0] == 100.0
    assert price_dict["BYND"][0] == 200.0

def test_custom_date_range():
    """Test with custom date range."""
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    price_dict, price_df = generate_price_series(start_date=start_date, end_date=end_date)
    
    # Check if date range matches
    expected_days = len(pd.date_range(start=start_date, end=end_date, freq="B"))
    assert len(price_df) == expected_days
    assert price_df.index[0].strftime("%Y-%m-%d") == "2024-01-01"

def test_custom_anchor_prices():
    """Test with custom anchor prices."""
    anchor_prices = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 1000.0}
    price_dict, price_df = generate_price_series(anchor_prices=anchor_prices)
    
    # Check if all tickers are present
    for ticker in anchor_prices.keys():
        assert ticker in price_dict
        assert ticker in price_df.columns
    
    # Check if initial prices match
    for ticker, price in anchor_prices.items():
        assert price_dict[ticker][0] == price
        assert price_df[ticker].iloc[0] == price

def test_price_series_statistics():
    """Test statistical properties of generated price series."""
    price_dict, price_df = generate_price_series()
    
    # Test for each ticker
    for ticker in price_df.columns:
        series = price_df[ticker]
        
        # Since we're using Gaussian with mean=0, the mean of differences should be close to 0
        diff = series.diff().dropna()
        assert abs(diff.mean()) < 1.0  # Should be close to 0 with some tolerance
        
        # Standard deviation should be close to 1 (as sigma=1 in the random.gauss)
        assert 0.5 < diff.std() < 1.5  # Allow some tolerance

def test_decimal_precision():
    """Test that values are rounded to 4 decimal places."""
    _, price_df = generate_price_series()
    
    # Check if all values are rounded to 4 decimal places
    for column in price_df.columns:
        # Extract the decimal part for each value and check its length
        decimal_lengths = price_df[column].apply(
            lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0
        )
        assert (decimal_lengths <= 4).all()

# tests/test_data_processor_advanced.py

import pytest
import pandas as pd
import numpy as np
from generalized_timeseries.data_processor import (
    fill_data, scale_data, stationarize_data, 
    log_stationarity,
    DataScaler, DataScalerFactory, StationaryReturnsProcessor
)

@pytest.fixture
def sample_data_with_missing():
    """Fixture providing sample data with missing values."""
    data = {
        "A": [1, 2, None, 4, 5],
        "B": [None, 2, 3, None, 5],
        "C": [1, 2, 3, 4, 5]  # No missing values
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_for_scaling():
    """Fixture providing sample data for scaling tests."""
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": [100, 200, 300, 400, 500]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_for_stationarity():
    """Fixture providing non-stationary and stationary data."""
    # Random walk (non-stationary)
    np.random.seed(42)
    random_walk = np.cumsum(np.random.normal(0, 1, 100))
    
    # White noise (stationary)
    white_noise = np.random.normal(0, 1, 100)
    
    data = {
        "random_walk": random_walk,
        "white_noise": white_noise
    }
    return pd.DataFrame(data)

def test_scale_data_standardize(sample_data_for_scaling):
    """Test standardization scaling."""
    scaled_df = scale_data(sample_data_for_scaling, method="standardize")
    
    # Check mean and std for each column
    for column in scaled_df.columns:
        assert abs(scaled_df[column].mean()) < 1e-10  # Mean should be close to 0
        assert abs(scaled_df[column].std() - 1.0) < 1e-10  # Std should be close to 1

def test_scale_data_minmax(sample_data_for_scaling):
    """Test min-max scaling."""
    scaled_df = scale_data(sample_data_for_scaling, method="minmax")
    
    # Check min and max for each column
    for column in scaled_df.columns:
        assert abs(scaled_df[column].min()) < 1e-10  # Min should be close to 0
        assert abs(scaled_df[column].max() - 1.0) < 1e-10  # Max should be close to 1

def test_stationarize_data(sample_data_for_stationarity):
    """Test making data stationary through differencing."""
    stationary_df = stationarize_data(sample_data_for_stationarity)
    
    # Check that differenced columns were created
    assert "random_walk_diff" in stationary_df.columns
    assert "white_noise_diff" in stationary_df.columns
    
    # Original columns should still exist
    assert "random_walk" in stationary_df.columns
    assert "white_noise" in stationary_df.columns


def test_data_scaler_factory_invalid_strategy():
    """Test DataScalerFactory with invalid strategy."""
    with pytest.raises(ValueError):
        DataScalerFactory.create_handler("invalid_strategy")

# tests/test_stats_model_advanced.py

import pytest
import pandas as pd
import numpy as np
from generalized_timeseries.stats_model import (
    ModelARIMA, ModelGARCH, ModelFactory,
    run_arima, run_garch
)

@pytest.fixture
def stationary_sample_data():
    """Fixture to provide stationary data for testing."""
    np.random.seed(42)
    
    # Create an AR(1) process
    n = 100
    ar_param = 0.7
    ar_series = np.zeros(n)
    
    for i in range(1, n):
        ar_series[i] = ar_param * ar_series[i-1] + np.random.normal(0, 1)
    
    # Create a series with GARCH effects
    garch_series = np.zeros(n)
    volatility = np.ones(n)
    
    for i in range(1, n):
        volatility[i] = 0.1 + 0.2 * garch_series[i-1]**2 + 0.7 * volatility[i-1]
        garch_series[i] = np.random.normal(0, np.sqrt(volatility[i]))
    
    data = {
        "AR": ar_series,
        "GARCH": garch_series
    }
    return pd.DataFrame(data)

def test_model_factory_arima(stationary_sample_data):
    """Test ModelFactory for creating ARIMA models."""
    model = ModelFactory.create_model(
        model_type="ARIMA",
        data=stationary_sample_data,
        order=(1, 0, 0),
        steps=3
    )
    
    assert isinstance(model, ModelARIMA)
    assert model.order == (1, 0, 0)
    assert model.steps == 3

def test_model_factory_garch(stationary_sample_data):
    """Test ModelFactory for creating GARCH models."""
    model = ModelFactory.create_model(
        model_type="GARCH",
        data=stationary_sample_data,
        p=1,
        q=1,
        dist="normal"
    )
    
    assert isinstance(model, ModelGARCH)
    assert model.p == 1
    assert model.q == 1
    assert model.dist == "normal"

def test_model_factory_invalid():
    """Test ModelFactory with invalid model type."""
    with pytest.raises(ValueError):
        ModelFactory.create_model(
            model_type="INVALID",
            data=pd.DataFrame({"A": [1, 2, 3]})
        )

def test_model_garch_methods(stationary_sample_data):
    """Test the methods of ModelGARCH."""
    model = ModelGARCH(data=stationary_sample_data, p=1, q=1)
    
    # Test fit method
    fits = model.fit()
    assert isinstance(fits, dict)
    assert "AR" in fits
    assert "GARCH" in fits
    
    # Test summary method
    summaries = model.summary()
    assert isinstance(summaries, dict)
    assert "AR" in summaries
    assert "GARCH" in summaries
    
    # Test forecast method
    forecasts = model.forecast(steps=3)
    assert isinstance(forecasts, dict)
    assert "AR" in forecasts
    assert "GARCH" in forecasts

def test_run_arima(stationary_sample_data):
    """Test the run_arima convenience function."""
    arima_fit, arima_forecast = run_arima(
        df_stationary=stationary_sample_data,
        p=1,
        d=0,
        q=0,
        forecast_steps=3
    )
    
    assert isinstance(arima_fit, dict)
    assert isinstance(arima_forecast, dict)
    assert "AR" in arima_fit
    assert "AR" in arima_forecast

def test_run_garch(stationary_sample_data):
    """Test the run_garch convenience function."""
    garch_fit, garch_forecast = run_garch(
        df_stationary=stationary_sample_data,
        p=1,
        q=1,
        dist="normal",
        forecast_steps=3
    )
    
    assert isinstance(garch_fit, dict)
    assert isinstance(garch_forecast, dict)
    assert "AR" in garch_fit
    assert "GARCH" in garch_forecast

