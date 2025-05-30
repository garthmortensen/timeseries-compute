#!/usr/bin/env python3
# tests/test_stats_model.py
import pytest
import pandas as pd
import numpy as np
from timeseries_compute.stats_model import ModelARIMA


@pytest.fixture
def sample_data():
    """gen ts data for arima tests"""
    np.random.seed(42)  # fixed seed
    n = 50  # sample size

    # ar(1) process parameters
    auto_coef = 0.7  # how much past values affect future (memory strength)
    noise_level_a = 1.0  # noise magnitude for series A (more volatile)
    noise_level_b = 0.5  # noise magnitude for series B (less volatile)
    mean_value = 0  # centered around zero

    # initialize empty series
    ar_series_a = np.zeros(n)
    ar_series_b = np.zeros(n)

    # generate two ar(1) series with different noise levels
    for i in range(1, n):
        # formula: current = (memory_factor * previous) + random_noise

        # series A: higher noise
        noise_a = np.random.normal(mean_value, noise_level_a)
        ar_series_a[i] = auto_coef * ar_series_a[i - 1] + noise_a

        # series B: lower noise
        noise_b = np.random.normal(mean_value, noise_level_b)
        ar_series_b[i] = auto_coef * ar_series_b[i - 1] + noise_b

    # create timeseries dataframe
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    data = {"A": ar_series_a, "B": ar_series_b}
    return pd.DataFrame(data, index=dates)


def test_model_arima_initialization(sample_data):
    """test arima init"""
    # arima components
    ar_terms = 1  # p: autoregressive lags
    diff_terms = 0  # d: differencing level
    ma_terms = 0  # q: moving average lags
    pred_steps = 2  # forecast horizon

    model = ModelARIMA(
        data=sample_data, order=(ar_terms, diff_terms, ma_terms), steps=pred_steps
    )
    assert model.data.equals(sample_data)
    assert model.order == (ar_terms, diff_terms, ma_terms)
    assert model.steps == pred_steps


def test_model_arima_fit(sample_data):
    """test fit method"""
    # ar(1) model specs
    ar_lag = 1  # num of autoregressive lags
    diff_level = 0  # no differencing needed
    ma_lag = 0  # no moving average component

    model = ModelARIMA(data=sample_data, order=(ar_lag, diff_level, ma_lag), steps=2)
    fits = model.fit()
    assert isinstance(fits, dict)
    assert "A" in fits
    assert "B" in fits
    assert hasattr(fits["A"], "params")
    assert hasattr(fits["B"], "params")


def test_model_arima_summary(sample_data):
    """test summary output"""
    # model order parameters
    p_val = 1  # autoregressive order
    d_val = 0  # integration order
    q_val = 0  # moving average order

    model = ModelARIMA(data=sample_data, order=(p_val, d_val, q_val), steps=2)
    model.fit()
    summaries = model.summary()
    assert isinstance(summaries, dict)
    assert "A" in summaries
    assert "B" in summaries
    assert isinstance(summaries["A"], str)
    assert isinstance(summaries["B"], str)


def test_model_arima_forecast(sample_data):
    """test forecast result"""
    # model settings
    model_order = (1, 0, 0)  # ar order, diff order, ma order
    forecast_horizon = 2  # steps to predict

    model = ModelARIMA(data=sample_data, order=model_order, steps=forecast_horizon)
    model.fit()
    forecasts = model.forecast()
    assert isinstance(forecasts, dict)
    assert "A" in forecasts
    assert "B" in forecasts
    
    # For multi-step forecasts (steps > 1), we should get lists
    assert isinstance(forecasts["A"], list)
    assert isinstance(forecasts["B"], list)
    assert len(forecasts["A"]) == forecast_horizon
    assert len(forecasts["B"]) == forecast_horizon

    # test multi-step forecasting - ensure more than 1 data point is forecasted
    multi_step_horizon = 5  # forecast 5 steps ahead
    multi_model = ModelARIMA(data=sample_data, order=model_order, steps=multi_step_horizon)
    multi_model.fit()
    multi_forecasts = multi_model.forecast()

    # verify that we get forecasts for multiple steps
    assert isinstance(multi_forecasts, dict)
    assert "A" in multi_forecasts
    assert "B" in multi_forecasts
    
    # Should get lists with the correct number of forecast steps
    assert isinstance(multi_forecasts["A"], list)
    assert isinstance(multi_forecasts["B"], list)
    assert len(multi_forecasts["A"]) == multi_step_horizon
    assert len(multi_forecasts["B"]) == multi_step_horizon
    
    # Test single-step forecast returns a float
    single_step_model = ModelARIMA(data=sample_data, order=model_order, steps=1)
    single_step_model.fit()
    single_forecasts = single_step_model.forecast()
    assert isinstance(single_forecasts["A"], float)
    assert isinstance(single_forecasts["B"], float)
