#!/usr/bin/env python3
# tests/test_stats_model_garch.py
import pytest
import pandas as pd
import numpy as np
from timeseries_compute.stats_model import ModelGARCH, run_garch


@pytest.fixture
def stationary_sample_data():
    """gen stationary data for modeling"""
    np.random.seed(42)
    n_points = 100

    # ar(1) params
    ar_coef = 0.7
    noise_std = 1.0

    # garch params
    constant = 0.1
    arch_coef = 0.2
    garch_coef = 0.7

    # create ar(1) series
    ar_series = np.zeros(n_points)
    for i in range(1, n_points):
        ar_series[i] = ar_coef * ar_series[i - 1] + np.random.normal(0, noise_std)

    # create garch series
    garch_series = np.zeros(n_points)
    volatility = np.ones(n_points)

    for i in range(1, n_points):
        volatility[i] = (
            constant
            + arch_coef * garch_series[i - 1] ** 2
            + garch_coef * volatility[i - 1]
        )
        garch_series[i] = np.random.normal(0, np.sqrt(volatility[i]))

    # Create dates
    start_date = pd.Timestamp("2023-01-01")
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_points)]
    
    data = {"Date": dates, "AR": ar_series, "GARCH": garch_series}
    return pd.DataFrame(data)


def test_model_garch_initialization(garch_sample_data):
    """test garch init"""
    # model parameters
    garch_lag = 1  # p: volatility lag
    arch_lag = 1  # q: shock lag
    distribution = "normal"

    model = ModelGARCH(
        data=garch_sample_data, p=garch_lag, q=arch_lag, dist=distribution
    )
    assert model.data.equals(garch_sample_data)
    assert model.p == garch_lag
    assert model.q == arch_lag
    assert model.dist == distribution


def test_model_garch_fit(garch_sample_data):
    """test fitting garch"""
    # use simplest garch model
    p_val = 1
    q_val = 1

    model = ModelGARCH(data=garch_sample_data, p=p_val, q=q_val)
    fits = model.fit()
    assert isinstance(fits, dict)
    assert "returns1" in fits
    assert "returns2" in fits
    assert hasattr(fits["returns1"], "params")
    assert hasattr(fits["returns2"], "params")


def test_model_garch_summary(garch_sample_data):
    """test summary method"""
    model = ModelGARCH(data=garch_sample_data, p=1, q=1)
    model.fit()
    summaries = model.summary()
    assert isinstance(summaries, dict)
    assert "returns1" in summaries
    assert "returns2" in summaries
    assert isinstance(summaries["returns1"], str)
    assert isinstance(summaries["returns2"], str)


def test_model_garch_forecast(garch_sample_data):
    """test forecast output"""
    # forecast settings
    horizon = 3  # steps ahead to forecast

    model = ModelGARCH(data=garch_sample_data, p=1, q=1)
    model.fit()
    forecasts = model.forecast(steps=horizon)
    assert isinstance(forecasts, dict)
    assert "returns1" in forecasts
    assert "returns2" in forecasts
    # vol forecasts should have right len
    assert len(forecasts["returns1"]) == horizon
    assert len(forecasts["returns2"]) == horizon


def test_run_garch_function(garch_sample_data):
    """test convenience func"""
    # model settings
    p_order = 1
    q_order = 1
    dist_type = "normal"
    forecast_steps = 3

    garch_fit, garch_forecast = run_garch(
        df_stationary=garch_sample_data,
        p=p_order,
        q=q_order,
        dist=dist_type,
        forecast_steps=forecast_steps,
    )

    assert isinstance(garch_fit, dict)
    assert isinstance(garch_forecast, dict)
    assert "returns1" in garch_fit
    assert "returns1" in garch_forecast
    assert "returns2" in garch_fit
    assert "returns2" in garch_forecast


def test_model_garch_different_params(garch_sample_data):
    """test diff p,q params"""
    # higher order model
    p_lag = 2  # 2 lags of volatility
    q_lag = 1  # 1 lag of shocks

    model = ModelGARCH(data=garch_sample_data, p=p_lag, q=q_lag, dist="normal")
    fits = model.fit()
    assert model.p == p_lag
    assert model.q == q_lag

    # expected param count
    min_params = 1 + p_lag + q_lag  # constant + garch terms + arch terms

    # check params count
    for col in fits:
        assert len(fits[col].params) >= min_params  # garch(2,1)


def test_model_garch_student_t_dist(garch_sample_data):
    """test t-dist"""
    # t-distribution allows for fat tails
    model = ModelGARCH(data=garch_sample_data, p=1, q=1, dist="t")
    fits = model.fit()

    # t-dist has extra degrees of freedom param
    for col in fits:
        assert "nu" in fits[col].params.index
