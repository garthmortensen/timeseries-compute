# tests/test_stats_model_garch.py
import pytest
import pandas as pd
import numpy as np
from generalized_timeseries.stats_model import ModelGARCH, run_garch


@pytest.fixture
def garch_sample_data():
    """gen data w/ vol clustering for garch tests"""
    np.random.seed(42)  # fixed seed
    n = 100

    # model 1 parameters
    constant_1 = 0.1  # base volatility level
    arch_coef_1 = 0.2  # impact of past shocks
    garch_coef_1 = 0.7  # persistence of past volatility
    mean_ret = 0  # expected return

    # garch series 1
    returns = np.zeros(n)
    volatility = np.ones(n)

    # garch(1,1) process - model 1
    for i in range(1, n):
        # calculate today's volatility based on yesterday's values
        prev_shock = returns[i - 1] ** 2  # previous squared return
        prev_vol = volatility[i - 1]  # previous volatility

        # today's volatility equation
        volatility[i] = constant_1 + arch_coef_1 * prev_shock + garch_coef_1 * prev_vol

        # generate random return based on volatility
        std_dev = np.sqrt(volatility[i])
        returns[i] = np.random.normal(mean_ret, std_dev)

    # model 2 parameters
    constant_2 = 0.05  # lower base volatility
    arch_coef_2 = 0.15  # less shock impact
    garch_coef_2 = 0.8  # higher persistence

    # garch series 2
    returns2 = np.zeros(n)
    volatility2 = np.ones(n)

    # garch(1,1) process - model 2
    for i in range(1, n):
        prev_shock2 = returns2[i - 1] ** 2
        prev_vol2 = volatility2[i - 1]

        volatility2[i] = (
            constant_2 + arch_coef_2 * prev_shock2 + garch_coef_2 * prev_vol2
        )

        std_dev2 = np.sqrt(volatility2[i])
        returns2[i] = np.random.normal(mean_ret, std_dev2)

    # make df w/ dates
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    data = {"returns1": returns, "returns2": returns2}
    return pd.DataFrame(data, index=dates)


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
