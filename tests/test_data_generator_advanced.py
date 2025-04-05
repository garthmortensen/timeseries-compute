# tests/test_data_generator_advanced.py

import pytest
import pandas as pd
import numpy as np
from generalized_timeseries.data_generator import (
    generate_price_series,
    PriceSeriesGenerator,
)
from generalized_timeseries.data_processor import (
    fill_data,
    scale_data,
    stationarize_data,
    log_stationarity,
    DataScaler,
    DataScalerFactory,
    StationaryReturnsProcessor,
)
from generalized_timeseries.stats_model import (
    ModelARIMA,
    ModelGARCH,
    ModelFactory,
    run_arima,
    run_garch,
)

# -- price generator tests --


def test_generate_price_series_default_params():
    """test default wrapper func"""
    price_dict, price_df = generate_price_series()

    # check types & content
    assert isinstance(price_dict, dict)
    assert isinstance(price_df, pd.DataFrame)
    assert "GME" in price_dict
    assert "BYND" in price_dict

    # check dates (trading days in 2023)
    expected_days = len(pd.date_range(start="2023-01-01", end="2023-12-31", freq="B"))
    assert len(price_df) == expected_days

    # check initial values
    assert price_dict["GME"][0] == 100.0
    assert price_dict["BYND"][0] == 200.0


def test_custom_date_range():
    """test custom dates"""
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    price_dict, price_df = generate_price_series(
        start_date=start_date, end_date=end_date
    )

    # check dates
    expected_days = len(pd.date_range(start=start_date, end=end_date, freq="B"))
    assert len(price_df) == expected_days
    assert price_df.index[0].strftime("%Y-%m-%d") == "2024-01-01"


def test_custom_anchor_prices():
    """test custom starting prices"""
    anchor_prices = {"ESPO": 150.0, "HERO": 250.0, "GAMR": 1000.0}
    price_dict, price_df = generate_price_series(anchor_prices=anchor_prices)

    # check tickers
    for ticker in anchor_prices:
        assert ticker in price_dict
        assert ticker in price_df.columns

    # check values
    for ticker, price in anchor_prices.items():
        assert price_dict[ticker][0] == price
        assert price_df[ticker].iloc[0] == price


def test_price_series_statistics():
    """test stats properties"""
    _, price_df = generate_price_series()

    # gaussian params
    expected_mean = 0
    expected_std = 1
    tolerance = 0.5  # allow some variation

    for ticker in price_df.columns:
        series = price_df[ticker]
        diff = series.diff().dropna()

        # check stats are close to expected
        assert abs(diff.mean() - expected_mean) < tolerance
        assert abs(diff.std() - expected_std) < tolerance * 2


def test_decimal_precision():
    """test 4-decimal rounding"""
    _, price_df = generate_price_series()

    max_decimals = 4
    for column in price_df.columns:
        # get decimal digit count
        decimal_lengths = price_df[column].apply(
            lambda x: len(str(x).split(".")[-1]) if "." in str(x) else 0
        )
        assert (decimal_lengths <= max_decimals).all()


# -- data processor tests --


@pytest.fixture
def sample_data_with_missing():
    """gen data w/ missing vals"""
    data = {
        "A": [1, 2, None, 4, 5],
        "B": [None, 2, 3, None, 5],
        "C": [1, 2, 3, 4, 5],  # complete column
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_for_scaling():
    """gen data for scaling tests"""
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": [100, 200, 300, 400, 500],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_for_stationarity():
    """gen non-stationary + stationary data"""
    np.random.seed(42)

    # random walk (non-stationary) - each point depends on previous point
    n_points = 100
    random_walk = np.zeros(n_points)
    for i in range(1, n_points):
        random_walk[i] = random_walk[i - 1] + np.random.normal(0, 1)

    # white noise (stationary) - points are independent
    white_noise = np.random.normal(0, 1, n_points)

    data = {"random_walk": random_walk, "white_noise": white_noise}
    return pd.DataFrame(data)


def test_scale_data_standardize(sample_data_for_scaling):
    """test z-score scaling"""
    scaled_df = scale_data(sample_data_for_scaling, method="standardize")

    target_mean = 0
    target_std = 1
    epsilon = 1e-10  # small error tolerance

    # check stats match targets
    for column in scaled_df.columns:
        assert abs(scaled_df[column].mean() - target_mean) < epsilon
        assert abs(scaled_df[column].std() - target_std) < epsilon


def test_scale_data_minmax(sample_data_for_scaling):
    """test 0-1 scaling"""
    scaled_df = scale_data(sample_data_for_scaling, method="minmax")

    target_min = 0
    target_max = 1
    epsilon = 1e-10  # small error tolerance

    # check bounds match targets
    for column in scaled_df.columns:
        assert abs(scaled_df[column].min() - target_min) < epsilon
        assert abs(scaled_df[column].max() - target_max) < epsilon


def test_stationarize_data(sample_data_for_stationarity):
    """test differencing for stationarity"""
    stationary_df = stationarize_data(sample_data_for_stationarity)

    # check differenced cols exist
    assert "random_walk_diff" in stationary_df.columns
    assert "white_noise_diff" in stationary_df.columns
    assert "random_walk" in stationary_df.columns
    assert "white_noise" in stationary_df.columns


def test_data_scaler_factory_invalid_strategy():
    """test factory error handling"""
    with pytest.raises(ValueError):
        DataScalerFactory.create_handler("invalid_strategy")


# -- stats model tests --


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

    data = {"AR": ar_series, "GARCH": garch_series}
    return pd.DataFrame(data)


def test_model_factory_arima(stationary_sample_data):
    """test factory creates arima"""
    ar_lag = 1
    diff_order = 0
    ma_lag = 0
    steps = 3

    model = ModelFactory.create_model(
        model_type="ARIMA",
        data=stationary_sample_data,
        order=(ar_lag, diff_order, ma_lag),
        steps=steps,
    )

    assert isinstance(model, ModelARIMA)
    assert model.order == (ar_lag, diff_order, ma_lag)
    assert model.steps == steps


def test_model_factory_garch(stationary_sample_data):
    """test factory creates garch"""
    p_order = 1  # garch lag
    q_order = 1  # arch lag
    dist_type = "normal"

    model = ModelFactory.create_model(
        model_type="GARCH",
        data=stationary_sample_data,
        p=p_order,
        q=q_order,
        dist=dist_type,
    )

    assert isinstance(model, ModelGARCH)
    assert model.p == p_order
    assert model.q == q_order
    assert model.dist == dist_type


def test_model_factory_invalid():
    """test factory invalid type"""
    test_data = pd.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(ValueError):
        ModelFactory.create_model(model_type="INVALID", data=test_data)


def test_model_garch_methods(stationary_sample_data):
    """test garch model methods"""
    model = ModelGARCH(data=stationary_sample_data, p=1, q=1)

    # check fit
    fits = model.fit()
    assert isinstance(fits, dict)
    assert "AR" in fits
    assert "GARCH" in fits

    # check summary
    summaries = model.summary()
    assert isinstance(summaries, dict)
    assert "AR" in summaries
    assert "GARCH" in summaries

    # check forecast
    forecast_steps = 3
    forecasts = model.forecast(steps=forecast_steps)
    assert isinstance(forecasts, dict)
    assert "AR" in forecasts
    assert "GARCH" in forecasts


def test_run_arima(stationary_sample_data):
    """test arima convenience func"""
    p_val = 1
    d_val = 0
    q_val = 0
    forecast_steps = 3

    arima_fit, arima_forecast = run_arima(
        df_stationary=stationary_sample_data,
        p=p_val,
        d=d_val,
        q=q_val,
        forecast_steps=forecast_steps,
    )

    assert isinstance(arima_fit, dict)
    assert isinstance(arima_forecast, dict)
    assert "AR" in arima_fit
    assert "AR" in arima_forecast


def test_run_garch(stationary_sample_data):
    """test garch convenience func"""
    p_val = 1
    q_val = 1
    dist_type = "normal"
    forecast_steps = 3

    garch_fit, garch_forecast = run_garch(
        df_stationary=stationary_sample_data,
        p=p_val,
        q=q_val,
        dist=dist_type,
        forecast_steps=forecast_steps,
    )

    assert isinstance(garch_fit, dict)
    assert isinstance(garch_forecast, dict)
    assert "AR" in garch_fit
    assert "GARCH" in garch_forecast
