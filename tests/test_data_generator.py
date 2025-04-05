# tests/test_data_generator.py

import pytest
import pandas as pd
from generalized_timeseries.data_generator import PriceSeriesGenerator


def test_price_series_generator_initialization():
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    generator = PriceSeriesGenerator(start_date=start_date, end_date=end_date)

    assert generator.start_date == start_date
    assert generator.end_date == end_date


def test_generate_prices():
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    anchor_prices = {"GM": 51.1, "LM": 2.2}
    generator = PriceSeriesGenerator(start_date=start_date, end_date=end_date)

    price_dict = generator.generate_prices(anchor_prices=anchor_prices)

    assert isinstance(price_dict, dict)
    assert "GM" in price_dict
    assert "LM" in price_dict
