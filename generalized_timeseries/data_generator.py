#!/usr/bin/env python3
# data_generator.py

import logging as l

# script specific imports
import pandas as pd
import random
from tabulate import tabulate  # pretty print dfs
from typing import Dict, Tuple, Optional


class PriceSeriesGenerator:
    """
    Class generates a series of prices for given tickers over a specified date range.

    Attributes:
        start_date (str): The start date of the price series in YYYY-MM-DD format.
        end_date (str): The end date of the price series in YYYY-MM-DD format.
        dates (pd.DatetimeIndex): A range of dates from start_date to end_date, including only weekdays.

    Methods:
        __init__(start_date: str, end_date: str):
            Initializes the PriceSeriesGenerator with the given date range.

    generate_prices(anchor_prices: dict) -> Tuple[dict, pd.DataFrame]:
        Generates a series of prices for the given tickers with initial prices.
            anchor_prices (dict): A dictionary where keys are tickers and values are initial prices.
            dict: A dictionary where keys are tickers and values are lists of generated prices.
            pd.DataFrame: A DataFrame containing the generated price series for all tickers.
    """

    def __init__(self, start_date: str, end_date: str):
        """
        Given data range, initialize the generator

        Args:
            start_date (str): start, YYYY-MM-DD
            end_date (str): end, YYYY-MM-DD
        """
        ascii_banner = """\n\n\t> PriceSeriesGenerator <\n"""
        l.info(ascii_banner)

        self.start_date = start_date
        self.end_date = end_date
        self.dates = pd.date_range(
            start=start_date, end=end_date, freq="B"
        )  # weekdays only

    def generate_prices(
        self, anchor_prices: Dict[str, float]
    ) -> Tuple[Dict[str, list], pd.DataFrame]:
        """
        Create price series for given tickers with initial prices.

        Args:
            anchor_prices (Dict[str, float]): keys = tickers, values = initial prices

        Returns:
            Tuple[Dict[str, list], pd.DataFrame]:
                - dict: keys = tickers, values = prices
                - pd.DataFrame: df of all series
        """
        data = {}
        l.info("generating prices...")
        for ticker, initial_price in anchor_prices.items():
            prices = [initial_price]
            for _ in range(1, len(self.dates)):
                # create price changes using gaussian distribution
                # statquest book has a good explanation
                change = random.gauss(mu=0, sigma=1)  # mean = 0, standev = 1
                prices.append(prices[-1] + change)
            data[ticker] = prices

        df = pd.DataFrame(data, index=self.dates).round(4)  # strictly 4

        l.info("generated prices:")
        l.info("\n" + tabulate(df.head(5), headers="keys", tablefmt="fancy_grid"))

        return data, df


# convenience wrapper around the class
def generate_price_series(
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    anchor_prices: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, list], pd.DataFrame]:
    """
    Generates a series of price data based on the provided parameters.

    Args:
        start_date (str, optional): The start date for the price series. Defaults to "2023-01-01".
        end_date (str, optional): The end date for the price series. Defaults to "2023-12-31".
        anchor_prices (Dict[str, float], optional): A dictionary of tickers and their initial prices.
            Defaults to {"GME": 100.0, "BYND": 200.0} if None.

    Returns:
        Tuple[Dict[str, list], pd.DataFrame]:
            - price_dict: A dictionary of generated prices.
            - price_df: A DataFrame of generated prices.
    """
    if anchor_prices is None:
        anchor_prices = {"GME": 100.0, "BYND": 200.0}

    l.info("Generating price series data")
    generator = PriceSeriesGenerator(
        start_date=start_date,
        end_date=end_date,
    )
    price_dict, price_df = generator.generate_prices(anchor_prices=anchor_prices)
    return price_dict, price_df
