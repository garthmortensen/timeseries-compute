#!/usr/bin/env python3
# data_generator.py

import logging as l

# script specific imports
import pandas as pd
import random
from tabulate import tabulate  # pretty print dfs
from typing import Dict, List, Optional  # type hints for better code readability

# set random seed for reproducibility
DEFAULT_RANDOM_SEED = 2025  # applied to all random module functions
random.seed(DEFAULT_RANDOM_SEED)


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

    generate_prices(anchor_prices: dict) -> List[dict]:
        Generates a series of prices for the given tickers with initial prices.
            anchor_prices (dict): A dictionary where keys are tickers and values are initial prices.
            List[dict]: A list of dictionaries, each containing date, symbol, and price.
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

    def generate_prices(self, anchor_prices: Dict[str, float]) -> List[dict]:
        """
        Create price series for given tickers with initial prices.

        Args:
            anchor_prices (Dict[str, float]): keys = tickers, values = initial prices

        Returns:
            List[dict]: List of dictionaries with date, symbol, and price keys
        """
        # First generate prices for each ticker
        price_data = {}
        l.info("generating prices...")
        for ticker, initial_price in anchor_prices.items():
            prices = [initial_price]
            for _ in range(1, len(self.dates)):
                # create price changes using gaussian distribution
                # statquest book has a good explanation
                change = random.gauss(mu=0, sigma=1)  # mean = 0, standev = 1
                prices.append(prices[-1] + change)
            price_data[ticker] = prices

        # For logging purposes
        df = pd.DataFrame(price_data, index=self.dates).round(4)
        l.info("generated prices:")
        l.info("\n" + tabulate(df.head(5), headers="keys", tablefmt="fancy_grid"))

        # Convert to the new format: list of dictionaries
        records = []
        for date in self.dates:
            date_str = date.strftime("%Y-%m-%d")
            for ticker in anchor_prices.keys():
                idx = self.dates.get_loc(date)
                price = price_data[ticker][idx]
                records.append(
                    {
                        "date": date_str,
                        "symbol": ticker,
                        "price": round(price, 4),  # Ensure 4 decimal places
                    }
                )

        return records


# set new random seed using a "convenience" function, which is a wrapper around the class
def set_random_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """
    Sets the random seed for the random module.

    Args:
        seed (int): Seed value for random number generator.
    """
    l.info(f"Setting random seed to {seed}")
    random.seed(seed)


# convenience wrapper around the class
def generate_price_series(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    anchor_prices: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
) -> List[dict]:
    """
    Generates a series of price data based on the provided parameters.

    Args:
        start_date (str, optional): The start date for the price series. Defaults to "2023-01-01".
        end_date (str, optional): The end date for the price series. Defaults to "2023-12-31".
        anchor_prices (Dict[str, float], optional): A dictionary of tickers and their initial prices.
            Defaults to {"GME": 100.0, "BYND": 200.0} if None.
        random_seed (int, optional): Seed for random number generation. If provided, overrides the module-level seed. Defaults to None.

    Returns:
        List[dict]: A list of dictionaries with date, symbol, and price keys.
    """
    if anchor_prices is None:
        anchor_prices = {"GME": 100.0, "BYND": 200.0}

    if random_seed is not None:
        set_random_seed(random_seed)

    l.info("Generating price series data")
    generator = PriceSeriesGenerator(
        start_date=start_date,
        end_date=end_date,
    )
    records = generator.generate_prices(anchor_prices=anchor_prices)
    return records
