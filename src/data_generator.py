#!/usr/bin/env python3
# data_generator.py

# handle relative directory imports for chronicler
import logging as l

# script specific imports
import pandas as pd
import random
from tabulate import tabulate  # pretty print dfs


class PriceSeriesGenerator:
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

    def generate_prices(self, anchor_prices: dict):
        """
        create price series for given tickers with initial prices

        Args:
            anchor_prices (dict): keys = tickers, values = initial prices

        Returns:
            dict: keys = tickers, values = prices
            pd.DataFrame: df of all series
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


def generate_price_series(config):
    l.info("Generating price series data")
    generator = PriceSeriesGenerator(
        start_date=config.data_generator.start_date,
        end_date=config.data_generator.end_date,
    )
    price_dict, price_df = generator.generate_prices(
        anchor_prices=config.data_generator.anchor_prices
    )
    return price_dict, price_df
