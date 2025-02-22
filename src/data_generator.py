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

    def generate_prices(self, ticker_initial_prices: dict):
        """
        create price series for given tickers with initial prices

        Args:
            ticker_initial_prices (dict): keys = tickers, values = initial prices

        Returns:
            dict: keys = tickers, values = prices
            pd.DataFrame: df of all series
        """
        data = {}
        l.info("generating prices...")
        for ticker, initial_price in ticker_initial_prices.items():
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
