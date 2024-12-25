# handle relative directory imports for chronicler
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import chronicler
import logging as l
from chronicler import Chronicler

# pass current script name
current_script_path = os.path.abspath(__file__)  # "/myproject/run.py"
chronicler = Chronicler(current_script_path)

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
        ascii_banner = """\n
        ╔═╗┬─┐┬┌─┐┌─┐╔═╗┌─┐┬─┐┬┌─┐┌─┐╔═╗┌─┐┌┐┌┌─┐┬─┐┌─┐┌┬┐┌─┐┬─┐
        ╠═╝├┬┘││  ├┤ ╚═╗├┤ ├┬┘│├┤ └─┐║ ╦├┤ │││├┤ ├┬┘├─┤ │ │ │├┬┘
        ╩  ┴└─┴└─┘└─┘╚═╝└─┘┴└─┴└─┘└─┘╚═╝└─┘┘└┘└─┘┴└─┴ ┴ ┴ └─┘┴└─
        """
        l.info(ascii_banner)

        self.start_date = start_date
        self.end_date = end_date
        self.dates = pd.date_range(start=start_date, end=end_date, freq="B")  # weekdays only

    def generate_prices(self, ticker_prices: dict):
        """
        create price series for given tickers with initial prices

        Args:
            ticker_prices (dict): keys = tickers, values = initial prices

        Returns:
            dict: keys = tickers, values = prices
            pd.DataFrame: df of all series
        """
        data = {}
        for ticker, initial_price in ticker_prices.items():
            prices = [initial_price]
            for _ in range(1, len(self.dates)):
                # create price changes using gaussian distribution
                # statquest book has a good explanation
                change = random.gauss(mu=0, sigma=1)  # mean = 0, standev = 1
                prices.append(prices[-1] + change)
            data[ticker] = prices

        df = pd.DataFrame(data, index=self.dates)

        return data, df

# Example usage
generator = PriceSeriesGenerator(start_date="2023-01-01", end_date="2023-12-31")
ticker_prices = {"GME": 150.0, "BYND": 200.0}
price_dict, price_df = generator.generate_prices(ticker_prices=ticker_prices)

l.info("generated prices:")
l.info(tabulate(price_df.head(5), headers="keys", tablefmt="fancy_grid"))

