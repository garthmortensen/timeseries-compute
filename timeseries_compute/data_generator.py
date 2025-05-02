#!/usr/bin/env python3
# timeseries_compute/data_generator.py

"""
Time Series Data Generation Module.

This module provides functionality for generating synthetic price series data with
controlled statistical properties. It's designed as the first step in a typical
time series analysis pipeline, creating test data with known characteristics.

Key Components:
- PriceSeriesGenerator: Class for generating correlated price series
- generate_price_series: Convenience function with simplified interface
- set_random_seed: Function to ensure reproducible results

Typical Usage Flow:
1. Create a PriceSeriesGenerator instance with desired date range
2. Generate price series with specific initial values and correlations
3. Proceed with the generated data to data_processor.py for preparation

The generated price series follow a random walk with drift, with options
to control cross-series correlations.
"""

import logging as l

# script specific imports
import numpy as np  # for sqrt function
import pandas as pd
import random
from tabulate import tabulate  # pretty print dfs
from typing import Dict, Tuple, Optional, List  # type hints for better code readability

# set random seed for reproducibility
DEFAULT_RANDOM_SEED = 2025  # this is used by random module
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

        generate_correlated_prices(anchor_prices: dict, correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None) -> Dict[str, list]:
            Generates a series of correlated prices for the given tickers with initial prices.
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

    def generate_correlated_prices(
        self,
        anchor_prices: Dict[str, float],
        correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Dict[str, list]:
        """
        Create price series for given tickers with initial prices and correlations.

        Args:
            anchor_prices (Dict[str, float]): Dictionary where keys are ticker symbols 
                (e.g., 'AAPL', 'MSFT') and values are their respective initial prices.
            correlation_matrix (Dict[Tuple[str, str], float], optional): Dictionary specifying 
                correlations between ticker pairs. Each key should be a tuple of two ticker 
                symbols (e.g., ('AAPL', 'MSFT')), and each value should be the desired 
                correlation coefficient between -1.0 and 1.0. For example:
                {('AAPL', 'MSFT'): 0.7, ('AAPL', 'GOOG'): 0.5, ('MSFT', 'GOOG'): 0.6}
                If None, a default correlation of 0.6 will be used for all pairs.

        Returns:
            Dict[str, list]: Dictionary where keys are ticker symbols and values are lists 
                containing the generated price series for each ticker.
                
        Example:
            >>> generator = PriceSeriesGenerator(start_date="2023-01-01", end_date="2023-01-31")
            >>> anchor_prices = {"AAA": 150.0, "BBB": 250.0}
            >>> correlations = {("AAA", "BBB"): 0.7}
            >>> prices = generator.generate_correlated_prices(anchor_prices, correlations)
        """
        # Initialize price data
        price_data = {}
        l.info("generating correlated prices...")

        # Create list of tickers
        tickers = list(anchor_prices.keys())
        num_tickers = len(tickers)

        # Set default correlation if not provided
        if correlation_matrix is None:
            # Default moderate positive correlation between all pairs
            correlation_matrix = {}
            for i in range(num_tickers):
                for j in range(i + 1, num_tickers):
                    correlation_matrix[(tickers[i], tickers[j])] = 0.6

        # Initialize with starting prices
        for ticker, initial_price in anchor_prices.items():
            price_data[ticker] = [initial_price]

        # Number of time steps to generate
        time_steps = len(self.dates) - 1

        # Generate correlated random changes
        for t in range(time_steps):
            # First, generate uncorrelated normal random variables
            uncorrelated_changes = {}
            for ticker in tickers:
                uncorrelated_changes[ticker] = random.gauss(mu=0, sigma=1)

            # Apply correlations to create correlated changes
            correlated_changes = {}

            # Start with uncorrelated values
            for ticker in tickers:
                correlated_changes[ticker] = uncorrelated_changes[ticker]

            # Apply correlations
            for (ticker1, ticker2), corr in correlation_matrix.items():
                # Update both directions with partial correlation
                if ticker1 in tickers and ticker2 in tickers:
                    # Mix in some of ticker2's change into ticker1
                    correlated_changes[ticker1] = (
                        uncorrelated_changes[ticker1] * np.sqrt(1 - corr**2)
                        + uncorrelated_changes[ticker2] * corr
                    )
                    # Mix in some of ticker1's change into ticker2
                    correlated_changes[ticker2] = (
                        uncorrelated_changes[ticker2] * np.sqrt(1 - corr**2)
                        + uncorrelated_changes[ticker1] * corr
                    )

            # Apply the correlated changes to prices
            for ticker in tickers:
                new_price = round(
                    price_data[ticker][-1] + correlated_changes[ticker], 4
                )
                price_data[ticker].append(new_price)

        return price_data


# set new random seed using a "convenience" function, which is a wrapper around the class
def set_random_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """
    Sets the random seed for the random module.

    Args:
        seed (int): Seed value for random number generator.
    """
    l.info(f"Setting random seed to {seed}")
    random.seed(seed)


def generate_price_series(
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    anchor_prices: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
    correlations: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[Dict[str, list], pd.DataFrame]:
    """
    Generates a series of price data based on the provided parameters.

    I return both a dict and a df. Supporting both means i can stop second guessing which to return.

    Args:
        start_date (str, optional): The start date for the price series. Defaults to "2023-01-01".
        end_date (str, optional): The end date for the price series. Defaults to "2023-12-31".
        anchor_prices (Dict[str, float], optional): A dictionary of tickers and their initial prices.
            Defaults to {"GME": 100.0, "BYND": 200.0} if None.
        random_seed (int, optional): Seed for random number generation. If provided, overrides the module-level seed.
        correlations (Dict[Tuple[str, str], float], optional): Dictionary specifying correlations between ticker pairs.

    Returns:
        Tuple[Dict[str, list], pd.DataFrame]: A dictionary of generated prices and a DataFrame.
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
    price_dict = generator.generate_correlated_prices(
        anchor_prices=anchor_prices, correlation_matrix=correlations
    )

    # Use the dates from the generator instead of creating a new date range

    price_df = pd.DataFrame(price_dict, index=generator.dates)

    return price_dict, price_df
