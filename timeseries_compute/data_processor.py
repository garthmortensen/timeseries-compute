#!/usr/bin/env python3
# data_processor.py

import logging as l

# handle data transformation and preparation tasks
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate  # pretty print dfs
from typing import Callable, Dict, Tuple, Optional, List, Union


class MissingDataHandler:
    """Handles missing data through various strategies such as dropping or forward filling."""

    def __init__(self) -> None:
        """
        Initializes the MissingDataHandler class.
        Logs an ASCII banner for initialization.
        """
        ascii_banner = """
        \n
        \t> MissingDataHandler <\n"""
        l.info(ascii_banner)

    def drop_na(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows with missing values from the given DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame from which to drop rows with missing values.

        Returns:
            pd.DataFrame: A DataFrame with rows containing missing values removed.
        """
        l.info("Dropping rows with missing values")
        l.info("df filled:")
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data.dropna()

    def forward_fill(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the DataFrame using the forward fill method.

        Args:
            data (pd.DataFrame): The DataFrame containing missing values to be filled.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled using forward fill.
        """
        l.info("Filling missing values with forward fill")
        l.info("df filled:")
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data.fillna(method="ffill")


class MissingDataHandlerFactory:
    """Factory for creating missing data handlers based on a specified strategy."""

    @staticmethod
    def create_handler(strategy: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """
        Creates a handler function based on the specified strategy.

        Args:
            strategy (str): The strategy to handle missing data. Options are "drop" or "forward_fill".

        Returns:
            Callable[[pd.DataFrame], pd.DataFrame]: A function that handles missing data accordingly.

        Raises:
            ValueError: If an unknown strategy is provided.
        """
        handler = MissingDataHandler()
        l.info(f"Creating handler for strategy: {strategy}")
        if strategy.lower() == "drop":
            return handler.drop_na
        elif strategy.lower() == "forward_fill":
            return handler.forward_fill
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")


def fill_data(df: pd.DataFrame, strategy: str = "forward_fill") -> pd.DataFrame:
    """
    Fills missing data in the given DataFrame according to the specified strategy.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be processed.
        strategy (str, optional): Strategy for handling missing values.
            Options are "drop" or "forward_fill". Defaults to "forward_fill".

    Returns:
        pd.DataFrame: The DataFrame with missing values handled according to the specified strategy.
    """
    l.info("\n# Processing: handling missing values")
    handler_missing = MissingDataHandlerFactory.create_handler(strategy=strategy)
    df_filled = handler_missing(df)
    return df_filled


class DataScaler:
    """
    Provides methods to scale numeric data in a pandas DataFrame.

    Methods:
        scale_data_standardize(data: pd.DataFrame) -> pd.DataFrame:
            Standardizes all numeric columns except the index by subtracting the mean and dividing by the standard deviation.

        scale_data_minmax(data: pd.DataFrame) -> pd.DataFrame:
            Scales all numeric columns using MinMaxScaler by dividing each value by the range (max - min) of the column.
    """

    def scale_data_standardize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes all numeric columns in the given DataFrame except the index.

        Args:
            data (pd.DataFrame): The input DataFrame containing numeric columns to be standardized.

        Returns:
            pd.DataFrame: The DataFrame with standardized numeric columns.

        Notes:
            - The standardization is performed by subtracting the mean and dividing by the standard deviation for each numeric column.
            - The index of the DataFrame is not modified.
            - Logs the process of scaling and displays the first 5 rows of the scaled DataFrame.
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            data[column] = (data[column] - data[column].mean()) / data[column].std()
        l.info("Scaling data using standardization")
        l.info("df scaled:")
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data

    def scale_data_minmax(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the numeric columns of the given DataFrame using Min-Max scaling.

        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be scaled.

        Returns:
            pd.DataFrame: The DataFrame with scaled numeric columns.

        Notes:
            - This function scales each numeric column to a range between 0 and 1.
            - Non-numeric columns are not affected by this scaling.
            - The function logs the scaling process and the first 5 rows of the scaled DataFrame.
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            data[column] = (data[column] - data[column].min()) / (
                data[column].max() - data[column].min()
            )
        l.info("Scaling data using minmax")
        l.info("df scaled:")
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data


class DataScalerFactory:
    """
    Factory class for creating data scaling handlers based on the specified strategy.

    Methods:
        create_handler(strategy: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
            Returns the appropriate scaling function based on the provided strategy.
    """

    @staticmethod
    def create_handler(strategy: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """
        Returns the appropriate scaling function based on the provided strategy.

        Args:
            strategy (str): The scaling strategy to use. Supported values are "standardize" and "minmax".

        Returns:
            Callable[[pd.DataFrame], pd.DataFrame]: The scaling function corresponding to the specified strategy.

        Raises:
            ValueError: If the provided strategy is not recognized.
        """
        scaler = DataScaler()
        l.info(f"Creating scaler for strategy: {strategy}")
        if strategy.lower() == "standardize":
            return scaler.scale_data_standardize
        elif strategy.lower() == "minmax":
            return scaler.scale_data_minmax
        else:
            raise ValueError(f"Unknown data scaling strategy: {strategy}")


def scale_data(df: pd.DataFrame, method: str = "standardize") -> pd.DataFrame:
    """
    Scales the input DataFrame according to the specified method.

    Args:
        df (pd.DataFrame): The input data to be scaled.
        method (str, optional): Scaling method to use. Options are "standardize" or "minmax".
            Defaults to "standardize".

    Returns:
        pd.DataFrame: The scaled DataFrame.
    """
    l.info("\n# Processing: scaling data")
    handler_scaler = DataScalerFactory.create_handler(strategy=method)
    df_scaled = handler_scaler(df)
    return df_scaled


class StationaryReturnsProcessor:
    """
    A class to process and test the stationarity of time series data.

    Methods:
        make_stationary(data: pd.DataFrame, method: str) -> pd.DataFrame:
            Apply the chosen method to make the data stationary.

        test_stationarity(data: pd.DataFrame, test: str) -> Dict[str, Dict[str, float]]:
            Perform the Augmented Dickey-Fuller test to check for stationarity.

        log_adf_results(data: Dict[str, Dict[str, float]], p_value_threshold: float) -> None:
            Log the interpreted results of the ADF test.
    """

    def make_stationary(
        self, data: pd.DataFrame, method: str = "difference"
    ) -> pd.DataFrame:
        """
        Apply the chosen method to make the data stationary.

        Args:
            data (pd.DataFrame): The input data to be made stationary.
            method (str, optional): The method to use for making the data stationary.
                Currently supported method is "difference". Defaults to "difference".

        Returns:
            pd.DataFrame: The transformed data with the applied stationarity method.

        Raises:
            ValueError: If an unknown method is provided.
        """
        l.info(f"Applying stationarity method: {method}")
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        if method.lower() == "difference":
            for column in numeric_columns:
                data[f"{column}_diff"] = data[column].diff()
            data = data.dropna()
        else:
            raise ValueError(f"Unknown make_stationary method: {method}")

        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data

    def test_stationarity(
        self, data: pd.DataFrame, test: str = "adf"
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform the Augmented Dickey-Fuller (ADF) test for stationarity on the given data.

        The null hypothesis (H0) is that the series is non-stationary (has a unit root).
        The alternative hypothesis (H1) is that the series is stationary.

        Args:
            data (pd.DataFrame): The input data containing time series to be tested.
            test (str, optional): The type of stationarity test to perform.
                Currently, only "adf" is supported. Defaults to "adf".

        Returns:
            Dict[str, Dict[str, float]]: A dictionary where keys are column names and values are dictionaries containing
                the ADF Statistic and p-value for each numeric column in the input data.

        Raises:
            ValueError: If an unsupported stationarity test is specified.
        """
        if test.lower() != "adf":
            raise ValueError(f"Unsupported stationarity test: {test}")
        else:
            l.info(f"Test_stationarity: {test} test for stationarity")
            results = {}
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                if data[column].isnull().any() or not np.isfinite(data[column]).all():
                    l.warning(
                        f"Column {column} contains NaN or Inf values. Skipping ADF test."
                    )
                    continue
                result = adfuller(data[column])
                results[column] = {"ADF Statistic": result[0], "p-value": result[1]}
        return results

    def log_adf_results(
        self, data: Dict[str, Dict[str, float]], p_value_threshold: float = 0.05
    ) -> None:
        """
        Logs interpreted Augmented Dickey-Fuller (ADF) test results.

        Args:
            data (Dict[str, Dict[str, float]]): A dictionary where keys are series names and values are dictionaries containing ADF test results.
                Each value dictionary should have the keys "ADF Statistic" and "p-value".
            p_value_threshold (float, optional): The threshold for the p-value to determine if the series is stationary.
                Defaults to 0.05.

        Returns:
            None
        """
        for series_name, result in data.items():
            adf_stat = result["ADF Statistic"]
            p_value = result["p-value"]
            if p_value < p_value_threshold:
                interpretation = f"p_value {p_value:.2e} < p_value_threshold {p_value_threshold}. Data is stationary (reject null hypothesis)."
            else:
                interpretation = f"p_value {p_value:.2e} >= p_value_threshold {p_value_threshold}. Data is non-stationary (fail to reject null hypothesis)."

            l.info(
                f"series_name: {series_name}\n"
                f"   adf_stat: {adf_stat:.2f}\n"
                f"   p_value: {p_value:.2e}\n"
                f"   interpretation: {interpretation}\n"
            )


def price_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prices to log returns, similar to MATLAB's price2ret function.

    Args:
        prices: DataFrame of price series

    Returns:
        DataFrame of log returns
    """
    return np.log(prices / prices.shift(1)).dropna()


class StationaryReturnsProcessorFactory:
    """
    Factory class for creating handlers for stationary returns processing strategies.

    Methods:
        create_handler(strategy: str) -> Callable:
            Returns the appropriate processing function based on the provided strategy.
    """

    @staticmethod
    def create_handler(strategy: str) -> StationaryReturnsProcessor:
        """
        Returns the appropriate processing function based on the provided strategy.

        Args:
            strategy (str): The name of the strategy for which the processing function is to be created.
                Supported strategies are:
                - "transform_to_stationary_returns"
                - "test_stationarity"
                - "log_stationarity"

        Returns:
            StationaryReturnsProcessor: A processor instance for the specified strategy.

        Raises:
            ValueError: If an unknown strategy is provided.
        """
        stationary_returns_processor = StationaryReturnsProcessor()
        l.info(f"Creating processor for strategy: {strategy}")
        if strategy.lower() == "transform_to_stationary_returns":
            return stationary_returns_processor
        elif strategy.lower() == "test_stationarity":
            return stationary_returns_processor
        elif strategy.lower() == "log_stationarity":
            return stationary_returns_processor
        else:
            raise ValueError(
                f"Unknown stationary returns processing strategy: {strategy}"
            )


def stationarize_data(df: pd.DataFrame, method: str = "difference") -> pd.DataFrame:
    """
    Processes the given DataFrame to make the data stationary.

    Args:
        df (pd.DataFrame): The input data to be made stationary.
        method (str, optional): Method to use for making data stationary.
            Currently only "difference" is supported. Defaults to "difference".

    Returns:
        pd.DataFrame: The stationary version of the input data.
    """
    l.info("\n# Processing: making data stationary")
    stationary_returns_processor = StationaryReturnsProcessor()
    df_stationary = stationary_returns_processor.make_stationary(data=df, method=method)
    return df_stationary


def test_stationarity(
    df: pd.DataFrame, method: str = "adf"
) -> Dict[str, Dict[str, float]]:
    """
    Tests the stationarity of a given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be tested for stationarity.
        method (str, optional): Method to use for testing stationarity.
            Currently only "adf" is supported. Defaults to "adf".

    Returns:
        Dict[str, Dict[str, float]]: Results of the stationarity test.
    """
    l.info("\n# Testing: stationarity")
    stationary_returns_processor = StationaryReturnsProcessorFactory.create_handler(
        "test_stationarity"
    )
    adf_results = stationary_returns_processor.test_stationarity(data=df, test=method)
    return adf_results


def log_stationarity(
    adf_results: Dict[str, Dict[str, float]], p_value_threshold: float = 0.05
) -> None:
    """
    Logs the stationarity of the given DataFrame using the Augmented Dickey-Fuller (ADF) test.

    Args:
        adf_results (Dict[str, Dict[str, float]]): Results from test_stationarity function.
        p_value_threshold (float, optional): The p-value threshold for the ADF test.
            Defaults to 0.05.

    Returns:
        None
    """
    l.info("\n# Logging: stationarity")
    stationary_returns_processor = StationaryReturnsProcessorFactory.create_handler(
        "log_stationarity"
    )
    stationary_returns_processor.log_adf_results(
        data=adf_results,
        p_value_threshold=p_value_threshold,
    )


def prepare_timeseries_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares time series data for analysis by:
    1. Converting date column to datetime and setting as index (if not already)
    2. Ensuring numeric columns are properly typed
    3. Removing non-numeric columns

    Args:
        df (pd.DataFrame): Input DataFrame with time series data

    Returns:
        pd.DataFrame: Properly formatted DataFrame for time series analysis
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Handle date column if it exists and isn't already the index
    if not isinstance(df.index, pd.DatetimeIndex):
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df.set_index(date_col, inplace=True)

    # Convert numeric columns to proper type
    for col in df.columns:
        if col.lower() not in ["date", "time", "datetime", "timestamp"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        raise ValueError("No numeric columns found after data preparation")

    l.info("Data prepared for time series analysis")
    l.info("\n" + tabulate(numeric_df.head(5), headers="keys", tablefmt="fancy_grid"))

    return numeric_df


def calculate_ewma_covariance(
    series1: pd.Series, series2: pd.Series, lambda_val: float = 0.95
) -> pd.Series:
    """
    Calculate Exponentially Weighted Moving Average covariance between two series.

    Args:
        series1: First time series
        series2: Second time series
        lambda_val: Decay factor (0.95 or 0.97 in your thesis)

    Returns:
        Series of EWMA covariances
    """
    # Initialize covariance series
    cov_series = pd.Series(index=series1.index)

    # Calculate initial covariance (first 20 observations)
    init_window = min(20, len(series1))
    init_cov = series1.iloc[:init_window].cov(series2.iloc[:init_window])
    cov_series.iloc[0] = init_cov

    # Calculate EWMA covariance
    for t in range(1, len(series1)):
        cov_series.iloc[t] = (
            lambda_val * cov_series.iloc[t - 1]
            + (1 - lambda_val) * series1.iloc[t - 1] * series2.iloc[t - 1]
        )

    return cov_series


def calculate_ewma_volatility(series: pd.Series, lambda_val: float = 0.95) -> pd.Series:
    """
    Calculate Exponentially Weighted Moving Average volatility for a series.

    Args:
        series: Time series
        lambda_val: Decay factor

    Returns:
        Series of EWMA volatilities
    """
    # Square the returns
    squared_returns = series**2

    # Calculate EWMA variance
    ewma_variance = squared_returns.ewm(alpha=1 - lambda_val, adjust=False).mean()

    # Convert variance to volatility (standard deviation)
    ewma_volatility = np.sqrt(ewma_variance)

    return ewma_volatility
