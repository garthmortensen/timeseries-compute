#!/usr/bin/env python3
# data_processor.py

# handle relative directory imports for chronicler
import logging as l

# handle data transformation and preparation tasks
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate  # pretty print dfs


class MissingDataHandler:
    """TODO: fill_missing_values_with_mean, fill_missing_values_with_mode, interpolation"""

    def __init__(self):  
        """
        Initializes the MissingDataHandler class.

        Constructor logs an ASCII banner and sets up the class for dependency injection.
        Dependency injection is used to pass dependencies as arguments instead of creating objects
        within the class or function. This approach makes the code easier to test, change, and understand.
        """
        ascii_banner = """\n\n\t> MissingDataHandler <\n"""
        l.info(ascii_banner)

    def drop_na(self, data):
        """
        Drops rows with missing values from the given DataFrame.

        Parameters:
        data (pandas.DataFrame): The DataFrame from which to drop rows with missing values.

        Returns:
        pandas.DataFrame: A DataFrame with rows containing missing values removed.
        """
        l.info(f"Dropping rows with missing values")
        l.info("df filled:")
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))

        return data.dropna()

    def forward_fill(self, data):
        """
        Fills missing values in the DataFrame using forward fill method.

        Parameters:
        data (pd.DataFrame): The DataFrame containing missing values to be filled.

        Returns:
        pd.DataFrame: The DataFrame with missing values filled using forward fill method.
        """
        l.info(f"Filling missing values with forward fill")
        l.info("df filled:")
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data.fillna(method="ffill")


class MissingDataHandlerFactory:
    # static method can be called without creating an instance of the class
    # e.g. MissingDataHandlerFactory.create_handler(strategy, config)
    @staticmethod
    def create_handler(strategy) -> callable:
        """Creates a handler function based on the specified strategy.

        Args:
            strategy (str): The strategy to handle missing data. Options are "drop" or "forward_fill".

        Returns:
            callable: A function that handles missing data according to the specified strategy.

        Raises:
            ValueError: If an unknown strategy is provided.
        """

        handler = MissingDataHandler()
        l.info(f"Creating handler for strategy: {strategy}")
        # centralize logic to choose method
        if strategy.lower() == "drop":
            return handler.drop_na
        elif strategy == "forward_fill":
            return handler.forward_fill
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")


def fill_data(df, config):
    """
    Fills missing data in the given DataFrame according to the specified configuration.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be processed.
        config (Config): Configuration object containing the strategy for handling missing values.

    Returns:
        pandas.DataFrame: The DataFrame with missing values handled according to the specified strategy.
    """
    l.info("\n# Processing: handling missing values")
    handler_missing = MissingDataHandlerFactory.create_handler(
        strategy=config.data_processor.handle_missing_values.strategy
    )
    df_filled = handler_missing(df)
    return df_filled


class DataScaler:
    """
    DataScaler is a class that provides methods to scale numeric data in a pandas DataFrame.

    Methods
    -------
    scale_data_standardize(data)
        Standardize all numeric columns except the index by subtracting the mean and dividing by the standard deviation.

    scale_data_minmax(data)
        Scale all numeric columns using MinMaxScaler by dividing each value by the range (max - min) of the column.
    """

    def scale_data_standardize(self, data):
        """
        Standardize all numeric columns in the given DataFrame except the index.

        Parameters:
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
        l.info(f"Scaling data using standardization")
        l.info("df scaled:")
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data

    def scale_data_minmax(self, data):
        """
        Scales the numeric columns of the given DataFrame using Min-Max scaling.
        Parameters:
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
            data[column] = (data[column].min()) / (
                data[column].max() - data[column].min()
            )
        l.info(f"Scaling data using minmax")
        l.info("df scaled:")
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data


class DataScalerFactory:
    """
    Factory class for creating data scaling handlers based on the specified strategy.
    Methods
    -------
    create_handler(strategy: str) -> Callable
        Returns the appropriate scaling function based on the provided strategy.
        Parameters
        ----------
        strategy : str
            The scaling strategy to use. Supported strategies are "standardize" and "minmax".
        Returns
        -------
        Callable
            The scaling function corresponding to the specified strategy.
        Raises
        ------
        ValueError
            If the provided strategy is not recognized.
    """

    @staticmethod
    def create_handler(strategy):
        """
        Return the appropriate scaling function based on the provided strategy.

        Parameters:
        strategy (str): The scaling strategy to use. Supported values are "standardize" and "minmax".

        Returns:
        function: A function that performs the specified scaling on data.

        Raises:
        ValueError: If the provided strategy is not supported.
        """
        scaler = DataScaler()
        l.info(f"creating scaler for strategy: {strategy}")
        if strategy.lower() == "standardize":
            return scaler.scale_data_standardize
        elif strategy == "minmax":  # TODO: fixme. This turns everything into a constant
            return scaler.scale_data_minmax
        else:
            raise ValueError(f"Unknown data scaling strategy: {strategy}")


def scale_data(df, config):
    """
    Scales the input DataFrame according to the specified configuration.

    Parameters:
    df (pandas.DataFrame): The input data to be scaled.
    config (object): Configuration object containing scaling method details.

    Returns:
    pandas.DataFrame: The scaled DataFrame.
    """
    l.info("\n# Processing: scaling data")
    handler_scaler = DataScalerFactory.create_handler(
        strategy=config.data_processor.scaling.method
    )
    df_scaled = handler_scaler(df)
    return df_scaled


class StationaryReturnsProcessor:
    """A class to process and test the stationarity of time series data.

    Methods
    -------
    make_stationary(data, method)
        Apply the chosen method to make the data stationary.
        
    test_stationarity(data, test)
        Perform the Augmented Dickey-Fuller test to check for stationarity.
        
    log_adf_results(data, p_value_threshold)
        Log the interpreted results of the ADF test.
    """
    def make_stationary(self, data, method):
        """
        Apply the chosen method to make the data stationary.

        Parameters:
        data (pd.DataFrame): The input data to be made stationary.
        method (str): The method to use for making the data stationary. Currently supported method is "difference".

        Returns:
        pd.DataFrame: The transformed data with the applied stationarity method.

        Raises:
        ValueError: If an unknown method is provided.
        """

        l.info(f"applying stationarity method: {method}")
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        if method.lower() == "difference":
            for column in numeric_columns:
                data[f"{column}_diff"] = data[column].diff()
            data = data.dropna()
        else:
            raise ValueError(f"unknown make_stationary method: {method}")

        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))

        return data

    def test_stationarity(self, data, test):
        """
        Perform the Augmented Dickey-Fuller (ADF) test for stationarity on the given data.

        The null hypothesis (H0) is that the series is non-stationary (has a unit root).
        The alternative hypothesis (H1) is that the series is stationary.

        Parameters:
        data (pd.DataFrame): The input data containing time series to be tested.
        test (str): The type of stationarity test to perform. Currently, only "adf" is supported.

        Returns:
        dict: A dictionary where keys are column names and values are dictionaries containing
              the ADF Statistic and p-value for each numeric column in the input data.

        Raises:
        ValueError: If an unsupported stationarity test is specified.

        Notes:
        - Columns containing NaN or infinite values will be skipped.
        - Only numeric columns in the input data will be tested.
        """
        if test.lower() != "adf":
            raise ValueError(f"Unsupported stationarity test: {test}")
        else:
            l.info(f"test_stationarity: {test} test for stationarity")
            results = {}
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                # NaN and Infs will break ADF
                if data[column].isnull().any() or not np.isfinite(data[column]).all():
                    l.warning(
                        f"Column {column} contains NaN or Inf values. Skipping ADF test."
                    )
                    continue
                result = adfuller(data[column])
                results[column] = {"ADF Statistic": result[0], "p-value": result[1]}
            l.info(f"results: {results}")
        return results

    def log_adf_results(self, data, p_value_threshold):
        """Logs interpreted Augmented Dickey-Fuller (ADF) test results.

        Parameters:
        data (dict): A dictionary where keys are series names and values are dictionaries containing ADF test results.
        Each value dictionary should have the keys "ADF Statistic" and "p-value".
        p_value_threshold (float): The threshold for the p-value to determine if the series is stationary.

        Returns:
        None

        Notes:
        - If the p-value is less than the p_value_threshold, the series is considered stationary (reject null hypothesis).
        - If the p-value is greater than or equal to the p_value_threshold, the series is considered non-stationary (fail to reject null hypothesis).
        - Logs the series name, ADF statistic, p-value, and interpretation of the result.

        # TODO: plot for visual inspection that series is stationary
        # TODO: rename from log, bc i think logorithm when i see log here. interpert_adf_results?
        """
        for series_name, result in data.items():
            adf_stat = result["ADF Statistic"]
            p_value = result["p-value"]
            if p_value < p_value_threshold:
                interpretation = f"p_value {p_value:.2e} < p_value_threshold {p_value_threshold}. Data is stationary (reject null hypothesis that series is non-stationary)"
            else:
                interpretation = f"p_value {p_value:.2e} >= p_value_threshold {p_value_threshold}. Data is non-stationary (fail to reject null hypothesis that series is non-stationary)"

            # If ADF is more negative than the critical value, reject the null H. The series is stationary
            # If p-value is less than the chosen significance level (0.05), reject null H that the series is stationary
            l.info(
                f"series_name: {series_name}\n"
                f"   adf_stat: {adf_stat:.2f}\n"  # More negative suggests stronger stationarity evidence
                f"   p_value: {p_value:.2e}\n"
                f"   interpretation: {interpretation}\n"
            )


class StationaryReturnsProcessorFactory:
    """Factory class for creating handlers for stationary returns processing strategies.

    This factory class provides a method to create and return appropriate processing 
    functions based on the specified strategy. The processing functions are stateless 
    and do not require instantiation, hence they are returned directly.

    Methods
    -------
    create_handler(strategy: str) -> function
        Returns the appropriate processing function based on the provided strategy.

    Parameters
    ----------
    strategy : str
        The name of the strategy for which the processing function is to be created. 
        Supported strategies are:
        - "transform_to_stationary_returns"
        - "test_stationarity"
        - "log_stationarity"

    Raises
    ------
    ValueError
        If an unknown strategy is provided.
    
    Notes:
    - `staticmethod` is used to define a method that doesn't operate on an instance.
    """

    @staticmethod
    def create_handler(strategy):
        """Return the appropriate processing function based on strategy."""
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


def stationarize_data(df, config):
    """Processes the given DataFrame to make the data stationary.

    Parameters:
    df (pandas.DataFrame): The input data to be made stationary.
    config (object): Configuration object containing the method to be used for making the data stationary.

    Returns:
    pandas.DataFrame: The stationary version of the input data.
    """
    l.info("\n# Processing: making data stationary")
    # recreating the object each time is not efficient, but it's simple
    stationary_returns_processor = StationaryReturnsProcessor()
    df_stationary = stationary_returns_processor.make_stationary(
        data=df, method=config.data_processor.make_stationary.method
    )
    return df_stationary


def test_stationarity(df, config):
    """Tests the stationarity of a given DataFrame using the specified configuration.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be tested for stationarity.
    config (object): Configuration object containing the method to be used for testing stationarity.

    Returns:
    dict: Results of the stationarity test.
    """
    l.info("\n# Testing: stationarity")
    stationary_returns_processor = StationaryReturnsProcessorFactory.create_handler(
        "test_stationarity"
    )
    adf_results = stationary_returns_processor.test_stationarity(
        data=df, test=config.data_processor.test_stationarity.method
    )

    return adf_results


def log_stationarity(df, config):
    """
    Logs the stationarity of the given DataFrame using the Augmented Dickey-Fuller (ADF) test.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be tested for stationarity.
    config (object): Configuration object containing the parameters for the stationarity test.
        - config.data_processor.test_stationarity.p_value_threshold (float): The p-value threshold for the ADF test.

    Returns:
    None
    """
    l.info("\n# Logging: stationarity")
    stationary_returns_processor = StationaryReturnsProcessorFactory.create_handler(
        "log_stationarity"
    )
    stationary_returns_processor.log_adf_results(
        data=df,
        p_value_threshold=config.data_processor.test_stationarity.p_value_threshold,
    )
