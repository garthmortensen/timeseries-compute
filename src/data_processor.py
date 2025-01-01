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
    def __init__(self,):  # this is a constructor
        ascii_banner = """\n\n\t> MissingDataHandler <\n"""
        l.info(ascii_banner)
        # dependency injection
        # instead of creating objects in class or function, you inject/pass as args
        # why? easier to test, easier to change, easier to understand

    def drop_na(self, data):
        """Drop rows with missing values."""
        l.info(f"Dropping rows with missing values")
        l.info("df filled:")
        l.info(tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))

        return data.dropna()

    def forward_fill(self, data):
        """Use the last known value to fill missing values."""
        l.info(f"Filling missing values with forward fill")
        l.info("df filled:")
        l.info(tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data.fillna(method="ffill")

class MissingDataHandlerFactory:
    """Factory for handling MissingDataHandler strategies."""
    # static method can be called without creating an instance of the class
    # e.g. MissingDataHandlerFactory.create_handler(strategy, config)
    @staticmethod
    def create_handler(strategy):
        """Return the appropriate handler function based on strategy."""
        handler = MissingDataHandler()
        l.info(f"Creating handler for strategy: {strategy}")
        # centralize logic to choose method
        if strategy == "drop":
            return handler.drop_na
        elif strategy == "forward_fill":
            return handler.forward_fill
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")

class StationaryReturnsProcessor:
    def transform_to_stationary_returns(self, data):
        """apply differencing to make data stationary.
        That is, diff to convert prices to zero-mean returns"""
        l.info("Differencing for stationarity")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            data[f"{column}_diff"] = data[column].diff().dropna()
            data = data.dropna()  # drop NaNs created by differencing
        l.info(tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data

    def check_stationarity(self, data):
        """Augmented Dickey-Fuller test for stationarity.
        Stationary time series have constant mean, variance, and autocorrelation. 
        Null H = series is non-stationary (has a unit root). Alt H = series is stationary.
        """
        l.info("Augmented Dickey-Fuller test for stationarity")
        results = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            # NaN and Infs will break ADF
            if data[column].isnull().any() or not np.isfinite(data[column]).all():
                l.warning(f"Column {column} contains NaN or Inf values. Skipping ADF test.")
                continue
            result = adfuller(data[column])
            results[column] = {"ADF Statistic": result[0], "p-value": result[1]}
        l.info(f"results: {results}")
        return results
    
    def log_adf_results(self, data, p_value_threshold):
        """logs interpreted ADF test results.
        # TODO: plot for visual inspection that series is stationary"""
        for series_name, result in data.items():
            adf_stat = result['ADF Statistic']
            p_value = result['p-value']
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
    """Factory for handling StationaryReturnsProcessor strategies."""
    @staticmethod
    def create_handler(strategy):
        """Return the appropriate processing function based on strategy."""
        processor = StationaryReturnsProcessor()
        l.info(f"Creating processor for strategy: {strategy}")
        if strategy == "transform_to_stationary_returns":
            return processor.transform_to_stationary_returns
        elif strategy == "check_stationarity":
            return processor.check_stationarity
        else:
            raise ValueError(f"Unknown stationary returns processing strategy: {strategy}")

class DataScaler:
    """TODO: scale_data_percent, scale_data_log"""
    def scale_data_standardize(self, data):
        """Standardize all numeric columns except the index."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            data[column] = (data[column] - data[column].mean()) / data[column].std()
        l.info(f"Scaling data using standardization")
        l.info("df scaled:")
        l.info(tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data

    def scale_data_minmax(self, data):
        """Scale all numeric columns using MinMaxScaler."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            data[column] = (data[column].min()) / (
                data[column].max() - data[column].min()
            )
        l.info(f"Scaling data using minmax")
        l.info("df scaled:")
        l.info(tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data


class DataScalerFactory:
    """Factory for handling DataScaler strategies."""
    @staticmethod
    def create_handler(strategy):
        """Return the appropriate scaling function based on strategy."""
        scaler = DataScaler()
        l.info(f"creating scaler for strategy: {strategy}")
        if strategy == "standardize":
            return scaler.scale_data_standardize
        elif strategy == "minmax":
            return scaler.scale_data_minmax
        else:
            raise ValueError(f"Unknown data scaling strategy: {strategy}")


