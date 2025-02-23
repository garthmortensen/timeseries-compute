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
        # why? easier to test, easier to change, easier to understand (someday!)

    def drop_na(self, data):
        """Drop rows with missing values."""
        l.info(f"Dropping rows with missing values")
        l.info("df filled:")
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))

        return data.dropna()

    def forward_fill(self, data):
        """Use the last known value to fill missing values."""
        l.info(f"Filling missing values with forward fill")
        l.info("df filled:")
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
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
        if strategy.lower() == "drop":
            return handler.drop_na
        elif strategy == "forward_fill":
            return handler.forward_fill
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")

def fill_data(df, config):
    l.info("\n# Processing: handling missing values")
    handler_missing = MissingDataHandlerFactory.create_handler(
        strategy=config.data_processor.handle_missing_values.strategy
    )
    df_filled = handler_missing(df)
    return df_filled

class DataScaler:
    """TODO: scale_data_percent, scale_data_log"""
    def scale_data_standardize(self, data):
        """Standardize all numeric columns except the index."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            data[column] = (data[column] - data[column].mean()) / data[column].std()
        l.info(f"Scaling data using standardization")
        l.info("df scaled:")
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
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
        l.info("\n" + tabulate(data.head(5), headers="keys", tablefmt="fancy_grid"))
        return data

class DataScalerFactory:
    """Factory for handling DataScaler strategies."""
    @staticmethod
    def create_handler(strategy):
        """Return the appropriate scaling function based on strategy."""
        scaler = DataScaler()
        l.info(f"creating scaler for strategy: {strategy}")
        if strategy.lower() == "standardize":
            return scaler.scale_data_standardize
        elif strategy == "minmax":  # TODO: fixme. This turns everything into a constant
            return scaler.scale_data_minmax
        else:
            raise ValueError(f"Unknown data scaling strategy: {strategy}")


def scale_data(df, config):
    l.info("\n# Processing: scaling data")
    handler_scaler = DataScalerFactory.create_handler(
        strategy=config.data_processor.scaling.method
        )
    df_scaled = handler_scaler(df)
    return df_scaled

class StationaryReturnsProcessor:
    def make_stationary(self, data, method):
        """Apply the chosen method to make the data stationary."""
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
        """Augmented Dickey-Fuller test for stationarity.
        Stationary time series have constant mean, variance, and autocorrelation. 
        Null H = series is non-stationary (has a unit root). Alt H = series is stationary.
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
                    l.warning(f"Column {column} contains NaN or Inf values. Skipping ADF test.")
                    continue
                result = adfuller(data[column])
                results[column] = {"ADF Statistic": result[0], "p-value": result[1]}
            l.info(f"results: {results}")
        return results
    
    def log_adf_results(self, data, p_value_threshold):
        """logs interpreted ADF test results.
        # TODO: plot for visual inspection that series is stationary
        # TODO: rename from log, bc i think logorithm when i see log here. interpert_adf_results?"""
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
    """Factory for handling StationaryReturnsProcessor strategies.
    Factories typically return objects, but here we return functions.
    This is because the processing functions are stateless and don't need to be instantiated.
    `staticmethod` is used to define a method that doesn't operate on an instance.
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
            raise ValueError(f"Unknown stationary returns processing strategy: {strategy}")

def stationarize_data(df, config):
    l.info("\n# Processing: making data stationary")
    # recreating the object each time is not efficient, but it's simple
    stationary_returns_processor = StationaryReturnsProcessor()
    df_stationary = stationary_returns_processor.make_stationary(
        data=df,
        method=config.data_processor.make_stationary.method
        )
    return df_stationary

def test_stationarity(df, config):
    l.info("\n# Testing: stationarity")
    stationary_returns_processor = StationaryReturnsProcessorFactory.create_handler("test_stationarity")
    adf_results = stationary_returns_processor.test_stationarity(
        data=df,
        test=config.data_processor.test_stationarity.method
        )

    return adf_results

def log_stationarity(df, config):
    l.info("\n# Logging: stationarity")
    stationary_returns_processor = StationaryReturnsProcessorFactory.create_handler("log_stationarity")
    stationary_returns_processor.log_adf_results(
        data=df,
        p_value_threshold=config.data_processor.test_stationarity.p_value_threshold
        )

