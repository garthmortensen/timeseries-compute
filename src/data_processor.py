# handle relative directory imports for chronicler
import logging as l
from chronicler_loader import init_chronicler

chronicler = init_chronicler()

# handle data transformation and preparation tasks
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


class MissingDataHandler:
    def __init__(self, config):
        ascii_banner = """\n\n\t> MissingDataHandler <\n"""
        l.info(ascii_banner)
        self.config = config

    def drop_na(self, data):
        """Drop rows with missing values."""
        return data.dropna()

    def forward_fill(self, data):
        """Use the last known value to fill missing values."""
        return data.fillna(method="ffill")

    # def fill_missing_values_with_mean(self, data):
    # pass

    # def fill_missing_values_with_mode(self, data):
    #     """Fill missing values with mode for all columns except the index."""
    # pass

    # def interpolation(self, data):
    #     """Fill missing values using interpolation for numeric columns."""
    # pass


class StationaryReturnsProcessor:
    def transform_to_stationary_returns(self, data, column):
        """Apply differencing to make data stationary.
        That is, diff to convert prices to zero-mean returns"""
        data[f"{column}_diff"] = data[column].diff().dropna()
        return data

    def check_stationarity(self, data, column):
        """Perform stationarity tests."""
        result = adfuller(data[column])
        return {"ADF Statistic": result[0], "p-value": result[1]}


class DataScaler:
    def scale_data_standardize(self, data):
        """Standardize all numeric columns except the index."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            data[column] = (data[column] - data[column].mean()) / data[column].std()
        return data

    def scale_data_minmax(self, data):
        """Scale all numeric columns using MinMaxScaler."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            data[column] = (data[column] - data[column].min()) / (
                data[column].max() - data[column].min()
            )
        return data

    # def scale_data_percent(self, data):
    #     """Scale all numeric columns using percentage change."""
    # pass

    # def scale_data_log(self, data):
    #     """Scale data using log transformation for numeric columns."""
    # pass
