#!/usr/bin/env python3
# stats_model.py

import logging as l

# handle data transformation and preparation tasks
import pandas as pd

# import model specific libraries
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# type hinting
from typing import Dict, Any, Tuple

class ModelARIMA:
    """
    Applies the ARIMA (AutoRegressive Integrated Moving Average) model on all columns of a DataFrame.

    Attributes:
        data (pd.DataFrame): The input data on which ARIMA models will be applied.
        order (Tuple[int, int, int]): The (p, d, q) order of the ARIMA model.
        steps (int): The number of steps to forecast.
        models (Dict[str, ARIMA]): A dictionary to store ARIMA models for each column.
        fits (Dict[str, ARIMA]): A dictionary to store fitted ARIMA models for each column.
    """

    def __init__(
        self, data: pd.DataFrame, order: Tuple[int, int, int], steps: int
    ) -> None:
        """
        Initializes the ARIMA model with the given data, order, and steps.

        Args:
            data (pd.DataFrame): The input data for the ARIMA model.
            order (Tuple[int, int, int]): The (p, d, q) order of the ARIMA model.
            steps (int): The number of steps to forecast.
        """
        ascii_banner = """
        \n
        \t> ARIMA <\n"""
        l.info(ascii_banner)
        self.data = data
        self.order = order
        self.steps = steps
        self.models: Dict[str, ARIMA] = {}  # Store models for each column
        self.fits: Dict[str, ARIMA] = {}  # Store fits for each column

    def fit(self) -> Dict[str, ARIMA]:
        """
        Fits an ARIMA model to each column in the dataset.

        Returns:
            Dict[str, ARIMA]: A dictionary where the keys are column names and the values are the
                fitted ARIMA models for each column.
        """
        for column in self.data.columns:
            model = ARIMA(self.data[column], order=self.order)
            self.fits[column] = model.fit()
        return self.fits

    def summary(self) -> Dict[str, str]:
        """
        Returns the model summaries for all columns.

        Returns:
            Dict[str, str]: A dictionary containing the model summaries for each column.
        """
        summaries = {}
        for column, fit in self.fits.items():
            summaries[column] = str(fit.summary())
        return summaries

    def forecast(self) -> Dict[str, float]:
        """
        Generates forecasts for each fitted model.

        Returns:
            Dict[str, float]: A dictionary where the keys are the column names and the values
                are the forecasted values for the first step.
        """
        forecasts = {}
        for column, fit in self.fits.items():
            forecasts[column] = fit.forecast(steps=self.steps).iloc[0]
        return forecasts


def run_arima(
    df_stationary: pd.DataFrame, config
) -> Tuple[Dict[str, object], Dict[str, float]]:
    """
    Runs the ARIMA model on the provided stationary DataFrame using the given configuration.

    Args:
        df_stationary (pd.DataFrame): The stationary DataFrame to be used for ARIMA modeling.
        config: Configuration object containing ARIMA parameters.

    Returns:
        Tuple[Dict[str, object], Dict[str, float]]: A tuple containing the fitted ARIMA model and the forecasted values.

    Logs:
        Logs the ARIMA model summary and forecasted values.
    """
    l.info("\n## Running ARIMA")
    model_arima = ModelFactory.create_model(
        model_type="ARIMA",
        data=df_stationary,
        order=(
            config.stats_model.ARIMA.parameters_fit.get("p"),
            config.stats_model.ARIMA.parameters_fit.get("d"),
            config.stats_model.ARIMA.parameters_fit.get("q"),
        ),
        steps=config.stats_model.ARIMA.parameters_predict_steps,
    )
    arima_fit = model_arima.fit()
    l.info("\n## ARIMA summary")
    l.info(model_arima.summary())
    l.info("\n## ARIMA forecast")
    arima_forecast = (
        model_arima.forecast()
    )  # Steps arg is already in object initialization
    l.info(f"arima_forecast: {arima_forecast}")

    return arima_fit, arima_forecast


class ModelGARCH:
    """
    Represents a GARCH model for time series data.

    Attributes:
        data (pd.DataFrame): The input time series data.
        p (int): The order of the GARCH model for the lag of the squared residuals.
        q (int): The order of the GARCH model for the lag of the conditional variance.
        dist (str): The distribution to use for the GARCH model (e.g., 'normal', 't').
        models (Dict[str, arch_model]): A dictionary to store models for each column of the data.
        fits (Dict[str, arch_model]): A dictionary to store fitted models for each column of the data.
    """

    def __init__(self, data: pd.DataFrame, p: int, q: int, dist: str) -> None:
        """
        Initializes the GARCH model with the given parameters.

        Args:
            data (pd.DataFrame): The input data for the GARCH model.
            p (int): The order of the GARCH model.
            q (int): The order of the ARCH model.
            dist (str): The distribution to be used in the model (e.g., 'normal', 't').
        """
        ascii_banner = """
        \n\t> GARCH <\n"""
        l.info(ascii_banner)
        self.data = data
        self.p = p
        self.q = q
        self.dist = dist
        self.models: Dict[str, arch_model] = {}  # Store models for each column
        self.fits: Dict[str, arch_model] = {}  # Store fits for each column

    def fit(self) -> Dict[str, arch_model]:
        """
        Fits a GARCH model to each column of the data.

        Returns:
            Dict[str, arch_model]: A dictionary where the keys are column names and the values
                are the fitted GARCH models.
        """
        for column in self.data.columns:
            model = arch_model(
                self.data[column], vol="Garch", p=self.p, q=self.q, dist=self.dist
            )
            self.fits[column] = model.fit(disp="off")
        return self.fits

    def summary(self) -> Dict[str, str]:
        """
        Returns the model summaries for all columns.

        Returns:
            Dict[str, str]: A dictionary containing the model summaries for each column.
        """
        summaries = {}
        for column, fit in self.fits.items():
            summaries[column] = str(fit.summary())
        return summaries

    def forecast(self, steps: int) -> Dict[str, float]:
        """
        Generates forecasted variance for each fitted model.

        Args:
            steps (int): The number of steps ahead to forecast.

        Returns:
            Dict[str, float]: A dictionary where keys are column names and values are the forecasted variances for the specified horizon.
        """
        forecasts = {}
        for column, fit in self.fits.items():
            forecasts[column] = fit.forecast(horizon=steps).variance.iloc[-1]
        return forecasts


class ModelFactory:
    """
    Factory class for creating instances of different statistical models.

    Methods:
        create_model(model_type: str, **kwargs) -> Any:
            Static method that creates and returns an instance of a model based on the provided model_type.
    """

    @staticmethod
    def create_model(model_type: str, **kwargs) -> Any:
        """
        Creates and returns a statistical model based on the specified type.

        Args:
            model_type (str): The type of model to create. Supported values are "arima" and "garch".
            **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
            Any: An instance of the specified model type.

        Raises:
            ValueError: If the specified model type is not supported.
        """
        l.info(f"Creating model type: {model_type}")
        if model_type.lower() == "arima":
            return ModelARIMA(**kwargs)
        elif model_type.lower() == "garch":
            return ModelGARCH(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def run_garch(
    df_stationary: pd.DataFrame, config
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Runs the GARCH model on the provided stationary DataFrame using the given configuration.

    Args:
        df_stationary (pd.DataFrame): The stationary time series data to fit the GARCH model on.
        config: Configuration object containing parameters for fitting and forecasting with the GARCH model.

    Returns:
        Tuple[Dict[str, Any], Dict[str, float]]: A tuple containing the fitted GARCH model and the forecasted values.
    """
    l.info("\n## Running GARCH")
    model_garch = ModelFactory.create_model(
        model_type="GARCH",
        data=df_stationary,
        p=config.stats_model.GARCH.parameters_fit.p,
        q=config.stats_model.GARCH.parameters_fit.q,
        dist=config.stats_model.GARCH.parameters_fit.dist,
    )
    garch_fit = model_garch.fit()
    l.info("\n## GARCH summary")
    l.info(model_garch.summary())
    l.info("\n## GARCH forecast")
    garch_forecast = model_garch.forecast(
        steps=config.stats_model.GARCH.parameters_predict_steps
    )
    l.info(f"garch_forecast: {garch_forecast}")

    return garch_fit, garch_forecast
