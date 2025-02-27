#!/usr/bin/env python3
# stats_model.py

# handle relative directory imports for chronicler
import logging as l

# handle data transformation and preparation tasks
import pandas as pd
import numpy as np
from tabulate import tabulate  # pretty print dfs

# import model specific libraries
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


class ModelARIMA:
    """
    A class to apply ARIMA (AutoRegressive Integrated Moving Average) model on all columns of a DataFrame.

    NB: It's a bit strange to apply ARIMA on all df columns...but i'm going with it for lack of a better (simple) idea.

    Attributes:
    -----------
    data : pandas.DataFrame
        The input data on which ARIMA models will be applied.
    order : tuple
        The (p, d, q) order of the ARIMA model.
    steps : int
        The number of steps to forecast.
    models : dict
        A dictionary to store ARIMA models for each column.
    fits : dict
        A dictionary to store fitted ARIMA models for each column.

    Methods:
    --------
    fit():
        Fits ARIMA models to each column of the DataFrame.
    summary():
        Returns the summary of the fitted ARIMA models for all columns.
    forecast():
        Forecasts future values for each column based on the fitted ARIMA models.
    """

    def __init__(self, data, order, steps):
        """
        Initialize the ARIMA model with the given data, order, and steps.

        Parameters:
        data (pd.DataFrame): The input data for the ARIMA model.
        order (tuple): The (p, d, q) order of the ARIMA model.
        steps (int): The number of steps to forecast.

        Attributes:
        data (pd.DataFrame): The input data for the ARIMA model.
        order (tuple): The (p, d, q) order of the ARIMA model.
        steps (int): The number of steps to forecast.
        models (dict): A dictionary to store models for each column.
        fits (dict): A dictionary to store fits for each column.
        """
        ascii_banner = """\n\n\t> ARIMA <\n"""
        l.info(ascii_banner)
        self.data = data
        self.order = order
        self.steps = steps
        self.models = {}  # store models for each column
        self.fits = {}  # store fits for each column

    def fit(self):
        """
        Fits an ARIMA model to each column in the dataset.

        This method iterates over each column in the dataset, fits an ARIMA model
        with the specified order, and stores the fitted model in the `fits` dictionary.

        Returns:
            dict: A dictionary where the keys are column names and the values are the
                  fitted ARIMA models for each column.
        """
        for column in self.data.columns:
            model = ARIMA(self.data[column], order=self.order)
            self.fits[column] = model.fit()
        return self.fits

    def summary(self):
        """
        Return the model summaries for all columns.

        This method iterates over the fitted models stored in the `fits` attribute
        and generates a summary for each model. The summaries are stored in a 
        dictionary where the keys are the column names and the values are the 
        corresponding model summaries.

        Returns:
            dict: A dictionary containing the model summaries for each column.
        """
        summaries = {}
        for column, fit in self.fits.items():
            summaries[column] = fit.summary()
        return summaries

    def forecast(self):
        """
        Generate forecasts for each fitted model.

        This method iterates over the fitted models stored in the `fits` attribute
        and generates forecasts for each model. The number of steps to forecast
        is determined by the `steps` attribute.

        Returns:
            dict: A dictionary where the keys are the column names and the values
                  are the forecasted values for the first step.
        """
        forecasts = {}
        for column, fit in self.fits.items():
            forecasts[column] = fit.forecast(steps=self.steps).iloc[0]
        return forecasts


def run_arima(df_stationary, config):
    """
    Runs the ARIMA model on the provided stationary dataframe using the given configuration.

    Args:
        df_stationary (pd.DataFrame): The stationary dataframe to be used for ARIMA modeling.
        config (Config): Configuration object containing ARIMA parameters.

    Returns:
        tuple: A tuple containing the fitted ARIMA model and the forecasted values.

    Logs:
        Logs the ARIMA model summary and forecasted values.
    """
    l.info("\n## Running ARIMA")
    model_arima = ModelFactory.create_model(
        model_type="ARIMA",
        data=df_stationary,
        order=(
            config.stats_model.ARIMA.parameters_fit.get(
                "p",
            ),
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
    )  # dont include steps arg here bc its already in object initialization
    l.info(f"arima_forecast: {arima_forecast}")

    return arima_fit, arima_forecast


class ModelGARCH:
    """
    A class used to represent a GARCH model for time series data.

    Attributes
    ----------
    data : pandas.DataFrame
        The input time series data.
    p : int
        The order of the GARCH model for the lag of the squared residuals.
    q : int
        The order of the GARCH model for the lag of the conditional variance.
    dist : str
        The distribution to use for the GARCH model (e.g., 'normal', 't').

    Methods
    -------
    fit():
        Fits the GARCH model to each column of the input data.
    summary():
        Returns the model summaries for all columns.
    forecast(steps):
        Forecasts the variance for a given number of steps ahead for each column.
    """
    def __init__(self, data, p, q, dist):
        """
        Initialize the GARCH model with the given parameters.

        Parameters:
        data (pd.DataFrame or np.ndarray): The input data for the GARCH model.
        p (int): The order of the GARCH model.
        q (int): The order of the ARCH model.
        dist (str): The distribution to be used in the model (e.g., 'normal', 't').

        Attributes:
        data (pd.DataFrame or np.ndarray): The input data for the GARCH model.
        p (int): The order of the GARCH model.
        q (int): The order of the ARCH model.
        dist (str): The distribution to be used in the model.
        models (dict): A dictionary to store models for each column of the data.
        fits (dict): A dictionary to store fits for each column of the data.
        """
        ascii_banner = """\n\n\t> GARCH <\n"""
        l.info(ascii_banner)
        self.data = data
        self.p = p
        self.q = q
        self.dist = dist
        self.models = {}  # store models for each column
        self.fits = {}  # store fits for each column

    def fit(self):
        """
        Fits a GARCH model to each column of the data.

        This method iterates over each column in the data and fits a GARCH model
        using the specified parameters for p, q, and dist. The fitted models are
        stored in the `fits` attribute.

        Returns:
            dict: A dictionary where the keys are the column names and the values
                  are the fitted GARCH models.
        """
        for column in self.data.columns:
            model = arch_model(
                self.data[column], vol="Garch", p=self.p, q=self.q, dist=self.dist
            )
            self.fits[column] = model.fit(disp="off")
        return self.fits

    def summary(self):
        """
        Return the model summaries for all columns.

        This method iterates over the fitted models stored in the `fits` attribute
        and generates a summary for each model. The summaries are returned as a 
        dictionary where the keys are the column names and the values are the 
        corresponding model summaries.

        Returns:
            dict: A dictionary containing the model summaries for each column.
        """
        summaries = {}
        for column, fit in self.fits.items():
            summaries[column] = fit.summary()
        return summaries

    def forecast(self, steps: int) -> dict:
        """Generate forecasted variance for each fitted model.

        Args:
            steps (int): The number of steps ahead to forecast.

        Returns:
            dict: A dictionary where keys are column names and values are the forecasted variances for the specified horizon.
        """
        forecasts = {}
        for column, fit in self.fits.items():
            forecasts[column] = fit.forecast(horizon=steps).variance.iloc[-1]
        return forecasts


class ModelFactory:
    """
    ModelFactory is a factory class for creating instances of different statistical models.

    Methods
    -------
    create_model(model_type, **kwargs)
        Static method that creates and returns an instance of a model based on the provided model_type.
        Supported model types are 'arima' and 'garch'.

        Parameters
        ----------
        model_type : str
            The type of model to create. Must be either 'arima' or 'garch'.
        **kwargs : dict
            Additional keyword arguments to pass to the model's constructor.

        Returns
        -------
        object
            An instance of the specified model type.

        Raises
        ------
        ValueError
            If the provided model_type is not supported.
    """

    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Creates and returns a statistical model based on the specified type.

        Parameters:
        model_type (str): The type of model to create. Supported values are "arima" and "garch".
        **kwargs: Additional keyword arguments to pass to the model constructor.

        Returns:
        object: An instance of the specified model type.

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


def run_garch(df_stationary, config):
    """
    Runs the GARCH model on the provided stationary dataframe and configuration.

    Args:
        df_stationary (pd.DataFrame): The stationary time series data to fit the GARCH model on.
        config (Config): Configuration object containing parameters for fitting and forecasting with the GARCH model.

    Returns:
        tuple: A tuple containing the fitted GARCH model and the forecasted values.
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
