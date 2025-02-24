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
    """It's a bit strange to apply ARIMA on all df columns...but i'm going with it for lack of a better (simple) idea"""
    def __init__(self, data, order, steps):
        ascii_banner = """\n\n\t> ARIMA <\n"""
        l.info(ascii_banner)
        self.data = data
        self.order = order
        self.steps = steps
        self.models = {}  # store models for each column
        self.fits = {}  # store fits for each column

    def fit(self):
        for column in self.data.columns:
            model = ARIMA(self.data[column], order=self.order)
            self.fits[column] = model.fit()
        return self.fits
    
    def summary(self):
        """Return the model summaries for all columns."""
        summaries = {}
        for column, fit in self.fits.items():
            summaries[column] = fit.summary()
        return summaries

    def forecast(self):
        forecasts = {}
        for column, fit in self.fits.items():
            forecasts[column] = fit.forecast(steps=self.steps).iloc[0]
        return forecasts

def run_arima(df_stationary, config):
        l.info("\n## Running ARIMA")
        model_arima = ModelFactory.create_model(
            model_type="ARIMA", 
            data=df_stationary, 
            order=(
                config.stats_model.ARIMA.parameters_fit.get("p",),
                config.stats_model.ARIMA.parameters_fit.get("d"),
                config.stats_model.ARIMA.parameters_fit.get("q")
                ),
            steps=config.stats_model.ARIMA.parameters_predict_steps
            )
        arima_fit = model_arima.fit()
        l.info("\n## ARIMA summary")
        l.info(model_arima.summary())
        l.info("\n## ARIMA forecast")
        arima_forecast = model_arima.forecast()  # dont include steps arg here bc its already in object initialization
        l.info(f"arima_forecast: {arima_forecast}")

        return arima_fit, arima_forecast


class ModelGARCH:
    def __init__(self, data, p, q, dist):
        ascii_banner = """\n\n\t> GARCH <\n"""
        l.info(ascii_banner)
        self.data = data
        self.p = p
        self.q = q
        self.dist = dist
        self.models = {}  # store models for each column
        self.fits = {}  # store fits for each column
    
    def fit(self):
        for column in self.data.columns:
            model = arch_model(self.data[column], vol="Garch", p=self.p, q=self.q, dist=self.dist)
            self.fits[column] = model.fit(disp="off")
        return self.fits

    def summary(self):
        """Return the model summaries for all columns."""
        summaries = {}
        for column, fit in self.fits.items():
            summaries[column] = fit.summary()
        return summaries

    def forecast(self, steps):
        forecasts = {}
        for column, fit in self.fits.items():
            forecasts[column] = fit.forecast(horizon=steps).variance.iloc[-1]
        return forecasts

class ModelFactory:
    """Factory for creating model instances."""
    @staticmethod
    def create_model(model_type, **kwargs):
        l.info(f"Creating model type: {model_type}")
        if model_type.lower() == "arima":
            return ModelARIMA(**kwargs)
        elif model_type.lower() == "garch":
            return ModelGARCH(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

def run_garch(df_stationary, config):
        l.info("\n## Running GARCH")
        model_garch = ModelFactory.create_model(
            model_type="GARCH", 
            data=df_stationary,
            p=config.stats_model.GARCH.parameters_fit.p,
            q=config.stats_model.GARCH.parameters_fit.q,
            dist=config.stats_model.GARCH.parameters_fit.dist
            )
        garch_fit = model_garch.fit()
        l.info("\n## GARCH summary")
        l.info(model_garch.summary())
        l.info("\n## GARCH forecast")
        garch_forecast = model_garch.forecast(steps=config.stats_model.GARCH.parameters_predict_steps)
        l.info(f"garch_forecast: {garch_forecast}")

        return garch_fit, garch_forecast