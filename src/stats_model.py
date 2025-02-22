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

# import garch
from arch.univariate import GARCH

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


class ModelGARCH:
    def __init__(self, data, order, dist):
        ascii_banner = """\n\n\t> GARCH <\n"""
        l.info(ascii_banner)
        self.data = data
        self.order = order
        self.dist = dist
    
    def fit(self):
        model = arch_model(self.data, vol="GARCH", p=self.order["p"], q=self.order["q"], dist=self.dist)
        fit = model.fit()
        return fit


class ModelFactory:
    """Factory for creating model instances."""
    @staticmethod
    def create_model(model_type, **kwargs):
        l.info(f"Creating model type: {model_type}")
        if model_type.lower() == "arima":
            return ModelARIMA(**kwargs)
        elif model_type.lower() == "garch":
            return GARCH(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

