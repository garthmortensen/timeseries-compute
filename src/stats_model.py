#!/usr/bin/env python3

# stats_model.py

# handle relative directory imports for chronicler
import logging as l

# handle data transformation and preparation tasks
import pandas as pd
import numpy as np
from tabulate import tabulate  # pretty print dfs

# import model specific libraries
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model



class ARIMA:
    def __init__(self, data, order):
        ascii_banner = """\n\n\t> ARIMA <\n"""
        l.info(ascii_banner)
        self.data = data
        self.order = order

    def fit(self,):
        model = ARIMA(self.data, order=self.order)
        return model.fit()
    
    def summary(self,):
        """Return the model summary."""
        return self.fit().summary()

    def forecast(self, steps):
        return self.fit().forecast(steps=steps)[0]




# The arguments mean="Zero", vol="GARCH" specify the GARCH model
# model = arch_model(returns, mean="Zero", vol="GARCH", p=1, q=1)
# res = model.fit(disp="off")
# res.summary()

# # predict volatility
# forecast_horizon = 5
# forecasts = res.forecast(start='2019-12-08', horizon=forecast_horizon)

# # plot volatility forecast
# intermediate = np.sqrt(forecasts.variance * 252)
# final = intermediate.dropna().T
# final.plot()