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


model = ARIMA(df['NumberPassengers'], order=(2, 1, 2))
results = model.fit()
results.summary()
# 3 day forecast
pd.DataFrame(results.forecast(steps=3)[0]).plot(title="Passenger Travel Forecast")

# A second model with a different order
model2 = ARIMA(df['NumberPassengers'], order=(2, 1, 4))
res2 = model2.fit()
res2.summary()


# The arguments mean="Zero", vol="GARCH" specify the GARCH model
model = arch_model(returns, mean="Zero", vol="GARCH", p=1, q=1)
res = model.fit(disp="off")
res.summary()

# predict volatility
forecast_horizon = 5
forecasts = res.forecast(start='2019-12-08', horizon=forecast_horizon)

# plot volatility forecast
intermediate = np.sqrt(forecasts.variance * 252)
final = intermediate.dropna().T
final.plot()