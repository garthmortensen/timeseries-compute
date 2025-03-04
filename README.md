# Generalized Timeseries

A package for time series data processing and modeling using ARIMA and GARCH models.

## Features

- Price series generation for simulation.
- Data preprocessing including missing data handling and scaling.
- Stationarity testing and transformation.
- ARIMA and GARCH models for time series forecasting.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install generalized-timeseries
```
## Usage

```python
from generalized_timeseries import data_generator, data_processor, stats_model

# generate price series data
price_series = data_generator.generate_price_series(length=1000)

# preprocess the data
processed_data = data_processor.preprocess_data(price_series)

# fit ARIMA model
arima_model = stats_model.fit_arima(processed_data)

# fit GARCH model
garch_model = stats_model.fit_garch(processed_data)

# forecast using ARIMA model
arima_forecast = stats_model.forecast_arima(arima_model, steps=10)

# forecast using GARCH model
garch_forecast = stats_model.forecast_garch(garch_model, steps=10)

print("ARIMA Forecast:", arima_forecast)
print("GARCH Forecast:", garch_forecast)
```
