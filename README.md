# Generalized Timeseries

![CI/CD](https://github.com/garthmortensen/garch/actions/workflows/execute_pytest.yml/badge.svg)

![Read the Docs](https://img.shields.io/readthedocs/generalized-timeseries)

![PyPI](https://img.shields.io/pypi/v/generalized-timeseries?color=blue&label=PyPI)

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

## External resources

[pypi repo](https://pypi.org/project/generalized-timeseries/)

[readthedocs.io](https://generalized-timeseries.readthedocs.io/en/latest/)

## Publishing Maintenance

### pypi

```shell
pip install --upgrade build
pip install --upgrade twine
python -m build  # build the package
twine check dist/  # check it works
twine upload dist/

rm -rf dist build *.egg-info # restart if needed
```

## Publishing via Github Actions

Pypi publication occurs when pushing a tag:

```shell
git tag v0.1.7
git push origin v0.1.7
```
