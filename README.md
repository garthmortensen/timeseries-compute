# Generalized Timeseries

![Python Versions](https://img.shields.io/pypi/pyversions/generalized-timeseries) 
[![PyPI](https://img.shields.io/pypi/v/generalized-timeseries?color=blue&label=PyPI)](https://pypi.org/project/generalized-timeseries/)

![CI/CD](https://github.com/garthmortensen/garch/actions/workflows/execute_CICD.yml/badge.svg) 
[![readthedocs.io](https://img.shields.io/readthedocs/generalized-timeseries)](https://generalized-timeseries.readthedocs.io/en/latest/)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-generalized--timeseries-blue)](https://hub.docker.com/r/goattheprofessionalmeower/generalized-timeseries)

[![codecov](https://codecov.io/gh/garthmortensen/generalized-timeseries/graph/badge.svg?token=L1L5OBSF3Z)](https://codecov.io/gh/garthmortensen/generalized-timeseries)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/a55633cfb8324f379b0b5ec16f03c268)](https://app.codacy.com/gh/garthmortensen/generalized-timeseries/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

A package for timeseries data processing and modeling using ARIMA and GARCH models.

## Features

- Price series generation for simulation.
- Data preprocessing including missing data handling and scaling.
- Stationarity testing and transformation.
- ARIMA and GARCH models for time series forecasting.

## Installation

Install from pypi:

```bash
python -m venv venv
source venv/bin/activate
pip install generalized-timeseries
```

Install from github:

```bash
python -m venv venv
source venv/bin/activate
pip install git+https://github.com/garthmortensen/generalized-timeseries.git
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

Alternatively,

```bash
python generalized_timeseries/examples/example.py
```

## Publishing Maintenance

Reminder on how to manually push to pypi. This step, along with autodoc build, is automated with CI/CD.


### Steps

1. bump `version` in `pyproject.toml` to `v0.1.11`
2. git add, commit:

    ```bash
    git add pyproject.toml
    git commit -m "Ver: bump"
    git tag v0.1.11
    git push && git push --tags
    ```

### pypi

```shell
pip install --upgrade build
pip install --upgrade twine
python -m build  # build the package
twine check dist/  # check it works
twine upload dist/

rm -rf dist build *.egg-info # restart if needed
```

## Design Decisions

### Package project

This package was created in order to carve out code from a larger thesis remake project. Doing so increased modularity. `pyproject.toml` is used instead of `setup.py` to achieve a modern build process for publishing to pypi. It also supports direct pip installation via github clone.

pypi project: https://pypi.org/project/generalized-timeseries/

### Self documenting code

To achieve self-documenting code, docstrings are used. 

Due to there being a fair amount of modules, classes and functions, type hints are used. Variable annotations (`myvar: int = 5`) are just too visually distracting, and are not used.

Sphinx converts all this metadata into `.html` and hosted on readthedocs.io.

readthedocs: https://generalized-timeseries.readthedocs.io/en/latest/

### Test covereage

Unit tests cover the majority of the codebase, to ensure code changes don't break existing functionality. Codecov.io is used to analyze code coverage.

code coverage report: https://app.codecov.io/gh/garthmortensen/generalized-timeseries

### Code quality

Codacy is used to analyze code quality, and highly any insecure programming issues.

code quality report: https://app.codacy.com/gh/garthmortensen/generalized-timeseries/dashboard

### Branching

Branches were used early on in the project for the sake of purity, but eventually pragmatism took over.

git branches: https://github.com/garthmortensen/generalized-timeseries/branches

### `venv`

Virtual environments and [`requirements.txt`](https://github.com/garthmortensen/generalized-timeseries/blob/dev/requirements.txt) enable sandboxed development.

### OS support

The code is OS-agnostic, and [tested across ubuntu and mac-os](https://github.com/garthmortensen/generalized-timeseries/blob/dev/.github/workflows/execute_CICD.yml#L21). Due to the slow spin-up times of testing on a Windows build, it's been excluded from testing since v0.1.7.

### CI/CD

A Github Actions workflow automates the labor involved with `pytest`, package build (wheels included), pypi distribution, producing `sphinx` autodocs, publishing docs to readthedocs.io, and publishing to docker hub.

CD/CD: https://github.com/garthmortensen/generalized-timeseries/tree/dev/.github/workflows

### Docker

In addition to `venv`, docker encapsulates the app and dependencies. This helps run it across different environments.

https://hub.docker.com/r/goattheprofessionalmeower/generalized-timeseries

```bash
docker build -t generalized-timeseries:latest ./
docker run -it generalized-timeseries:latest /app/generalized_timeseries/examples/example.py  # run example pipeline
docker run -it --entrypoint /bin/bash generalized-timeseries:latest  # interactive shell in container
```

### Makefile

Inspired by [deepseek repos makefiles](https://github.com/deepseek-ai/DeepSeek-LLM/blob/main/Makefile), I want to add a makefile.
