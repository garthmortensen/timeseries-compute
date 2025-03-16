# Generalized Timeseries

[![Python Versions](https://img.shields.io/pypi/pyversions/generalized-timeseries)]((https://pypi.org/project/generalized-timeseries/))
[![PyPI](https://img.shields.io/pypi/v/generalized-timeseries?color=blue&label=PyPI)](https://pypi.org/project/generalized-timeseries/)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-generalized--timeseries-blue)](https://hub.docker.com/r/goattheprofessionalmeower/generalized-timeseries)

![CI/CD](https://github.com/garthmortensen/garch/actions/workflows/execute_CICD.yml/badge.svg) 
[![codecov](https://codecov.io/gh/garthmortensen/generalized-timeseries/graph/badge.svg?token=L1L5OBSF3Z)](https://codecov.io/gh/garthmortensen/generalized-timeseries)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/a55633cfb8324f379b0b5ec16f03c268)](https://app.codacy.com/gh/garthmortensen/generalized-timeseries/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

[![readthedocs.io](https://img.shields.io/readthedocs/generalized-timeseries)](https://generalized-timeseries.readthedocs.io/en/latest/)

```ascii
 ▗▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖▗▄▄▖  ▗▄▖ ▗▖   ▗▄▄▄▖▗▄▄▄▄▖▗▄▄▄▖▗▄▄▄ 
▐▌   ▐▌   ▐▛▚▖▐▌▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌     █     ▗▞▘▐▌   ▐▌  █
▐▌▝▜▌▐▛▀▀▘▐▌ ▝▜▌▐▛▀▀▘▐▛▀▚▖▐▛▀▜▌▐▌     █   ▗▞▘  ▐▛▀▀▘▐▌  █
▝▚▄▞▘▐▙▄▄▖▐▌  ▐▌▐▙▄▄▖▐▌ ▐▌▐▌ ▐▌▐▙▄▄▖▗▄█▄▖▐▙▄▄▄▖▐▙▄▄▖▐▙▄▄▀
     ▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖ ▗▄▄▖▗▄▄▄▖▗▄▄▖ ▗▄▄▄▖▗▄▄▄▖ ▗▄▄▖
       █    █  ▐▛▚▞▜▌▐▌   ▐▌   ▐▌   ▐▌ ▐▌  █  ▐▌   ▐▌
       █    █  ▐▌  ▐▌▐▛▀▀▘ ▝▀▚▖▐▛▀▀▘▐▛▀▚▖  █  ▐▛▀▀▘ ▝▀▚▖
       █  g▄█▄m▐▌  ▐▌▐▙▄▄▖▗▄▄▞▘▐▙▄▄▖▐▌ ▐▌▗▄█▄▖▐▙▄▄▖▗▄▄▞▘
```

A Python package for timeseries data processing and modeling using ARIMA and GARCH models.

## Architecture and Design Decisions

This package follows best development practices:

- **Modular Design**: Extracted from a larger thesis replication project to increase maintainability and reusability, [published on pypi as package](https://pypi.org/project/generalized-timeseries/)
- **Modern Build System**: Uses `pyproject.toml` instead of legacy `setup.py` for more robust dependency management and packaging. `requirements.txt` allows users to easily construct `venv` virtual environments.
- **Self-Documenting Code**: Extensive docstrings and type hints allow automated documentation generation via Sphinx and publication on [readthedocs.io](https://generalized-timeseries.readthedocs.io/en/latest/). Variable annotations (`myvar: int = 5`) are just too visually distracting, and are not used
- **Test Coverage**: Comprehensive unit tests with pytest with goal of 70%+ code coverage, as indicated by [Codecov dashboard](https://app.codecov.io/gh/garthmortensen/generalized-timeseries)
- **CI/CD Pipeline**: Automated testing, documentation, package publishing and Docker image building
- **Cross-Platform**: OS-agnostic design with automated testing across [OS environments](https://github.com/garthmortensen/generalized-timeseries/blob/dev/.github/workflows/execute_CICD.yml#L21)
- **Multi-Version Support**: Compatible with Python 3.11+ with automated testing across [language versions](https://github.com/garthmortensen/generalized-timeseries/blob/dev/.github/workflows/execute_CICD.yml#L20)
- **Containerization**: Docker support with optimized multi-stage builds for reproducible environments
- **Code Quality**: OOP, DRY, secure programming practices are followed, resulting in [high code quality score](https://app.codacy.com/gh/garthmortensen/generalized-timeseries/dashboard)
- **Minimal reinvention**: Relies on well-maintained libraries instead of DIY solutions

## Features

- Price series generation
- Data preprocessing with configurable missing data handling and scaling options
- Stationarity testing and transformation for time series analysis
- ARIMA modeling for time series forecasting
- GARCH modeling for volatility forecasting and risk assessment

## Installation

Install from PyPI (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install generalized-timeseries
```

Install from GitHub (latest development version):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install git+https://github.com/garthmortensen/generalized-timeseries.git
```

## Quick Start

```python
from generalized_timeseries import data_generator, data_processor, stats_model

# Generate synthetic price series
price_dict, price_df = data_generator.generate_price_series(length=1000)

# Preprocess the data (handle missing values, stationarize)
processed_data = data_processor.preprocess_data(price_df)

# Fit ARIMA model
arima_model = stats_model.fit_arima(processed_data)

# Fit GARCH model for volatility forecasting
garch_model = stats_model.fit_garch(processed_data)

# Generate forecasts
arima_forecast = stats_model.forecast_arima(arima_model, steps=10)
garch_forecast = stats_model.forecast_garch(garch_model, steps=10)

print("ARIMA Forecast:", arima_forecast)
print("GARCH Forecast:", garch_forecast)
```

For a near-complete working example:

```bash
python -m generalized_timeseries.examples.example
```

## Docker Support

Run with Docker for isolated environments:

```bash
# build the image
docker build -t generalized-timeseries:latest ./

# R\run the example pipeline
docker run -it generalized-timeseries:latest /app/generalized_timeseries/examples/example.py

# get into interactive shell
docker run -it --entrypoint /bin/bash generalized-timeseries:latest
```

## Development

### Environment Setup

Option 1 (recommended):

```bash
mkdir generalized-timeseries
cd generalized-timeseries

# create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install generalized-timeseries
```

Option 2

```bash
# clone the repository
git clone https://github.com/garthmortensen/generalized-timeseries.git
cd generalized-timeseries

# create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e ".[dev]"
```

### Testing

```bash
pytest --cov=generalized_timeseries
```

### Tag & Publish

Iirc, much of the CI/CD actions are gated not behind pushed branches, but pushed tags. It will fail if you don't version += 1

1. Bump version in `pyproject.toml` (e.g., `v0.1.13`)
2. Commit and tag:
   ```bash
   git add pyproject.toml
   git commit -m "bump: version to v0.1.13"
   git tag v0.1.13
   git push && git push --tags  # push branch and tag at once
   ```

#### Overall Process

- Triggers: Runs when code is pushed to branches `main` or `dev`
- `pytest`: Validates code across multiple Python versions and OS
- The following are gated behind all tests passing:
    - Building: Creates package distributions and documentation
    - Publishing: Deploys to PyPI, Docker Hub and ReadTheDocs.

## Documentation

Full documentation is available at [generalized-timeseries.readthedocs.io](https://generalized-timeseries.readthedocs.io/en/latest/).

## License

Released under the MIT License.

glhf...
