# Generalized Timeseries

[![Python Versions](https://img.shields.io/pypi/pyversions/generalized-timeseries)]((https://pypi.org/project/generalized-timeseries/))
[![PyPI](https://img.shields.io/pypi/v/generalized-timeseries?color=blue&label=PyPI)](https://pypi.org/project/generalized-timeseries/)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-generalized--timeseries-blue)](https://hub.docker.com/r/goattheprofessionalmeower/generalized-timeseries)

![CI/CD](https://github.com/garthmortensen/generalized-timeseries/actions/workflows/cicd.yml/badge.svg) 
[![codecov](https://codecov.io/gh/garthmortensen/generalized-timeseries/graph/badge.svg?token=L1L5OBSF3Z)](https://codecov.io/gh/garthmortensen/generalized-timeseries)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/a55633cfb8324f379b0b5ec16f03c268)](https://app.codacy.com/gh/garthmortensen/generalized-timeseries/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

[![readthedocs.io](https://img.shields.io/readthedocs/generalized-timeseries)](https://generalized-timeseries.readthedocs.io/en/latest/)

## Overview

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

A Python package for timeseries data processing and modeling using ARIMA and GARCH models with both univariate and bivariate capabilities.

### Features

- Price series generation for single and multiple assets
- Data preprocessing with configurable missing data handling and scaling options
- Stationarity testing and transformation for time series analysis
- ARIMA modeling for time series forecasting
- GARCH modeling for volatility forecasting and risk assessment
- Bivariate GARCH modeling with both Constant Conditional Correlation (CCC) and Dynamic Conditional Correlation (DCC) methods
- EWMA covariance calculation for dynamic correlation analysis
- Portfolio risk assessment using volatility and correlation matrices

### Architectural Overview

```mermaid
flowchart TB
    %% Styling
    classDef person fill:#08427B,color:#fff,stroke:#052E56,stroke-width:1px
    classDef system fill:#1168BD,color:#fff,stroke:#0B4884,stroke-width:1px
    classDef external fill:#999999,color:#fff,stroke:#6B6B6B,stroke-width:1px
    
    %% Actors and Systems
    User((User)):::person
    
    %% Main Systems
    TimeSeriesFrontend["Timeseries Frontend
    (Visualization Apps)"]:::system
    TimeSeriesPipeline["Timeseries Pipeline
    (API Service)"]:::system
    GeneralizedTimeseries["Generalized Timeseries
    (Python Package)"]:::system
    
    %% External Systems
    ExternalDataSource[(External Data Source)]:::external
    AnalysisTool["Data Analysis Tools"]:::external
    PyPI["PyPI Package Registry"]:::external
    
    %% Relationships
    User -- "Uses package directly" --> GeneralizedTimeseries
    TimeSeriesPipeline -- "Imports and uses" --> GeneralizedTimeseries
    TimeSeriesFrontend -- "Imports and uses" --> GeneralizedTimeseries
    
    ExternalDataSource -- "Provides time series data" --> GeneralizedTimeseries
    GeneralizedTimeseries -- "Exports analysis to" --> AnalysisTool
    
    GeneralizedTimeseries -- "Published to" --> PyPI
    User -- "Installs from" --> PyPI
```

## Quick Start

### Installation

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

### Example Usage

For univariate time series analysis:

```bash
python -m generalized_timeseries.examples.example_univariate_garch
```

For bivariate GARCH analysis (correlation between two assets):

```bash
python -m generalized_timeseries.examples.example_bivariate_garch
```

### Docker Support

Run with Docker for isolated environments:

```bash
# build the image
docker build -t generalized-timeseries:latest ./

# Run the univariate example
docker run -it generalized-timeseries:latest /app/generalized_timeseries/examples/example_univariate_garch.py

# Run the bivariate example
docker run -it generalized-timeseries:latest /app/generalized_timeseries/examples/example_bivariate_garch.py

# Get into interactive shell
docker run -it --entrypoint /bin/bash generalized-timeseries:latest
```

### Project Structure

```text
generalized_timeseries/...........
├── __init__.py                     # Package initialization
├── data_generator.py               # For creating synthetic price data with random walks and specific statistical properties
├── data_processor.py               # For handling missing data, scaling, stationarizing, and testing time series stationarity
├── stats_model.py                  # For implementing ARIMA, GARCH, and multivariate GARCH models with factory pattern
├── examples/...................
│   ├── __init__.py                 # Makes examples importable as a module
│   ├── example_bivariate_garch.py  # For demonstrating correlation analysis between two markets with CC-GARCH and DCC-GARCH
│   └── example_univariate_garch.py # For showing basic usage of ARIMA and GARCH for single-series forecasting
├── tests/.....................
    ├── __init__.py                 # Makes tests discoverable as a module
    ├── test_data_generator.py      # For testing basic price generation functionality
    ├── test_data_generator_advanced.py # For testing advanced features like customization and statistical properties
    ├── test_data_processor.py      # For testing data transformation, scaling, and stationarity testing
    ├── test_stats_model_arima.py   # For testing ARIMA modeling separately with specialized fixtures
    └── test_stats_model_garch.py   # For testing GARCH volatility modeling with different distributions
```

### Architectural Diagrams

#### Level 2: Container Diagram

```mermaid
flowchart TB
    %% Styling
    classDef person fill:#08427B,color:#fff,stroke:#052E56,stroke-width:1px
    classDef container fill:#438DD5,color:#fff,stroke:#2E6295,stroke-width:1px
    classDef external fill:#999999,color:#fff,stroke:#6B6B6B,stroke-width:1px
    classDef system fill:#1168BD,color:#fff,stroke:#0B4884,stroke-width:1px
    
    %% Person
    User((User)):::person
    
    %% System boundary
    subgraph GeneralizedTimeseriesSystem["Generalized Timeseries System"]
        PythonPackage["Python Package<br>[Library]<br>Core functions for analysis"]:::container
        Dockerized["Docker Container<br>[Linux]<br>Containerized deployment"]:::container
        ExampleScripts["Example Scripts<br>[Python]<br>Demonstration use cases"]:::container
        CIpipeline["CI/CD Pipeline<br>[GitHub Actions]<br>Automates testing/deployment"]:::container
        Documentation["Documentation<br>[ReadTheDocs]<br>API and usage docs"]:::container
    end
    
    %% External Systems
    ExternalDataSource[(External Data Source)]:::external
    AnalysisTool[Analysis & Visualization Tools]:::external
    PyPI[PyPI Repository]:::external
    
    %% Relationships
    User -- "Imports [Python]" --> PythonPackage
    User -- "Runs [CLI]" --> ExampleScripts
    User -- "Reads [Web]" --> Documentation
    ExampleScripts -- "Uses" --> PythonPackage
    PythonPackage -- "Packaged into" --> Dockerized
    CIpipeline -- "Builds and tests" --> Dockerized
    CIpipeline -- "Publishes" --> PyPI
    CIpipeline -- "Updates" --> Documentation
    ExternalDataSource -- "Provides data to" --> PythonPackage
    PythonPackage -- "Exports analysis to" --> AnalysisTool
    User -- "Downloads from" --> PyPI
```

#### Level 3: Component Diagram

```mermaid
flowchart TB
    %% Styling
    classDef person fill:#08427B,color:#fff,stroke:#052E56,stroke-width:1px
    classDef component fill:#85BBF0,color:#000,stroke:#5D82A8,stroke-width:1px
    classDef container fill:#438DD5,color:#fff,stroke:#2E6295,stroke-width:1px
    classDef external fill:#999999,color:#fff,stroke:#6B6B6B,stroke-width:1px
    
    %% Person
    User((User)):::person
    
    %% Package Container
    subgraph PythonPackage["Python Package"]
        DataGenerator["Data Generator<br>[Python]<br>Creates synthetic time series"]:::component
        DataProcessor["Data Processor<br>[Python]<br>Transforms and tests data"]:::component
        StatsModels["Statistical Models<br>[Python]<br>ARIMA and GARCH models"]:::component
        ExampleScripts["Example Scripts<br>[Python]<br>Usage demonstrations"]:::component
        TestSuite["Test Suite<br>[pytest]<br>Validates functionality"]:::component
        
        %% Component relationships
        ExampleScripts --> DataGenerator
        ExampleScripts --> DataProcessor
        ExampleScripts --> StatsModels
        DataProcessor --> DataGenerator
        StatsModels --> DataProcessor
        TestSuite --> DataGenerator
        TestSuite --> DataProcessor
        TestSuite --> StatsModels
    end
    
    %% External Components
    StatsLibraries[(Statistical Libraries<br>statsmodels, arch)]:::external
    DataLibraries[(Data Libraries<br>pandas, numpy)]:::external
    VisualizationLibraries[(Visualization<br>matplotlib)]:::external
    
    %% Relationships
    User -- "Uses" --> ExampleScripts
    User -- "Uses directly" --> DataGenerator
    User -- "Uses directly" --> DataProcessor
    User -- "Uses directly" --> StatsModels
    DataGenerator -- "Uses" --> DataLibraries
    DataProcessor -- "Uses" --> DataLibraries
    StatsModels -- "Uses" --> StatsLibraries
    StatsModels -- "Uses" --> DataLibraries
    ExampleScripts -- "Uses" --> VisualizationLibraries
```

#### Level 4: Code/Class Diagram

```mermaid
classDiagram
    %% Main Classes
    class PriceSeriesGenerator {
        +start_date: str
        +end_date: str
        +dates: pd.DatetimeIndex
        +__init__(start_date, end_date)
        +generate_prices(anchor_prices): Dict[str, list]
    }
    
    class MissingDataHandler {
        +drop_na(data): pd.DataFrame
        +forward_fill(data): pd.DataFrame
    }
    
    class DataScaler {
        +scale_data_standardize(data): pd.DataFrame
        +scale_data_minmax(data): pd.DataFrame
    }
    
    class StationaryReturnsProcessor {
        +make_stationary(data, method): pd.DataFrame
        +test_stationarity(data, test): Dict
        +log_adf_results(data, p_value_threshold): None
    }
    
    class ModelARIMA {
        +data: pd.DataFrame
        +order: Tuple[int, int, int]
        +steps: int
        +models: Dict[str, ARIMA]
        +fits: Dict[str, ARIMA]
        +__init__(data, order, steps)
        +fit(): Dict[str, ARIMA]
        +summary(): Dict[str, str]
        +forecast(): Dict[str, float]
    }
    
    class ModelGARCH {
        +data: pd.DataFrame
        +p: int
        +q: int
        +dist: str
        +models: Dict[str, arch_model]
        +fits: Dict[str, arch_model]
        +__init__(data, p, q, dist)
        +fit(): Dict[str, arch_model]
        +summary(): Dict[str, str]
        +forecast(steps): Dict[str, float]
    }
    
    class ModelMultivariateGARCH {
        +data: pd.DataFrame
        +p: int
        +q: int
        +model_type: str
        +fits: Dict
        +__init__(data, p, q, model_type)
        +fit_cc_garch(): Dict[str, Any]
        +fit_dcc_garch(lambda_val): Dict[str, Any]
    }
    
    class ModelFactory {
        <<static>>
        +create_model(model_type, data, **kwargs): Model
    }
    
    %% Factory Classes
    class MissingDataHandlerFactory {
        <<static>>
        +create_handler(strategy): Callable
    }
    
    class DataScalerFactory {
        <<static>>
        +create_handler(strategy): Callable
    }
    
    class StationaryReturnsProcessorFactory {
        <<static>>
        +create_handler(strategy): StationaryReturnsProcessor
    }
    
    %% Helper Functions
    class DataGeneratorHelpers {
        <<static>>
        +set_random_seed(seed): None
        +generate_price_series(start_date, end_date, anchor_prices, random_seed): Tuple[Dict, pd.DataFrame]
    }
    
    class DataProcessorHelpers {
        <<static>>
        +fill_data(df, strategy): pd.DataFrame
        +scale_data(df, method): pd.DataFrame
        +stationarize_data(df, method): pd.DataFrame
        +test_stationarity(df, method): Dict
        +log_stationarity(adf_results, p_value_threshold): None
        +price_to_returns(prices): pd.DataFrame
        +prepare_timeseries_data(df): pd.DataFrame
        +calculate_ewma_covariance(series1, series2, lambda_val): pd.Series
        +calculate_ewma_volatility(series, lambda_val): pd.Series
    }
    
    class StatsModelHelpers {
        <<static>>
        +run_arima(df_stationary, p, d, q, forecast_steps): Tuple[Dict, Dict]
        +run_garch(df_stationary, p, q, dist, forecast_steps): Tuple[Dict, Dict]
        +run_multivariate_garch(df_stationary, arima_fits, garch_fits, lambda_val): Dict
        +calculate_correlation_matrix(standardized_residuals): pd.DataFrame
        +calculate_dynamic_correlation(ewma_cov, ewma_vol1, ewma_vol2): pd.Series
        +construct_covariance_matrix(volatilities, correlation): np.ndarray
        +calculate_portfolio_risk(weights, cov_matrix): Tuple[float, float]
        +calculate_stats(series): Dict
    }
    
    %% Example Scripts
    class ExampleUnivariateGARCH {
        <<static>>
        +main(): None
    }
    
    class ExampleBivariateGARCH {
        <<static>>
        +main(): None
    }
    
    %% Relationships
    DataGeneratorHelpers --> PriceSeriesGenerator: uses
    
    DataProcessorHelpers --> MissingDataHandler: uses
    DataProcessorHelpers --> DataScaler: uses
    DataProcessorHelpers --> StationaryReturnsProcessor: uses
    DataProcessorHelpers --> MissingDataHandlerFactory: uses
    DataProcessorHelpers --> DataScalerFactory: uses
    DataProcessorHelpers --> StationaryReturnsProcessorFactory: uses
    
    StatsModelHelpers --> ModelARIMA: uses
    StatsModelHelpers --> ModelGARCH: uses
    StatsModelHelpers --> ModelMultivariateGARCH: uses
    StatsModelHelpers --> ModelFactory: uses
    StatsModelHelpers --> DataProcessorHelpers: uses
    
    ExampleUnivariateGARCH --> DataGeneratorHelpers: uses
    ExampleUnivariateGARCH --> DataProcessorHelpers: uses
    ExampleUnivariateGARCH --> StatsModelHelpers: uses
    
    ExampleBivariateGARCH --> DataGeneratorHelpers: uses
    ExampleBivariateGARCH --> DataProcessorHelpers: uses
    ExampleBivariateGARCH --> StatsModelHelpers: uses
    
    MissingDataHandlerFactory --> MissingDataHandler: creates
    DataScalerFactory --> DataScaler: creates
    StationaryReturnsProcessorFactory --> StationaryReturnsProcessor: creates
    ModelFactory --> ModelARIMA: creates
    ModelFactory --> ModelGARCH: creates
    ModelFactory --> ModelMultivariateGARCH: creates
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

Option 2:

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

```bash
# Bump version in pyproject.toml and README.md
git add pyproject.toml README.md
git commit -m "version bump"
git tag v0.1.31
git push && git push --tags
```

### CI/CD Process

- Triggers: Runs when code is pushed to branches `main` or `dev`
- `pytest`: Validates code across multiple Python versions and OS
- Building: Creates package distributions and documentation
- Publishing: Deploys to PyPI, Docker Hub and ReadTheDocs

```mermaid
flowchart TB
    %% Styling
    classDef person fill:#08427B,color:#fff,stroke:#052E56,stroke-width:1px
    classDef system fill:#1168BD,color:#fff,stroke:#0B4884,stroke-width:1px
    classDef external fill:#999999,color:#fff,stroke:#6B6B6B,stroke-width:1px
    classDef pipeline fill:#ff9900,color:#fff,stroke:#cc7700,stroke-width:1px
    
    %% Actors
    Developer((Developer)):::person
    
    %% Main Systems
    GeneralizedTimeseries["Generalized Timeseries\nPython Package"]:::system
    
    %% Source Control
    GitHub["GitHub\nSource Repository"]:::external
    
    %% CI/CD Pipeline and Tools
    GitHubActions["GitHub Actions\nCI/CD Pipeline"]:::pipeline
    
    %% Distribution Platforms
    PyPI["PyPI Registry"]:::external
    DockerHub["Docker Hub"]:::external
    ReadTheDocs["ReadTheDocs"]:::external
    
    %% Code Quality Services
    Codecov["Codecov\nCode Coverage"]:::external
    
    %% Flow
    Developer -- "Commits code to" --> GitHub
    GitHub -- "Triggers on push\nto main/dev" --> GitHubActions
    
    %% Primary Jobs
    subgraph TestJob["Test Job"]
        Test["Run Tests\nPytest"]:::pipeline
        Lint["Lint with Flake8"]:::pipeline
        
        Lint --> Test
    end
    
    subgraph DockerJob["Docker Job"]
        BuildDocker["Build Docker Image"]:::pipeline
    end
    
    subgraph BuildJob["Build Job"]
        BuildPackage["Build Package\nSDist & Wheel"]:::pipeline
        VerifyPackage["Verify with Twine"]:::pipeline
        
        BuildPackage --> VerifyPackage
    end
    
    subgraph DocsJob["Docs Job"]
        BuildDocs["Generate Docs\nSphinx"]:::pipeline
        BuildUML["Generate UML\nDiagrams"]:::pipeline
        
        BuildDocs --> BuildUML
    end
    
    subgraph PublishJob["Publish Job"]
        PublishPyPI["Publish to PyPI"]:::pipeline
    end
    
    %% Job Dependencies
    GitHubActions --> TestJob
    
    TestJob --> DockerJob
    TestJob --> BuildJob
    TestJob --> DocsJob
    
    BuildJob --> PublishJob
    DocsJob --> PublishJob
    
    %% External Services Connections
    Test -- "Upload Results" --> Codecov
    BuildDocker -- "Push Image" --> DockerHub
    DocsJob -- "Deploy Documentation" --> ReadTheDocs
    PublishPyPI -- "Deploy Package" --> PyPI
    
    %% Final Products
    PyPI --> GeneralizedTimeseries
    DockerHub --> GeneralizedTimeseries
    ReadTheDocs -- "Documents" --> GeneralizedTimeseries
```

## Documentation

Full documentation is available at [generalized-timeseries.readthedocs.io](https://generalized-timeseries.readthedocs.io/en/latest/).

## License

Released under the MIT License.
