![CI/CD](https://github.com/garthmortensen/garch/actions/workflows/execute_pytest.yml/badge.svg)

![Read the Docs](https://img.shields.io/readthedocs/garch)


# Productionalize Thesis Project

Build a py production-level model pipeline from scratch, deployed on the cloud.

Take pdf writings and convert entirely. Then add supplementary generalized code.

[archive source](https://github.com/garthmortensen/finance/tree/master/15_thesis)

## Documentation

Documentation available at:  

[https://garch.readthedocs.io/en/latest/](https://garch.readthedocs.io/en/latest/)

## tech stack

- py + django + docker + github actions  
- cloud: aws

## architecture & infrastructure

### 12-factor app
- store configs in yaml or environment variables  
- output logs to stdout/stderr for cloudwatch logging  
- containerize each microservice

### amazon api mandate
- use api gateway + lambda/ecs/eks  
- implement api with Django REST Framework, which "seamlessly integrates" with Django
- handle auto-scaling and load balancing (auto-scaling groups)

### environment & secrets management
- use `.env` for local development  
- store environment variables securely in aws secrets manager  

## model configuration
- use `.yml` to store hyperparameters, model settings, data paths  
- maintain separate yaml files for dev/test/prod  

## testing

### unit tests
- aim for 80%+ coverage with pytest  
- keep tests organized in a dedicated `tests/` folder  
- test each module (data loading, etl, model classes)

## Github Actions CI/CD
- run tests automatically  
- Black linting
- security checks (dependabot)  
- build & push docker images to aws elastic container registry (ecr)  
- deploy to staging/prod with aws ecs or eks  

## Docker containerization
- docker handles packaging/isolation, while github actions handles automation
- `dockerfile` installs dependencies, runs migrations, starts the server  
- tag and version images for rollbacks  
- deploy docker containers to aws ecs (fargate for serverless containers)  

## logging & performance tracking
- track execution time, cpu, memory usage  
- use py’s built-in `logging`  
- aggregate logs with aws cloudwatch for centralized monitoring  

## OOP
- encapsulate logic in classes (e.g., `garchmodel`, `modelmanager`)  
- keep methods cohesive and modular  
- use sqlalchemy for db models and queries  

## DB & ETL

### etl process
- pull data from an api or csv  
- clean/transform data  
- load into sqlite for local dev (switch to aws rds for prod)

## Stats modeling

### GARCH for cross-index volatility
- pipeline steps: data prep, parameter estimation, forecasting  
- possibly store multiple index results in separate tables  

### statistical tests
- dickey-fuller for stationarity  
- other tests: kpss, shapiro-wilk  
- explore arima, var, ml-based models if needed  

## Presentation

### dropdown indices
- use django forms or a small js snippet  
- query the model based on user selection  

### django & django rest framework
- serve results via html or json endpoints  
- keep logic separate in views/apis  

### plotly
- create interactive charts for volatility/price movements  
- embed in django templates or a separate frontend  

### standard statistics reports
- show confidence intervals, p-values, test outcomes, trends in tables and charts  

## final thoughts
1. keep environment-specific configs (dev/test/prod)  
2. plan a clear deployment strategy (staging vs prod using aws ecs)  
3. automate everything possible (lint, build, test, security scans)...  


## great idea

oh this is exciting. i should generalize to time series in general. and give it the ability to generate price series. danger - scope creep.

https://editor.plantuml.com/uml/VP0zQm9148Rx-HKlMbmbrdU5Q0eXS5L84x0PTyTpOPsvxEuf_lki5mWHGjgPv_sOsUR2gKoNIEoA9RpO4SiadXgydmMcyGuVJYT9eavmb78JKSmmDQmUOzK75qRMWf1HgiedlWKTwFTg5uEJfydY5MU-2XX9ECRxGQFf0EMBBC0PFPPjkz-tBQqRVDW4npKuPeCN5pb9Hy1JVF-G_MlxOmaqQwAvr6fJZ-wmcsfrhLvSWYdUv7EINAZkLfkP-mF9escMRwRlqDk0abXpVKy5Q2lg7w_z0W00


```pseudocode
class GeneralizedTimeSeriesModel:
    Attributes:
        - data: input price series
        - model_type: define model type (GARCH, ARIMA, VAR)
        - model_params: model parames
        - fitted_model: fitted model instance
        - diagnostics: model diagnostics

    Methods:
        - __init__(data, model_type, **kwargs): initialize object
        - preprocess_data(): prep data (e.g. stationarity checks)
        - specify_model(): given model type, dynamically select and configure
        - fit_model(): fit model
        - diagnose_model(): perform model diagnostics (e.g. residuals)
        - forecast(steps): forecast
        - validate_model(): compare model performance metrics
        - visualize_results(): plot data, residuals, and forecasts
        - save_model(filepath): save a model
        - load_model(filepath): load a model

class ModelFactory:
    Methods:
        - create_model(data, model_type, **kwargs): factory method to create a specific model instance
```


DataProcessor:
- log returns
- stationarity checks
- outlier filtering for robust analysis
- PCA for dimensionality reduction

DataScaler:
- Data normalization and scaling for compatibility across datasets
    - % changes
    - standardization

MissingDataHandler:
- Handle missing data
    - interpolation
    - forward-fill
    - drop

- volatility measures
    - simple moving averages
    - exponentially weighted moving averages
- "Risk Factors and Their Sensitivities" - volatility decomposition

Perform simple tests
- ADF for stationarity
- KPSS
- Shapiro-Wilk for normality
- basic descriptives
    - mean
    - variance
    - skewness
    - kurtosis
    - quantiles
- distribution fits to assess if time series conforms
    - normal
    - Student-t
    - lognormal

Modeling
- GARCH or EGARCH can handle volatility clustering
- factor models or PCA can help decompose risk drivers
- linear regression for factor exposures
- check residuals for autocorrelation or heteroscedasticity

Backtesting
- coverage/backtesting of VaR to ensure realistic forecasts


