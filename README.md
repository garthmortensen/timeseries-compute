# Productionalize Thesis Project

Build a py production-level model pipeline from scratch, deployed on the cloud.

Take pdf writings and convert entirely. Then add supplementary generalized code.

archive source: https://github.com/garthmortensen/finance/tree/master/15_thesis

## tech stack

- py + django + docker + github actions  
- cloud options: heroku (easier), aws (harder)
  - Which???

## architecture & infrastructure

### 12-factor app
- store configs in yaml or environment variables  
- output logs to stdout/stderr for cloud logging  
- maybe containerize each microservice

### amazon api mandate
- use api gateway + lambda/ecs/eks if on aws  
- handle auto-scaling and load balancing (auto-scaling groups)

### environment & secrets management
- use `.env` for local, secure storage in production (aws secrets manager / heroku config vars)

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
- build & push docker images  
- deploy to staging/prod if all checks pass

## Docker containerization
Note: docker handles packaging/isolation, github actions handles automation

- `Dockerfile` installs dependencies, runs migrations, starts the server  
- tag and version images for rollbacks 

## logging & performance tracking
- track execution time, cpu, memory usage  
- use py’s built-in `logging` plus tools like datadog, cloudwatch  
- keep logs in a centralized logging service

## OOP
- encapsulate logic in classes (e.g., `garchmodel`, `modelmanager`)  
- keep methods cohesive and modular
- sqlalchemy

## DB & ETL

### etl process
- pull data from an api or csv  
- clean/transform data  
- load into sqlite bc ez

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

### plotly?
- create interactive charts for volatility/price movements  
- embed in django templates or a separate frontend

### standard statistics reports
- show confidence intervals, p-values, test outcomes, trends in tables and charts

## final thoughts
1. keep environment-specific configs (dev/test/prod)  
2. plan a clear deployment strategy (stage vs prod)  
3. automate everything possible (lint, build, test, security scans)...

