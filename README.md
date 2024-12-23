# Productionalize Thesis Project

Build a py production-level model pipeline from scratch, deployed on the cloud.

Take pdf writings and convert entirely. Then add supplementary generalized code.

[archive source](https://github.com/garthmortensen/finance/tree/master/15_thesis)

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
