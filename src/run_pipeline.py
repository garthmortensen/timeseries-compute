#!/usr/bin/env python3

# run_pipeline.py

# handle relative directory imports for chronicler
import logging as l
from chronicler_loader import init_chronicler
chronicler = init_chronicler()

from configurator import load_configuration
from data_generator import PriceSeriesGenerator
from data_processor import MissingDataHandlerFactory
from data_processor import DataScalerFactory
from data_processor import StationaryReturnsProcessor
from stats_model import ModelFactory


# TODO: bad idea to include .dev in file name
config_file = "config.yml"
config = load_configuration(config_file)

l.info("# Generating: price series data")
generator = PriceSeriesGenerator(
    start_date=config.data_generator.start_date,
    end_date=config.data_generator.end_date
    )
price_dict, price_df = generator.generate_prices(
    ticker_initial_prices=config.data_generator.ticker_initial_prices
)

l.info("# Processing: handling missing data")
handler_missing = MissingDataHandlerFactory.create_handler(
    strategy=config.data_processor.missing_data_handler_strategy
)
filled_df = handler_missing(price_df)

l.info("# Processing: scaling data")
handler_scaler = DataScalerFactory.create_handler(
    strategy=config.data_processor.scaler_method
    )
scaled_df = handler_scaler(filled_df)
stationary_returns_processor = StationaryReturnsProcessor()

l.info("# Processing: making data stationary")
diffed_df = stationary_returns_processor.make_stationary(scaled_df, config.data_processor.make_stationarity_method)

l.info("# Testing: stationarity")
adf_results = stationary_returns_processor.check_stationarity(diffed_df, config.data_processor.test_stationarity_method)
stationary_returns_processor.log_adf_results(adf_results, config.data_processor.test_stationarity_p_value_threshold)


l.info("# Modeling")

# if arima_run:
#     l.info("## Running ARIMA")
#     model_arima = ModelFactory.create_model("ARIMA", data=diffed_df, order=arima_order, steps=arima_steps)
#     arima_fit = model_arima.fit()
#     l.info("## ARIMA summary")
#     l.info(model_arima.summary())
#     l.info("## ARIMA forecast")
#     arima_forecast = model_arima.forecast()  # dont include steps arg here bc its already in object initialization
#     l.info(f"arima_forecast: {arima_forecast}")

# if garch_run:
#     l.info("## Running GARCH")
#     # model_garch = ModelFactory.create_model("GARCH", data=diffed_df, p=garch_p, q=garch_q, dist=garch_dist)


# GARCH models, like ARMA models, predict volatility rather than values. 
# Volatility = changes in variance over time, making it a function of time. 
# GARCH handles uneven variance (heteroskedasticity).
# GARCH models assume stationarity, similar to ARMA models, and include both AR and MA components.
# Since volatility often clusters, GARCH is designed to capture and leverage this behavior.
