#!/usr/bin/env python3

# run_pipeline.py

# handle relative directory imports for chronicler
import logging as l
from chronicler_loader import init_chronicler
chronicler = init_chronicler()

from config_loader import load_config
from data_generator import PriceSeriesGenerator
from data_processor import MissingDataHandlerFactory
from data_processor import DataScalerFactory
from data_processor import StationaryReturnsProcessor

l.info("Loading configuration")
config = load_config("config.dev.yml")

l.info("Configuring: data generator")
config_sub = config["data_generator"]
start_date = config_sub.get("start_date", "2023-01-01")
end_date = config_sub.get("end_date", "2023-12-31")
ticker_initial_prices = config_sub.get("ticker_initial_prices", {"AAPL": 100.0})
scaling_data_strategy = config_sub.get("scaler", {}).get(
    "method", "standardize"
)
missing_data_strategy = config_sub.get("missing_data_handler", {}).get(
    "strategy", "drop"
)
# TODO: update this to use the config file
p_value_threshold = config_sub.get("stationarity_test", {}).get(    
    "p_value_threshold", 0.05
)
make_stationarity

generator = PriceSeriesGenerator(start_date=start_date, end_date=end_date)
price_dict, price_df = generator.generate_prices(
    ticker_initial_prices=ticker_initial_prices
)

l.info("Configuring: data processor")
handler_missing = MissingDataHandlerFactory.create_handler(missing_data_strategy)
filled_df = handler_missing(price_df)

handler_scaler = DataScalerFactory.create_handler("standardize")
scaled_df = handler_scaler(filled_df)
stationary_returns_processor = StationaryReturnsProcessor()
diffed_df = stationary_returns_processor.make_stationary(scaled_df)
adf_results = stationary_returns_processor.check_stationarity(diffed_df)
stationary_returns_processor.log_adf_results(adf_results, p_value_threshold)

# GARCH models, like ARMA models, predict volatility rather than values. 
# Volatility = changes in variance over time, making it a function of time. 
# GARCH handles uneven variance (heteroskedasticity).
# GARCH models assume stationarity, similar to ARMA models, and include both AR and MA components.
# Since volatility often clusters, GARCH is designed to capture and leverage this behavior.
