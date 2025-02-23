#!/usr/bin/env python3

# run_pipeline.py

# handle relative directory imports for chronicler
import logging as l
from chronicler_loader import init_chronicler
chronicler = init_chronicler()

from configurator import load_configuration
from data_generator import generate_price_series
from data_processor import fill_data, scale_data, stationarize_data, test_stationarity, log_stationarity
from data_processor import StationaryReturnsProcessor
from stats_model import ModelFactory

try:
    config_file = "config.yml"
    config = load_configuration(config_file=config_file)

    # Generate price data
    price_dict, price_df = generate_price_series(config=config)

    # Fill data
    df_filled = fill_data(df=price_df, config=config)

    # Scale data
    df_scaled = scale_data(df=df_filled, config=config)

    # Stationarize data
    df_stationary = stationarize_data(df=df_scaled, config=config)

    # TODO: test stationarity
    adf_results = test_stationarity(df=df_stationary, config=config)
    
    # TODO: log stationarity results
    log_stationarity(df=adf_results, config=config)

    l.info("\n# Modeling")

    # if config.stats_model.ARIMA.run:
    #     l.info("\n## Running ARIMA")
    #     model_arima = ModelFactory.create_model(
    #         model_type="ARIMA", 
    #         data=diffed_df, 
    #         order=(
    #             config.stats_model.ARIMA.parameters_fit.get("p",),
    #             config.stats_model.ARIMA.parameters_fit.get("d"),
    #             config.stats_model.ARIMA.parameters_fit.get("q")
    #             ),
    #         steps=config.stats_model.ARIMA.parameters_predict_steps
    #         )
    #     arima_fit = model_arima.fit()
    #     l.info("\n## ARIMA summary")
    #     l.info(model_arima.summary())
    #     l.info("\n## ARIMA forecast")
    #     arima_forecast = model_arima.forecast()  # dont include steps arg here bc its already in object initialization
    #     l.info(f"arima_forecast: {arima_forecast}")

    # if config.stats_model.GARCH.run:
    #     l.info("\n## Running GARCH")
    #     model_garch = ModelFactory.create_model(
    #         model_type="GARCH", 
    #         data=diffed_df,
    #         p=config.stats_model.GARCH.parameters_fit.p,
    #         q=config.stats_model.GARCH.parameters_fit.q,
    #         dist=config.stats_model.GARCH.parameters_fit.dist
    #         )
    #     garch_fit = model_garch.fit()
    #     l.info("\n## GARCH summary")
    #     l.info(model_garch.summary())
    #     l.info("\n## GARCH forecast")
    #     garch_forecast = model_garch.forecast(steps=config.stats_model.GARCH.parameters_predict_steps)
    #     l.info(f"garch_forecast: {garch_forecast}")

except Exception as e:
    l.exception(f"\nError in pipeline:\n{e}")
    raise
