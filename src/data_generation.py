import logging as l
from data_generator import PriceSeriesGenerator

def generate_price_series(config):
    l.info("Generating price series data")
    generator = PriceSeriesGenerator(
        start_date=config.data_generator.start_date,
        end_date=config.data_generator.end_date
    )
    price_dict, price_df = generator.generate_prices(
        ticker_initial_prices=config.data_generator.ticker_initial_prices
    )
    return price_dict, price_df
