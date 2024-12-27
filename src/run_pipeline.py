import logging as l
from config_loader import load_config
from data_generator import PriceSeriesGenerator
from data_processor import MissingDataHandler

l.info("Loading configuration")
config = load_config("config.dev.yml")

l.info("Configuring: data generator")
config_sub = config["data_generator"]
start_date = config_sub.get("start_date", "2023-01-01")
end_date = config_sub.get("end_date", "2023-12-31")
ticker_initial_prices = config_sub.get("ticker_initial_prices", {"AAPL": 100.0})

generator = PriceSeriesGenerator(start_date=start_date, end_date=end_date)
price_dict, price_df = generator.generate_prices(
    ticker_initial_prices=ticker_initial_prices
)

l.info("Configuring: data processor")
config_sub = config["data_processor"]

# Extract configuration for each component with defaults
missing_data_strategy = config_sub.get("missing_data_handler", {}).get(
    "strategy", "drop"
)

l.info("Configuring: data processor")
handler = MissingDataHandler(config)

# Use the match statement to process missing data based on the strategy
match missing_data_strategy:
    case "forward_fill":
        price_df = handler.forward_fill(price_df)
    case "drop":
        price_df = handler.drop_na(price_df)
    case _:
        raise ValueError(f"Unknown missing data strategy: {missing_data_strategy}")

