#!/usr/bin/env python3
# timeseries_compute/examples/example_univariate_garch.py

"""
Example: Univariate GARCH Analysis with Timeseries Compute.

This example demonstrates the core functionality of the timeseries_compute
package for univariate time series analysis. It shows a complete workflow:
1. Generating synthetic price data
2. Making the data stationary
3. Testing for stationarity
4. Fitting ARIMA models for conditional mean
5. Fitting GARCH models for volatility
6. Generating and interpreting forecasts

The example uses simple AR(1) and GARCH(1,1) models with default parameters
to demonstrate the basic usage pattern.

To run this example:
python -m timeseries_compute.examples.example_univariate_garch
"""

import logging
import pandas as pd
import numpy as np
from tabulate import tabulate

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from timeseries_compute import data_generator, data_processor, stats_model

# logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Main function demonstrating the package usage."""
    print("START: GENERALIZED TIMESERIES EXAMPLE")

    # 1 Generate price series
    price_dict, price_df = data_generator.generate_price_series(
        start_date="2023-01-01",
        end_date="2025-12-31",
        anchor_prices={"AAA": 150.0, "BBB": 250.0, "CCC": 1000.0},
    )

    # 2 Make data stationary
    df_stationary = data_processor.stationarize_data(price_df)

    # 3 Test for stationarity
    adf_results = data_processor.test_stationarity(df_stationary)

    print("\nStationarity test results:")
    for col, result in adf_results.items():
        print(f"Column: {col}")
        print(f"  ADF Statistic: {result['ADF Statistic']:.4f}")
        print(f"  p-value: {result['p-value']:.4e}")
        print(f"  Stationary: {'Yes' if result['p-value'] < 0.05 else 'No'}")
        print()

    # 4 arima model
    try:
        arima_fit, arima_forecast = stats_model.run_arima(
            df_stationary=df_stationary, p=1, d=0, q=0, forecast_steps=5
        )

        # Display forecasts: arima
        print("\nARIMA Forecasts:")
        for col, forecast in arima_forecast.items():
            print(f"  {col}: {forecast:.4f}")
    except Exception as e:
        print(f"ARIMA modeling failed: {str(e)}")

    # 5 Fit GARCH model
    try:
        garch_fit, garch_forecast = stats_model.run_garch(
            df_stationary=df_stationary, p=1, q=1, forecast_steps=5
        )

        # display forecasts: garch
        print("\nGARCH Volatility Forecasts:")
        for col, forecast in garch_forecast.items():
            if hasattr(forecast, "iloc"):
                print(f"  {col}:")
                for i, value in enumerate(forecast):
                    print(f"    Step {i+1}: {value:.6f}")
            else:
                print(f"  {col}: {forecast:.6f}")
    except Exception as e:
        print(f"GARCH modeling failed: {str(e)}")

    print("FINISH: GENERALIZED TIMESERIES EXAMPLE")


if __name__ == "__main__":
    main()
