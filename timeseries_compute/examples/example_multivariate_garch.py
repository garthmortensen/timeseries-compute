#!/usr/bin/env python3
# timeseries_compute/examples/example_multivariate_garch.py
"""
Multivariate GARCH Analysis Example with Spillover Effects.
This script demonstrates the multivariate GARCH analysis functionality with added spillover analysis.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add the parent directory to the PYTHONPATH if running as a standalone script
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import our modules
from timeseries_compute import (
    data_generator,
    data_processor,
    stats_model,
    spillover_processor,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
l = logging.getLogger(__name__)


def main():
    """Main function implementing multivariate GARCH analysis with spillover effects."""
    l.info("START: MULTIVARIATE GARCH ANALYSIS WITH SPILLOVER EFFECTS EXAMPLE")

    # 1. Generate correlated price series for multiple markets
    price_dict, price_df = data_generator.generate_price_series(
        start_date="2020-01-01",
        end_date="2025-12-31",
        anchor_prices={"DJ": 150.0, "SZ": 250.0, "EU": 300.0, "JP": 1000.0},
        correlations={
            ("DJ", "SZ"): 0.8,
            ("DJ", "EU"): 0.7,
            ("SZ", "EU"): 0.6,
            ("DJ", "JP"): 0.5,
            ("SZ", "JP"): 0.4,
            ("EU", "JP"): 0.3,
        },
    )
    l.info(f"Generated price series for markets: {list(price_df.columns)}")
    l.info(f"Number of observations: {len(price_df)}")

    # 2. Calculate returns (similar to MATLAB's price2ret function)
    returns_df = data_processor.price_to_returns(price_df)
    l.info("Calculated log returns")
    l.info(f"First 5 return values:\n{returns_df.head()}")

    # 3. Test stationarity of returns (adding this step)
    adf_results = data_processor.test_stationarity(returns_df)
    l.info("Stationarity test results:")
    for col, result in adf_results.items():
        l.info(f"Column: {col}")
        l.info(f"  ADF Statistic: {result['ADF Statistic']:.4f}")
        l.info(f"  p-value: {result['p-value']:.4e}")
        l.info(f"  Stationary: {'Yes' if result['p-value'] < 0.05 else 'No'}")

    # 4. Scale returns for GARCH modeling (adding this step)
    scaled_returns_df = data_processor.scale_for_garch(returns_df)
    l.info("Scaled returns for GARCH modeling")

    # 5. Fit ARMA models to filter out conditional mean
    try:
        arima_fits, arima_forecasts = stats_model.run_arima(
            df_stationary=scaled_returns_df,  # Use scaled data now
            p=1,
            d=0,
            q=1,
            forecast_steps=5,
        )

        l.info("ARIMA parameters:")
        for column in returns_df.columns:
            l.info(f"  {column}:")
            for param, value in arima_fits[column].params.items():
                l.info(f"    {param}: {value:.4f}")

    except Exception as e:
        l.error(f"ARIMA modeling failed: {str(e)}")
        arima_fits = None

    # 6. Run multivariate GARCH analysis with spillover effects
    try:
        # Run the analysis with spillover effects
        combined_results = spillover_processor.run_spillover_analysis(
            df_stationary=scaled_returns_df,  # Use scaled data here too
            arima_fits=arima_fits,
            lambda_val=0.95,
            max_lag=5,
            window_size=40,  # Smaller window for this example
            forecast_horizon=10,
            response_periods=10,
            significance_level=0.05,
        )

        # Extract standard results
        arima_residuals = combined_results["arima_residuals"]
        cond_vol_df = combined_results["conditional_volatilities"]
        cc_corr = combined_results["cc_correlation"]

        # Extract spillover analysis results
        spillover_results = combined_results["spillover_analysis"]

        # Display correlation results
        l.info("Unconditional correlation between markets:")
        uncond_corr = returns_df.corr()
        l.info(f"\n{tabulate(uncond_corr, headers='keys', tablefmt='fancy_grid')}")

        l.info("Constant conditional correlation (CCC-GARCH):")
        l.info(f"\n{tabulate(cc_corr, headers='keys', tablefmt='fancy_grid')}")

        # Display spillover analysis results
        l.info("Granger Causality Results:")
        for pair, result in spillover_results["granger_causality"].items():
            l.info(f"  {pair}: {'Yes' if result['causality'] else 'No'}")
            if result["causality"]:
                l.info(f"    Optimal lag: {result['optimal_lag']}")

        l.info("Significant Shock Spillover Relationships:")
        for pair, result in spillover_results["shock_spillover"].items():
            if result["significant_lags"]:
                l.info(f"  {pair}:")
                l.info(f"    Significant lags: {result['significant_lags']}")
                l.info(f"    R-squared: {result['r_squared']:.4f}")

        # Generate and save spillover analysis plots
        spillover_fig = spillover_processor.plot_spillover_analysis(
            spillover_results, output_path="spillover_analysis.png"
        )
        l.info("Spillover analysis plots saved to 'spillover_analysis.png'")

        # Create additional visualization for volatility
        plt.figure(figsize=(12, 8))
        for column in cond_vol_df.columns:
            annualized_vol = cond_vol_df[column] * np.sqrt(252)  # Annualize
            plt.plot(annualized_vol.index, annualized_vol, label=f"{column} Volatility")
        plt.title("Conditional Volatilities (Annualized)")
        plt.legend()
        plt.grid(True)
        plt.savefig("volatility_analysis.png")
        l.info("Volatility analysis plot saved to 'volatility_analysis.png'")

    except Exception as e:
        l.error(f"GARCH modeling or spillover analysis failed: {str(e)}")
        import traceback

        traceback.print_exc()

    l.info("FINISH: MULTIVARIATE GARCH ANALYSIS WITH SPILLOVER EFFECTS EXAMPLE")


if __name__ == "__main__":
    main()
