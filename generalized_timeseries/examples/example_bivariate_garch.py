#!/usr/bin/env python3
"""
Bivariate GARCH Analysis Example.
This script demonstrates the bivariate GARCH analysis functionality that replicates the MATLAB thesis work in Python.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # For visualization
from tabulate import tabulate

# Add the parent directory to the PYTHONPATH if running as a standalone script
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import our modules
from generalized_timeseries import data_generator, data_processor, stats_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function implementing bivariate GARCH analysis."""
    logger.info("START: BIVARIATE GARCH ANALYSIS EXAMPLE")

    # 1. Generate price series (representing two markets like DJ and SZ from the thesis)
    price_dict, price_df = data_generator.generate_price_series(
        start_date="2023-01-01",
        end_date="2023-12-31",
        anchor_prices={"DJ": 150.0, "SZ": 250.0},
    )

    logger.info(f"Generated price series for markets: {list(price_df.columns)}")
    logger.info(f"Number of observations: {len(price_df)}")

    # 2. Calculate returns (similar to MATLAB's price2ret function)
    returns_df = data_processor.price_to_returns(price_df)
    logger.info("Calculated log returns")
    logger.info(f"First 5 return values:\n{returns_df.head()}")

    # 3. Fit ARMA models to filter out conditional mean
    try:
        arima_fits, arima_forecasts = stats_model.run_arima(
            df_stationary=returns_df, p=1, d=0, q=1, forecast_steps=5
        )

        logger.info("ARIMA parameters:")
        for column in returns_df.columns:
            logger.info(f"  {column}:")
            for param, value in arima_fits[column].params.items():
                logger.info(f"    {param}: {value:.4f}")

    except Exception as e:
        logger.error(f"ARIMA modeling failed: {str(e)}")
        arima_fits = None

    # 4. Fit GARCH models for each series
    try:
        # Run the bivariate GARCH analysis
        mvgarch_results = stats_model.run_multivariate_garch(
            df_stationary=returns_df, arima_fits=arima_fits, lambda_val=0.95
        )

        # Extract results
        arima_residuals = mvgarch_results["arima_residuals"]
        cond_vol_df = mvgarch_results["conditional_volatilities"]
        std_resid_df = mvgarch_results["standardized_residuals"]
        cc_corr = mvgarch_results["cc_correlation"]
        cc_cov_matrix = mvgarch_results["cc_covariance_matrix"]
        dcc_corr = mvgarch_results["dcc_correlation"]

        # Display results
        logger.info("Unconditional correlation between markets:")
        uncond_corr = returns_df.corr()
        logger.info(f"\n{tabulate(uncond_corr, headers='keys', tablefmt='fancy_grid')}")

        logger.info("Constant conditional correlation (CCC-GARCH):")
        logger.info(f"\n{tabulate(cc_corr, headers='keys', tablefmt='fancy_grid')}")

        logger.info(f"Dynamic conditional correlation statistics (EWMA lambda=0.95):")
        logger.info(f"  Mean: {dcc_corr.mean():.4f}")
        logger.info(f"  Min: {dcc_corr.min():.4f}")
        logger.info(f"  Max: {dcc_corr.max():.4f}")

        # Calculate portfolio risk for a 50/50 portfolio
        weights = np.array([0.5, 0.5])
        portfolio_variance, portfolio_volatility = stats_model.calculate_portfolio_risk(
            weights=weights, cov_matrix=cc_cov_matrix
        )

        logger.info("Portfolio risk (50/50 allocation):")
        logger.info(f"  Daily volatility: {portfolio_volatility:.6f}")
        logger.info(
            f"  Annualized volatility: {portfolio_volatility * np.sqrt(252):.6f}"
        )

        # Create plots
        plt.figure(figsize=(12, 9))

        # Plot prices
        plt.subplot(3, 1, 1)
        for column in price_df.columns:
            plt.plot(price_df.index, price_df[column], label=column)
        plt.title("Market Prices")
        plt.legend()
        plt.grid(True)

        # Plot conditional volatilities
        plt.subplot(3, 1, 2)
        for column in cond_vol_df.columns:
            annualized_vol = cond_vol_df[column] * np.sqrt(252)  # Annualize
            plt.plot(annualized_vol.index, annualized_vol, label=f"{column} Volatility")
        plt.title("Conditional Volatilities (Annualized)")
        plt.legend()
        plt.grid(True)

        # Plot dynamic correlation
        plt.subplot(3, 1, 3)
        plt.plot(dcc_corr.index, dcc_corr)
        plt.axhline(y=cc_corr.iloc[0, 1], color="r", linestyle="--", label="CC-GARCH")
        plt.axhline(
            y=uncond_corr.iloc[0, 1], color="g", linestyle=":", label="Unconditional"
        )
        plt.title("Dynamic Conditional Correlation")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        plt.savefig("bivariate_garch_results.png")
        logger.info("Plot saved to 'bivariate_garch_results.png'")

    except Exception as e:
        logger.error(f"GARCH modeling or correlation analysis failed: {str(e)}")

    logger.info("FINISH: BIVARIATE GARCH ANALYSIS EXAMPLE")


if __name__ == "__main__":
    main()
