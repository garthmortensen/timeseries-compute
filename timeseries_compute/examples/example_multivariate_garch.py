#!/usr/bin/env python3
# timeseries_compute/examples/example_multivariate_garch.py

"""
Example: Multivariate GARCH Analysis with Timeseries Compute.

To run this example:
python -m timeseries_compute.examples.example_multivariate_garch
"""

import logging
import pandas as pd
import numpy as np
from tabulate import tabulate

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from timeseries_compute import data_generator, data_processor, stats_model
from timeseries_compute.export_util import export_data 

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
l = logging.getLogger(__name__)


def main():
    """Main function demonstrating the package usage."""
    l.info("START: UNIVARIATE GARCH ANALYSIS EXAMPLE")

    # 1. Generate price series
    l.info("Generating synthetic price series data...")
    price_dict, price_df = data_generator.generate_price_series(
        start_date="2020-01-01",
        end_date="2025-12-31",
        anchor_prices={"AAA": 150.0, "BBB": 250.0, "CCC": 1000.0}
    )
    l.info(f"Generated price series for assets: {list(price_df.columns)}")
    l.info(f"Number of observations: {len(price_df)}")
    export_data(price_df)

    # 2. Calculate log returns
    l.info("Calculating log returns...")
    returns_df = data_processor.price_to_returns(price_df)
    export_data(returns_df)

    # 3. Test for stationarity
    l.info("Testing stationarity of returns...")
    adf_results = data_processor.test_stationarity(returns_df)
    for col, result in adf_results.items():
        l.info(f"{col}: p-value={result['p-value']:.4e} {'(Stationary)' if result['p-value'] < 0.05 else '(Non-stationary)'}")

    # 4. Scale data for GARCH modeling
    l.info("Scaling data for GARCH modeling...")
    scaled_returns_df = data_processor.scale_for_garch(returns_df)
    export_data(scaled_returns_df)

    # 5. Fit ARIMA models for conditional mean
    l.info("Fitting ARIMA models...")
    try:
        arima_fits, arima_forecasts = stats_model.run_arima(
            df_stationary=scaled_returns_df, 
            p=1, d=0, q=1,
            forecast_steps=5
        )
        
        # Extract ARIMA residuals for GARCH modeling
        arima_residuals = pd.DataFrame(index=scaled_returns_df.index)
        for column in scaled_returns_df.columns:
            arima_residuals[column] = arima_fits[column].resid
            
        l.info("ARIMA forecasts (5-step ahead):")
        for col, forecast in arima_forecasts.items():
            l.info(f"  {col}: {forecast:.6f}")
            
    except Exception as e:
        l.error(f"ARIMA modeling failed: {str(e)}")
        arima_residuals = scaled_returns_df  # Use original returns if ARIMA fails

    # 6. Fit GARCH models for volatility
    l.info("Fitting GARCH models...")
    try:
        garch_fit, garch_forecast = stats_model.run_garch(
            df_stationary=arima_residuals,
            p=1, q=1,
            forecast_steps=5
        )
        
        # Extract conditional volatilities
        cond_vol = pd.DataFrame(index=arima_residuals.index)
        for column in arima_residuals.columns:
            cond_vol[column] = np.sqrt(garch_fit[column].conditional_volatility)
        
        # Display GARCH forecasts
        l.info("GARCH volatility forecasts (5-step ahead):")
        for col, forecast in garch_forecast.items():
            if hasattr(forecast, '__iter__'):
                # Convert variance forecasts to volatility
                forecast_vols = np.sqrt(forecast)
                l.info(f"  {col} volatility forecast: {', '.join([f'{v:.6f}' for v in forecast_vols])}")
            else:
                l.info(f"  {col}: {np.sqrt(forecast):.6f}")
                
        # 7. Calculate correlation between assets
        if len(arima_residuals.columns) > 1:
            # Calculate standardized residuals
            std_residuals = pd.DataFrame(index=arima_residuals.index)
            for column in arima_residuals.columns:
                std_residuals[column] = arima_residuals[column] / np.sqrt(garch_fit[column].conditional_volatility)
                
            # Calculate correlation matrix
            correlation_matrix = std_residuals.corr()
            l.info("Correlation matrix of standardized residuals:")
            l.info(f"\n{tabulate(correlation_matrix, headers='keys', tablefmt='fancy_grid')}")
            export_data(correlation_matrix)

            # Calculate portfolio metrics for equal weights
            num_assets = len(returns_df.columns)
            weights = np.ones(num_assets) / num_assets
            export_data(weights)
            
            # Get recent volatilities for a simplified covariance matrix
            latest_vol_vector = np.array([cond_vol[col].iloc[-1] for col in cond_vol.columns])
            cov_matrix = np.outer(latest_vol_vector, latest_vol_vector) * correlation_matrix.values
            export_data(cov_matrix)
            
            # Calculate portfolio risk
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            annualized_volatility = portfolio_volatility * np.sqrt(252)  # Annualized
            
            l.info("Portfolio risk metrics (equal-weighted):")
            l.info(f"  Daily volatility: {portfolio_volatility:.6f}")   
            l.info(f"  Annualized volatility: {annualized_volatility:.6f}")
            l.info(f"  1-day Value at Risk (99%): {2.326 * portfolio_volatility:.6f}")
        
    except Exception as e:
        l.error(f"GARCH modeling failed: {str(e)}")
        import traceback
        traceback.print_exc()

    l.info("FINISH: UNIVARIATE GARCH ANALYSIS EXAMPLE")


if __name__ == "__main__":
    main()
