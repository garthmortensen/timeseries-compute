#!/usr/bin/env python3
# timeseries_compute/examples/example_univariate_garch.py

"""
Example: Univariate GARCH Analysis with Timeseries Compute.

This example demonstrates the core functionality of the timeseries_compute
package for univariate time series analysis. It shows a complete workflow:
1. Generating synthetic price data
2. Converting to returns and testing stationarity
3. Scaling data appropriately for GARCH modeling
4. Fitting ARIMA models for conditional mean
5. Fitting GARCH models for volatility
6. Generating and interpreting forecasts
7. Calculating risk metrics for analysis

The example uses simple AR(1) and GARCH(1,1) models with default parameters
to demonstrate the basic usage pattern.

To run this example:
python -m timeseries_compute.examples.example_univariate_garch
"""

import logging
import pandas as pd
import numpy as np
from tabulate import tabulate
from typing import Dict, Any

# Add the parent directory to the PYTHONPATH if running as a standalone script
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import our modules
from timeseries_compute import data_generator, data_processor, stats_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
l = logging.getLogger(__name__)


def main():
    """Main function demonstrating the package usage."""
    l.info("START: UNIVARIATE GARCH ANALYSIS EXAMPLE")

    # 1. Generate price series
    l.info("Step 1: Generating synthetic price series data")
    price_dict, price_df = data_generator.generate_price_series(
        start_date="2023-01-01",
        end_date="2025-12-31",
        anchor_prices={"AAA": 150.0, "BBB": 250.0, "CCC": 1000.0},
        random_seed=42,  # For reproducibility
    )
    l.info(f"Generated price series for {list(price_df.columns)}")
    l.info(f"Dataset spans from {price_df.index[0]} to {price_df.index[-1]}")
    l.info(f"Number of observations: {len(price_df)}")
    l.info(f"First 5 price values:\n{price_df.head()}")

    # 2. Calculate log returns
    l.info("Step 2: Calculating log returns")
    returns_df = data_processor.price_to_returns(price_df)
    l.info(f"Calculated log returns with {len(returns_df)} observations")
    l.info(f"First 5 return values:\n{returns_df.head()}")

    # Display basic return statistics
    for column in returns_df.columns:
        stats = stats_model.calculate_stats(returns_df[column])
        l.info(f"Stats for {column}:")
        l.info(f"  Mean: {stats['mean']:.6f}")
        l.info(f"  Std Dev: {stats['std']:.6f}")
        l.info(f"  Skewness: {stats['skew']:.6f}")
        l.info(f"  Kurtosis: {stats['kurt']:.6f}")
        l.info(f"  Annualized Vol: {stats['annualized_vol']:.6f}")

    # 3. Test stationarity of returns
    l.info("Step 3: Testing stationarity of returns")
    adf_results = data_processor.test_stationarity(returns_df)
    l.info("Stationarity test results:")
    for col, result in adf_results.items():
        l.info(f"Column: {col}")
        l.info(f"  ADF Statistic: {result['ADF Statistic']:.4f}")
        l.info(f"  p-value: {result['p-value']:.4e}")
        l.info(f"  Stationary: {'Yes' if result['p-value'] < 0.05 else 'No'}")

    # Log summary statistics of prices and returns
    for column in price_df.columns:
        l.info(f"Price statistics for {column}:")
        l.info(f"  Min: {price_df[column].min():.2f}")
        l.info(f"  Max: {price_df[column].max():.2f}")
        l.info(f"  Mean: {price_df[column].mean():.2f}")
        l.info(f"  Std Dev: {price_df[column].std():.2f}")
    
    l.info("--- Summary statistics of prices and returns ---")
    price_summary = price_df.describe()
    returns_summary = returns_df.describe()
    l.info("Price summary statistics:")
    l.info(f"\n{tabulate(price_summary, headers='keys', tablefmt='fancy_grid')}")
    l.info("Returns summary statistics:")
    l.info(f"\n{tabulate(returns_summary, headers='keys', tablefmt='fancy_grid')}")

    # 4. Scale returns for GARCH modeling
    l.info("Step 4: Scaling returns for GARCH modeling")
    scaled_returns_df = data_processor.scale_for_garch(returns_df)
    l.info("Scaled returns for GARCH modeling")
    l.info(f"First 5 scaled returns:\n{scaled_returns_df.head()}")

    # 5. Fit ARIMA models to filter out conditional mean
    l.info("Step 5: Fitting ARIMA models for conditional mean")
    try:
        arima_fits, arima_forecasts = stats_model.run_arima(
            df_stationary=scaled_returns_df,
            p=1,
            d=0,
            q=1,
            forecast_steps=5,
        )

        l.info("ARIMA model parameters:")
        for column in returns_df.columns:
            l.info(f"  {column}:")
            for param, value in arima_fits[column].params.items():
                l.info(f"    {param}: {value:.4f}")

        l.info("ARIMA forecasts (5-step ahead):")
        for column, forecast in arima_forecasts.items():
            l.info(f"  {column}: {forecast:.6f}")

        # Extract ARIMA residuals for GARCH modeling
        arima_residuals = pd.DataFrame(index=scaled_returns_df.index)
        for column in scaled_returns_df.columns:
            arima_residuals[column] = arima_fits[column].resid
        
        l.info("Extracted ARIMA residuals for GARCH modeling")
        l.info(f"First 5 residuals:\n{arima_residuals.head()}")

    except Exception as e:
        l.error(f"ARIMA modeling failed: {str(e)}")
        arima_residuals = scaled_returns_df  # Use original returns if ARIMA fails
        arima_fits = None

    # 6. Fit GARCH models for volatility
    l.info("Step 6: Fitting GARCH models for volatility")
    try:
        garch_params = {
            'p': 1,  # GARCH order
            'q': 1,  # ARCH order
            'dist': 'normal',  # Distribution: 'normal', 't', etc.
            'forecast_steps': 10  # Number of steps to forecast
        }
        
        garch_fit, garch_forecast = stats_model.run_garch(
            df_stationary=arima_residuals,
            **garch_params
        )

        l.info("GARCH model parameters:")
        for column in arima_residuals.columns:
            l.info(f"  {column}:")
            for param, value in garch_fit[column].params.items():
                l.info(f"    {param}: {value:.6f}")

        l.info(f"GARCH volatility forecasts ({garch_params['forecast_steps']}-step ahead):")
        for col, forecast in garch_forecast.items():
            if hasattr(forecast, '__iter__'):
                l.info(f"  {col}:")
                for i, value in enumerate(forecast):
                    l.info(f"    Step {i+1}: {value:.6f}")
            else:
                l.info(f"  {col}: {forecast:.6f}")

        # Extract conditional volatilities
        conditional_volatilities = pd.DataFrame(index=arima_residuals.index)
        for column in arima_residuals.columns:
            conditional_volatilities[column] = np.sqrt(garch_fit[column].conditional_volatility)
        
        l.info("Extracted conditional volatilities")
        l.info(f"First 5 conditional volatilities:\n{conditional_volatilities.head()}")

        # Calculate standardized residuals
        standardized_residuals = pd.DataFrame(index=arima_residuals.index)
        for column in arima_residuals.columns:
            standardized_residuals[column] = arima_residuals[column] / np.sqrt(garch_fit[column].conditional_volatility)
        
        l.info("Calculated standardized residuals")
        l.info(f"First 5 standardized residuals:\n{standardized_residuals.head()}")

        # 7. Analyze GARCH results
        l.info("Step 7: Analyzing GARCH results")
        
        # Analyze volatility patterns
        for column in conditional_volatilities.columns:
            vol_stats = stats_model.calculate_stats(conditional_volatilities[column])
            l.info(f"Volatility statistics for {column}:")
            l.info(f"  Mean volatility: {vol_stats['mean']:.6f}")
            l.info(f"  Max volatility: {vol_stats['max']:.6f}")
            l.info(f"  Min volatility: {vol_stats['min']:.6f}")
            l.info(f"  Volatility of volatility: {vol_stats['std']:.6f}")
            
            # Identify volatility clusters (periods of high volatility)
            high_vol_threshold = vol_stats['mean'] + vol_stats['std']
            high_vol_periods = conditional_volatilities[column] > high_vol_threshold
            cluster_count = sum(high_vol_periods.diff().fillna(0) != 0) // 2
            l.info(f"  Number of volatility clusters: {cluster_count}")
            l.info(f"  Percentage of high volatility days: {sum(high_vol_periods)/len(high_vol_periods)*100:.2f}%")
        
        # Analyze standardized residuals
        for column in standardized_residuals.columns:
            l.info(f"Standardized residuals statistics for {column}:")
            stats = stats_model.calculate_stats(standardized_residuals[column])
            l.info(f"  Mean: {stats['mean']:.6f} (should be close to 0)")
            l.info(f"  Std Dev: {stats['std']:.6f} (should be close to 1)")
            l.info(f"  Skewness: {stats['skew']:.6f}")
            l.info(f"  Excess Kurtosis: {stats['kurt']:.6f}")
            
            # Check normality (if standardization worked correctly)
            exceed_2sd = sum(abs(standardized_residuals[column]) > 2) / len(standardized_residuals[column])
            exceed_3sd = sum(abs(standardized_residuals[column]) > 3) / len(standardized_residuals[column])
            l.info(f"  Exceeding 2 std deviations: {exceed_2sd*100:.2f}% (5% expected for normal)")
            l.info(f"  Exceeding 3 std deviations: {exceed_3sd*100:.2f}% (0.3% expected for normal)")
        
        # Summarize forecast results
        l.info("Volatility forecast summary:")
        forecast_dates = pd.date_range(
            start=arima_residuals.index[-1] + pd.Timedelta(days=1),
            periods=garch_params['forecast_steps'],
            freq='B'  # Business days
        )
        
        for column in arima_residuals.columns:
            forecast_values = garch_forecast[column]
            if hasattr(forecast_values, '__iter__'):
                l.info(f"  {column} volatility forecast:")
                # Convert forecast variances to volatilities
                forecast_vols = np.sqrt(forecast_values)
                
                # Create a DataFrame for better formatting
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Volatility': forecast_vols
                })
                l.info(f"\n{tabulate(forecast_df, headers='keys', tablefmt='fancy_grid')}")
                
                # Trend analysis
                if forecast_vols[-1] > forecast_vols[0]:
                    l.info(f"  Trend: Increasing volatility expected for {column}")
                else:
                    l.info(f"  Trend: Decreasing volatility expected for {column}")
                
                l.info(f"  Average forecast volatility: {np.mean(forecast_vols):.6f}")
                l.info(f"  Max forecast volatility: {np.max(forecast_vols):.6f}")
                l.info(f"  Min forecast volatility: {np.min(forecast_vols):.6f}")

    except Exception as e:
        l.error(f"GARCH modeling or visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()

    # 8. Calculate risk metrics (for a portfolio)
    l.info("Step 8: Calculating portfolio risk metrics")
    try:
        if 'conditional_volatilities' in locals():
            # Get the last day's volatilities
            latest_vols = {col: conditional_volatilities[col].iloc[-1] for col in conditional_volatilities.columns}
            
            # Calculate correlation matrix from standardized residuals
            correlation_matrix = standardized_residuals.corr()
            l.info("Correlation matrix of standardized residuals:")
            l.info(f"\n{tabulate(correlation_matrix, headers='keys', tablefmt='fancy_grid')}")
            
            # Equal-weighted portfolio
            num_assets = len(returns_df.columns)
            weights = np.ones(num_assets) / num_assets
            
            # Construct covariance matrix for the last day
            latest_vol_vector = np.array([latest_vols[col] for col in conditional_volatilities.columns])
            cov_matrix = np.outer(latest_vol_vector, latest_vol_vector) * correlation_matrix.values
            
            # Calculate portfolio risk
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            annualized_volatility = portfolio_volatility * np.sqrt(252)  # Annualized
            
            l.info("Portfolio risk metrics (equal-weighted):")
            l.info(f"  Daily volatility: {portfolio_volatility:.6f}")
            l.info(f"  Annualized volatility: {annualized_volatility:.6f}")
            l.info(f"  1-day Value at Risk (95%): {1.645 * portfolio_volatility:.6f}")
            l.info(f"  1-day Value at Risk (99%): {2.326 * portfolio_volatility:.6f}")
            
            # Display the correlation matrix
            l.info("Asset correlation matrix:")
            l.info(f"\n{tabulate(correlation_matrix, headers='keys', tablefmt='fancy_grid')}")
            
            # Analyze correlation strength
            for i in range(len(correlation_matrix.index)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    asset_i = correlation_matrix.index[i]
                    asset_j = correlation_matrix.columns[j]
                    
                    if abs(corr_val) >= 0.8:
                        strength = "Very strong"
                    elif abs(corr_val) >= 0.6:
                        strength = "Strong"
                    elif abs(corr_val) >= 0.4:
                        strength = "Moderate"
                    elif abs(corr_val) >= 0.2:
                        strength = "Weak"
                    else:
                        strength = "Very weak"
                        
                    direction = "positive" if corr_val >= 0 else "negative"
                    l.info(f"  Correlation between {asset_i} and {asset_j}: {corr_val:.4f} ({strength} {direction})")
    
    except Exception as e:
        l.error(f"Portfolio risk calculation failed: {str(e)}")

    l.info("FINISH: UNIVARIATE GARCH ANALYSIS EXAMPLE")


if __name__ == "__main__":
    main()
    