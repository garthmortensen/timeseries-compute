#!/usr/bin/env python3
"""
Bivariate GARCH Analysis Example.
This script demonstrates the bivariate GARCH analysis functionality that replicates the MATLAB thesis work in Python.
"""

import logging
import pandas as pd
import numpy as np
from tabulate import tabulate

# Add the parent directory to the PYTHONPATH if running as a standalone script
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our modules
from generalized_timeseries import data_generator, data_processor, stats_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function implementing bivariate GARCH analysis."""
    print("START: BIVARIATE GARCH ANALYSIS EXAMPLE")
    
    # 1. Generate price series (representing two markets like DJ and SZ from the thesis)
    price_dict, price_df = data_generator.generate_price_series(
        start_date="2023-01-01",
        end_date="2023-12-31",
        anchor_prices={"DJ": 150.0, "SZ": 250.0}
    )
    
    print(f"\nGenerated price series for markets: {list(price_df.columns)}")
    print(f"Number of observations: {len(price_df)}")
    
    # TODO: Add to data_processor.py
    # 2. Calculate returns (similar to MATLAB's price2ret function)
    # Using log returns for financial modeling
    returns_df = np.log(price_df / price_df.shift(1)).dropna()
    print(f"\nCalculated log returns")
    print(f"First 5 return values:\n{returns_df.head()}")
    
    # 3. Test for stationarity of returns
    adf_results = data_processor.test_stationarity(returns_df)
    
    print("\nStationarity test results:")
    for col, result in adf_results.items():
        print(f"Column: {col}")
        print(f"  ADF Statistic: {result['ADF Statistic']:.4f}")
        print(f"  p-value: {result['p-value']:.4e}")
        print(f"  Stationary: {'Yes' if result['p-value'] < 0.05 else 'No'}")
        print()
    
    # TODO: Add to stats_model.py
    # 4. Calculate unconditional correlation
    uncond_corr = returns_df.corr().iloc[0, 1]
    print(f"\nUnconditional correlation between markets: {uncond_corr:.4f}")
    
    # 5. Fit ARMA models to filter out conditional mean
    try:
        # For bivariate GARCH, we need to use ARIMA to filter out the mean effects
        arima_fits, arima_forecasts = stats_model.run_arima(
            df_stationary=returns_df,
            p=1,
            d=0,
            q=1,
            forecast_steps=5
        )
        
        # Get residuals from ARIMA models
        arima_residuals = pd.DataFrame(index=returns_df.index)
        
        for column in returns_df.columns:
            arima_residuals[column] = arima_fits[column].resid
        
        print("\nARIMA parameters:")
        for column in returns_df.columns:
            print(f"  {column}:")
            for param, value in arima_fits[column].params.items():
                print(f"    {param}: {value:.4f}")
    
    except Exception as e:
        print(f"ARIMA modeling failed: {str(e)}")
        # If ARIMA fails, use the returns directly
        arima_residuals = returns_df
        print("Using returns directly as residuals")
    
    # 6. Fit GARCH models for each series
    try:
        garch_fits, garch_forecasts = stats_model.run_garch(
            df_stationary=arima_residuals,
            p=1,
            q=1,
            forecast_steps=5
        )
        
        # Display GARCH parameter estimates
        print("\nGARCH parameters:")
        for column, fit in garch_fits.items():
            print(f"  {column}:")
            for param_name in ['omega', 'alpha[1]', 'beta[1]']:
                if param_name in fit.params:
                    print(f"    {param_name}: {fit.params[param_name]:.6f}")
        
        # Extract conditional volatilities
        cond_vol = {}
        for column in arima_residuals.columns:
            cond_vol[column] = np.sqrt(garch_fits[column].conditional_volatility)
        
        cond_vol_df = pd.DataFrame(cond_vol, index=arima_residuals.index)
        
        # Display volatility forecasts
        print("\nGARCH Volatility Forecasts:")
        for col, forecast in garch_forecasts.items():
            if hasattr(forecast, 'iloc'):
                print(f"  {col}:")
                for i, value in enumerate(forecast):
                    print(f"    Step {i+1}: {value:.6f}")
            else:
                print(f"  {col}: {forecast:.6f}")
        
        # 7. Calculate standardized residuals
        std_resid = {}
        for column in arima_residuals.columns:
            std_resid[column] = arima_residuals[column] / cond_vol[column]
        
        std_resid_df = pd.DataFrame(std_resid, index=arima_residuals.index)
        
        # TODO: Add to stats_model.py
        # 8. Fit constant conditional correlation (CCC-GARCH)
        # Calculate constant correlation of standardized residuals
        cc_corr = std_resid_df.corr().iloc[0, 1]
        print(f"\nConstant conditional correlation: {cc_corr:.4f}")
        
        # 9. Fit dynamic conditional correlation (DCC-GARCH using EWMA)
        # Calculate EWMA covariance between standardized residuals
        columns = list(std_resid_df.columns)
        ewma_lambda = 0.95  # EWMA decay factor (similar to thesis)
        
        # Calculate EWMA correlation
        ewma_cov = data_processor.calculate_ewma_covariance(
            std_resid_df[columns[0]], 
            std_resid_df[columns[1]], 
            lambda_val=ewma_lambda
        )
        
        # Calculate EWMA volatilities for standardized residuals
        ewma_vol1 = data_processor.calculate_ewma_volatility(
            std_resid_df[columns[0]], 
            lambda_val=ewma_lambda
        )
        
        ewma_vol2 = data_processor.calculate_ewma_volatility(
            std_resid_df[columns[1]],
            lambda_val=ewma_lambda
        )
        
        # TODO: Add to stats_model.py
        # Calculate dynamic correlation
        dcc_corr = ewma_cov / (ewma_vol1 * ewma_vol2)
        
        print(f"\nDynamic conditional correlation statistics (lambda={ewma_lambda}):")
        print(f"  Mean: {dcc_corr.mean():.4f}")
        print(f"  Min: {dcc_corr.min():.4f}")
        print(f"  Max: {dcc_corr.max():.4f}")
        
        # 10. Compare correlation methods
        print("\nCorrelation comparison:")
        print(f"  Unconditional correlation: {uncond_corr:.4f}")
        print(f"  Constant conditional correlation: {cc_corr:.4f}")
        print(f"  Dynamic conditional correlation (mean): {dcc_corr.mean():.4f}")
        
        # 11. Calculate bivariate portfolio risk
        # For a simple 50/50 portfolio
        weights = np.array([0.5, 0.5])
        
        # Get latest volatilities
        latest_vols = [cond_vol[col].iloc[-1] for col in columns]
        
        # TODO: Add to stats_model.py
        # Construct covariance matrix using CCC
        cov_matrix = np.outer(latest_vols, latest_vols)
        cov_matrix[0, 1] = cov_matrix[0, 1] * cc_corr
        cov_matrix[1, 0] = cov_matrix[1, 0] * cc_corr
        
        # TODO: Add to stats_model.py
        # Calculate portfolio variance
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        print("\nPortfolio risk (50/50 allocation):")
        print(f"  Daily volatility: {portfolio_volatility:.6f}")
        print(f"  Annualized volatility: {portfolio_volatility * np.sqrt(252):.6f}")
        
    except Exception as e:
        print(f"GARCH modeling or correlation analysis failed: {str(e)}")
    
    print("\nFINISH: BIVARIATE GARCH ANALYSIS EXAMPLE")

if __name__ == "__main__":
    main()
