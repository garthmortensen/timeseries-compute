#!/usr/bin/env python3
"""
Bivariate GARCH Analysis Example.
This script demonstrates the bivariate GARCH analysis functionality that replicates the MATLAB thesis work in Python.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # TODO: replace with another library
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import seaborn as sns  # TODO: replace with another library

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the PYTHONPATH if running as a standalone script
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our modules
from generalized_timeseries import data_generator, data_processor

# EWMA covariance function (similar to your MATLAB implementation)
def calculate_ewma_covariance(series1, series2, lambda_val=0.95):
    """
    Calculate Exponentially Weighted Moving Average covariance between two series.
    
    Args:
        series1: First time series
        series2: Second time series
        lambda_val: Decay factor (0.95 or 0.97 in the thesis)
        
    Returns:
        Series of EWMA covariances
    """
    # Initialize covariance series
    cov_series = pd.Series(index=series1.index)
    
    # Calculate initial covariance (first 20 observations or all if less than 20)
    init_window = min(20, len(series1))
    init_cov = series1.iloc[:init_window].cov(series2.iloc[:init_window])
    cov_series.iloc[0] = init_cov
    
    # Calculate EWMA covariance
    for t in range(1, len(series1)):
        cov_series.iloc[t] = lambda_val * cov_series.iloc[t-1] + \
                            (1 - lambda_val) * series1.iloc[t-1] * series2.iloc[t-1]
    
    return cov_series

# Price to returns function (similar to MATLAB's price2ret)
def price_to_returns(prices):
    """Convert prices to log returns, similar to MATLAB's price2ret function."""
    return np.log(prices / prices.shift(1)).dropna()

# CC-GARCH model implementation
def cc_garch(data, p=1, q=1):
    """
    Constant Conditional Correlation (CC) GARCH model.
    
    Args:
        data: DataFrame with multiple time series
        p: GARCH order
        q: ARCH order
        
    Returns:
        Dictionary with model results
    """
    logger.info(f"Fitting CC-GARCH model with p={p}, q={q}")
    
    # Fit univariate GARCH models
    models = {}
    volatilities = {}
    
    for column in data.columns:
        # Fit GARCH model
        model = arch_model(data[column], vol="Garch", p=p, q=q)
        result = model.fit(disp="off")
        models[column] = result
        
        # Extract conditional volatility
        volatilities[column] = pd.Series(np.sqrt(result.conditional_volatility), 
                                         index=data.index)
    
    # Calculate standardized residuals
    std_resid = pd.DataFrame(index=data.index)
    for column in data.columns:
        std_resid[column] = data[column] / volatilities[column]
    
    # Calculate constant correlation matrix
    correlation = std_resid.corr()
    
    # Extract parameters
    params = {}
    for column in data.columns:
        params[column] = {
            'omega': models[column].params['omega'],
            'alpha': models[column].params[f'alpha[1]'],
            'beta': models[column].params[f'beta[1]']
        }
    
    # Collect results
    results = {
        'models': models,
        'volatilities': volatilities,
        'std_resid': std_resid,
        'correlation': correlation,
        'params': params
    }
    
    return results

# DCC-GARCH model implementation using EWMA
def dcc_garch_ewma(data, cc_results, lambda_val=0.95):
    """
    Dynamic Conditional Correlation (DCC) GARCH model using EWMA for correlation.
    
    Args:
        data: DataFrame with multiple time series
        cc_results: Results from CC-GARCH model
        lambda_val: EWMA decay factor
        
    Returns:
        Dictionary with model results
    """
    logger.info(f"Fitting DCC-GARCH model with lambda={lambda_val}")
    
    # Get standardized residuals from CC-GARCH
    std_resid = cc_results['std_resid']
    
    # Calculate dynamic correlations for each pair of series
    dynamic_corr = {}
    columns = data.columns
    
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            col_pair = f"{columns[i]}_{columns[j]}"
            # Calculate EWMA covariance
            cov_ewma = calculate_ewma_covariance(
                std_resid[columns[i]], 
                std_resid[columns[j]], 
                lambda_val
            )
            
            # Calculate EWMA volatilities
            vol_i = std_resid[columns[i]].ewm(alpha=1-lambda_val).std()
            vol_j = std_resid[columns[j]].ewm(alpha=1-lambda_val).std()
            
            # Calculate correlation
            dynamic_corr[col_pair] = cov_ewma / (vol_i * vol_j)
    
    # Collect results
    results = {
        'dynamic_correlations': dynamic_corr,
        'lambda': lambda_val
    }
    
    return results

def main():
    """Main function implementing bivariate GARCH analysis."""
    print("START: BIVARIATE GARCH ANALYSIS EXAMPLE")
    
    # 1. Generate price series (representing two markets like DJ and SZ from your thesis)
    price_dict, price_df = data_generator.generate_price_series(
        start_date="2023-01-01",
        end_date="2023-12-31",
        anchor_prices={"DJ": 150.0, "SZ": 250.0}
    )
    
    print(f"\nGenerated price series for markets: {list(price_df.columns)}")
    print(f"Number of observations: {len(price_df)}")
    
    # 2. Calculate returns (similar to MATLAB's price2ret function)
    returns_df = price_to_returns(price_df)
    print(f"\nCalculated log returns")
    print(f"First 5 return values:\n{returns_df.head()}")
    
    # 3. Fit ARMA models to filter out conditional mean
    arma_residuals = pd.DataFrame(index=returns_df.index)
    arma_fits = {}
    
    # ARMA parameters (similar to your thesis)
    arma_p, arma_d, arma_q = 1, 0, 1
    
    print(f"\nFitting ARMA({arma_p},{arma_d},{arma_q}) models to filter conditional mean")
    
    for column in returns_df.columns:
        model = ARIMA(returns_df[column], order=(arma_p, arma_d, arma_q))
        fit = model.fit()
        arma_fits[column] = fit
        arma_residuals[column] = fit.resid
        
        print(f"  {column} ARMA parameters:")
        for param, value in fit.params.items():
            print(f"    {param}: {value:.4f}")
    
    # 4. Calculate unconditional correlation
    uncond_corr = returns_df.corr().iloc[0, 1]
    print(f"\nUnconditional correlation between markets: {uncond_corr:.4f}")
    
    # 5. Fit CC-GARCH model
    # GARCH parameters (similar to your thesis)
    garch_p, garch_q = 1, 1
    
    print(f"\nFitting CC-GARCH({garch_p},{garch_q}) model")
    cc_results = cc_garch(arma_residuals, p=garch_p, q=garch_q)
    
    # Display CC-GARCH parameters and correlation
    print("  CC-GARCH parameters:")
    for market, params in cc_results['params'].items():
        print(f"    {market}:")
        for param_name, param_value in params.items():
            print(f"      {param_name}: {param_value:.6f}")
    
    cc_corr = cc_results['correlation'].iloc[0, 1]
    print(f"  Constant conditional correlation: {cc_corr:.4f}")
    
    # 6. Fit DCC-GARCH model using EWMA
    ewma_lambda = 0.95  # Similar to your thesis
    
    print(f"\nFitting DCC-GARCH model with EWMA (lambda={ewma_lambda})")
    dcc_results = dcc_garch_ewma(arma_residuals, cc_results, lambda_val=ewma_lambda)
    
    # Get dynamic correlation series for the market pair
    dynamic_corr_key = f"{arma_residuals.columns[0]}_{arma_residuals.columns[1]}"
    dynamic_corr = dcc_results['dynamic_correlations'][dynamic_corr_key]
    
    print(f"  Dynamic correlation range: {dynamic_corr.min():.4f} to {dynamic_corr.max():.4f}")
    print(f"  Dynamic correlation mean: {dynamic_corr.mean():.4f}")
    
    # 7. Plot results
    plt.figure(figsize=(12, 9))
    
    # Plot prices
    plt.subplot(3, 1, 1)
    for column in price_df.columns:
        plt.plot(price_df.index, price_df[column], label=column)
    plt.title('Market Prices')
    plt.legend()
    plt.grid(True)
    
    # Plot conditional volatilities
    plt.subplot(3, 1, 2)
    for column in cc_results['volatilities']:
        annualized_vol = cc_results['volatilities'][column] * np.sqrt(250)  # Annualize
        plt.plot(annualized_vol.index, annualized_vol, label=f"{column} Volatility")
    plt.title('Conditional Volatilities (Annualized)')
    plt.legend()
    plt.grid(True)
    
    # Plot dynamic correlation
    plt.subplot(3, 1, 3)
    plt.plot(dynamic_corr.index, dynamic_corr)
    plt.axhline(y=cc_corr, color='r', linestyle='--', label='CC-GARCH')
    plt.axhline(y=uncond_corr, color='g', linestyle=':', label='Unconditional')
    plt.title('Dynamic Conditional Correlation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('bivariate_garch_results.png')
    print("\nPlot saved to 'bivariate_garch_results.png'")
    
    print("\nFINISH: BIVARIATE GARCH ANALYSIS EXAMPLE")

if __name__ == "__main__":
    main()
