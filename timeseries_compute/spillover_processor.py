#!/usr/bin/env python3
# Add to spillover_processor.py
import pandas as pd
from typing import Dict, Any
from timeseries_compute.stats_model import run_multivariate_garch

def test_granger_causality(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 5,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Test for Granger causality between two time series.
    
    Args:
        series1: First time series (potential cause)
        series2: Second time series (potential effect)
        max_lag: Maximum number of lags to test
        significance_level: Significance level for hypothesis testing
        
    Returns:
        Dictionary containing:
        - 'causality': Boolean indicating if series1 Granger-causes series2
        - 'p_values': p-values for each lag
        - 'f_statistics': F-statistics for each lag
        - 'optimal_lag': Optimal lag based on AIC
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # Combine series into a DataFrame
    data = pd.concat([series1, series2], axis=1)
    data.columns = ['series1', 'series2']
    data = data.dropna()
    
    # Run Granger causality tests for different lags
    test_results = grangercausalitytests(data, maxlag=max_lag)
    
    # Extract p-values and F-statistics
    p_values = {}
    f_statistics = {}
    for lag, results in test_results.items():
        p_values[lag] = results[0]['ssr_ftest'][1]
        f_statistics[lag] = results[0]['ssr_ftest'][0]
    
    # Determine if there is Granger causality at any lag
    causality = any(p < significance_level for p in p_values.values())
    
    # Find optimal lag based on AIC
    aic_values = {}
    for lag, results in test_results.items():
        aic_values[lag] = results[0]['ssr_chi2test'][1]
    optimal_lag = min(aic_values, key=aic_values.get)
    
    return {
        'causality': causality,
        'p_values': p_values,
        'f_statistics': f_statistics,
        'optimal_lag': optimal_lag
    }


def analyze_shock_spillover(
    residuals1: pd.Series,
    volatility2: pd.Series,
    max_lag: int = 5,
) -> Dict[str, Any]:
    """
    Analyze how shocks in one market affect volatility in another market.
    
    Args:
        residuals1: Residuals (shocks) from the first market
        volatility2: Conditional volatility of the second market
        max_lag: Maximum number of lags to consider
        
    Returns:
        Dictionary containing:
        - 'coefficients': Coefficients for each lag
        - 'p_values': p-values for each coefficient
        - 'r_squared': R-squared value of the model
        - 'significant_lags': Lags with significant spillover effects
    """
    import statsmodels.api as sm
    
    # Square the residuals to get the shock magnitude
    shock_magnitude = residuals1 ** 2
    
    # Create lagged shock variables
    lagged_shocks = {}
    for lag in range(1, max_lag + 1):
        lagged_shocks[f'lag_{lag}'] = shock_magnitude.shift(lag)
    
    # Combine into a DataFrame
    df = pd.DataFrame(lagged_shocks)
    df['volatility'] = volatility2
    df = df.dropna()
    
    # Prepare X and y for regression
    X = df.drop('volatility', axis=1)
    X = sm.add_constant(X)
    y = df['volatility']
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Extract results
    coefficients = model.params[1:]  # Exclude the constant
    p_values = model.pvalues[1:]  # Exclude the constant
    r_squared = model.rsquared
    
    # Identify significant lags
    significant_lags = [lag for lag, p_val in enumerate(p_values, 1) if p_val < 0.05]
    
    return {
        'coefficients': coefficients,
        'p_values': p_values,
        'r_squared': r_squared,
        'significant_lags': significant_lags,
        'summary': model.summary()
    }


def measure_spillover_effects(
    returns_df: pd.DataFrame,
    window_size: int = 60,
    forecast_horizon: int = 10,
) -> Dict[str, Any]:
    """
    Measure the magnitude and direction of spillover effects between markets.
    """
    from statsmodels.tsa.api import VAR
    import numpy as np
    import logging
    
    # Get list of markets
    markets = returns_df.columns.tolist()
    n_markets = len(markets)
    safe_forecast_horizon = min(forecast_horizon, n_markets)
    
    # Initialize results
    spillover_indices = pd.DataFrame(index=returns_df.index[window_size:])
    
    # Calculate maximum lag based on data constraints
    max_possible_lag = max(1, int((window_size - n_markets) / n_markets) - 1)
    actual_maxlag = min(5, max_possible_lag)
    
    # Perform rolling window analysis
    for i in range(len(returns_df) - window_size):
        # Get data for current window
        window_data = returns_df.iloc[i:i+window_size]
        
        try:
            # Fit VAR model with forced lag=1 instead of using criteria
            model = VAR(window_data)
            # Force a lag of 1 instead of letting AIC choose, which might choose 0
            results = model.fit(maxlags=1, ic=None)
            
            # Check if we got meaningful results
            if len(results.coefs) == 0:
                logging.warning(f"Window {i}: No valid coefficients found, skipping")
                continue
                
            # Get forecast error variance decomposition
            fevd = results.fevd(safe_forecast_horizon)
        
            # Create spillover matrix for this window
            spillover_matrix = np.zeros((n_markets, n_markets))
            for j in range(n_markets):
                for k in range(n_markets):
                    # The decomposition at the final forecast horizon
                    spillover_matrix[j, k] = fevd.decomp[j, k, safe_forecast_horizon-1]
            
            # Rest of function remains the same...
            # Normalize the matrix
            row_sums = spillover_matrix.sum(axis=1)
            normalized_matrix = spillover_matrix / row_sums[:, np.newaxis]
            
            # Calculate total spillover index
            total_spillover = (normalized_matrix.sum() - np.trace(normalized_matrix)) / normalized_matrix.sum()
            
            # Calculate directional spillovers
            directional_from = {}
            directional_to = {}
            
            for j, market in enumerate(markets):
                from_market = (normalized_matrix[j, :].sum() - normalized_matrix[j, j]) / normalized_matrix.sum()
                directional_from[market] = from_market
                
                to_market = (normalized_matrix[:, j].sum() - normalized_matrix[j, j]) / normalized_matrix.sum()
                directional_to[market] = to_market
            
            # Calculate net spillover
            net_spillover = {}
            for market in markets:
                net_spillover[market] = directional_from[market] - directional_to[market]
            
            # Store results for this window
            current_date = returns_df.index[i+window_size]
            spillover_indices.loc[current_date, 'total'] = total_spillover
            
            for market in markets:
                spillover_indices.loc[current_date, f'from_{market}'] = directional_from[market]
                spillover_indices.loc[current_date, f'to_{market}'] = directional_to[market]
                spillover_indices.loc[current_date, f'net_{market}'] = net_spillover[market]
                
        except Exception as e:
            logging.warning(f"Error in window {i}: {str(e)}")
            continue
    
    return {
        'spillover_indices': spillover_indices,
        'markets': markets
    }

def calculate_impulse_response(
    returns_df: pd.DataFrame,
    response_periods: int = 10,
    method: str = 'generalized'  # We'll handle this parameter differently
) -> Dict[str, Any]:
    """
    Calculate impulse response functions to visualize spillover effects over time.
    
    Args:
        returns_df: DataFrame of returns from multiple markets
        response_periods: Number of periods to calculate response for
        method: IRF identification method (ignored in older statsmodels versions)
        
    Returns:
        Dictionary containing impulse responses between all markets
    """
    from statsmodels.tsa.api import VAR
    import numpy as np
    import logging
    
    # Get list of markets
    markets = returns_df.columns.tolist()
    
    try:
        # Fit VAR model with fixed lag=1
        model = VAR(returns_df)
        results = model.fit(maxlags=1, ic=None)  # Force lag=1
        
        # Call irf without the method parameter
        # This will work with older versions of statsmodels
        try:
            # First try without the method parameter
            irf = results.irf(response_periods)
        except TypeError as e:
            logging.warning(f"Error in IRF calculation: {str(e)}")
            return {
                'irfs': {},
                'periods': np.arange(response_periods),
                'markets': markets,
                'error': str(e)
            }
        
        # Get the IRF values for all market combinations
        irf_results = {}
        for i, shock_market in enumerate(markets):
            for j, response_market in enumerate(markets):
                key = f"{shock_market}_to_{response_market}"
                irf_results[key] = irf.irfs[:, j, i]
        
        return {
            'irfs': irf_results,
            'periods': np.arange(response_periods),
            'markets': markets
        }
        
    except Exception as e:
        logging.warning(f"Error calculating impulse response: {str(e)}")
        return {
            'irfs': {},
            'periods': np.arange(response_periods),
            'markets': markets,
            'error': str(e)
        }
    

def run_spillover_analysis(
    df_stationary: pd.DataFrame,
    arima_fits: dict = None,
    garch_fits: dict = None,
    lambda_val: float = 0.95,
    max_lag: int = 5,
    window_size: int = 60,
    forecast_horizon: int = 10,
    response_periods: int = 10,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Comprehensive analysis of spillover effects between multiple markets.
    
    This function runs the multivariate GARCH analysis and then performs
    additional spillover effect analysis between markets.
    
    Args:
        df_stationary (pd.DataFrame): The stationary time series data
        arima_fits (dict, optional): Dictionary of fitted ARIMA models
        garch_fits (dict, optional): Dictionary of fitted GARCH models
        lambda_val (float): EWMA decay factor for DCC model
        max_lag (int): Maximum lag for spillover analysis
        window_size (int): Window size for rolling spillover estimation
        forecast_horizon (int): Forecast horizon for variance decomposition
        response_periods (int): Periods for impulse response function
        significance_level (float): Significance level for hypothesis testing
        
    Returns:
        Dict[str, Any]: Dictionary containing comprehensive spillover analysis results
    """
    import itertools
    

    # Run the standard multivariate GARCH analysis first
    mvgarch_results = run_multivariate_garch(
        df_stationary=df_stationary,
        arima_fits=arima_fits,
        garch_fits=garch_fits,
        lambda_val=lambda_val
    )
    
    # Extract the results needed for spillover analysis
    arima_residuals = mvgarch_results["arima_residuals"]
    cond_vol_df = mvgarch_results["conditional_volatilities"]
    
    # Initialize spillover analysis results
    results = {
        'granger_causality': {},
        'shock_spillover': {},
    }
    
    # Get list of markets
    markets = df_stationary.columns.tolist()
    n_markets = len(markets)

    # Get list of markets
    markets = df_stationary.columns.tolist()
    safe_forecast_horizon = min(forecast_horizon, n_markets)

    # 1. Granger causality tests for all market pairs
    for market_i, market_j in itertools.permutations(markets, 2):
        pair_key = f"{market_i}_to_{market_j}"
        
        # Run Granger causality test
        results['granger_causality'][pair_key] = test_granger_causality(
            df_stationary[market_i], 
            df_stationary[market_j],
            max_lag=max_lag,
            significance_level=significance_level
        )
    
    # 2. Shock spillover analysis for all market pairs
    for market_i, market_j in itertools.permutations(markets, 2):
        pair_key = f"{market_i}_to_{market_j}"
        
        # Run shock spillover analysis
        results['shock_spillover'][pair_key] = analyze_shock_spillover(
            arima_residuals[market_i],
            cond_vol_df[market_j],
            max_lag=max_lag
        )
    
    # 3. Measure spillover magnitude and direction
    results['spillover_magnitude'] = measure_spillover_effects(
        returns_df=df_stationary,
        window_size=window_size,
        forecast_horizon=safe_forecast_horizon  # Use the adjusted horizon
    )
    safe_response_periods = min(response_periods, n_markets * 3)
    
    # 4. Calculate impulse response functions
    results['impulse_response'] = calculate_impulse_response(
        returns_df=df_stationary,
        response_periods=safe_response_periods,  # Use the adjusted periods
        method='generalized'  # This will be ignored if not supported
    )
    
    # Combine with GARCH results
    combined_results = {**mvgarch_results, 'spillover_analysis': results}
    
    return combined_results


def plot_spillover_analysis(spillover_results: Dict[str, Any], output_path: str = None):
    """
    Create visualizations of the spillover analysis results.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    import seaborn as sns
    import logging
    
    # Extract data
    spillover_magnitude = spillover_results.get('spillover_magnitude', {})
    impulse_response = spillover_results.get('impulse_response', {})
    granger_results = spillover_results.get('granger_causality', {})
    shock_spillover = spillover_results.get('shock_spillover', {})
    
    # Extract market names
    markets = spillover_magnitude.get('markets', [])
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 2, figure=fig)
    
    # ... rest of the code for first 4 charts
    
    # Fix the impulse response plotting section
    ax5 = fig.add_subplot(gs[3, :])
    
    irfs = impulse_response.get('irfs', {})
    periods = impulse_response.get('periods', np.arange(10))  # Default if missing
    
    # For clarity, only plot responses to shocks in the first market
    has_valid_irf = False
    if markets and len(markets) > 0 and irfs:
        source_market = markets[0]
        for target_market in markets:
            if target_market != source_market:
                key = f"{source_market}_to_{target_market}"
                if key in irfs:
                    response_data = irfs[key]
                    # Initialize with original periods
                    plot_periods = periods
                    
                    # Handle length mismatches
                    if len(response_data) != len(periods):
                        # Use the shorter length
                        min_len = min(len(response_data), len(periods))
                        plot_periods = periods[:min_len]
                        response_data = response_data[:min_len]
                    
                    # Now plot with matching dimensions
                    ax5.plot(plot_periods, response_data, linewidth=2, 
                             label=f'Response of {target_market} to {source_market} shock')
                    has_valid_irf = True
    
    if not has_valid_irf:
        # If no valid IRF data, add a message to the plot
        ax5.text(0.5, 0.5, "No valid impulse response data available", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax5.transAxes, fontsize=14)
    
    title_text = "Impulse Response Functions"
    if markets and len(markets) > 0:
        title_text += f" to {markets[0]} Shock"
    ax5.set_title(title_text)
    ax5.set_xlabel('Periods')
    ax5.set_ylabel('Response')
    if has_valid_irf:
        ax5.legend()
    ax5.grid(True)
    ax5.axhline(y=0, color='black', linestyle='--')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

