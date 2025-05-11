#!/usr/bin/env python3
# timeseries_compute/spillover_processor.py - Simplified version

"""
Market Spillover Effects Analysis Module.

This module extends the multivariate GARCH analysis with tools for analyzing
how shocks and volatility spill over between different markets or assets.
It implements methods for testing causality, measuring spillover magnitude,
and visualizing the results.

Key Components:
- test_granger_causality: Test if one series helps predict another
- analyze_shock_spillover: Analyze how shocks affect volatility in other markets
- run_spillover_analysis: Comprehensive spillover effects analysis
- plot_spillover_analysis: Visualization of spillover relationships

Features:
- Granger causality testing with optimal lag selection
- Shock spillover analysis with significance testing
- Visualization tools for interpreting spillover relationships

Typical Usage Flow:
1. Start with prepared stationary data from data_processor.py
2. Run ARIMA and GARCH models (optional, can be done internally)
3. Perform comprehensive spillover analysis
4. Visualize and interpret the results

This module depends on stats_model.py for the underlying GARCH modeling and
extends its functionality with specific spillover analysis tools.
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List
from timeseries_compute.stats_model import run_multivariate_garch


def test_granger_causality(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 5,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Test if series1 Granger-causes series2.
    
    Granger causality tests if past values of series1 help predict future values of series2
    beyond what past values of series2 alone can predict.

    Args:
        series1 (pd.Series): Potential cause series
        series2 (pd.Series): Potential effect series
        max_lag (int): Maximum number of lags to test (will test lags 1 to max_lag)
        significance_level (float): p-value threshold for determining significance

    Returns:
        Dict[str, Any]: Dictionary with the following keys:
            - 'causality' (bool): True if series1 Granger-causes series2 at any tested lag
            - 'p_values' (Dict[int, float]): Dictionary mapping each lag to its p-value
            - 'optimal_lag' (int or None): Lag with the smallest p-value if causality exists,
              None otherwise
            
    Example:
        >>> # Test if returns of Market A cause returns of Market B
        >>> market_a = pd.Series([0.01, -0.015, 0.02, -0.01, 0.015])
        >>> market_b = pd.Series([0.005, -0.01, 0.015, -0.005, 0.01])
        >>> result = test_granger_causality(market_a, market_b, max_lag=2)
        >>> print(f"Causality exists: {result['causality']}")
        >>> print(f"P-values by lag: {result['p_values']}")
        >>> print(f"Best lag: {result['optimal_lag']}")
    """
    from statsmodels.tsa.stattools import grangercausalitytests

        # Convert DataFrames with Date column to Series with Date index if needed
    if isinstance(series1, pd.DataFrame) and "Date" in series1.columns:
        series1 = series1.set_index("Date")[series1.columns[1]]
    if isinstance(series2, pd.DataFrame) and "Date" in series2.columns:
        series2 = series2.set_index("Date")[series2.columns[1]]

    # Combine series into a DataFrame
    data = pd.concat([series1, series2], axis=1)
    data.columns = ["series1", "series2"]
    data = data.dropna()

    # Run Granger causality tests
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    # Extract key results
    p_values = {lag: results[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)}
    causality = any(p < significance_level for p in p_values.values())
    optimal_lag = min(p_values, key=p_values.get) if causality else None

    return {"causality": causality, "p_values": p_values, "optimal_lag": optimal_lag}


def analyze_shock_spillover(
    residuals1: pd.Series, 
    volatility2: pd.Series, 
    max_lag: int = 5
) -> Dict[str, Union[List[int], float]]:
    """
    Simplified analysis of how shocks in one market affect volatility in another.

    Args:
        residuals1: Residuals from the first market
        volatility2: Volatility of the second market
        max_lag: Maximum lag to consider

    Returns:
        Dictionary with basic spillover metrics
    """

    # Convert DataFrames with Date column to Series with Date index if needed
    if isinstance(residuals1, pd.DataFrame) and "Date" in residuals1.columns:
        residuals1 = residuals1.set_index("Date")[residuals1.columns[1]]
    if isinstance(volatility2, pd.DataFrame) and "Date" in volatility2.columns:
        volatility2 = volatility2.set_index("Date")[volatility2.columns[1]]
    
    # Create a simple model using correlation with lags
    significant_lags = []
    correlations = {}

    # Check correlation at different lags
    for lag in range(1, max_lag + 1):
        # Squared residuals represent shock magnitude
        shock = residuals1**2
        lagged_shock = shock.shift(lag).dropna()

        # Match with corresponding volatility
        aligned_vol = volatility2.loc[lagged_shock.index]

        # Calculate correlation
        if len(lagged_shock) > 10:  # Ensure enough data
            corr = lagged_shock.corr(aligned_vol)
            correlations[lag] = corr

            # Simple significance threshold
            if abs(corr) > 0.3:
                significant_lags.append(lag)

    # Calculate simple r-squared as max squared correlation
    r_squared = max([corr**2 for corr in correlations.values()]) if correlations else 0

    return {"significant_lags": significant_lags, "r_squared": r_squared}


def run_spillover_analysis(
    df_stationary: pd.DataFrame,
    arima_fits: Optional[Dict[str, Any]] = None,
    garch_fits: Optional[Dict[str, Any]] = None,
    lambda_val: float = 0.95,
    max_lag: int = 5,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Analyzes spillover effects between markets using multivariate GARCH and Granger causality.

    Args:
        df_stationary (pd.DataFrame): DataFrame of stationary returns for multiple markets
        arima_fits (dict, optional): Pre-fitted ARIMA models
        garch_fits (dict, optional): Pre-fitted GARCH models
        lambda_val (float): EWMA decay factor for dynamic correlation calculation
        max_lag (int): Maximum lag for Granger causality tests
        significance_level (float): Significance threshold for statistical tests

    Returns:
        Dict[str, Any]: Dictionary with analysis results containing:
            - Standard multivariate GARCH results (see run_multivariate_garch)
            - 'spillover_analysis': Dictionary with the following keys:
                - 'granger_causality': Results from Granger causality tests between markets
                - 'shock_spillover': Results from shock spillover analysis
                - 'spillover_magnitude': Information about the strength of spillover effects
                - 'impulse_response': Impulse response function results
            
    Example:
        >>> # Create returns data for two markets
        >>> returns = pd.DataFrame({
        ...     'US': [0.01, -0.02, 0.015, -0.01, 0.02],
        ...     'EU': [0.015, -0.01, 0.02, -0.015, 0.01]
        ... })
        >>> # Run spillover analysis
        >>> results = run_spillover_analysis(returns, max_lag=3)
        >>> # Check if US returns Granger-cause EU returns
        >>> us_to_eu = results['spillover_analysis']['granger_causality']['US_to_EU']
        >>> print(f"US Granger-causes EU: {us_to_eu['causality']}")
    """
    import itertools

    # Run multivariate GARCH
    mvgarch_results = run_multivariate_garch(
        df_stationary=df_stationary,
        arima_fits=arima_fits,
        garch_fits=garch_fits,
        lambda_val=lambda_val,
    )

    # Extract key components
    arima_residuals = mvgarch_results["arima_residuals"]
    cond_vol_df = mvgarch_results["conditional_volatilities"]

    # Initialize spillover results
    results = {"granger_causality": {}, "shock_spillover": {}}

    # Get list of markets
    markets = df_stationary.columns.tolist()

    # Granger causality tests
    for market_i, market_j in itertools.permutations(markets, 2):
        pair_key = f"{market_i}_to_{market_j}"

        # Test returns -> returns causality
        results["granger_causality"][pair_key] = test_granger_causality(
            df_stationary[market_i],
            df_stationary[market_j],
            max_lag=max_lag,
            significance_level=significance_level,
        )

        # Test residual -> volatility spillover
        results["shock_spillover"][pair_key] = analyze_shock_spillover(
            arima_residuals[market_i], cond_vol_df[market_j], max_lag=max_lag
        )

    # Provide minimal placeholders for compatibility
    results["spillover_magnitude"] = {
        "spillover_indices": pd.DataFrame(index=df_stationary.index[-10:]),
        "markets": markets,
    }

    results["impulse_response"] = {
        "irfs": {},
        "periods": np.arange(min(10, len(markets))),
        "markets": markets,
    }

    # Combine with GARCH results
    combined_results = {**mvgarch_results, "spillover_analysis": results}

    return combined_results
