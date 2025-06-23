#!/usr/bin/env python3
# timeseries_compute/spillover_processor.py - Standard Diebold-Yilmaz Implementation

"""
Market Spillover Effects Analysis Module - Standard Diebold-Yilmaz Implementation.

This module implements the standard Diebold-Yilmaz (2012) methodology for measuring
spillover effects between financial markets using Vector Autoregression (VAR) models
and Forecast Error Variance Decomposition (FEVD).

Key Components:
- fit_var_model: Fit Vector Autoregression model to returns data
- calculate_fevd: Calculate Forecast Error Variance Decomposition
- calculate_spillover_index: Calculate Total Connectedness Index and directional spillovers
- run_diebold_yilmaz_analysis: Complete Diebold-Yilmaz spillover analysis
- test_granger_causality: Granger causality testing (supplementary)

The methodology follows these steps:
1. Fit VAR model to stationary returns
2. Calculate FEVD at specified horizon
3. Compute spillover indices from FEVD matrix
4. Extract directional and net spillovers

References:
- Diebold, F.X. & Yilmaz, K. (2012). Better to Give than to Receive: 
  Predictive Directional Measurement of Volatility Spillovers. 
  International Journal of Forecasting, 28(1), 57-66.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

from timeseries_compute.stats_model import fit_arima_model, fit_garch_model, fit_dcc_garch_model

logger = logging.getLogger(__name__)


def fit_var_model(
    returns_df: pd.DataFrame,
    max_lags: int = 5,
    ic: str = 'aic'
) -> Tuple[Any, int]:
    """
    Fit Vector Autoregression (VAR) model to returns data.
    
    Args:
        returns_df: DataFrame of stationary returns for multiple assets
        max_lags: Maximum number of lags to consider
        ic: Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
        
    Returns:
        Tuple of (fitted VAR model, selected lag order)
        
    Example:
        >>> returns = pd.DataFrame({'AAPL': [0.01, -0.02], 'MSFT': [0.015, -0.01]})
        >>> model, lag = fit_var_model(returns, max_lags=3)
        >>> print(f"Selected lag order: {lag}")
    """
    logger.info(f"Fitting VAR model with max_lags={max_lags}, ic={ic}")
    
    # Remove any NaN values
    clean_data = returns_df.dropna()
    
    if len(clean_data) < max_lags + 10:
        raise ValueError(f"Insufficient data: need at least {max_lags + 10} observations, got {len(clean_data)}")
    
    # Initialize VAR model
    var_model = VAR(clean_data)
    
    # Try automatic lag selection first
    try:
        fitted_model = var_model.fit(maxlags=max_lags, ic=ic)
        selected_lag = fitted_model.k_ar
        
        # If lag selection returns 0, force at least 1 lag for FEVD to work
        if selected_lag == 0:
            logger.warning("VAR lag selection returned 0 lags, forcing lag=1 for spillover analysis")
            fitted_model = var_model.fit(maxlags=1, ic=None)
            selected_lag = 1
            
    except Exception as e:
        logger.warning(f"VAR automatic lag selection failed: {e}, trying fixed lag=1")
        fitted_model = var_model.fit(maxlags=1, ic=None)
        selected_lag = 1
    
    logger.info(f"VAR model fitted with {selected_lag} lags")
    
    return fitted_model, selected_lag


def calculate_fevd(
    var_model: Any,
    horizon: int = 10,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate Forecast Error Variance Decomposition (FEVD) from fitted VAR model.
    
    Args:
        var_model: Fitted VAR model from statsmodels
        horizon: Forecast horizon for variance decomposition
        normalize: Whether to normalize FEVD to sum to 100% (recommended: True)
        
    Returns:
        FEVD matrix of shape (n_variables, n_variables) where entry (i,j) represents
        the percentage of forecast error variance of variable i explained by shocks to variable j
        
    Example:
        >>> # After fitting VAR model
        >>> fevd_matrix = calculate_fevd(fitted_var, horizon=10)
        >>> print(f"FEVD shape: {fevd_matrix.shape}")
        >>> print(f"Row sums: {fevd_matrix.sum(axis=1)}")  # Should be ~100 if normalized
    """
    logger.info(f"Calculating FEVD with horizon={horizon}")
    
    # Calculate FEVD using statsmodels
    fevd_result = var_model.fevd(horizon)
    
    # Extract the FEVD matrix for the final horizon
    # fevd_result.decomp has shape (n_variables, n_variables, horizon)
    # We want the final horizon (index -1)
    fevd_matrix = fevd_result.decomp[:, :, -1]  # Take the final horizon
    
    # Ensure the matrix is square (n_variables x n_variables)
    n_vars = len(var_model.names)
    if fevd_matrix.shape != (n_vars, n_vars):
        logger.error(f"FEVD matrix shape mismatch: got {fevd_matrix.shape}, expected ({n_vars}, {n_vars})")
        # If there's a shape issue, take only the square portion
        fevd_matrix = fevd_matrix[:n_vars, :n_vars]
    
    if normalize:
        # Ensure each row sums to 100% (sometimes there are small numerical errors)
        row_sums = fevd_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        fevd_matrix = (fevd_matrix / row_sums) * 100
    
    logger.info(f"FEVD matrix calculated, shape: {fevd_matrix.shape}")
    
    return fevd_matrix


def calculate_spillover_index(
    fevd_matrix: np.ndarray,
    variable_names: list
) -> Dict[str, Any]:
    """
    Calculate Diebold-Yilmaz spillover indices from FEVD matrix.
    
    Args:
        fevd_matrix: FEVD matrix from calculate_fevd()
        variable_names: Names of the variables (assets)
        
    Returns:
        Dictionary containing:
        - total_spillover_index: Total Connectedness Index (TCI) in percentage
        - directional_spillovers: Dict with 'to' and 'from' spillovers for each variable
        - net_spillovers: Net spillover for each variable (to - from)
        - pairwise_spillovers: Matrix of pairwise spillovers
        - fevd_table: FEVD table as DataFrame for inspection
        
    Example:
        >>> fevd = np.array([[80, 15, 5], [10, 75, 15], [5, 20, 75]])
        >>> names = ['AAPL', 'MSFT', 'TSLA']
        >>> spillovers = calculate_spillover_index(fevd, names)
        >>> print(f"Total spillover: {spillovers['total_spillover_index']:.1f}%")
    """
    logger.info("Calculating Diebold-Yilmaz spillover indices")
    
    n_vars = len(variable_names)
    
    # Create FEVD DataFrame for easier interpretation
    fevd_df = pd.DataFrame(
        fevd_matrix, 
        index=variable_names, 
        columns=variable_names
    )
    
    # 1. Total Connectedness Index (TCI)
    # Sum of off-diagonal elements divided by total sum, times 100
    off_diagonal_sum = fevd_matrix.sum() - np.trace(fevd_matrix)
    total_sum = fevd_matrix.sum()
    total_spillover_index = (off_diagonal_sum / total_sum) * 100
    
    # 2. Directional Spillovers
    directional_spillovers = {}
    
    for i, var_name in enumerate(variable_names):
        # "To" spillover: how much this variable contributes to others' variance
        # Sum of column i, excluding diagonal element
        to_spillover = (fevd_matrix[:, i].sum() - fevd_matrix[i, i]) / total_sum * 100
        
        # "From" spillover: how much this variable receives from others
        # Sum of row i, excluding diagonal element  
        from_spillover = (fevd_matrix[i, :].sum() - fevd_matrix[i, i]) / total_sum * 100
        
        directional_spillovers[var_name] = {
            'to': to_spillover,
            'from': from_spillover
        }
    
    # 3. Net Spillovers (To - From)
    net_spillovers = {}
    for var_name in variable_names:
        net_spillovers[var_name] = (
            directional_spillovers[var_name]['to'] - 
            directional_spillovers[var_name]['from']
        )
    
    # 4. Pairwise Spillovers (off-diagonal elements as percentages of total)
    pairwise_spillovers = pd.DataFrame(
        fevd_matrix / total_sum * 100,
        index=variable_names,
        columns=variable_names
    )
    # Zero out diagonal for pairwise interpretation
    np.fill_diagonal(pairwise_spillovers.values, 0)
    
    results = {
        'total_spillover_index': total_spillover_index,
        'directional_spillovers': directional_spillovers,
        'net_spillovers': net_spillovers,
        'pairwise_spillovers': pairwise_spillovers,
        'fevd_table': fevd_df
    }
    
    logger.info(f"Spillover analysis complete. TCI: {total_spillover_index:.2f}%")
    
    return results


def test_granger_causality(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 5,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Test if series1 Granger-causes series2.
    
    This is a supplementary analysis to the main Diebold-Yilmaz methodology.
    Now includes multi-level significance testing (1% and 5%) like stationarity tests.
    
    Args:
        series1: Potential cause series
        series2: Potential effect series  
        max_lag: Maximum number of lags to test
        significance_level: p-value threshold for significance (kept for backward compatibility)
        
    Returns:
        Dictionary with causality test results including multi-level significance
    """
    # Combine series into DataFrame
    data = pd.concat([series2, series1], axis=1)  # Note: order matters for grangercausalitytests
    data.columns = ['target', 'cause']
    data = data.dropna()
    
    if len(data) < max_lag + 10:
        return {
            'causality': False,
            'causality_1pct': False,
            'causality_5pct': False,
            'p_values': {},
            'optimal_lag': None,
            'optimal_lag_1pct': None,
            'optimal_lag_5pct': None,
            'error': 'Insufficient data for Granger causality test'
        }
    
    try:
        # Run Granger causality tests
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        
        # Extract p-values
        p_values = {lag: results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)}
        
        # Multi-level significance testing (like stationarity tests)
        causality_1pct = any(p < 0.01 for p in p_values.values())  # 1% significance
        causality_5pct = any(p < 0.05 for p in p_values.values())  # 5% significance
        
        # Optimal lags for each significance level
        optimal_lag_1pct = min((lag for lag, p in p_values.items() if p < 0.01), default=None)
        optimal_lag_5pct = min((lag for lag, p in p_values.items() if p < 0.05), default=None)
        
        return {
            'causality_1pct': causality_1pct,
            'causality_5pct': causality_5pct,
            'p_values': p_values,
            'optimal_lag_1pct': optimal_lag_1pct,
            'optimal_lag_5pct': optimal_lag_5pct,
            'significance_summary': {
                'significant_at_1pct': causality_1pct,
                'significant_at_5pct': causality_5pct,
                'min_p_value': min(p_values.values()) if p_values else 1.0
            }
        }
        
    except Exception as e:
        logger.warning(f"Granger causality test failed: {e}")
        return {
            'causality': False,
            'causality_1pct': False,
            'causality_5pct': False,
            'p_values': {},
            'optimal_lag': None,
            'optimal_lag_1pct': None,
            'optimal_lag_5pct': None,
            'error': str(e)
        }


def run_diebold_yilmaz_analysis(
    returns_df: pd.DataFrame,
    horizon: int = 10,
    max_lags: int = 5,
    ic: str = 'aic',
    include_granger: bool = True,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Complete Diebold-Yilmaz spillover analysis.
    
    This is the main function that implements the standard Diebold-Yilmaz methodology:
    1. Fit VAR model to returns
    2. Calculate FEVD
    3. Compute spillover indices
    4. Optionally include Granger causality tests
    
    Args:
        returns_df: DataFrame of stationary returns for multiple assets
        horizon: Forecast horizon for FEVD calculation  
        max_lags: Maximum lags for VAR model selection
        ic: Information criterion for VAR lag selection
        include_granger: Whether to include Granger causality tests
        significance_level: Significance level for Granger tests
        
    Returns:
        Dictionary containing complete spillover analysis results:
        - var_model: Fitted VAR model
        - var_lag: Selected VAR lag order
        - fevd_matrix: FEVD matrix
        - spillover_results: All spillover indices and measures
        - granger_causality: Granger causality test results (if requested)
        
    Example:
        >>> returns = pd.DataFrame({
        ...     'AAPL': np.random.normal(0, 0.02, 100),
        ...     'MSFT': np.random.normal(0, 0.02, 100),
        ...     'TSLA': np.random.normal(0, 0.03, 100)
        ... })
        >>> results = run_diebold_yilmaz_analysis(returns, horizon=10)
        >>> print(f"Total spillover: {results['spillover_results']['total_spillover_index']:.1f}%")
    """
    logger.info("Starting Diebold-Yilmaz spillover analysis")
    
    # Validate input
    if returns_df.empty:
        raise ValueError("Empty returns DataFrame provided")
    
    if len(returns_df.columns) < 2:
        raise ValueError("Need at least 2 assets for spillover analysis")
    
    # Step 1: Fit VAR model
    try:
        var_model, selected_lag = fit_var_model(returns_df, max_lags=max_lags, ic=ic)
    except Exception as e:
        raise RuntimeError(f"Failed to fit VAR model: {e}")
    
    # Step 2: Calculate FEVD
    try:
        fevd_matrix = calculate_fevd(var_model, horizon=horizon)
    except Exception as e:
        raise RuntimeError(f"Failed to calculate FEVD: {e}")
    
    # Step 3: Calculate spillover indices
    try:
        spillover_results = calculate_spillover_index(fevd_matrix, returns_df.columns.tolist())
    except Exception as e:
        raise RuntimeError(f"Failed to calculate spillover indices: {e}")
    
    # Step 4: Granger causality tests (optional)
    granger_results = {}
    if include_granger:
        logger.info("Performing Granger causality tests")
        asset_names = returns_df.columns.tolist()
        
        for i, asset_i in enumerate(asset_names):
            for j, asset_j in enumerate(asset_names):
                if i != j:  # Skip self-causality
                    pair_key = f"{asset_i}_to_{asset_j}"
                    try:
                        granger_results[pair_key] = test_granger_causality(
                            returns_df[asset_i],
                            returns_df[asset_j],
                            max_lag=max_lags,
                            significance_level=significance_level
                        )
                    except Exception as e:
                        logger.warning(f"Granger test failed for {pair_key}: {e}")
                        granger_results[pair_key] = {
                            'causality': False,
                            'error': str(e)
                        }
    
    # Compile final results
    final_results = {
        'var_model': var_model,
        'var_lag': selected_lag,
        'fevd_matrix': fevd_matrix,
        'spillover_results': spillover_results,
        'granger_causality': granger_results if include_granger else {},
        'metadata': {
            'horizon': horizon,
            'max_lags': max_lags,
            'ic': ic,
            'n_assets': len(returns_df.columns),
            'n_observations': len(returns_df),
            'asset_names': returns_df.columns.tolist()
        }
    }
    
    logger.info("Diebold-Yilmaz analysis completed successfully")
    
    return final_results
