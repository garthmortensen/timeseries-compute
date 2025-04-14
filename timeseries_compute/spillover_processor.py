#!/usr/bin/env python3
# spillover_processor.py - Simplified version

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from timeseries_compute.stats_model import run_multivariate_garch


def test_granger_causality(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 5,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Test if series1 Granger-causes series2.

    Args:
        series1: Potential cause series
        series2: Potential effect series
        max_lag: Maximum number of lags to test
        significance_level: Threshold for significance

    Returns:
        Dictionary with causality test results
    """
    from statsmodels.tsa.stattools import grangercausalitytests

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
    residuals1: pd.Series, volatility2: pd.Series, max_lag: int = 5
) -> Dict[str, Any]:
    """
    Simplified analysis of how shocks in one market affect volatility in another.

    Args:
        residuals1: Residuals from the first market
        volatility2: Volatility of the second market
        max_lag: Maximum lag to consider

    Returns:
        Dictionary with basic spillover metrics
    """
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
    arima_fits: dict = None,
    garch_fits: dict = None,
    lambda_val: float = 0.95,
    max_lag: int = 5,
    window_size: int = 60,  # Kept for compatibility
    forecast_horizon: int = 10,  # Kept for compatibility
    response_periods: int = 10,  # Kept for compatibility
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Simplified spillover analysis between markets.

    Args:
        df_stationary: DataFrame of stationary returns
        arima_fits: Pre-fitted ARIMA models (optional)
        garch_fits: Pre-fitted GARCH models (optional)
        lambda_val: EWMA decay factor
        max_lag: Maximum lag for Granger causality
        window_size: Window size (kept for backward compatibility)
        forecast_horizon: Forecast horizon (kept for backward compatibility)
        response_periods: Response periods (kept for backward compatibility)
        significance_level: Significance threshold

    Returns:
        Dictionary with analysis results
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


def plot_spillover_analysis(
    spillover_results: Dict[str, Any], output_path: Optional[str] = None
):
    """
    Create a simple visualization of spillover analysis results.

    Args:
        spillover_results: Results from run_spillover_analysis
        output_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    # Extract causality results - handle both direct and nested structure
    if "granger_causality" in spillover_results:
        causality_results = spillover_results["granger_causality"]
    else:
        causality_results = spillover_results.get("spillover_analysis", {}).get(
            "granger_causality", {}
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count significant spillovers by source market
    market_impacts = {}
    for pair, result in causality_results.items():
        if result.get("causality", False):
            source, target = pair.split("_to_")
            if source not in market_impacts:
                market_impacts[source] = 0
            market_impacts[source] += 1

    # Create bar chart
    if market_impacts:
        markets = list(market_impacts.keys())
        counts = [market_impacts[m] for m in markets]
        ax.bar(markets, counts)
        ax.set_title("Significant Spillover Effects (Granger Causality)")
        ax.set_ylabel("Number of Markets Affected")
        ax.set_xlabel("Source Market")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(
            0.5,
            0.5,
            "No significant spillovers detected",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=14,
        )

    plt.tight_layout()

    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig
