#!/usr/bin/env python3
# stats_model.py

import logging as l

# handle data transformation and preparation tasks
import numpy as np
import pandas as pd

# import model specific libraries
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# type hinting
from typing import Dict, Any, Tuple
from generalized_timeseries import data_processor
from generalized_timeseries.data_processor import (
    calculate_ewma_covariance,
    calculate_ewma_volatility,
)


class ModelARIMA:
    """
    Applies the ARIMA (AutoRegressive Integrated Moving Average) model on all columns of a DataFrame.

    Attributes:
        data (pd.DataFrame): The input data on which ARIMA models will be applied.
        order (Tuple[int, int, int]): The (p, d, q) order of the ARIMA model.
        steps (int): The number of steps to forecast.
        models (Dict[str, ARIMA]): A dictionary to store ARIMA models for each column.
        fits (Dict[str, ARIMA]): A dictionary to store fitted ARIMA models for each column.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        order: Tuple[int, int, int] = (1, 1, 1),
        steps: int = 5,
    ) -> None:
        """
        Initializes the ARIMA model with the given data, order, and steps.

        Args:
            data (pd.DataFrame): The input data for the ARIMA model.
            order (Tuple[int, int, int]): The (p, d, q) order of the ARIMA model.
            steps (int): The number of steps to forecast.
        """
        ascii_banner = """
        \n
        \t> ARIMA <\n"""
        l.info(ascii_banner)
        self.data = data
        self.order = order
        self.steps = steps
        self.models: Dict[str, ARIMA] = {}  # Store models for each column
        self.fits: Dict[str, ARIMA] = {}  # Store fits for each column

    def fit(self) -> Dict[str, ARIMA]:
        """
        Fits an ARIMA model to each column in the dataset.

        Returns:
            Dict[str, ARIMA]: A dictionary where the keys are column names and the values are the
                fitted ARIMA models for each column.
        """
        for column in self.data.columns:
            model = ARIMA(self.data[column], order=self.order)
            self.fits[column] = model.fit()
        return self.fits

    def summary(self) -> Dict[str, str]:
        """
        Returns the model summaries for all columns.

        Returns:
            Dict[str, str]: A dictionary containing the model summaries for each column.
        """
        summaries = {}
        for column, fit in self.fits.items():
            summaries[column] = str(fit.summary())
        return summaries

    def forecast(self) -> Dict[str, float]:
        """
        Generates forecasts for each fitted model.

        Returns:
            Dict[str, float]: A dictionary where the keys are the column names and the values
                are the forecasted values for the first step.
        """
        forecasts = {}
        for column, fit in self.fits.items():
            forecasts[column] = fit.forecast(steps=self.steps).iloc[0]
        return forecasts


def run_arima(
    df_stationary: pd.DataFrame,
    p: int = 1,
    d: int = 1,
    q: int = 1,
    forecast_steps: int = 5,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    """
    Runs an ARIMA model on stationary time series data.

    This function fits ARIMA(p,d,q) models to each column in the provided DataFrame
    and generates forecasts for the specified number of steps ahead. It performs minimal
    logging to display only core information about the model and forecasts.

    Args:
        df_stationary (pd.DataFrame): The DataFrame with stationary time series data
        p (int): Autoregressive lag order, default=1
        d (int): Degree of differencing, default=1
        q (int): Moving average lag order, default=1
        forecast_steps (int): Number of steps to forecast, default=5

    Returns:
        Tuple[Dict[str, object], Dict[str, float]]:
            - First element: Dictionary of fitted ARIMA models for each column
            - Second element: Dictionary of forecasted values for each column
    """
    l.info(f"\n## Running ARIMA(p={p}, d={d}, q={q})")

    # Ensure data is properly prepared
    df_stationary = data_processor.prepare_timeseries_data(df_stationary)

    model_arima = ModelFactory.create_model(
        model_type="ARIMA",
        data=df_stationary,
        order=(p, d, q),
        steps=forecast_steps,
    )
    arima_fit = model_arima.fit()

    # Log only core model information instead of full summary
    l.info(f"## ARIMA model fitted to columns: {list(arima_fit.keys())}")

    # Generate and log forecast values concisely
    arima_forecast = model_arima.forecast()
    l.info(f"## ARIMA {forecast_steps}-step forecast values:")
    for col, value in arima_forecast.items():
        l.info(f"   {col}: {value:.4f}")

    return arima_fit, arima_forecast


class ModelGARCH:
    """
    Represents a GARCH model for time series data.

    Attributes:
        data (pd.DataFrame): The input time series data.
        p (int): The order of the GARCH model for the lag of the squared residuals.
        q (int): The order of the GARCH model for the lag of the conditional variance.
        dist (str): The distribution to use for the GARCH model (e.g., 'normal', 't').
        models (Dict[str, arch_model]): A dictionary to store models for each column of the data.
        fits (Dict[str, arch_model]): A dictionary to store fitted models for each column of the data.
    """

    def __init__(
        self, data: pd.DataFrame, p: int = 1, q: int = 1, dist: str = "normal"
    ) -> None:
        """
        Initializes the GARCH model with the given parameters.

        Args:
            data (pd.DataFrame): The input data for the GARCH model.
            p (int): The order of the GARCH model.
            q (int): The order of the ARCH model.
            dist (str): The distribution to be used in the model (e.g., 'normal', 't').
        """
        ascii_banner = """
        \n\t> GARCH <\n"""
        l.info(ascii_banner)
        self.data = data
        self.p = p
        self.q = q
        self.dist = dist
        self.models: Dict[str, arch_model] = {}  # Store models for each column
        self.fits: Dict[str, arch_model] = {}  # Store fits for each column

    def fit(self) -> Dict[str, arch_model]:
        """
        Fits a GARCH model to each column of the data.

        Returns:
            Dict[str, arch_model]: A dictionary where the keys are column names and the values
                are the fitted GARCH models.
        """
        for column in self.data.columns:
            model = arch_model(
                self.data[column], vol="Garch", p=self.p, q=self.q, dist=self.dist
            )
            self.fits[column] = model.fit(disp="off")
        return self.fits

    def summary(self) -> Dict[str, str]:
        """
        Returns the model summaries for all columns.

        Returns:
            Dict[str, str]: A dictionary containing the model summaries for each column.
        """
        summaries = {}
        for column, fit in self.fits.items():
            summaries[column] = str(fit.summary())
        return summaries

    def forecast(self, steps: int) -> Dict[str, float]:
        """
        Generates forecasted variance for each fitted model.

        Args:
            steps (int): The number of steps ahead to forecast.

        Returns:
            Dict[str, float]: A dictionary where keys are column names and values are the forecasted variances for the specified horizon.
        """
        forecasts = {}
        for column, fit in self.fits.items():
            forecasts[column] = fit.forecast(horizon=steps).variance.iloc[-1]
        return forecasts


class ModelMultivariateGARCH:
    """Implements multivariate GARCH models including CC-GARCH and DCC-GARCH."""

    def __init__(
        self, data: pd.DataFrame, p: int = 1, q: int = 1, model_type: str = "cc"
    ):
        """
        Initialize multivariate GARCH model.

        Args:
            data: DataFrame with multiple time series
            p: GARCH order
            q: ARCH order
            model_type: 'cc' for Constant Correlation or 'dcc' for Dynamic Conditional Correlation
        """
        self.data = data
        self.p = p
        self.q = q
        self.model_type = model_type
        self.fits = {}

    def fit_cc_garch(self) -> Dict[str, Any]:
        """Fit Constant Conditional Correlation GARCH model."""
        # First fit univariate GARCH models
        univariate_models = {}
        for column in self.data.columns:
            model = arch_model(self.data[column], vol="Garch", p=self.p, q=self.q)
            univariate_models[column] = model.fit(disp="off")

        # Calculate constant correlation matrix
        residuals = pd.DataFrame()
        for column in self.data.columns:
            residuals[column] = univariate_models[column].resid

        correlation_matrix = residuals.corr()

        # Store results
        self.cc_results = {
            "univariate_models": univariate_models,
            "correlation": correlation_matrix,
        }
        return self.cc_results

    def fit_dcc_garch(self, lambda_val: float = 0.95):
        """
        Fit Dynamic Conditional Correlation GARCH model using EWMA for correlation.

        Args:
            lambda_val: EWMA decay factor

        Returns:
            Dictionary with DCC-GARCH results
        """
        # Fit univariate GARCH models
        univariate_models = {}
        conditional_vols = pd.DataFrame(index=self.data.index)

        for column in self.data.columns:
            model = arch_model(self.data[column], vol="Garch", p=self.p, q=self.q)
            fit = model.fit(disp="off")
            univariate_models[column] = fit
            conditional_vols[column] = np.sqrt(fit.conditional_volatility)

        # Calculate standardized residuals
        std_residuals = pd.DataFrame(index=self.data.index)
        for column in self.data.columns:
            std_residuals[column] = self.data[column] / conditional_vols[column]

        # Calculate EWMA correlation for all pairs
        correlations = {}
        columns = self.data.columns
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col_pair = f"{columns[i]}_{columns[j]}"
                correlations[col_pair] = calculate_ewma_covariance(
                    std_residuals[columns[i]], std_residuals[columns[j]], lambda_val
                )

        self.dcc_results = {
            "univariate_models": univariate_models,
            "conditional_vols": conditional_vols,
            "correlations": correlations,
        }

        return self.dcc_results


# These functions should be outside the class
def calculate_correlation_matrix(standardized_residuals: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate constant conditional correlation matrix from standardized residuals.

    Args:
        standardized_residuals (pd.DataFrame): DataFrame of standardized residuals from GARCH models

    Returns:
        pd.DataFrame: Correlation matrix
    """
    return standardized_residuals.corr()


def calculate_dynamic_correlation(
    ewma_cov: pd.Series, ewma_vol1: pd.Series, ewma_vol2: pd.Series
) -> pd.Series:
    """
    Calculate dynamic conditional correlation from EWMA covariance and volatilities.

    Args:
        ewma_cov (pd.Series): EWMA covariance between two series
        ewma_vol1 (pd.Series): EWMA volatility of first series
        ewma_vol2 (pd.Series): EWMA volatility of second series

    Returns:
        pd.Series: Dynamic conditional correlation
    """
    return ewma_cov / (ewma_vol1 * ewma_vol2)


def construct_covariance_matrix(volatilities: list, correlation: float) -> np.ndarray:
    """
    Construct a 2x2 covariance matrix using volatilities and correlation.

    Args:
        volatilities (list): List of volatilities [vol1, vol2]
        correlation (float): Correlation coefficient

    Returns:
        np.ndarray: 2x2 covariance matrix
    """
    cov_matrix = np.outer(volatilities, volatilities)
    cov_matrix[0, 1] *= correlation
    cov_matrix[1, 0] *= correlation
    return cov_matrix


def calculate_portfolio_risk(weights: np.ndarray, cov_matrix: np.ndarray) -> tuple:
    """
    Calculate portfolio variance and volatility for given weights and covariance matrix.

    Args:
        weights (np.ndarray): Array of portfolio weights
        cov_matrix (np.ndarray): Covariance matrix

    Returns:
        tuple: (portfolio_variance, portfolio_volatility)
    """
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    return portfolio_variance, portfolio_volatility


def run_multivariate_garch(
    df_stationary: pd.DataFrame,
    arima_fits: dict = None,
    garch_fits: dict = None,
    lambda_val: float = 0.95,
) -> dict:
    """
    Runs multivariate GARCH analysis on the provided stationary DataFrame.

    This function implements both Constant Conditional Correlation (CCC) and
    Dynamic Conditional Correlation (DCC) GARCH models. It either uses provided
    ARIMA and GARCH models or fits new ones if not provided.

    Args:
        df_stationary (pd.DataFrame): The stationary time series data for GARCH modeling
        arima_fits (dict, optional): Dictionary of fitted ARIMA models for each column
        garch_fits (dict, optional): Dictionary of fitted GARCH models for each column
        lambda_val (float, optional): EWMA decay factor for DCC model. Defaults to 0.95.

    Returns:
        dict: Dictionary containing multivariate GARCH results
            - 'arima_residuals': DataFrame of ARIMA residuals
            - 'conditional_volatilities': DataFrame of conditional volatilities
            - 'standardized_residuals': DataFrame of standardized residuals
            - 'cc_correlation': Constant conditional correlation matrix
            - 'cc_covariance_matrix': Covariance matrix using CCC
            - 'dcc_correlation': Series of dynamic conditional correlations
            - 'dcc_covariance': Series of dynamic conditional covariances
    """
    results = {}

    # 1. If ARIMA fits not provided, fit ARIMA models to filter out conditional mean
    if arima_fits is None:
        arima_fits, _ = run_arima(
            df_stationary=df_stationary, p=1, d=0, q=1, forecast_steps=1
        )

    # 2. Extract ARIMA residuals
    arima_residuals = pd.DataFrame(index=df_stationary.index)
    for column in df_stationary.columns:
        if hasattr(arima_fits[column], "resid"):
            arima_residuals[column] = arima_fits[column].resid
        else:
            # If no residuals available, use original series
            arima_residuals[column] = df_stationary[column]

    results["arima_residuals"] = arima_residuals

    # 3. If GARCH fits not provided, fit GARCH models
    if garch_fits is None:
        garch_fits, _ = run_garch(
            df_stationary=arima_residuals, p=1, q=1, forecast_steps=1
        )

    # 4. Extract conditional volatilities
    cond_vol = {}
    for column in arima_residuals.columns:
        cond_vol[column] = np.sqrt(garch_fits[column].conditional_volatility)

    cond_vol_df = pd.DataFrame(cond_vol, index=arima_residuals.index)
    results["conditional_volatilities"] = cond_vol_df

    # 5. Calculate standardized residuals
    std_resid = {}
    for column in arima_residuals.columns:
        std_resid[column] = arima_residuals[column] / cond_vol[column]

    std_resid_df = pd.DataFrame(std_resid, index=arima_residuals.index)
    results["standardized_residuals"] = std_resid_df

    # 6. Constant Conditional Correlation (CCC-GARCH)
    cc_corr = calculate_correlation_matrix(std_resid_df)
    results["cc_correlation"] = cc_corr

    # 7. Get latest volatilities for covariance matrix
    if len(arima_residuals.columns) == 2:
        columns = list(arima_residuals.columns)
        latest_vols = [cond_vol[col].iloc[-1] for col in columns]

        # Construct covariance matrix using CCC
        cc_cov_matrix = construct_covariance_matrix(
            volatilities=latest_vols, correlation=cc_corr.iloc[0, 1]
        )
        results["cc_covariance_matrix"] = cc_cov_matrix

    # 8. Dynamic Conditional Correlation (DCC-GARCH)
    if len(arima_residuals.columns) == 2:
        columns = list(std_resid_df.columns)

        # Calculate EWMA covariance
        ewma_cov = calculate_ewma_covariance(
            std_resid_df[columns[0]], std_resid_df[columns[1]], lambda_val=lambda_val
        )

        # Calculate EWMA volatilities for standardized residuals
        ewma_vol1 = calculate_ewma_volatility(
            std_resid_df[columns[0]], lambda_val=lambda_val
        )

        ewma_vol2 = calculate_ewma_volatility(
            std_resid_df[columns[1]], lambda_val=lambda_val
        )

        # Calculate dynamic correlation
        dcc_corr = calculate_dynamic_correlation(ewma_cov, ewma_vol1, ewma_vol2)
        results["dcc_correlation"] = dcc_corr

        # Calculate dynamic covariance
        dcc_cov = dcc_corr * (cond_vol_df[columns[0]] * cond_vol_df[columns[1]])
        results["dcc_covariance"] = dcc_cov

    return results


class ModelFactory:
    """
    Factory class for creating instances of different statistical models.

    Methods:
        create_model(model_type: str, **kwargs) -> Any:
            Static method that creates and returns an instance of a model based on the provided model_type.
    """

    @staticmethod
    def create_model(
        model_type: str,
        data: pd.DataFrame,
        # ARIMA parameters with defaults
        order: Tuple[int, int, int] = (1, 1, 1),
        steps: int = 5,
        # GARCH parameters with defaults
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        # Multivariate GARCH parameters
        mv_model_type: str = "cc",
    ) -> Any:
        l.info(f"Creating model type: {model_type}")
        if model_type.lower() == "arima":
            return ModelARIMA(data=data, order=order, steps=steps)
        elif model_type.lower() == "garch":
            return ModelGARCH(data=data, p=p, q=q, dist=dist)
        elif model_type.lower() == "mvgarch":
            return ModelMultivariateGARCH(data=data, p=p, q=q, model_type=mv_model_type)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


def run_garch(
    df_stationary: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    forecast_steps: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Runs the GARCH model on the provided stationary DataFrame.

    This function fits GARCH(p,q) models to each column in the provided DataFrame
    and generates volatility forecasts. It performs minimal logging to display only
    core information about the model and forecasts.

    Args:
        df_stationary (pd.DataFrame): The stationary time series data for GARCH modeling
        p (int): The GARCH lag order, default=1
        q (int): The ARCH lag order, default=1
        dist (str): The error distribution - 'normal', 't', etc., default="normal"
        forecast_steps (int): The number of steps to forecast, default=5

    Returns:
        Tuple[Dict[str, Any], Dict[str, float]]:
            - First element: Dictionary of fitted GARCH models for each column
            - Second element: Dictionary of forecasted volatility values for each column
    """
    l.info(f"\n## Running GARCH(p={p}, q={q}, dist={dist})")

    # Ensure data is properly prepared for time series analysis
    try:
        df_stationary = data_processor.prepare_timeseries_data(df_stationary)
    except Exception as e:
        l.error(f"Error preparing data for GARCH model: {e}")
        raise ValueError(f"Failed to prepare data for GARCH model: {str(e)}")

    # Check if we have enough data points for GARCH modeling (need at least p+q+1)
    min_points = p + q + 1
    if len(df_stationary) < min_points:
        raise ValueError(
            f"GARCH model requires at least {min_points} data points, but only {len(df_stationary)} provided"
        )

    # Verify data has variance (GARCH won't work on constant data)
    for col in df_stationary.columns:
        if df_stationary[col].std() == 0:
            l.warning(f"Column {col} has zero variance, GARCH modeling may fail")

    # Create and fit the GARCH model
    try:
        model_garch = ModelFactory.create_model(
            model_type="GARCH",
            data=df_stationary,
            p=p,
            q=q,
            dist=dist,
        )
        garch_fit = model_garch.fit()

        # Log only core model information instead of full summary
        l.info(f"## GARCH model fitted to columns: {list(garch_fit.keys())}")

        # Generate and log forecast values concisely
        garch_forecast = model_garch.forecast(steps=forecast_steps)
        l.info(f"## GARCH {forecast_steps}-step volatility forecast:")
        for col, value in garch_forecast.items():
            if hasattr(value, "iloc"):
                value_str = ", ".join(f"{v:.6f}" for v in value)
                l.info(f"   {col}: [{value_str}]")
            else:
                l.info(f"   {col}: {value:.6f}")

        return garch_fit, garch_forecast

    except Exception as e:
        l.error(f"Error during GARCH model fitting or forecasting: {e}")
        raise RuntimeError(f"GARCH model failed: {str(e)}")


def calculate_stats(series: pd.Series) -> dict:
    """
    Calculate comprehensive statistics for a time series.

    Args:
        series: Time series data

    Returns:
        Dictionary of statistics
    """
    return {
        "n": len(series),
        "mean": series.mean(),
        "median": series.median(),
        "min": series.min(),
        "max": series.max(),
        "std": series.std(),
        "skew": series.skew(),
        "kurt": series.kurtosis(),
        "annualized_vol": series.std() * np.sqrt(250),  # Assuming daily data
    }
