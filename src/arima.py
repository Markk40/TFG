"""ARIMA model fitting and diagnostics"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from typing import Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore")


def adf_test(series: pd.Series) -> Dict[str, Any]:
    """
    Perform Augmented Dickey-Fuller test for stationarity
    
    Args:
        series: Input time series
        
    Returns:
        dict: Test results including statistic, p-value, and recommended d
    """
    adf_stat, adf_p, _, _, critical, _ = adfuller(series, autolag="AIC")
    d_recommended = 0 if adf_p < 0.05 else 1
    
    return {
        "statistic": adf_stat,
        "p_value": adf_p,
        "critical_values": critical,
        "d_recommended": d_recommended,
        "is_stationary": adf_p < 0.05,
    }


def compute_acf_pacf(
    series: pd.Series,
    max_lags: int,
    d: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ACF and PACF for (differenced) series
    
    Args:
        series: Input time series
        max_lags: Maximum number of lags
        d: Differencing order
        
    Returns:
        Tuple: (lags, acf_values, pacf_lags, pacf_values)
    """
    # Apply differencing if needed
    series_diff = series.copy()
    for _ in range(d):
        series_diff = series_diff.diff().dropna()
    
    # Compute ACF and PACF
    acf_values = acf(series_diff, nlags=max_lags, fft=True)
    max_pacf_lags = min(max_lags, len(series_diff) // 4 - 1)
    pacf_values = pacf(series_diff, nlags=max_pacf_lags, method="ywmle")
    
    lags_acf = np.arange(len(acf_values))
    lags_pacf = np.arange(len(pacf_values))
    
    return lags_acf, acf_values, lags_pacf, pacf_values


def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int]
) -> Tuple[Any, pd.Series]:
    """
    Fit ARIMA model to time series
    
    Args:
        series: Input time series
        order: (p, d, q) ARIMA order
        
    Returns:
        Tuple: (fitted model, residuals)
    """
    model = SARIMAX(
        series,
        order=order,
        enforce_stationarity=True,
        enforce_invertibility=True
    )
    result = model.fit(disp=False)
    residuals = result.resid.dropna().iloc[1:]
    
    return result, residuals


def evaluate_arima_model(residuals: pd.Series) -> Dict[str, Any]:
    """
    Evaluate ARIMA model fit using residual diagnostics
    
    Args:
        residuals: Model residuals
        
    Returns:
        dict: Evaluation metrics
    """
    # Ljung-Box test for residual autocorrelation
    lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_result["lb_pvalue"].values[0]
    
    # Kurtosis and skewness
    kurtosis = stats.kurtosis(residuals)
    skewness = stats.skew(residuals)
    
    return {
        "lb_pvalue": lb_pvalue,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "aic": getattr(residuals, "model", None),
        "is_residuals_white_noise": lb_pvalue > 0.05,
        "has_heavy_tails": kurtosis > 10,
    }