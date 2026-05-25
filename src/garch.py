"""GARCH model fitting for volatility"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from typing import Tuple, Any, Dict, Optional


def fit_garch(
    residuals: pd.Series,
    p: int = 1,
    q: int = 1,
    distribution: str = "studentst"
) -> Tuple[Any, pd.Series, pd.Series]:
    """
    Fit GARCH(p, q) model to residuals
    
    Args:
        residuals: Input residuals from ARIMA
        p: GARCH order for lagged volatility
        q: GARCH order for lagged squared residuals
        distribution: Innovation distribution ('normal', 'studentst', 'skewstudent')
        
    Returns:
        Tuple: (fitted model, standardized residuals, conditional volatility)
    """
    model = arch_model(residuals, vol="GARCH", p=p, q=q, dist=distribution)
    result = model.fit(disp="off")
    
    std_resid = result.std_resid
    cond_vol = result.conditional_volatility
    
    return result, std_resid, cond_vol


def extract_garch_params(garch_result: Any) -> Dict[str, float]:
    """
    Extract GARCH model parameters
    
    Args:
        garch_result: Fitted GARCH model
        
    Returns:
        dict: Model parameters
    """
    params = garch_result.params
    
    omega = params.get("omega", 0)
    alpha = params.get("alpha[1]", 0)
    beta = params.get("beta[1]", 0)
    nu = params.get("nu", 10)
    
    persistence = alpha + beta
    
    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "nu": nu,
        "persistence": persistence,
    }


def evaluate_garch_fit(
    original_residuals: pd.Series,
    standardized_resid: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate GARCH model effectiveness
    
    Args:
        original_residuals: Original ARIMA residuals
        standardized_resid: Standardized residuals after GARCH
        
    Returns:
        dict: Evaluation metrics
    """
    orig_kurt = stats.kurtosis(original_residuals)
    std_kurt = stats.kurtosis(standardized_resid.dropna())
    
    kurtosis_reduction = orig_kurt - std_kurt
    kurtosis_reduction_pct = (1 - std_kurt / orig_kurt) * 100 if orig_kurt != 0 else 0
    
    return {
        "original_kurtosis": orig_kurt,
        "standardized_kurtosis": std_kurt,
        "kurtosis_reduction": kurtosis_reduction,
        "kurtosis_reduction_pct": kurtosis_reduction_pct,
        "is_improved": std_kurt < orig_kurt,
    }