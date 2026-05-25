"""Extreme Value Theory - GPD fitting for tails"""

import numpy as np
from scipy import stats
from scipy.stats import genpareto
from typing import Tuple, Dict, Any, Optional


def compute_mean_excess(
    data: np.ndarray,
    n_thresholds: int = 80,
    min_exceedances: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean excess function for threshold selection
    
    Args:
        data: Input data (positive values for upper tail)
        n_thresholds: Number of thresholds to evaluate
        min_exceedances: Minimum exceedances for valid threshold
        
    Returns:
        Tuple: (thresholds, mean_excess, ci_upper, ci_lower)
    """
    thresholds = np.linspace(data.min(), np.quantile(data, 0.98), n_thresholds)
    mean_excess = []
    ci_upper = []
    ci_lower = []
    
    for u in thresholds:
        excess = data[data > u] - u
        if len(excess) < min_exceedances:
            break
            
        m = excess.mean()
        se = excess.std() / np.sqrt(len(excess))
        mean_excess.append(m)
        ci_upper.append(m + 1.96 * se)
        ci_lower.append(m - 1.96 * se)
    
    thresholds = thresholds[:len(mean_excess)]
    
    return np.array(thresholds), np.array(mean_excess), np.array(ci_upper), np.array(ci_lower)


def fit_gpd_tail(
    data: np.ndarray,
    threshold: float,
    tail: str = "upper"
) -> Dict[str, Any]:
    """
    Fit Generalized Pareto Distribution to tail data
    
    Args:
        data: Full dataset
        threshold: Threshold for extreme events
        tail: 'upper' or 'lower' tail
        
    Returns:
        dict: GPD parameters and fit statistics
    """
    if tail == "upper":
        tail_data = data[data > 0]
        exceedances = tail_data[tail_data > threshold] - threshold
    else:
        tail_data = -data[data < 0]
        exceedances = tail_data[tail_data > threshold] - threshold
    
    if len(exceedances) < 10:
        return {
            "xi": np.nan,
            "sigma": np.nan,
            "n_exceedances": len(exceedances),
            "ks_pvalue": np.nan,
            "is_valid": False,
        }
    
    # Fit GPD
    xi, _, sigma = genpareto.fit(exceedances, floc=0)
    
    # Kolmogorov-Smirnov test
    ks_result = stats.kstest(exceedances, "genpareto", args=(xi, 0, sigma))
    
    return {
        "xi": xi,
        "sigma": sigma,
        "n_exceedances": len(exceedances),
        "ks_pvalue": ks_result.pvalue,
        "is_valid": ks_result.pvalue > 0.05,
        "is_heavy_tail": xi > 0,
        "is_bounded": xi < 0,
    }


def get_tail_data(
    data: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract upper and lower tail exceedances
    
    Args:
        data: Input data
        threshold: Threshold value
        
    Returns:
        Tuple: (upper_exceedances, lower_exceedances)
    """
    # Upper tail (positive spikes)
    upper_data = data[data > 0]
    upper_exceedances = upper_data[upper_data > threshold] - threshold
    
    # Lower tail (drops)
    lower_data = -data[data < 0]
    lower_exceedances = lower_data[lower_data > threshold] - threshold
    
    return upper_exceedances, lower_exceedances