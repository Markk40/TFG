"""Data loading and preprocessing utilities"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")


def load_data(file) -> pd.DataFrame:
    """
    Load CSV or Excel file into DataFrame
    
    Args:
        file: Uploaded file object
        
    Returns:
        pd.DataFrame: Loaded data
    """
    filename = file.name
    if filename.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, engine="openpyxl")
    return df


def build_power_series(
    df: pd.DataFrame,
    power_cols: list,
    time_col: Optional[str] = None
) -> Tuple[pd.Series, np.ndarray]:
    """
    Build power time series from selected columns
    
    Args:
        df: Input DataFrame
        power_cols: List of column names containing power measurements
        time_col: Column name for time values (None = use row index)
        
    Returns:
        Tuple[pd.Series, np.ndarray]: (power series, time values)
    """
    # Build power values
    if len(power_cols) > 1:
        power_values = df[power_cols].sum(axis=1)
    else:
        power_values = df[power_cols[0]]
    
    # Build time values
    if time_col is None or time_col == "(use row index)":
        time_values = np.arange(len(power_values)).astype(float)
    else:
        time_values = pd.to_numeric(df[time_col], errors="coerce").values
    
    series = pd.Series(power_values.values, index=time_values, name="power")
    
    return series, time_values


def remove_warmup(series: pd.Series, warmup_samples: int, trim_end: int = 0) -> pd.Series:
    """
    Remove warm-up period from the beginning of the series
    
    Args:
        series: Input time series
        warmup_samples: Number of samples to remove from start
        trim_end: Number of samples to remove from start
        
    Returns:
        pd.Series: Series with warm-up removed
    """
    if trim_end > 0:
        return series.iloc[warmup_samples:-trim_end].copy()
    return series.iloc[warmup_samples:].copy()


def resample_to_regular_grid(
    series: pd.Series,
    dt_median: Optional[float] = None
) -> Tuple[pd.Series, float]:
    """
    Resample irregular time series to regular grid
    
    Args:
        series: Input time series with irregular timestamps
        dt_median: Target time step (if None, compute from data)
        
    Returns:
        Tuple[pd.Series, float]: (Resampled series, time step)
    """
    time_idx = series.index
    diffs = np.diff(time_idx)
    dt = float(np.median(diffs)) if dt_median is None else dt_median
    
    try:
        s_td = pd.Series(
            series.values,
            index=pd.to_timedelta(series.index, unit="s")
        )
        rule = f"{dt:.4f}s"
        series_reg = s_td.resample(rule).mean().interpolate("linear")
        series_reg.index = np.arange(len(series_reg)) * dt
        series_reg.name = "power"
    except Exception:
        series_reg = series.copy()
        series_reg.index = np.arange(len(series_reg)) * dt
    
    return series_reg, dt


def compute_basic_stats(series: pd.Series) -> dict:
    """
    Compute basic statistics for a time series
    
    Args:
        series: Input time series
        
    Returns:
        dict: Dictionary of statistics
    """
    dt = series.index[1] - series.index[0] if len(series) > 1 else 1
    
    return {
        "samples": len(series),
        "duration": series.index[-1] if len(series) > 0 else 0,
        "fs": 1.0 / dt if dt > 0 else 0,
        "mean": series.mean(),
        "std": series.std(),
    }