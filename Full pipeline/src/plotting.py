"""Plotting utilities for the application"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from typing import Optional, Tuple, List, Dict, Any

from .config import PLOT_LAYOUT


def create_signal_preview(series: pd.Series, title: str = "Signal preview") -> go.Figure:
    """Create a preview plot of a signal"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(series)), y=series.values,
        mode="lines", line=dict(color="#4fc3f7", width=0.8),
        name="Signal"
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=250,
        title=title,
        xaxis_title="Sample index"
    )
    return fig


def create_signal_dashboard(
    series: pd.Series,
    title: str = "Signal overview"
) -> go.Figure:
    """Create a dashboard with signal trace and distribution"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Time series", "Distribution"]
    )
    
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", line=dict(color="#4fc3f7", width=0.7),
        name="Power"
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=series.values, nbinsx=80,
        marker_color="#4fc3f7", opacity=0.75,
        name="Distribution"
    ), row=1, col=2)
    
    fig.update_layout(**PLOT_LAYOUT, height=320, showlegend=False, title=title)
    fig.update_xaxes(gridcolor="#2a2d3a")
    fig.update_yaxes(gridcolor="#2a2d3a")
    
    return fig


def create_acf_pacf_plot(
    acf_lags: np.ndarray,
    acf_values: np.ndarray,
    pacf_lags: np.ndarray,
    pacf_values: np.ndarray,
    n_samples: int,
    d: int = 0,
) -> go.Figure:
    """Create ACF and PACF side-by-side plot"""
    ci = 1.96 / np.sqrt(n_samples)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"ACF {'(differenced)' if d>0 else ''} → guides MA order q",
            f"PACF {'(differenced)' if d>0 else ''} → guides AR order p"
        ]
    )
    
    # ACF plot
    for lag, val in zip(acf_lags[1:], acf_values[1:]):
        fig.add_trace(go.Bar(
            x=[lag], y=[val], marker_color="#4fc3f7", opacity=0.8,
            showlegend=False
        ), row=1, col=1)
    fig.add_hline(y=ci, line=dict(color="red", dash="dash", width=1), row=1, col=1)
    fig.add_hline(y=-ci, line=dict(color="red", dash="dash", width=1), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="white", width=0.5), row=1, col=1)
    
    # PACF plot
    for lag, val in zip(pacf_lags[1:], pacf_values[1:]):
        fig.add_trace(go.Bar(
            x=[lag], y=[val], marker_color="#f48fb1", opacity=0.8,
            showlegend=False
        ), row=1, col=2)
    fig.add_hline(y=ci, line=dict(color="red", dash="dash", width=1), row=1, col=2)
    fig.add_hline(y=-ci, line=dict(color="red", dash="dash", width=1), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="white", width=0.5), row=1, col=2)
    
    fig.update_layout(**PLOT_LAYOUT, height=380, showlegend=False)
    fig.update_xaxes(gridcolor="#2a2d3a", title_text="Lag")
    fig.update_yaxes(gridcolor="#2a2d3a", title_text="Correlation")
    
    return fig


def create_residuals_plot(residuals: pd.Series, title: str) -> go.Figure:
    """Create residuals diagnostic plot"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Residuals over time", "Residuals distribution"]
    )
    
    fig.add_trace(go.Scatter(
        x=np.arange(len(residuals)), y=residuals.values,
        mode="lines", line=dict(color="#4fc3f7", width=0.6),
        name="Residuals"
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="red", dash="dash", width=1), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=residuals.values, nbinsx=80,
        marker_color="#4fc3f7", opacity=0.75,
        name="Distribution"
    ), row=1, col=2)
    
    fig.update_layout(**PLOT_LAYOUT, height=320, showlegend=False, title=title)
    fig.update_xaxes(gridcolor="#2a2d3a")
    fig.update_yaxes(gridcolor="#2a2d3a")
    
    return fig


def create_garch_diagnostics(
    cond_vol: pd.Series,
    std_resid: pd.Series,
    title: str = "GARCH(1,1) diagnostics"
) -> go.Figure:
    """Create GARCH diagnostics plot"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Conditional volatility σ_t",
            "Standardized residuals z_t",
            "z_t distribution"
        ]
    )
    
    fig.add_trace(go.Scatter(
        x=np.arange(len(cond_vol)), y=cond_vol.values,
        mode="lines", line=dict(color="#ffcc80", width=0.8)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=np.arange(len(std_resid)), y=std_resid.values,
        mode="lines", line=dict(color="#4fc3f7", width=0.5), opacity=0.8
    ), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="red", dash="dash", width=0.8), row=1, col=2)
    
    fig.add_trace(go.Histogram(
        x=std_resid.values, nbinsx=80,
        marker_color="#4fc3f7", opacity=0.75
    ), row=1, col=3)
    
    fig.update_layout(**PLOT_LAYOUT, height=350, showlegend=False, title=title)
    fig.update_xaxes(gridcolor="#2a2d3a")
    fig.update_yaxes(gridcolor="#2a2d3a")
    
    return fig


def create_mean_excess_plot(
    upper_series: np.ndarray,
    lower_series: np.ndarray,
    n_thresholds: int = 80
) -> go.Figure:
    """
    Create mean excess plot for threshold selection
    
    The mean excess plot shows E[X - u | X > u] vs u.
    Choose u where the plot becomes approximately linear.
    
    Args:
        upper_series: Upper tail data (positive values)
        lower_series: Lower tail data (absolute values of negative)
        n_thresholds: Number of thresholds to evaluate
        
    Returns:
        go.Figure: Mean excess plot figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Upper tail (power spikes)", "Lower tail (power drops)"]
    )
    
    for col_idx, (data, color, tail_name) in enumerate(
        [(upper_series, "#f48fb1", "Upper"), (lower_series, "#ffcc80", "Lower")], 1
    ):
        thresholds = np.linspace(data.min(), np.quantile(data, 0.98), n_thresholds)
        thresholds_list = []
        mean_excess_list = []
        ci_upper_list = []
        ci_lower_list = []
        
        for u in thresholds:
            excess = data[data > u] - u
            if len(excess) < 10:
                break
            m = excess.mean()
            se = excess.std() / np.sqrt(len(excess))
            thresholds_list.append(u)
            mean_excess_list.append(m)
            ci_upper_list.append(m + 1.96 * se)
            ci_lower_list.append(m - 1.96 * se)
        
        thresholds_arr = np.array(thresholds_list)
        mean_excess_arr = np.array(mean_excess_list)
        
        # Add mean excess line (solid, thick, visible)
        fig.add_trace(go.Scatter(
            x=thresholds_arr,
            y=mean_excess_arr,
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=4, color=color, opacity=0.8),
            name=f"{tail_name} tail - Mean excess",
            showlegend=False
        ), row=1, col=col_idx)
        
        # Add confidence interval as a filled area (semi-transparent)
        fig.add_trace(go.Scatter(
            x=np.concatenate([thresholds_arr, thresholds_arr[::-1]]),
            y=np.concatenate([ci_upper_list, ci_lower_list[::-1]]),
            fill="toself",
            fillcolor=f"rgba{color[3:-1] if color.startswith('rgb') else '(79, 195, 247, 0.2)'}",
            line=dict(width=0),
            showlegend=False,
            name=f"{tail_name} tail - 95% CI"
        ), row=1, col=col_idx)
        
        # Add a horizontal line at y=0 for reference
        fig.add_hline(y=0, line=dict(color="white", width=1, dash="dot"), 
                      row=1, col=col_idx, opacity=0.5)
    
    fig.update_layout(
        **PLOT_LAYOUT,
        height=450,
        showlegend=True,
        title="Mean Excess Plot — Choose threshold u where curve becomes linear",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.6)", font=dict(size=10))
    )
    fig.update_xaxes(gridcolor="#2a2d3a", title_text="Threshold u")
    fig.update_yaxes(gridcolor="#2a2d3a", title_text="Mean excess E[X-u | X>u]")
    
    return fig


def create_gpd_fit_plot(
    exceedances: np.ndarray,
    xi: float,
    sigma: float,
    tail: str,
    color: str
) -> go.Figure:
    """Create GPD fit validation plot"""
    x_sorted = np.sort(exceedances)
    emp_cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    x_range = np.linspace(0, x_sorted.max(), 200)
    gpd_cdf = stats.genpareto.cdf(x_range, xi, loc=0, scale=sigma)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_sorted, y=emp_cdf,
        mode="markers", marker=dict(color=color, size=3, opacity=0.6),
        name="Empirical"
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=gpd_cdf,
        mode="lines", line=dict(color="white", width=2, dash="dash"),
        name="GPD fit"
    ))
    
    fig.update_layout(**PLOT_LAYOUT, height=380,
                      title=f"GPD fit — {tail} tail (ξ={xi:.3f})")
    fig.update_xaxes(gridcolor="#2a2d3a", title_text="Exceedance above u")
    fig.update_yaxes(gridcolor="#2a2d3a", title_text="CDF")
    
    return fig


def create_comparison_plot(
    original: pd.Series,
    synthetic: np.ndarray,
    dt: float,
    title: str = "Original vs Synthetic"
) -> go.Figure:
    """Create comparison plot between original and synthetic traces"""
    t_syn = np.arange(len(synthetic)) * dt
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Original signal", "Synthetic trace"]
    )
    
    fig.add_trace(go.Scatter(
        x=original.index, y=original.values,
        mode="lines", line=dict(color="#4fc3f7", width=0.7),
        name="Original"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=t_syn, y=synthetic,
        mode="lines", line=dict(color="#f48fb1", width=0.7),
        name="Synthetic"
    ), row=2, col=1)
    
    fig.update_layout(**PLOT_LAYOUT, height=500, showlegend=True, title=title)
    fig.update_xaxes(gridcolor="#2a2d3a", title_text="Elapsed time (s)")
    fig.update_yaxes(gridcolor="#2a2d3a", title_text="Power (W)")
    
    return fig


def create_distribution_comparison(
    original: np.ndarray,
    synthetic: np.ndarray
) -> go.Figure:
    """Create distribution comparison histogram"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=original, nbinsx=80, name="Original",
        marker_color="#4fc3f7", opacity=0.6,
        histnorm="probability density"
    ))
    
    fig.add_trace(go.Histogram(
        x=synthetic, nbinsx=80, name="Synthetic",
        marker_color="#f48fb1", opacity=0.6,
        histnorm="probability density"
    ))
    
    fig.update_layout(
        **PLOT_LAYOUT,
        height=300,
        title="Power distribution comparison",
        barmode="overlay"
    )
    
    return fig

def create_qq_plot(
    data: np.ndarray,
    distribution: str = "studentt",
    df: float = 5,
    title: str = "QQ-plot: Standardized Residuals vs Theoretical Distribution"
) -> go.Figure:
    """
    Create a QQ-plot comparing data distribution to theoretical distribution
    
    Args:
        data: Input data (standardized residuals)
        distribution: 'normal' or 'studentt'
        df: Degrees of freedom for Student-t distribution
        title: Plot title
        
    Returns:
        go.Figure: QQ-plot figure
    """
    # Sort data for QQ plot
    data_sorted = np.sort(data)
    n = len(data_sorted)
    
    # Theoretical quantiles
    if distribution == "normal":
        # Standard normal quantiles
        p = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = stats.norm.ppf(p)
        dist_name = "Normal"
    else:
        # Student-t quantiles
        p = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = stats.t.ppf(p, df=df)
        dist_name = f"Student-t (ν={df:.1f})"
    
    # Create QQ plot
    fig = go.Figure()
    
    # Add reference line (y = x)
    min_val = min(data_sorted.min(), theoretical_quantiles.min())
    max_val = max(data_sorted.max(), theoretical_quantiles.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="red", dash="dash", width=2),
        name="Reference (y=x)",
        showlegend=True
    ))
    
    # Add Q-Q points
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=data_sorted,
        mode="markers",
        marker=dict(color="#4fc3f7", size=4, opacity=0.7),
        name="Empirical Quantiles",
        showlegend=True
    ))
    
    # Add confidence bands (approximate)
    se = 1.0 / np.sqrt(n)  # Standard error for quantiles
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val + 1.96*se, max_val + 1.96*se],
        mode="lines",
        line=dict(color="gray", dash="dot", width=1),
        name="95% CI",
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val - 1.96*se, max_val - 1.96*se],
        mode="lines",
        line=dict(color="gray", dash="dot", width=1),
        showlegend=False
    ))
    
    # Layout
    fig.update_layout(
        **PLOT_LAYOUT,
        height=450,
        title=title,
        xaxis_title=f"Theoretical {dist_name} Quantiles",
        yaxis_title="Empirical Quantiles (Standardized Residuals)",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)")
    )
    
    fig.update_xaxes(gridcolor="#2a2d3a")
    fig.update_yaxes(gridcolor="#2a2d3a")
    
    return fig


def create_tail_comparison_plot(
    data: np.ndarray,
    theoretical_quantiles: np.ndarray,
    tail_percent: float = 5,
    title: str = "Tail Comparison"
) -> go.Figure:
    """
    Create a focused plot comparing only the tails
    
    Args:
        data: Input data
        theoretical_quantiles: Theoretical quantiles
        tail_percent: Percentage to show in each tail (default 5%)
        title: Plot title
        
    Returns:
        go.Figure: Tail comparison figure
    """
    n = len(data)
    tail_n = int(n * tail_percent / 100)
    
    # Get lower tail (negative values)
    lower_data = np.sort(data)[:tail_n]
    lower_theoretical = np.sort(theoretical_quantiles)[:tail_n]
    
    # Get upper tail (positive values)
    upper_data = np.sort(data)[-tail_n:]
    upper_theoretical = np.sort(theoretical_quantiles)[-tail_n:]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Lower tail (bottom {tail_percent}%)", 
                       f"Upper tail (top {tail_percent}%)"]
    )
    
    # Lower tail
    fig.add_trace(go.Scatter(
        x=lower_theoretical,
        y=lower_data,
        mode="markers",
        marker=dict(color="#f48fb1", size=5, opacity=0.8),
        name="Lower tail"
    ), row=1, col=1)
    
    min_val = min(lower_theoretical.min(), lower_data.min())
    max_val = max(lower_theoretical.max(), lower_data.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="red", dash="dash", width=2),
        showlegend=False
    ), row=1, col=1)
    
    # Upper tail
    fig.add_trace(go.Scatter(
        x=upper_theoretical,
        y=upper_data,
        mode="markers",
        marker=dict(color="#ffcc80", size=5, opacity=0.8),
        name="Upper tail"
    ), row=1, col=2)
    
    min_val = min(upper_theoretical.min(), upper_data.min())
    max_val = max(upper_theoretical.max(), upper_data.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="red", dash="dash", width=2),
        showlegend=False
    ), row=1, col=2)
    
    fig.update_layout(
        **PLOT_LAYOUT,
        height=400,
        title=title,
        showlegend=False
    )
    fig.update_xaxes(gridcolor="#2a2d3a", title_text="Theoretical Quantiles")
    fig.update_yaxes(gridcolor="#2a2d3a", title_text="Empirical Quantiles")
    
    return fig