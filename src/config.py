"""Configuration and constants for the application"""

from dataclasses import dataclass
from typing import Dict, Any

# Plot theme configuration
PLOT_LAYOUT = {
    "paper_bgcolor": "#0f1117",
    "plot_bgcolor": "#1a1d27",
    "font": {"color": "#ccc", "size": 11},
    "xaxis": {"gridcolor": "#2a2d3a", "zerolinecolor": "#333"},
    "yaxis": {"gridcolor": "#2a2d3a", "zerolinecolor": "#333"},
    "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
}

# Model default parameters
ARIMA_DEFAULTS = {"max_p": 5, "max_q": 5, "max_d": 2}
GARCH_DEFAULTS = {"p": 1, "q": 1, "distributions": ["studentst", "normal", "skewstudent"]}
EVT_DEFAULTS = {"threshold_range": (0.5, 3.5), "default_threshold": 1.5}

# CSS styles
CUSTOM_CSS = """
<style>
    .stApp { background-color: #0f1117; }
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #4fc3f7, #f48fb1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #888; font-size: 0.95rem; margin-bottom: 2rem; }
    .metric-card {
        background: #1a1d27; border: 1px solid #2a2d3a;
        border-radius: 10px; padding: 1rem 1.2rem; margin: 0.3rem 0;
    }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #4fc3f7; }
    .metric-label { font-size: 0.8rem; color: #888; }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #f48fb1;
        border-bottom: 1px solid #2a2d3a; padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
    }
    .info-box {
        background: #1a1d27; border-left: 3px solid #4fc3f7;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0; font-size: 0.88rem; color: #ccc;
    }
    .warn-box {
        background: #1a1d27; border-left: 3px solid #ffcc80;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0; font-size: 0.88rem; color: #ccc;
    }
    .good-box {
        background: #1a1d27; border-left: 3px solid #a5d6a7;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        margin: 0.5rem 0; font-size: 0.88rem; color: #ccc;
    }
    div[data-testid="stSidebar"] { background-color: #13151f; }
    .step-badge {
        display: inline-block; background: #4fc3f7; color: #0f1117;
        border-radius: 50%; width: 24px; height: 24px; text-align: center;
        line-height: 24px; font-weight: 800; font-size: 0.85rem;
        margin-right: 0.5rem;
    }
</style>
"""