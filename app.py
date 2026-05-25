"""Streamlit application entry point"""

import streamlit as st
import pandas as pd
import numpy as np
from src.config import CUSTOM_CSS

# Page configuration
st.set_page_config(
    page_title="Power Trace Modeler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
SESSION_KEYS = [
    "df", "serie", "serie_reg", "dt_median", "d_recommended",
    "arima_result", "arima_residuals", "garch_result",
    "std_resid", "cond_vol", "gpd_upper", "gpd_lower",
    "u_threshold", "synthetic", "preset"
]

for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar navigation
st.sidebar.markdown("## Power Trace Modeler")
st.sidebar.markdown("---")

pages = {
    "1 - Load Data": "load",
    "2 - Identify Model": "identify",
    "3 - Fit GARCH + EVT": "fit",
    "4 - Generate": "generate",
}

page = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
current = pages[page]

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.8rem; color:#555;'>
TFG · Marcos Alconchel Fernández<br>
Ingeniería Matemática e IA · Comillas ICAI<br><br>
<i>Controlling Dynamics from Data Centers</i>
</div>
""", unsafe_allow_html=True)


# Page routing - import and call show() for each page
if current == "load":
    from pages_modules.load_data import show
    show()
    
elif current == "identify":
    from pages_modules.identify_model import show
    show()
    
elif current == "fit":
    from pages_modules.fit_GARCH_EVT import show
    show()
    
elif current == "generate":
    from pages_modules.generate import show
    show()