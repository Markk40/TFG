"""Page 1: Load and preprocess data"""

import streamlit as st
import numpy as np
import pandas as pd
from src.preprocessing import load_data, build_power_series, remove_warmup, resample_to_regular_grid, compute_basic_stats
from src.plotting import create_signal_preview, create_signal_dashboard


def show():
    st.markdown('<div class="main-title">Load Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload your power trace dataset and configure the signal</div>', unsafe_allow_html=True)
    
    # Upload
    st.markdown('<div class="section-header">Upload file</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV or XLSX file", type=["csv", "xlsx"])
    
    if uploaded:
        try:
            df = load_data(uploaded)
            st.session_state.df = df
            st.markdown(f'<div class="good-box">File loaded: <b>{uploaded.name}</b> — {df.shape[0]} rows, {df.shape[1]} columns</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
    
    if st.session_state.df is None:
        return
    
    df = st.session_state.df
    
    # Column selection
    st.markdown('<div class="section-header">Select columns</div>', unsafe_allow_html=True)
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        time_col = st.selectbox("Time column", ["(use row index)"] + all_cols)
    with col2:
        power_cols = st.multiselect(
            "Power column(s) — select one or more to sum",
            num_cols,
            default=num_cols[:1]
        )
    
    if not power_cols:
        st.warning("Please select at least one power column")
        return
    
    # Build series
    if len(power_cols) > 1:
        st.markdown(f'<div class="info-box">Summing {len(power_cols)} columns → total power</div>', unsafe_allow_html=True)
    
    time_sel = time_col if time_col != "(use row index)" else None
    serie_raw, time_vals = build_power_series(df, power_cols, time_sel)
    
    # Warm-up removal
    st.markdown('<div class="section-header">Remove warm-up and end-of-job periods</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">GPU training traces typically have a warm-up phase at the start '
                'and an end-of-job idle drop at the end. Remove both before fitting.</div>',
                unsafe_allow_html=True)
    
    # Quick preview
    fig_prev = create_signal_preview(serie_raw, "Full signal — identify warm-up end and end-of-job drop")
    st.plotly_chart(fig_prev, width='stretch')
    
    col1, col2 = st.columns(2)
    with col1:
        warmup = st.slider(
            "Remove first N samples (warm-up)",
            min_value=0, max_value=len(serie_raw)//2,
            value=0, step=10
        )
    with col2:
        trim_end = st.slider(
            "Remove last N samples (end-of-job)",
            min_value=0, max_value=len(serie_raw)//2,
            value=0, step=10
        )

    # Apply both trims
    serie_clean = remove_warmup(serie_raw, warmup, trim_end)
    
    # Resampling to regular grid
    serie_reg, dt_median = resample_to_regular_grid(serie_clean)
    
    st.session_state.serie = serie_clean
    st.session_state.serie_reg = serie_reg
    st.session_state.dt_median = dt_median
    
    # Stats
    st.markdown('<div class="section-header">Signal overview</div>', unsafe_allow_html=True)
    
    stats_data = compute_basic_stats(serie_reg)
    
    cols = st.columns(5)
    metrics = [
        ("Samples", f"{stats_data['samples']:,}"),
        ("Duration", f"{stats_data['duration']:.1f} s"),
        ("Fs", f"{stats_data['fs']:.2f} Hz"),
        ("Mean", f"{stats_data['mean']:.1f} W"),
        ("Std", f"{stats_data['std']:.1f} W"),
    ]
    
    for col, (label, val) in zip(cols, metrics):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    
    # Final signal plot
    fig_final = create_signal_dashboard(serie_reg, "Steady-state signal ready for modeling")
    st.plotly_chart(fig_final, width='stretch')
    
    st.markdown('<div class="good-box">Data ready — proceed to <b>2 · Identify Model</b></div>', unsafe_allow_html=True)