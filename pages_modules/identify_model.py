"""Page 2: ARIMA model identification"""

import streamlit as st
import numpy as np
import pandas as pd
from src.arima import adf_test, compute_acf_pacf, fit_arima, evaluate_arima_model
from src.plotting import create_acf_pacf_plot, create_residuals_plot


def show():
    st.markdown('<div class="main-title">Identify Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Run diagnostics to choose ARIMA orders (p, d, q)</div>', unsafe_allow_html=True)
    
    if st.session_state.serie_reg is None:
        st.warning("Please load data first (Step 1)")
        return
    
    serie_reg = st.session_state.serie_reg
    
    # ADF Test
    st.markdown('<div class="section-header">Stationarity — ADF Test</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>What is this?</b> The ADF test checks whether the series needs differencing (d parameter). If p &lt; 0.05, the series is stationary and d=0. Otherwise use d=1.</div>', unsafe_allow_html=True)
    
    if st.button("Run ADF Test", type="primary"):
        with st.spinner("Running ADF test..."):
            adf_result = adf_test(serie_reg)
            st.session_state.d_recommended = adf_result["d_recommended"]
        
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-card"><div class="metric-value">{adf_result["statistic"]:.4f}</div><div class="metric-label">ADF Statistic</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="metric-value">{adf_result["p_value"]:.4f}</div><div class="metric-label">p-value</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div class="metric-value">d = {adf_result["d_recommended"]}</div><div class="metric-label">Recommended d</div></div>', unsafe_allow_html=True)
        
        if adf_result["is_stationary"]:
            st.markdown('<div class="good-box">Series is stationary: use d=0</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">Series is non-stationary: use d=1 (differencing)</div>', unsafe_allow_html=True)
    
    if st.session_state.d_recommended is None:
        st.info("Run the ADF test above before proceeding")
        st.stop()
    
    # Select d
    st.markdown('<div class="section-header">Select differencing order d</div>', unsafe_allow_html=True)
    d = st.slider("d — Differencing", 0, 2, st.session_state.d_recommended)
    
    # ACF / PACF
    st.markdown('<div class="section-header">ACF & PACF — choose p and q</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>How to read these plots:</b><br>
    • <b>PACF cuts off at lag p</b> → use that as your AR order (p)<br>
    • <b>ACF cuts off at lag q</b> → use that as your MA order (q)<br>
    • "Cuts off" means the bars drop inside the red confidence bands and stay there<br>
    • Start simple: p=1 or p=2, q=1 or q=2
    </div>
    """, unsafe_allow_html=True)
    
    max_lags = st.slider("Max lags to display", 20, 100, 40, 5)
    
    with st.spinner("Computing ACF/PACF..."):
        lags_acf, acf_vals, lags_pacf, pacf_vals = compute_acf_pacf(serie_reg, max_lags, d)
    
    fig_acf = create_acf_pacf_plot(lags_acf, acf_vals, lags_pacf, pacf_vals, len(serie_reg), d)
    st.plotly_chart(fig_acf, width='stretch')
    
    # ARIMA order selection and fitting
    st.markdown('<div class="section-header">Choose ARIMA orders and fit</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.slider("p — AR order (from PACF)", 0, 5, 2)
        st.markdown('<div class="info-box">How many past values predict the current one</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{d}</div><div class="metric-label">d — Differencing</div></div>', unsafe_allow_html=True)
    with col3:
        q = st.slider("q — MA order (from ACF)", 0, 5, 2)
        st.markdown('<div class="info-box">How many past errors correct the current prediction</div>', unsafe_allow_html=True)
    
    st.markdown(f"**Selected model: ARIMA({p}, {d}, {q})**")
    
    if st.button("Fit ARIMA", type="primary"):
        with st.spinner(f"Fitting ARIMA({p},{d},{q})... this may take a moment"):
            try:
                result, residuals = fit_arima(serie_reg, (p, d, q))
                eval_metrics = evaluate_arima_model(residuals)
                
                st.session_state.arima_result = result
                st.session_state.arima_residuals = residuals
                
                # Show metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.markdown(f'<div class="metric-card"><div class="metric-value">{result.aic:.0f}</div><div class="metric-label">AIC (lower = better)</div></div>', unsafe_allow_html=True)
                col2.markdown(f'<div class="metric-card"><div class="metric-value">{result.bic:.0f}</div><div class="metric-label">BIC (lower = better)</div></div>', unsafe_allow_html=True)
                col3.markdown(f'<div class="metric-card"><div class="metric-value">{eval_metrics["lb_pvalue"]:.3f}</div><div class="metric-label">Ljung-Box p (residuals)<br>want > 0.05</div></div>', unsafe_allow_html=True)
                col4.markdown(f'<div class="metric-card"><div class="metric-value">{eval_metrics["kurtosis"]:.1f}</div><div class="metric-label">Residual kurtosis<br>(normal = 3)</div></div>', unsafe_allow_html=True)
                
                # Residuals plot
                fig_res = create_residuals_plot(residuals, f"ARIMA({p},{d},{q}) residuals")
                st.plotly_chart(fig_res, width='stretch')
                
                if eval_metrics["is_residuals_white_noise"]:
                    st.markdown('<div class="good-box">Ljung-Box p > 0.05 — no autocorrelation in residuals. ARIMA captured the mean structure well.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warn-box">Ljung-Box p < 0.05 — some autocorrelation remains. Consider increasing p or q.</div>', unsafe_allow_html=True)
                
                if eval_metrics["has_heavy_tails"]:
                    st.markdown(f'<div class="warn-box">Kurtosis = {eval_metrics["kurtosis"]:.1f} (normal = 3) — heavy tails detected. GARCH + EVT will handle this in the next step.</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="good-box">ARIMA fitted — proceed to <b>3 · Fit GARCH + EVT</b></div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"ARIMA fitting failed: {e}")