"""Page 4: Generate synthetic traces"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy import stats
from src.config import SCENARIO_PRESETS
from src.generation import generate_synthetic_trace
from src.plotting import create_comparison_plot, create_distribution_comparison


def show():
    st.markdown('<div class="main-title">Generate Synthetic Trace</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Configure and generate synthetic power traces using the fitted model</div>', unsafe_allow_html=True)
    
    if st.session_state.garch_result is None or st.session_state.gpd_upper is None:
        st.warning("Please complete Steps 2 and 3 first")
        st.stop()
    
    garch_params = {
        "omega": st.session_state.garch_result.params.get("omega", 100),
        "alpha": st.session_state.garch_result.params.get("alpha[1]", 0.1),
        "beta": st.session_state.garch_result.params.get("beta[1]", 0.8),
        "nu": st.session_state.garch_result.params.get("nu", 10),
    }
    
    gpd_upper = st.session_state.gpd_upper
    gpd_lower = st.session_state.gpd_lower
    serie_reg = st.session_state.serie_reg
    dt_median = st.session_state.dt_median
    u_threshold = st.session_state.u_threshold
    
    st.markdown('<div class="section-header">⚙ Generation parameters</div>', unsafe_allow_html=True)
    
    # Parameter inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Signal shape**")
        n_samples = st.slider(
            "Number of samples to generate",
            1000, 50000, len(serie_reg), 500,
            help="More samples = longer synthetic trace"
        )
        base_level = st.slider(
            "Base power level (W)",
            int(serie_reg.mean() * 0.5),
            int(serie_reg.mean() * 1.5),
            int(serie_reg.mean()),
            50,
            help="Mean power during steady-state. Lower = less loaded GPU (e.g. inference vs training)"
        )
        volatility_scale = st.slider(
            "Bulk volatility scale",
            0.5, 5.0, 1.0, 0.5,
            help="Multiplier for background noise. >1 = noisier signal"
        )
    
    with col2:
        st.markdown("**Checkpoint events**")
        checkpoint_freq = st.slider(
            "Checkpoint frequency (samples between events)",
            100, 10000, 3500, 100,
            help="How often checkpoint/evaluation events occur. Lower = more frequent drops"
        )
        checkpoint_depth = st.slider(
            "Checkpoint depth (fraction of base level dropped)",
            0.1, 1.0, 0.75, 0.05,
            help="0.75 = drops to 25% of base level. 1.0 = drops to 0W (stress test)"
        )
        checkpoint_len = st.slider(
            "Checkpoint recovery length (samples)",
            10, 300, 80, 10,
            help="How many samples it takes to recover. Longer = slower recovery"
        )
    
    col3, col4 = st.columns(2)
    with col3:
        tail_scale = st.slider(
            "Tail scale (GPD extremes multiplier)",
            0.5, 3.0, 1.0, 0.25,
            help=">1 = more extreme spikes/drops. Useful for stress testing"
        )
    with col4:
        seed = st.number_input("Random seed (for reproducibility)", 0, 9999, 42, 1)
    
    # Generate
    if st.button("▶ Generate synthetic trace", type="primary"):
        with st.spinner("Generating synthetic trace..."):
            h, eps, synthetic = generate_synthetic_trace(
                n_samples=n_samples,
                base_level=base_level,
                garch_params=garch_params,
                gpd_upper=gpd_upper,
                gpd_lower=gpd_lower,
                u_threshold=u_threshold,
                checkpoint_freq=checkpoint_freq,
                checkpoint_depth=checkpoint_depth,
                checkpoint_len=checkpoint_len,
                volatility_scale=volatility_scale,
                tail_scale=tail_scale,
                seed=int(seed)
            )
            st.session_state.synthetic = synthetic
        
        # ── Results
        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
        
        orig = serie_reg.values
        t_syn = np.arange(n_samples) * dt_median
        
        # Comparison plot
        fig_comp = create_comparison_plot(serie_reg, synthetic, dt_median,
                                          f"Original vs Synthetic — {n_samples} samples")
        st.plotly_chart(fig_comp, width='stretch')
        
        # Distribution comparison
        fig_dist = create_distribution_comparison(orig, synthetic)
        st.plotly_chart(fig_dist, width='stretch')
        
        # Stats table
        st.markdown('<div class="section-header">Statistics comparison</div>', unsafe_allow_html=True)
        
        metrics = {
            "Mean (W)": (orig.mean(), synthetic.mean()),
            "Std (W)": (orig.std(), synthetic.std()),
            "Min (W)": (orig.min(), synthetic.min()),
            "Max (W)": (orig.max(), synthetic.max()),
            "Kurtosis": (stats.kurtosis(orig), stats.kurtosis(synthetic)),
            "Skewness": (stats.skew(orig), stats.skew(synthetic)),
        }
        
        cols = st.columns(len(metrics))
        for col, (name, (o, s)) in zip(cols, metrics.items()):
            rel_err = abs(o - s) / (abs(o) + 1e-9)
            color = "#a5d6a7" if rel_err < 0.15 else "#ffcc80" if rel_err < 0.35 else "#ef9a9a"
            col.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.75rem; color:#888;">{name}</div>
                <div style="color:#4fc3f7; font-size:1rem; font-weight:600;">{o:.2f}</div>
                <div style="color:{color}; font-size:1rem; font-weight:600;">{s:.2f}</div>
                <div style="font-size:0.7rem; color:#555;">orig → synth</div>
            </div>""", unsafe_allow_html=True)
        
        # ── Download
        st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
        
        df_out = pd.DataFrame({
            "Elapsed Time (s)": t_syn,
            "Total Power (W)": synthetic,
            **{f"GPU {i} Power (W)": synthetic / 8 for i in range(8)}
        })
        
        col_csv, col_xlsx = st.columns(2)
        
        csv_buf = io.StringIO()
        df_out.to_csv(csv_buf, index=False)
        
        col_csv.download_button(
            "⬇ Download CSV",
            csv_buf.getvalue(),
            file_name=f"synthetic_trace_base{int(base_level)}_ckpt{checkpoint_freq}_n{n_samples}.csv",
            mime="text/csv",
            width='stretch'
        )
        
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Synthetic Trace")
        col_xlsx.download_button(
            "⬇ Download XLSX",
            xlsx_buf.getvalue(),
            file_name=f"synthetic_trace_base{int(base_level)}_ckpt{checkpoint_freq}_n{n_samples}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch'
        )