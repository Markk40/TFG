"""Page 3: Fit GARCH and EVT models"""

import streamlit as st
import numpy as np
from scipy import stats
from src.garch import fit_garch, extract_garch_params, evaluate_garch_fit
from src.evt import compute_mean_excess, fit_gpd_tail, get_tail_data
from src.plotting import (
    create_garch_diagnostics, 
    create_mean_excess_plot, 
    create_gpd_fit_plot,
    create_qq_plot,
    create_tail_comparison_plot
)


def show():
    st.markdown('<div class="main-title">Fit GARCH + EVT</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Model volatility clustering and extreme tail events</div>', unsafe_allow_html=True)
    
    if st.session_state.arima_residuals is None:
        st.warning("Please fit ARIMA first (Step 2)")
        st.stop()
    
    residuals = st.session_state.arima_residuals
    
    # ── GARCH
    st.markdown('<div class="section-header">GARCH(1,1) — conditional volatility</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>What is GARCH?</b> It models the fact that volatility is not constant — some periods are 
    calm, others are turbulent. GARCH(1,1) estimates σ_t (the volatility at each instant) so 
    that dividing the residuals by σ_t gives a much more homogeneous signal.<br><br>
    <b>Distribution:</b> Student-t is recommended when residual kurtosis is high (heavy tails).
    </div>
    """, unsafe_allow_html=True)
    
    dist = st.selectbox(
        "Innovation distribution",
        ["studentst", "normal", "skewstudent"],
        help="Student-t recommended for heavy-tailed signals"
    )
    
    if st.button("Fit GARCH(1,1)", type="primary"):
        with st.spinner("Fitting GARCH(1,1)..."):
            try:
                garch_res, std_resid, cond_vol = fit_garch(residuals, 1, 1, dist)
                params = extract_garch_params(garch_res)
                eval_metrics = evaluate_garch_fit(residuals, std_resid)
                
                st.session_state.garch_result = garch_res
                st.session_state.std_resid = std_resid
                st.session_state.cond_vol = cond_vol
                
                # Parameters display
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.markdown(f'<div class="metric-card"><div class="metric-value">{params["omega"]:.2f}</div><div class="metric-label">ω — base variance</div></div>', unsafe_allow_html=True)
                col2.markdown(f'<div class="metric-card"><div class="metric-value">{params["alpha"]:.3f}</div><div class="metric-label">α — shock impact</div></div>', unsafe_allow_html=True)
                col3.markdown(f'<div class="metric-card"><div class="metric-value">{params["beta"]:.3f}</div><div class="metric-label">β — persistence</div></div>', unsafe_allow_html=True)
                col4.markdown(f'<div class="metric-card"><div class="metric-value">{params["persistence"]:.3f}</div><div class="metric-label">α+β — total persistence</div></div>', unsafe_allow_html=True)
                col5.markdown(f'<div class="metric-card"><div class="metric-value">{eval_metrics["standardized_kurtosis"]:.1f}</div><div class="metric-label">Kurtosis after GARCH<br>(was {eval_metrics["original_kurtosis"]:.0f})</div></div>', unsafe_allow_html=True)
                
                # GARCH diagnostics plot
                fig_g = create_garch_diagnostics(cond_vol, std_resid)
                st.plotly_chart(fig_g, width='stretch')
                
                # QQ-plot for standardized residuals
                st.markdown('<div class="section-header">QQ-plot — Distribution Fit Assessment</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                <b>What does the QQ-plot show?</b> It compares the quantiles of the standardized residuals 
                against the theoretical distribution (Student-t or Normal).<br><br>
                • <b>Points on the red line</b> → perfect fit<br>
                • <b>Points deviating in the tails</b> → heavier tails than the theoretical distribution, 
                which is why we need EVT (next step)<br>
                • <b>Typical pattern</b>: good fit in the center, deviations in the extremes
                </div>
                """, unsafe_allow_html=True)
                
                # Prepare data (drop NaNs)
                z_clean = std_resid.dropna().values
                nu_fitted = params.get("nu", 5)
                
                # QQ-plot against Student-t (or Normal)
                if dist in ["studentst", "skewstudent"]:
                    fig_qq = create_qq_plot(z_clean, distribution="studentt", df=nu_fitted,
                                            title=f"QQ-plot: Standardized Residuals vs Student-t(ν={nu_fitted:.1f})")
                else:
                    fig_qq = create_qq_plot(z_clean, distribution="normal",
                                            title="QQ-plot: Standardized Residuals vs Normal Distribution")
                
                st.plotly_chart(fig_qq, width='stretch')
                
                # Tail focus plot (optional, with expander)
                with st.expander("Zoom on tails (show 5% extremes)", expanded=False):
                    st.markdown("""
                    <div class="info-box">
                    <b>Tail focus:</b> This plot isolates the bottom 5% and top 5% of the distribution.
                    Deviations from the red line in the tails indicate heavy tails that EVT will capture.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Generate theoretical quantiles for tail plot
                    n = len(z_clean)
                    p = (np.arange(1, n + 1) - 0.5) / n
                    if dist in ["studentst", "skewstudent"]:
                        theoretical_q = stats.t.ppf(p, df=nu_fitted)
                    else:
                        theoretical_q = stats.norm.ppf(p)
                    
                    fig_tail = create_tail_comparison_plot(z_clean, theoretical_q, tail_percent=5,
                                                           title="Tail Focus: 5% Extremes")
                    st.plotly_chart(fig_tail, width='stretch')
                    
                    # Interpret tail behavior
                    z_sorted = np.sort(z_clean)
                    lower_5pct = z_sorted[:int(n*0.05)]
                    upper_5pct = z_sorted[-int(n*0.05):]
                    
                    st.markdown(f"""
                    <div class="info-box">
                    <b>Tail analysis:</b><br>
                    • Lower tail (extreme drops): min = {lower_5pct.min():.2f}, max = {lower_5pct.max():.2f}<br>
                    • Upper tail (extreme spikes): min = {upper_5pct.min():.2f}, max = {upper_5pct.max():.2f}<br>
                    • If |values| > 3-4, you have heavy tails that require EVT modeling
                    </div>
                    """, unsafe_allow_html=True)
                
                if eval_metrics["is_improved"]:
                    st.markdown(f'<div class="good-box">GARCH fitted — kurtosis reduced from {eval_metrics["original_kurtosis"]:.0f} to {eval_metrics["standardized_kurtosis"]:.1f} ({eval_metrics["kurtosis_reduction_pct"]:.0f}% reduction)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="warn-box">Kurtosis increased from {eval_metrics["original_kurtosis"]:.0f} to {eval_metrics["standardized_kurtosis"]:.1f}</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"GARCH fitting failed: {e}")
                return
    
    if st.session_state.std_resid is None:
        st.stop()
    
    # EVT
    st.markdown('<div class="section-header">EVT — Generalized Pareto Distribution for extreme tails</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>What is EVT?</b> Even after GARCH, the most extreme events (e.g. checkpoint drops) are 
    underestimated. EVT focuses exclusively on values beyond a threshold u and fits a 
    Generalized Pareto Distribution (GPD) to them.<br><br>
    <b>How to choose u:</b> Look at the Mean Excess Plot below. Find the lowest u where the 
    curve becomes approximately linear — that is where GPD starts to apply.
    </div>
    """, unsafe_allow_html=True)
    
    std_resid = st.session_state.std_resid
    z = std_resid.dropna().values
    
    # Mean Excess Plot
    z_upper = z[z > 0]
    z_lower = -z[z < 0]
    
    fig_me = create_mean_excess_plot(z_upper, z_lower)
    st.plotly_chart(fig_me, width='stretch')
    
    u = st.slider(
        "GPD threshold u",
        min_value=0.5, max_value=3.5, value=1.5, step=0.1,
        help="Choose the lowest u where the Mean Excess Plot is approximately linear"
    )
    st.session_state.u_threshold = u
    
    if st.button("Fit GPD to both tails", type="primary"):
        with st.spinner("Fitting GPD..."):
            try:
                upper_exc, lower_exc = get_tail_data(z, u)
                
                upper_fit = fit_gpd_tail(z, u, "upper")
                lower_fit = fit_gpd_tail(z, u, "lower")
                
                st.session_state.gpd_upper = upper_fit
                st.session_state.gpd_lower = lower_fit
                
                # Results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Upper tail (power spikes)**")
                    st.markdown(f'<div class="metric-card"><div class="metric-value">ξ = {upper_fit["xi"]:.4f}</div><div class="metric-label">{"→ Bounded tail (physical TDP limit)" if upper_fit["is_bounded"] else "→ Heavy tail"}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><div class="metric-value">σ = {upper_fit["sigma"]:.4f}</div><div class="metric-label">Scale — spread of extremes</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><div class="metric-value">n = {upper_fit["n_exceedances"]}</div><div class="metric-label">Exceedances above u={u}</div></div>', unsafe_allow_html=True)
                    
                    box_class = "good-box" if upper_fit["is_valid"] else "warn-box"
                    st.markdown(f'<div class="{box_class}">KS test p = {upper_fit["ks_pvalue"]:.4f} {"✓ good fit" if upper_fit["is_valid"] else "⚠ marginal fit — try adjusting u"}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Lower tail (power drops)**")
                    st.markdown(f'<div class="metric-card"><div class="metric-value">ξ = {lower_fit["xi"]:.4f}</div><div class="metric-label">{"→ Heavy tail (deep checkpoint drops)" if lower_fit["is_heavy_tail"] else "→ Bounded tail"}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><div class="metric-value">σ = {lower_fit["sigma"]:.4f}</div><div class="metric-label">Scale — spread of extremes</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card"><div class="metric-value">n = {lower_fit["n_exceedances"]}</div><div class="metric-label">Exceedances above u={u}</div></div>', unsafe_allow_html=True)
                    
                    box_class = "good-box" if lower_fit["is_valid"] else "warn-box"
                    st.markdown(f'<div class="{box_class}">KS test p = {lower_fit["ks_pvalue"]:.4f} {"✓ good fit" if lower_fit["is_valid"] else "⚠ marginal fit — try adjusting u"}</div>', unsafe_allow_html=True)
                
                # GPD CDF plots
                fig_upper = create_gpd_fit_plot(upper_exc, upper_fit["xi"], upper_fit["sigma"], "Upper", "#f48fb1")
                fig_lower = create_gpd_fit_plot(lower_exc, lower_fit["xi"], lower_fit["sigma"], "Lower", "#ffcc80")
                
                st.plotly_chart(fig_upper, width='stretch')
                st.plotly_chart(fig_lower, width='stretch')
                
                st.markdown('<div class="good-box">GPD fitted — proceed to <b>4 · Generate</b></div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"GPD fitting failed: {e}")