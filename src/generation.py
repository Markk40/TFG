"""Synthetic trace generation"""

import numpy as np
from scipy.stats import genpareto
from typing import Dict, Any, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


def generate_garch_innovations(
    n_samples: int,
    omega: float,
    alpha: float,
    beta: float,
    nu: float,
    rng: np.random.Generator,
    checkpoint_freq: int,
    checkpoint_depth: float,
    checkpoint_len: int,
    u_threshold: float,
    xi_up: float,
    sigma_up: float,
    xi_lo: float,
    sigma_lo: float,
    tail_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate GARCH innovations with checkpoint and extreme events
    
    Args:
        n_samples: Number of samples to generate
        omega, alpha, beta: GARCH parameters
        nu: Degrees of freedom for Student-t
        rng: Random number generator
        checkpoint_freq: Frequency of checkpoint events
        checkpoint_depth: Depth of checkpoint drops
        checkpoint_len: Recovery length
        u_threshold: Threshold for extreme events
        xi_up, sigma_up: Upper tail GPD parameters
        xi_lo, sigma_lo: Lower tail GPD parameters
        tail_scale: Multiplier for tail severity
        
    Returns:
        Tuple: (conditional variance, innovations)
    """
    h = np.zeros(n_samples)
    eps = np.zeros(n_samples)
    
    # Initialize
    h[0] = omega / max(1 - alpha - beta, 1e-6)
    
    for t in range(1, n_samples):
        # GARCH variance update
        h[t] = omega + alpha * eps[t-1]**2 + beta * h[t-1]
        
        # Checkpoint detection
        is_checkpoint = (t % checkpoint_freq) < 3
        
        if is_checkpoint:
            # Generate drop event
            exc = genpareto.rvs(xi_lo, loc=0, scale=sigma_lo * tail_scale, random_state=rng)
            # Limit the drop to prevent negative power later
            max_drop = 0.95  # No more than 95% drop from mean
            drop_magnitude = min(u_threshold + exc, max_drop * 10)  # Cap extreme drops
            eps[t] = -drop_magnitude * np.sqrt(h[t])
        else:
            # Normal innovation with Student-t
            z_b = rng.standard_t(df=float(nu))
            p_r = rng.random()
            
            # Upper tail extreme (power spike)
            if p_r > 0.97:
                exc = genpareto.rvs(xi_up, loc=0, scale=sigma_up * tail_scale, random_state=rng)
                # Cap extreme spikes to prevent unrealistic values
                max_spike = 5.0  # Maximum 5x the volatility
                z_b = min(u_threshold + exc, max_spike)
            # Lower tail extreme (power drop)
            elif p_r < 0.03:
                exc = genpareto.rvs(xi_lo, loc=0, scale=sigma_lo * tail_scale, random_state=rng)
                # Limit drops to prevent negative power
                max_drop = 0.95 * 10  # Scale appropriately
                z_b = -min(u_threshold + exc, max_drop)
            
            eps[t] = z_b * np.sqrt(h[t])
    
    return h, eps


def apply_bulk_noise(
    eps: np.ndarray,
    volatility_scale: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Add additional bulk noise to the innovations
    
    Args:
        eps: Base innovations
        volatility_scale: Scale factor for bulk noise (1.0 = no extra noise)
        rng: Random number generator
        
    Returns:
        np.ndarray: Modified innovations
    """
    if volatility_scale <= 1.0:
        return eps
    
    bulk_mask = np.abs(eps) < np.percentile(np.abs(eps), 95)
    bulk_std = np.std(eps[bulk_mask]) if bulk_mask.sum() > 0 else 1.0
    
    extra_noise = rng.normal(0, bulk_std * (volatility_scale - 1), len(eps))
    return eps + extra_noise


def generate_checkpoint_profile(
    n_samples: int,
    base_level: float,
    checkpoint_freq: int,
    checkpoint_depth: float,
    checkpoint_len: int,
) -> np.ndarray:
    """
    Generate checkpoint drop profile (mean component)
    
    Args:
        n_samples: Number of samples
        base_level: Base power level (W)
        checkpoint_freq: Frequency of checkpoints (samples)
        checkpoint_depth: Depth of drop (0.75 = drops to 25%)
        checkpoint_len: Recovery length in samples
        
    Returns:
        np.ndarray: Mean component with checkpoint drops
    """
    mean_comp = np.full(n_samples, base_level)
    
    # Ensure checkpoint_depth doesn't cause negative values
    safe_depth = min(checkpoint_depth, 0.99)  # At least 1% of base power remains
    
    for t in range(n_samples):
        phase = t % checkpoint_freq
        if phase < checkpoint_len:
            drop_frac = 1 - safe_depth
            recovery = 1 - np.exp(-5 * phase / checkpoint_len)
            mean_comp[t] = base_level * (drop_frac + safe_depth * recovery)
    
    return mean_comp


def clip_to_positive(
    synthetic: np.ndarray,
    min_power: float = 0.0
) -> np.ndarray:
    """
    Clip synthetic power values to ensure non-negative values
    
    Args:
        synthetic: Synthetic power trace
        min_power: Minimum allowed power (default 0)
        
    Returns:
        np.ndarray: Clipped synthetic trace
    """
    return np.maximum(synthetic, min_power)


def generate_synthetic_trace(
    n_samples: int,
    base_level: float,
    garch_params: Dict[str, float],
    gpd_upper: Dict[str, float],
    gpd_lower: Dict[str, float],
    u_threshold: float,
    checkpoint_freq: int,
    checkpoint_depth: float,
    checkpoint_len: int,
    volatility_scale: float = 1.0,
    tail_scale: float = 1.0,
    seed: int = 42,
    enforce_positivity: bool = True,
    min_power: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate complete synthetic power trace
    
    Args:
        n_samples: Number of samples to generate
        base_level: Base power level (W)
        garch_params: GARCH model parameters
        gpd_upper: Upper tail GPD parameters
        gpd_lower: Lower tail GPD parameters
        u_threshold: Threshold for extreme events
        checkpoint_freq: Frequency of checkpoint events
        checkpoint_depth: Depth of checkpoint drops
        checkpoint_len: Recovery length
        volatility_scale: Scale for bulk noise
        tail_scale: Scale for tail severity
        seed: Random seed
        enforce_positivity: If True, ensure all values are >= min_power
        min_power: Minimum allowed power (default 0)
        
    Returns:
        Tuple: (conditional variance, innovations, synthetic trace)
    """
    rng = np.random.default_rng(seed)
    
    # Extract parameters with safe defaults
    omega = max(garch_params.get("omega", 100), 1e-6)
    alpha = min(garch_params.get("alpha", 0.1), 0.99)
    beta = min(garch_params.get("beta", 0.8), 0.99)
    nu = max(garch_params.get("nu", 10), 2.1)  # Nu must be > 2 for finite variance
    
    xi_up = gpd_upper.get("xi", 0.5)
    sigma_up = max(gpd_upper.get("sigma", 1.0), 1e-6)
    xi_lo = gpd_lower.get("xi", 0.5)
    sigma_lo = max(gpd_lower.get("sigma", 1.0), 1e-6)
    
    # Ensure GARCH stationarity
    if alpha + beta >= 0.999:
        alpha = alpha * 0.9
        beta = beta * 0.9
    
    # Generate GARCH innovations
    h, eps = generate_garch_innovations(
        n_samples, omega, alpha, beta, nu, rng,
        checkpoint_freq, checkpoint_depth, checkpoint_len,
        u_threshold, xi_up, sigma_up, xi_lo, sigma_lo,
        tail_scale
    )
    
    # Apply bulk noise
    eps = apply_bulk_noise(eps, volatility_scale, rng)
    
    # Generate mean component with checkpoints
    mean_comp = generate_checkpoint_profile(
        n_samples, base_level, checkpoint_freq, checkpoint_depth, checkpoint_len
    )
    
    # Combine
    synthetic = mean_comp + eps
    
    # Enforce non-negative values
    if enforce_positivity:
        synthetic = clip_to_positive(synthetic, min_power)
        # Also ensure the mean component was reasonable
        if np.any(mean_comp < min_power):
            print(f"Warning: mean component has {np.sum(mean_comp < min_power)} values below {min_power}")
    
    # Final sanity check
    if np.any(synthetic < min_power):
        synthetic = clip_to_positive(synthetic, min_power)
    
    return h, eps, synthetic