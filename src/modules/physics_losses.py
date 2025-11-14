from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import torch

from .ser_inference import compute_local_order_params, estimate_omega_and_gap
from .kuramoto_analytics import (
    KuramotoAnalytics,
    covariance_partially_synchronized,
    covariance_incoherent,
)


def ser_residual_loss_1d(
    phases: np.ndarray,
    times: np.ndarray,
    x_positions: np.ndarray,
    subset: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Compute basic SER residual for Eq. (7):
      Ω_j ≈ ω + (Ω - ω) η_j, η_j = 2|ζ_j|^2 / (1 + |ζ_j|^2)
    Steps:
      - compute ζ_j, Ω_j, Ω
      - fit ω, (Ω - ω) by regression
      - compute MSE residual between Ω_j and linear fit
    Returns:
      dict with 'ser_mse', 'omega', 'Omega_minus_omega'
    """
    # Validate inputs
    if np.any(np.isnan(phases)) or np.any(np.isinf(phases)):
        raise ValueError("ser_residual_loss_1d: phases contains NaN or Inf")
    if np.any(np.isnan(times)) or np.any(np.isinf(times)):
        raise ValueError("ser_residual_loss_1d: times contains NaN or Inf")
    
    zeta, Omega_j, Omega = compute_local_order_params(phases, times, indices=subset)
    
    # Check for NaN/inf in computed values
    if np.any(np.isnan(zeta)) or np.any(np.isinf(zeta)):
        raise ValueError("ser_residual_loss_1d: zeta contains NaN or Inf")
    if np.any(np.isnan(Omega_j)) or np.any(np.isinf(Omega_j)):
        raise ValueError("ser_residual_loss_1d: Omega_j contains NaN or Inf")
    if np.isnan(Omega) or np.isinf(Omega):
        raise ValueError(f"ser_residual_loss_1d: Omega is NaN or Inf (value={Omega})")
    
    omega, gap = estimate_omega_and_gap(Omega_j, zeta)
    
    # Check fitted parameters
    if np.isnan(omega) or np.isinf(omega):
        raise ValueError(f"ser_residual_loss_1d: omega is NaN or Inf (value={omega})")
    if np.isnan(gap) or np.isinf(gap):
        raise ValueError(f"ser_residual_loss_1d: gap is NaN or Inf (value={gap})")
    
    eta = (2.0 * (np.abs(zeta) ** 2)) / (1.0 + (np.abs(zeta) ** 2))
    pred = omega + gap * eta
    mse = float(np.mean((Omega_j - pred) ** 2))
    
    # Validate output
    if np.isnan(mse) or np.isinf(mse):
        raise ValueError(f"ser_residual_loss_1d: mse is NaN or Inf (value={mse})")
    
    return {"ser_mse": mse, "omega": float(omega), "Omega_minus_omega": float(gap), "Omega": float(Omega)}


def empirical_covariance(Z: np.ndarray, times: np.ndarray, max_lags: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical covariance R(τ) from complex order parameter series Z(t):
      R(k) = E[(Z - m)(Z_{t+k} - m)^*], τ_k = k * dt
    Returns:
      (R_hat [L], tau [L])
    """
    Z = np.asarray(Z, dtype=np.complex128)
    times = np.asarray(times, dtype=float)
    
    # Validate inputs
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        raise ValueError("empirical_covariance: Z contains NaN or Inf")
    if np.any(np.isnan(times)) or np.any(np.isinf(times)):
        raise ValueError("empirical_covariance: times contains NaN or Inf")
    if len(Z) < 2:
        raise ValueError(f"empirical_covariance: Z must have length >= 2, got {len(Z)}")
    if len(times) != len(Z):
        raise ValueError(f"empirical_covariance: times length ({len(times)}) != Z length ({len(Z)})")
    
    T = Z.shape[0]
    m = np.mean(Z)
    
    # Check mean
    if np.isnan(m) or np.isinf(m):
        raise ValueError(f"empirical_covariance: mean of Z is NaN or Inf (value={m})")
    
    Zc = Z - m
    L = min(max_lags, T - 1)
    if L < 0:
        raise ValueError(f"empirical_covariance: max_lags={max_lags}, T={T}, L={L} < 0")
    
    R = np.zeros(L + 1, dtype=np.complex128)
    for k in range(L + 1):
        R[k] = np.mean(Zc[: T - k] * np.conj(Zc[k:]))
        # Check for NaN/inf in covariance
        if np.isnan(R[k]) or np.isinf(R[k]):
            raise ValueError(f"empirical_covariance: R[{k}] is NaN or Inf (value={R[k]})")
    
    dt = float(np.mean(np.diff(times)))
    if np.isnan(dt) or np.isinf(dt) or dt <= 0:
        raise ValueError(f"empirical_covariance: dt is invalid (value={dt})")
    
    tau = dt * np.arange(L + 1)
    return R, tau


def covariance_match_loss(
    Z: np.ndarray,
    times: np.ndarray,
    mode: str,
    dist: str,
    dist_params: Dict[str, float],
    K: float,
    lambda_param: float,
    r_inf: Optional[float] = None,
    omega_inf: float = 0.0,
    max_lags: int = 256,
) -> Dict[str, float]:
    """
    Compare empirical covariance of Z(t) to theory; returns relative L2 error.
    """
    # Validate inputs
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        raise ValueError("covariance_match_loss: Z contains NaN or Inf")
    if np.isnan(K) or np.isinf(K):
        raise ValueError(f"covariance_match_loss: K is NaN or Inf (value={K})")
    if np.isnan(lambda_param) or np.isinf(lambda_param):
        raise ValueError(f"covariance_match_loss: lambda_param is NaN or Inf (value={lambda_param})")
    
    R_emp, tau = empirical_covariance(Z, times, max_lags=max_lags)
    
    if dist == "gaussian":
        ka = KuramotoAnalytics("gaussian", {"mu": dist_params.get("mu", 0.0), "sigma": dist_params.get("sigma", 1.0)})
    elif dist == "lorentzian":
        ka = KuramotoAnalytics("lorentzian", {"gamma": dist_params.get("gamma", 1.0), "omega_0": dist_params.get("omega_0", 0.0)})
    else:
        raise ValueError(f"Unsupported dist: {dist}")
    
    if mode == "ps":
        if r_inf is None:
            # Estimate r_inf from |Z|
            r_inf = float(np.mean(np.abs(Z)))
        if np.isnan(r_inf) or np.isinf(r_inf) or r_inf < 0:
            raise ValueError(f"covariance_match_loss: r_inf is invalid (value={r_inf})")
        
        R_th = np.array([covariance_partially_synchronized(t, K=K, r_inf=r_inf, omega_inf=omega_inf, lambda_param=lambda_param, g_omega=ka.g_omega) for t in tau], dtype=np.complex128)
        
        # Check theoretical covariance
        if np.any(np.isnan(R_th)) or np.any(np.isinf(R_th)):
            raise ValueError("covariance_match_loss: R_th contains NaN or Inf")
    elif mode == "incoh":
        R_th = np.array([covariance_incoherent(t, g_omega=ka.g_omega) for t in tau], dtype=np.complex128)
        if np.any(np.isnan(R_th)) or np.any(np.isinf(R_th)):
            raise ValueError("covariance_match_loss: R_th contains NaN or Inf")
    else:
        raise ValueError("mode must be 'ps' or 'incoh'")
    
    denom = np.linalg.norm(R_emp) + 1e-12
    if np.isnan(denom) or np.isinf(denom) or denom <= 0:
        raise ValueError(f"covariance_match_loss: denominator is invalid (value={denom})")
    
    rel_err = float(np.linalg.norm(R_emp - R_th) / denom)
    
    # Validate output
    if np.isnan(rel_err) or np.isinf(rel_err):
        raise ValueError(f"covariance_match_loss: rel_err is NaN or Inf (value={rel_err})")
    
    return {"cov_rel_err": rel_err}


def variance_scaling_loss(
    Z: np.ndarray,
    mode: str,
) -> Dict[str, float]:
    """
    Check variance scaling and mean |Z| expectations:
      - incoh: V = 1 - π/4, ⟨r⟩ ~ √(π/N) / 2
      - ps: no strict target here beyond V ≈ 0.5 R_ps(0) (requires theory)
    Returns:
      simple penalty terms for incoherent regime
    """
    # Validate inputs
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        raise ValueError("variance_scaling_loss: Z contains NaN or Inf")
    
    r = np.abs(Z)
    r_mean = float(np.mean(r))
    
    # Check mean
    if np.isnan(r_mean) or np.isinf(r_mean):
        raise ValueError(f"variance_scaling_loss: r_mean is NaN or Inf (value={r_mean})")
    
    N = 1  # For |Z| as mean of exp(i θ), effective N is hidden; use |Z| directly
    losses = {}
    if mode == "incoh":
        vincoh = 1.0 - np.pi / 4.0
        # Rescale variance to dimensionless compare using |Z|
        var_r = float(np.var(r))
        
        # Check variance
        if np.isnan(var_r) or np.isinf(var_r):
            raise ValueError(f"variance_scaling_loss: var_r is NaN or Inf (value={var_r})")
        
        losses["var_err"] = abs(var_r - vincoh)
        # Soft target for mean magnitude in finite-N: not directly comparable; keep as diagnostic
        losses["r_mean"] = r_mean
    elif mode == "ps":
        # For ps mode, still compute variance for diagnostics
        var_r = float(np.var(r))
        if np.isnan(var_r) or np.isinf(var_r):
            raise ValueError(f"variance_scaling_loss: var_r is NaN or Inf (value={var_r})")
        losses["var_r"] = var_r
        losses["r_mean"] = r_mean
    
    return losses


def metastability_band_loss(
    Z: np.ndarray,
    var_lo: float,
    var_hi: float,
) -> Dict[str, float]:
    """
    Encourage Var(|Z|) to lie within a target band [var_lo, var_hi].
    Penalizes both too-low and too-high variance (metastability band objective).

    Returns:
      {
        "band_err": max(0, var_lo - Var(|Z|)) + max(0, Var(|Z|) - var_hi),
        "var_r": Var(|Z|)
      }
    """
    # Validate inputs
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        raise ValueError("metastability_band_loss: Z contains NaN or Inf")
    if np.isnan(var_lo) or np.isinf(var_lo):
        raise ValueError(f"metastability_band_loss: var_lo is NaN or Inf (value={var_lo})")
    if np.isnan(var_hi) or np.isinf(var_hi):
        raise ValueError(f"metastability_band_loss: var_hi is NaN or Inf (value={var_hi})")
    if var_lo >= var_hi:
        raise ValueError(f"metastability_band_loss: var_lo ({var_lo}) >= var_hi ({var_hi})")
    
    r = np.abs(Z)
    var_r = float(np.var(r))
    
    # Check variance
    if np.isnan(var_r) or np.isinf(var_r):
        raise ValueError(f"metastability_band_loss: var_r is NaN or Inf (value={var_r})")
    
    err_low = max(0.0, float(var_lo) - var_r)
    err_high = max(0.0, var_r - float(var_hi))
    band_err = err_low + err_high
    
    # Validate output
    if np.isnan(band_err) or np.isinf(band_err):
        raise ValueError(f"metastability_band_loss: band_err is NaN or Inf (value={band_err})")
    
    return {"band_err": band_err, "var_r": var_r}


def torch_scalar(t: Optional[float], device: torch.device) -> torch.Tensor:
    """
    Convert a scalar to a torch tensor, handling None and NaN/Inf cases.
    """
    if t is None:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    
    val = float(t)
    if np.isnan(val) or np.isinf(val):
        raise ValueError(f"torch_scalar: value is NaN or Inf (value={val})")
    
    return torch.tensor(val, device=device, dtype=torch.float32)


def aggregate_physics_losses_torch(
    Z: np.ndarray,
    times: np.ndarray,
    ser_mse: Optional[float] = None,
    cov_rel_err: Optional[float] = None,
    var_err: Optional[float] = None,
    band_err: Optional[float] = None,
    w_ser: float = 1.0,
    w_cov: float = 1.0,
    w_var: float = 0.25,
    w_meta: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Combine available physics loss terms into a single scalar torch tensor.
    Includes optional metastability band penalty (w_meta * band_err).
    """
    device = device or torch.device("cpu")
    
    # Validate weights
    if np.isnan(w_ser) or np.isinf(w_ser):
        raise ValueError(f"aggregate_physics_losses_torch: w_ser is NaN or Inf (value={w_ser})")
    if np.isnan(w_cov) or np.isinf(w_cov):
        raise ValueError(f"aggregate_physics_losses_torch: w_cov is NaN or Inf (value={w_cov})")
    if np.isnan(w_var) or np.isinf(w_var):
        raise ValueError(f"aggregate_physics_losses_torch: w_var is NaN or Inf (value={w_var})")
    if np.isnan(w_meta) or np.isinf(w_meta):
        raise ValueError(f"aggregate_physics_losses_torch: w_meta is NaN or Inf (value={w_meta})")
    
    L = (
        w_ser * torch_scalar(ser_mse, device)
        + w_cov * torch_scalar(cov_rel_err, device)
        + w_var * torch_scalar(var_err, device)
        + w_meta * torch_scalar(band_err, device)
    )
    
    # Validate final loss
    if torch.isnan(L) or torch.isinf(L):
        raise ValueError(f"aggregate_physics_losses_torch: final loss L is NaN or Inf (value={L.item()})")
    
    return L
