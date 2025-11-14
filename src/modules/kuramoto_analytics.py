"""
Analytical Kuramoto-Sakaguchi Theory

Implements analytical expressions for finite-size fluctuations in the Kuramoto-Sakaguchi model,
based on: "Mean-field approach to finite-size fluctuations in the Kuramoto-Sakaguchi model"
(arXiv:2511.03700v1, Omel'chenko & Gottwald, 2025).

This module provides:
- Analytical covariance functions for order parameter fluctuations
- Variance formulas for partially synchronized and incoherent states
- Comparison utilities for validating numerical simulations

Key Results:
- R_ps(τ): Covariance function for partially synchronized states (Eq. 9)
- R_incoh(τ): Covariance function for incoherent states (Eq. 11)
- V_ps: Variance for partially synchronized states (Eq. 10)
- V_incoh: Variance for incoherent states (Eq. 12)
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict
import numpy as np
from dataclasses import dataclass


def h_function(s: np.ndarray) -> np.ndarray:
    """
    Compute the h(s) function from the Ott-Antonsen ansatz (Eq. 5).
    
    h(s) = {
        (1 - sqrt(1 - s^-2)) * s    for |s| > 1
        s - i*sqrt(1 - s^2)          for |s| <= 1
    }
    
    Args:
        s: Array of normalized frequency differences s = (ω - Ω_∞) / (K * r_∞)
        
    Returns:
        Complex array h(s) [same shape as s]
    """
    s = np.asarray(s, dtype=np.complex128)
    result = np.zeros_like(s, dtype=np.complex128)
    
    # Case 1: |s| > 1 (incoherent oscillators)
    mask_sup = np.abs(s) > 1.0
    if np.any(mask_sup):
        result[mask_sup] = (1.0 - np.sqrt(1.0 - s[mask_sup] ** (-2))) * s[mask_sup]
    
    # Case 2: |s| <= 1 (coherent oscillators)
    mask_sub = np.abs(s) <= 1.0
    if np.any(mask_sub):
        s_sub = s[mask_sub]
        result[mask_sub] = s_sub - 1j * np.sqrt(1.0 - s_sub ** 2)
    
    return result


def covariance_partially_synchronized(
    tau: float,
    K: float,
    r_inf: float,
    omega_inf: float,
    lambda_param: float,
    g_omega: callable,
    s_min: float = -10.0,
    s_max: float = 10.0,
    n_points: int = 1000,
) -> complex:
    """
    Compute analytical covariance function R_ps(τ) for partially synchronized state (Eq. 9).
    
    R_ps(τ) = K * r_∞ * ∫_{-∞}^{∞} (h²(s) - 1)(1 - |h²(s)|) * g(Ω_∞ + K*r_∞*s) / 
              (h²(s) - exp(i*K*r_∞*s*sqrt(1 - s^-2)*τ)) ds
    
    Args:
        tau: Time lag τ
        K: Coupling strength
        r_inf: Thermodynamic order parameter magnitude r_∞
        omega_inf: Global rotation frequency Ω_∞
        lambda_param: Phase lag parameter λ
        g_omega: Frequency distribution function g(ω) (callable)
        s_min: Lower integration bound (default: -10)
        s_max: Upper integration bound (default: 10)
        n_points: Number of integration points (default: 1000)
        
    Returns:
        Complex covariance value R_ps(τ)
    """
    # Integration variable: s = (ω - Ω_∞) / (K * r_∞)
    s_vals = np.linspace(s_min, s_max, n_points)
    ds = (s_max - s_min) / (n_points - 1)
    
    # Compute h(s) for all s values
    h_vals = h_function(s_vals)
    h_squared = h_vals ** 2
    
    # Compute integrand
    integrand = np.zeros_like(s_vals, dtype=np.complex128)
    
    for i, s in enumerate(s_vals):
        omega = omega_inf + K * r_inf * s
        
        # Evaluate frequency distribution
        try:
            g_val = g_omega(omega)
        except:
            g_val = 0.0
        
        if g_val <= 0 or np.isnan(g_val):
            continue
        
        # Compute denominator: h²(s) - exp(i*K*r_∞*s*sqrt(1 - s^-2)*τ)
        if np.abs(s) > 1.0:
            # For |s| > 1: sqrt(1 - s^-2) is real
            sqrt_term = np.sqrt(1.0 - s ** (-2))
            exponent = 1j * K * r_inf * s * sqrt_term * tau
        else:
            # For |s| <= 1: use imaginary part
            sqrt_term = np.sqrt(1.0 - s ** 2)
            exponent = 1j * K * r_inf * s * sqrt_term * tau
        
        denominator = h_squared[i] - np.exp(exponent)
        
        if np.abs(denominator) < 1e-10:
            continue
        
        # Numerator: (h²(s) - 1)(1 - |h²(s)|)
        numerator = (h_squared[i] - 1.0) * (1.0 - np.abs(h_squared[i]))
        
        integrand[i] = numerator * g_val / denominator
    
    # Integrate
    result = K * r_inf * np.trapz(integrand, dx=ds)
    
    return result


def covariance_incoherent(
    tau: float,
    g_omega: callable,
    omega_min: float = -10.0,
    omega_max: float = 10.0,
    n_points: int = 1000,
) -> complex:
    """
    Compute analytical covariance function R_incoh(τ) for incoherent state (Eq. 11).
    
    R_incoh(τ) = ∫_{-∞}^{∞} g(ω) * exp(-i*ω*τ) dω
    
    This is the Fourier transform of the frequency distribution.
    
    Args:
        tau: Time lag τ
        g_omega: Frequency distribution function g(ω) (callable)
        omega_min: Lower integration bound (default: -10)
        omega_max: Upper integration bound (default: 10)
        n_points: Number of integration points (default: 1000)
        
    Returns:
        Complex covariance value R_incoh(τ)
    """
    omega_vals = np.linspace(omega_min, omega_max, n_points)
    domega = (omega_max - omega_min) / (n_points - 1)
    
    # Compute integrand: g(ω) * exp(-i*ω*τ)
    integrand = np.zeros_like(omega_vals, dtype=np.complex128)
    
    for i, omega in enumerate(omega_vals):
        try:
            g_val = g_omega(omega)
        except:
            g_val = 0.0
        
        if g_val > 0 and not np.isnan(g_val):
            integrand[i] = g_val * np.exp(-1j * omega * tau)
    
    # Integrate
    result = np.trapz(integrand, dx=domega)
    
    return result


def variance_partially_synchronized(
    K: float,
    r_inf: float,
    omega_inf: float,
    lambda_param: float,
    g_omega: callable,
    s_min: float = -10.0,
    s_max: float = 10.0,
    n_points: int = 1000,
) -> Tuple[float, float]:
    """
    Compute analytical variance V_ps for partially synchronized state (Eq. 10).
    
    V_ps = (1/2) * R_ps(0) + O(1/√N)
    
    Args:
        K: Coupling strength
        r_inf: Thermodynamic order parameter magnitude r_∞
        omega_inf: Global rotation frequency Ω_∞
        lambda_param: Phase lag parameter λ
        g_omega: Frequency distribution function g(ω) (callable)
        s_min: Lower integration bound (default: -10)
        s_max: Upper integration bound (default: 10)
        n_points: Number of integration points (default: 1000)
        
    Returns:
        Tuple of (variance, R_ps(0)) where variance is the order parameter variance
    """
    # Compute R_ps(0)
    R_ps_0 = covariance_partially_synchronized(
        tau=0.0,
        K=K,
        r_inf=r_inf,
        omega_inf=omega_inf,
        lambda_param=lambda_param,
        g_omega=g_omega,
        s_min=s_min,
        s_max=s_max,
        n_points=n_points,
    )
    
    # Variance: V_ps = (1/2) * Re(R_ps(0))
    variance = 0.5 * np.real(R_ps_0)
    
    return variance, np.real(R_ps_0)


def variance_incoherent() -> Tuple[float, float]:
    """
    Compute analytical variance V_incoh and mean order parameter for incoherent state (Eq. 12).
    
    For completely incoherent state:
    - Mean order parameter: ⟨r(t)⟩ = √(π/N) / 2
    - Variance: V_incoh = 1 - π/4
    
    Note: These are asymptotic results for N >> 1.
    
    Returns:
        Tuple of (variance, mean_order_param)
    """
    variance = 1.0 - np.pi / 4.0
    # Mean order parameter depends on N, but the variance formula is N-independent
    mean_order_param = np.sqrt(np.pi) / 2.0  # For N=1 normalization
    
    return variance, mean_order_param


def gaussian_frequency_distribution(omega: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Gaussian frequency distribution g(ω) = (1/(σ√(2π))) * exp(-(ω-μ)²/(2σ²))
    
    Args:
        omega: Frequency value
        mu: Mean (default: 0.0)
        sigma: Standard deviation (default: 1.0)
        
    Returns:
        Probability density at ω
    """
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((omega - mu) / sigma) ** 2)


def lorentzian_frequency_distribution(omega: float, gamma: float = 1.0, omega_0: float = 0.0) -> float:
    """
    Lorentzian (Cauchy) frequency distribution g(ω) = (γ/π) / ((ω - ω₀)² + γ²)
    
    Args:
        omega: Frequency value
        gamma: Scale parameter (default: 1.0)
        omega_0: Location parameter (default: 0.0)
        
    Returns:
        Probability density at ω
    """
    return (gamma / np.pi) / ((omega - omega_0) ** 2 + gamma ** 2)


class KuramotoAnalytics:
    """
    Analytical Kuramoto-Sakaguchi theory calculator.
    
    Provides analytical predictions for order parameter statistics that can be
    compared with numerical simulations from ResonanceAttentionHead.
    """
    
    def __init__(
        self,
        frequency_distribution: str = "gaussian",
        distribution_params: Optional[Dict] = None,
    ):
        """
        Initialize analytical calculator.
        
        Args:
            frequency_distribution: Type of distribution ("gaussian" or "lorentzian")
            distribution_params: Parameters for the distribution
                - Gaussian: {"mu": 0.0, "sigma": 1.0}
                - Lorentzian: {"gamma": 1.0, "omega_0": 0.0}
        """
        self.frequency_distribution = frequency_distribution
        self.distribution_params = distribution_params or {}
        
        # Set up frequency distribution function
        if frequency_distribution == "gaussian":
            mu = self.distribution_params.get("mu", 0.0)
            sigma = self.distribution_params.get("sigma", 1.0)
            self.g_omega = lambda omega: gaussian_frequency_distribution(omega, mu, sigma)
        elif frequency_distribution == "lorentzian":
            gamma = self.distribution_params.get("gamma", 1.0)
            omega_0 = self.distribution_params.get("omega_0", 0.0)
            self.g_omega = lambda omega: lorentzian_frequency_distribution(omega, gamma, omega_0)
        else:
            raise ValueError(f"Unknown frequency distribution: {frequency_distribution}")
    
    def compute_covariance_ps(
        self,
        tau: float,
        K: float,
        r_inf: float,
        omega_inf: float = 0.0,
        lambda_param: float = 0.0,
    ) -> complex:
        """
        Compute covariance for partially synchronized state.
        
        Args:
            tau: Time lag
            K: Coupling strength
            r_inf: Order parameter magnitude
            omega_inf: Global rotation frequency
            lambda_param: Phase lag parameter
            
        Returns:
            Complex covariance value
        """
        return covariance_partially_synchronized(
            tau=tau,
            K=K,
            r_inf=r_inf,
            omega_inf=omega_inf,
            lambda_param=lambda_param,
            g_omega=self.g_omega,
        )
    
    def compute_covariance_incoherent(self, tau: float) -> complex:
        """
        Compute covariance for incoherent state.
        
        Args:
            tau: Time lag
            
        Returns:
            Complex covariance value
        """
        return covariance_incoherent(tau=tau, g_omega=self.g_omega)
    
    def compute_variance_ps(
        self,
        K: float,
        r_inf: float,
        omega_inf: float = 0.0,
        lambda_param: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Compute variance for partially synchronized state.
        
        Args:
            K: Coupling strength
            r_inf: Order parameter magnitude
            omega_inf: Global rotation frequency
            lambda_param: Phase lag parameter
            
        Returns:
            Tuple of (variance, R_ps(0))
        """
        return variance_partially_synchronized(
            K=K,
            r_inf=r_inf,
            omega_inf=omega_inf,
            lambda_param=lambda_param,
            g_omega=self.g_omega,
        )
    
    def compute_variance_incoherent(self) -> Tuple[float, float]:
        """
        Compute variance for incoherent state.
        
        Returns:
            Tuple of (variance, mean_order_param)
        """
        return variance_incoherent()

    def fit_covariance_params(
        self,
        tau: np.ndarray,
        R_emp: np.ndarray,
        mode: str = "ps",
        K_grid: Tuple[float, float, int] = (0.5, 10.0, 25),
        lambda_grid: Tuple[float, float, int] = (0.0, np.pi / 2.0, 13),
        r_inf: Optional[float] = None,
        omega_inf: float = 0.0,
    ) -> Dict[str, float]:
        """
        Coarse grid search to fit K, lambda to empirical covariance R_emp(tau).
        For 'ps' mode, requires r_inf; for 'incoh', r_inf is ignored.
        Args:
            tau: [T] time-lag grid
            R_emp: [T] complex empirical covariance
            mode: 'ps' or 'incoh'
            K_grid: (min, max, num)
            lambda_grid: (min, max, num)
            r_inf: required for 'ps' (use empirical mean |Z| as proxy)
            omega_inf: rotation frequency (use estimate from Z phase drift)
        Returns:
            dict with best 'K', 'lambda', and 'loss'
        """
        tau = np.asarray(tau, dtype=float)
        R_emp = np.asarray(R_emp, dtype=np.complex128)
        K_vals = np.linspace(*K_grid)
        lam_vals = np.linspace(*lambda_grid)
        best = {"loss": float("inf"), "K": None, "lambda": None}
        for K in K_vals:
            for lam in lam_vals:
                if mode == "ps":
                    if r_inf is None:
                        raise ValueError("fit_covariance_params: r_inf is required for 'ps'")
                    R_th = np.array(
                        [
                            covariance_partially_synchronized(
                                t,
                                K=K,
                                r_inf=float(r_inf),
                                omega_inf=omega_inf,
                                lambda_param=lam,
                                g_omega=self.g_omega,
                            )
                            for t in tau
                        ],
                        dtype=np.complex128,
                    )
                else:
                    R_th = np.array(
                        [covariance_incoherent(t, g_omega=self.g_omega) for t in tau],
                        dtype=np.complex128,
                    )
                # Relative L2 loss
                denom = np.linalg.norm(R_emp) + 1e-12
                loss = float(np.linalg.norm(R_emp - R_th) / denom)
                if loss < best["loss"]:
                    best.update({"loss": loss, "K": float(K), "lambda": float(lam)})
        return best
    
    def estimate_thermodynamic_params(
        self,
        K: float,
        lambda_param: float = 0.0,
        r_init: float = 0.5,
        omega_init: float = 0.0,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> Tuple[float, float]:
        """
        Estimate thermodynamic parameters (r_∞, Ω_∞) using self-consistency equation (Eq. 4).
        
        This is a simplified estimation - full solution requires solving the integral equation.
        
        Args:
            K: Coupling strength
            lambda_param: Phase lag parameter
            r_init: Initial guess for r_∞
            omega_init: Initial guess for Ω_∞
            tol: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            Tuple of (r_inf, omega_inf)
        """
        # Simplified estimation - in practice, this requires solving Eq. 4
        # For now, return reasonable estimates based on coupling strength
        if K < 1.0:
            # Subcritical: incoherent state
            r_inf = 0.0
            omega_inf = omega_init
        else:
            # Supercritical: partially synchronized
            # Rough estimate: r_∞ increases with K
            r_inf = min(0.9, 0.5 * (K - 1.0) / max(1.0, K))
            omega_inf = omega_init
        
        return r_inf, omega_inf


def compare_analytical_numerical(
    analytical_variance: float,
    numerical_variance: torch.Tensor,
    numerical_order_param: torch.Tensor,
    N: int,
) -> Dict[str, float]:
    """
    Compare analytical predictions with numerical simulation results.
    
    Args:
        analytical_variance: Analytical variance prediction
        numerical_variance: Numerical variance from simulation [batch]
        numerical_order_param: Numerical order parameter [batch]
        N: Number of oscillators (sequence length)
        
    Returns:
        Dictionary with comparison metrics
    """
    num_var_mean = float(numerical_variance.mean().item())
    num_var_std = float(numerical_variance.std().item())
    num_R_mean = float(numerical_order_param.mean().item())
    
    # Scale analytical variance by 1/N (finite-size correction)
    analytical_variance_scaled = analytical_variance / N
    
    comparison = {
        "analytical_variance": analytical_variance_scaled,
        "numerical_variance_mean": num_var_mean,
        "numerical_variance_std": num_var_std,
        "numerical_order_param_mean": num_R_mean,
        "variance_ratio": num_var_mean / max(1e-10, analytical_variance_scaled),
        "variance_error": abs(num_var_mean - analytical_variance_scaled),
        "relative_error": abs(num_var_mean - analytical_variance_scaled) / max(1e-10, analytical_variance_scaled),
    }
    
    return comparison

