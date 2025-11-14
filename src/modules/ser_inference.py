from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Dict
import math
import numpy as np


def _as_time_major(phases: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Normalize phases array to shape [T, N] matching times length.
    Accepts input as [T, N] or [N, T].
    """
    phases = np.asarray(phases)
    times = np.asarray(times)
    if phases.ndim != 2:
        raise ValueError(f"phases must be 2D (T,N) or (N,T); got shape {phases.shape}")
    T = times.shape[0]
    if phases.shape[0] == T:
        return phases
    if phases.shape[1] == T:
        return np.swapaxes(phases, 0, 1)
    raise ValueError(f"phases shape {phases.shape} is incompatible with times length {T}")


def _compute_periodic_trapezoid_deltas(x_positions: np.ndarray) -> np.ndarray:
    """
    Compute periodic trapezoidal half-widths Δx_k = (x_{k+1} - x_{k-1}) / 2 with periodic boundaries.
    Assumes x_positions are on a ring and ordered.
    """
    x = np.asarray(x_positions, dtype=float)
    N = x.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 positions for periodic trapezoidal rule")
    # Periodize by appending wrapped neighbors
    x_prev = np.roll(x, 1)
    x_next = np.roll(x, -1)
    deltas = 0.5 * (x_next - x_prev)
    # For a ring on [-π, π], adjust wrap differences to remain in principal interval
    # This ensures Δx is positive for uniform grids like x_j = -π + 2π j/N.
    two_pi = 2.0 * math.pi
    deltas = (deltas + math.pi) % (2.0 * math.pi) - math.pi
    return deltas


def compute_global_order_parameter(phases: np.ndarray) -> np.ndarray:
    """
    Compute the global order parameter time series Z(t) = (1/N) Σ_k exp(i θ_k(t)).
    Args:
        phases: array of shape [T, N] time-major (use _as_time_major to normalize).
    Returns:
        complex array Z of shape [T]
    """
    phases = np.asarray(phases)
    return np.mean(np.exp(1j * phases), axis=1)


def compute_effective_frequencies(phases: np.ndarray, times: np.ndarray, indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Compute per-oscillator effective frequencies Ω_j via phase unwrapping:
        Ω_j ≈ (θ_j(t_T) - θ_j(t_0)) / (t_T - t_0)  using unwrapped phase.
    Works with irregular sampling.
    Args:
        phases: [T, N] or [N, T] array of phases in radians
        times: [T] time stamps (monotonic)
        indices: optional subset of oscillator indices to compute; defaults to all
    Returns:
        Ω_j for j in chosen set; shape [N'] where N' = len(indices) or N
    """
    phases_tm = _as_time_major(phases, times)
    T, N = phases_tm.shape
    t0 = float(times[0])
    tT = float(times[-1])
    if not np.isfinite(tT - t0) or (tT <= t0):
        raise ValueError("times must be strictly increasing")
    if indices is None:
        indices = range(N)
    thetas = phases_tm[:, indices]
    thetas_unwrapped = np.unwrap(thetas, axis=0)
    delta = thetas_unwrapped[-1, :] - thetas_unwrapped[0, :]
    return delta / (tT - t0)


def compute_local_order_params(
    phases: np.ndarray,
    times: np.ndarray,
    indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-oscillator local order parameters ζ_j and per-oscillator Ω_j, plus global Ω.
        ζ_j = ⟨ exp(i θ_j(t)) * conj(Z(t)) / |Z(t)| ⟩_t
        Ω_j via unwrapped slope; Ω via unwrapped argument of Z(t)
    Args:
        phases: [T, N] or [N, T]
        times: [T]
        indices: optional subset S; if provided:
            - Z(t) uses only indices in S
            - returns ζ_j and Ω_j for j ∈ S
    Returns:
        (zeta: [N'], Omega_j: [N'], Omega: float)
    """
    phases_tm = _as_time_major(phases, times)
    T, N = phases_tm.shape
    if indices is None:
        indices = np.arange(N)
    indices = np.asarray(list(indices), dtype=int)
    # Z(t) using selected subset
    subset_phases = phases_tm[:, indices]
    Z_t = np.mean(np.exp(1j * subset_phases), axis=1)  # [T]
    Z_abs = np.abs(Z_t)
    # Avoid divide-by-zero: mask near-zero magnitude
    eps = 1e-12
    unit_Z = np.where(Z_abs > eps, np.conj(Z_t) / Z_abs, 0.0)
    # ζ_j time-average
    zeta_t = np.exp(1j * subset_phases) * unit_Z[:, None]  # [T, N']
    zeta = np.mean(zeta_t, axis=0)  # [N']
    # Ω_j and Ω (global from Z(t))
    Omega_j = compute_effective_frequencies(subset_phases, times)
    Z_phase = np.unwrap(np.angle(Z_t))
    Omega = (Z_phase[-1] - Z_phase[0]) / float(times[-1] - times[0])
    return zeta, Omega_j, float(Omega)


def estimate_omega_and_gap(Omega_j: np.ndarray, zeta: np.ndarray) -> Tuple[float, float]:
    """
    Estimate ω and (Ω - ω) using SER (7):
        Ω_j = ω + (Ω - ω) * η_j, where η_j = 2|ζ_j|^2 / (1 + |ζ_j|^2).
    Linear regression yields:
        ω = (S_Ω S_ηη - S_η S_Ωη) / (S_ηη - S_η^2)
        Ω - ω = (S_Ωη - S_η S_Ω) / (S_ηη - S_η^2)
    Returns:
        (omega, Omega_minus_omega)
    """
    Omega_j = np.asarray(Omega_j, dtype=float)
    zeta = np.asarray(zeta, dtype=np.complex128)
    eta = (2.0 * (np.abs(zeta) ** 2)) / (1.0 + (np.abs(zeta) ** 2))
    N = float(eta.shape[0])
    S_eta = np.sum(eta) / N
    S_Omega = np.sum(Omega_j) / N
    S_etaeta = np.sum(eta * eta) / N
    S_OmegaEta = np.sum(Omega_j * eta) / N
    denom = (S_etaeta - (S_eta ** 2))
    if abs(denom) < 1e-14:
        raise ValueError("Degenerate regression: insufficient variation in η_j")
    omega = (S_Omega * S_etaeta - S_eta * S_OmegaEta) / denom
    Omega_minus_omega = (S_OmegaEta - S_eta * S_Omega) / denom
    return float(omega), float(Omega_minus_omega)


def _fourier_basis_cos(x: np.ndarray, M: int) -> np.ndarray:
    """
    Compute cosine basis q_m(x) = cos(m x) for m = 0..M.
    Args:
        x: array of shape [J] of angles
        M: max harmonic
    Returns:
        Q: [J, M+1], Q[:, m] = cos(m * x)
    """
    J = x.shape[0]
    Q = np.empty((J, M + 1), dtype=float)
    for m in range(M + 1):
        Q[:, m] = np.cos(m * x)
    return Q


def _build_Qjm(
    beta: float,
    x_positions: np.ndarray,
    zeta: np.ndarray,
    omega: float,
    Omega: float,
    M: int,
) -> np.ndarray:
    """
    Build Q_jm(β) matrix used in linear system A(β) c = b(β):
        Q_jm(β) = Σ_k [ q_m(x_j - x_k) * Δx_k / (ω - Ω) ] * Re{ e^{iβ} * ζ_k / ζ_j^2 }
    where Δx_k is periodic trapezoid half-width (x_{k+1} - x_{k-1}) / 2.
    """
    x = np.asarray(x_positions, dtype=float)
    zeta = np.asarray(zeta, dtype=np.complex128)
    N = x.shape[0]
    if zeta.shape[0] != N:
        raise ValueError("zeta and x_positions must have the same length")
    if M < 0:
        raise ValueError("M must be non-negative")
    deltas = _compute_periodic_trapezoid_deltas(x)
    # Adjust deltas by 1/2 factor per discrete formula (10)
    deltas = deltas / 2.0
    inv_gap = 1.0 / max(1e-20, (omega - Omega))
    # Precompute pairwise differences x_j - x_k wrapped to [-π, π]
    xx = x[:, None] - x[None, :]
    xx = (xx + math.pi) % (2.0 * math.pi) - math.pi
    Qcos = _fourier_basis_cos(xx.reshape(-1), M).reshape(N, N, M + 1)  # [N, N, M+1]
    # Ratio term per j,k
    ratio = np.exp(1j * beta) * (zeta[None, :] / (zeta[:, None] ** 2))
    re_ratio = np.real(ratio)  # [N, N]
    # Weight by deltas and inv_gap
    weights = inv_gap * (deltas[None, :])  # [1, N]
    # Compute Qjm via sum over k
    Qjm = np.empty((N, M + 1), dtype=float)
    for m in range(M + 1):
        contrib = Qcos[:, :, m] * re_ratio * weights  # [N, N]
        Qjm[:, m] = np.sum(contrib, axis=1)
    return Qjm


def _build_Q_tilde(
    x_positions: np.ndarray,
    zeta: np.ndarray,
    omega: float,
    Omega: float,
    M: int,
) -> np.ndarray:
    """
    Build ˜Q_jm (beta-independent) used in Jincoh(β):
        ˜Q_jm = Σ_k [ q_m(x_j - x_k) * Δx_k / (ω - Ω) ] * ζ_k
    (note: no e^{iβ} nor Re; β enters outside as 2 e^{iβ} c_m in the sum)
    """
    x = np.asarray(x_positions, dtype=float)
    zeta = np.asarray(zeta, dtype=np.complex128)
    N = x.shape[0]
    if zeta.shape[0] != N:
        raise ValueError("zeta and x_positions must have the same length")
    deltas = _compute_periodic_trapezoid_deltas(x) / 2.0
    inv_gap = 1.0 / max(1e-20, (omega - Omega))
    xx = x[:, None] - x[None, :]
    xx = (xx + math.pi) % (2.0 * math.pi) - math.pi
    Qcos = _fourier_basis_cos(xx.reshape(-1), M).reshape(N, N, M + 1)
    weights = inv_gap * (deltas[None, :])
    Qt = np.empty((N, M + 1), dtype=np.complex128)
    for m in range(M + 1):
        contrib = Qcos[:, :, m] * weights * (zeta[None, :])
        Qt[:, m] = np.sum(contrib, axis=1)
    return Qt


def _solve_cm_for_beta(Qjm: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """
    Solve A(β) c = b(β):
        A_nm = Σ_j Q_jn Q_jm
        b_n  = Σ_j 2 Q_jn / (1 + |ζ_j|^2)
    Returns:
        c vector of shape [M+1]
    """
    zeta = np.asarray(zeta, dtype=np.complex128)
    denom = 1.0 + (np.abs(zeta) ** 2)
    A = Qjm.T @ Qjm
    b = 2.0 * (Qjm.T @ (1.0 / denom))
    # Regularize slightly for numerical stability if needed
    reg = 1e-10
    A_reg = A + reg * np.eye(A.shape[0], dtype=float)
    c = np.linalg.solve(A_reg, b)
    return c


def _jincoh(
    beta: float,
    x_positions: np.ndarray,
    zeta: np.ndarray,
    omega: float,
    Omega: float,
    M: int,
) -> Tuple[float, np.ndarray]:
    """
    Compute Jincoh(β) and c(β).
    Jincoh(β) = (1/N*) Σ_{j: |ζ_j| < 1 - 1/√N} || 2 ζ_j / (1 + |ζ_j|^2) - Σ_m 2 e^{iβ} c_m ˜Q_jm ||^2
    Returns:
        (J, c) where c are the Fourier coefficients at this β
    """
    zeta = np.asarray(zeta, dtype=np.complex128)
    N = zeta.shape[0]
    Qjm = _build_Qjm(beta, x_positions, zeta, omega, Omega, M)
    c = _solve_cm_for_beta(Qjm, zeta)
    Qtilde = _build_Q_tilde(x_positions, zeta, omega, Omega, M)  # complex
    incoh_mask = (np.abs(zeta) < (1.0 - 1.0 / math.sqrt(max(1.0, float(N)))))
    if not np.any(incoh_mask):
        # Fall back to all j if mask empty
        incoh_mask = np.ones(N, dtype=bool)
    target = (2.0 * zeta) / (1.0 + (np.abs(zeta) ** 2))  # complex
    recon = 2.0 * np.exp(1j * beta) * (Qtilde @ c)  # complex
    diff = target - recon
    err = np.mean(np.abs(diff[incoh_mask]) ** 2)
    return float(err), c


def _minimize_beta_bruteforce(
    x_positions: np.ndarray,
    zeta: np.ndarray,
    omega: float,
    Omega: float,
    M: int,
    n_grid: int = 60,
) -> Tuple[float, np.ndarray, float]:
    """
    Minimize Jincoh(β) over β ∈ [0, π] via grid search, return (β*, c(β*), J*).
    """
    betas = np.linspace(0.0, math.pi, n_grid)
    best_beta = 0.0
    best_J = float("inf")
    best_c = None
    for b in betas:
        J, c = _jincoh(b, x_positions, zeta, omega, Omega, M)
        if J < best_J:
            best_J = J
            best_beta = float(b)
            best_c = c
    return best_beta, np.asarray(best_c), float(best_J)


def estimate_beta_and_kernel(
    x_positions: np.ndarray,
    zeta: np.ndarray,
    omega: float,
    Omega: float,
    M: int = 10,
    n_grid: int = 60,
) -> Tuple[float, np.ndarray]:
    """
    Estimate phase lag β and Fourier coefficients c_m of symmetric coupling kernel
        G(x) ≈ Σ_{m=0..M} c_m cos(m x)
    using SERs (8), (9) and the Jincoh(β) criterion.
    Args:
        x_positions: [N] oscillator positions on ring (e.g., [-π, π] spaced)
        zeta: [N] local order parameters
        omega: scalar ω
        Omega: scalar Ω
        M: number of spatial Fourier modes (default 10)
        n_grid: number of β grid points in [0, π] (default 60)
    Returns:
        (beta_min, c) where beta_min ∈ [0, π], c has shape [M+1]
    """
    beta_min, c, _ = _minimize_beta_bruteforce(
        x_positions=x_positions,
        zeta=zeta,
        omega=omega,
        Omega=Omega,
        M=M,
        n_grid=n_grid,
    )
    return beta_min, c


def reconstruct_parameters(
    phases: np.ndarray,
    times: np.ndarray,
    x_positions: np.ndarray,
    subset: Optional[Sequence[int]] = None,
    M: int = 10,
    n_grid: int = 60,
) -> Dict[str, np.ndarray]:
    """
    Full pipeline:
      1) Compute ζ_j, Ω_j, Ω (global) from phases and times (optionally using subset indices)
      2) Estimate ω and (Ω - ω) via regression (SER 7)
      3) Estimate β and c_m via Jincoh minimization (SERs 8–9)
    Args:
      phases: [T, N] or [N, T]
      times: [T]
      x_positions: [N]
      subset: optional indices used to compute observables; parameters are reconstructed for that subset
      M: number of Fourier modes
      n_grid: β grid resolution
    Returns:
      dict with keys:
        - zeta: complex [N'] local order parameters
        - Omega_j: float [N'] per-oscillator effective frequencies
        - Omega: float global frequency
        - omega: float estimated natural frequency
        - beta: float estimated phase lag β
        - c: float [M+1] Fourier coefficients of G(x)
    """
    zeta, Omega_j, Omega = compute_local_order_params(phases, times, indices=subset)
    omega, Omega_minus_omega = estimate_omega_and_gap(Omega_j, zeta)
    beta, c = estimate_beta_and_kernel(
        x_positions=(np.asarray(x_positions)[subset] if subset is not None else np.asarray(x_positions)),
        zeta=zeta,
        omega=omega,
        Omega=Omega,
        M=M,
        n_grid=n_grid,
    )
    return {
        "zeta": zeta,
        "Omega_j": Omega_j,
        "Omega": np.array(Omega, dtype=float),
        "omega": np.array(omega, dtype=float),
        "beta": np.array(beta, dtype=float),
        "c": np.asarray(c, dtype=float),
    }


