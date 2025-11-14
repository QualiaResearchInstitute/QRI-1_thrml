from __future__ import annotations

from typing import Dict, Optional, Tuple
import math
import numpy as np


def compute_local_order_parameter_2d(phases: np.ndarray, times: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute time-averaged local order parameter field on a 2D grid:
        A(x, y) e^{iϕ(x, y)} = ⟨e^{i θ(x, y, t)}⟩_t
    Args:
        phases: array [T, Ny, Nx] or [Ny, Nx, T] of phases (radians)
        times: optional [T] timestamps (unused; included for API symmetry)
    Returns:
        complex array M of shape [Ny, Nx]
    """
    arr = np.asarray(phases)
    if arr.ndim != 3:
        raise ValueError(f"phases must be 3D [T, Ny, Nx] or [Ny, Nx, T], got {arr.shape}")
    # Normalize to [T, Ny, Nx]
    if arr.shape[0] not in (arr.shape[-1],) or arr.shape[0] != arr.shape[-1]:
        # If ambiguous, assume [T, Ny, Nx] when T != Ny and T != Nx
        pass
    if arr.shape[0] in (arr.shape[1], arr.shape[2]):
        # If T equals Ny or Nx, prefer time-major if provided as [T, Ny, Nx]; else swap
        # Prefer interpreting last axis as time if shape[-1] is "T-like" (more samples)
        if arr.shape[-1] > 8 and arr.shape[-1] != arr.shape[1] and arr.shape[-1] != arr.shape[2]:
            arr = np.moveaxis(arr, -1, 0)
    if arr.shape[0] <= 2 and arr.shape[-1] > 2:
        # Likely [Ny, Nx, T]
        arr = np.moveaxis(arr, -1, 0)
    if arr.ndim != 3:
        raise ValueError("Failed to normalize phases shape")
    # Time average of complex phase
    M = np.mean(np.exp(1j * arr), axis=0)
    return M


def estimate_plane_wave_k_and_A(
    phases: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    times: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Estimate dominant plane-wave wavevector k and amplitude A for a partially coherent twisted state:
      M(x,y) = ⟨e^{i θ(x,y,t)}⟩_t ≈ A * exp(i (k_x x + k_y y))
    Steps:
      1) Compute M = ⟨exp(i θ)⟩_t
      2) Compute 2D FFT of M; pick (kx, ky) at peak magnitude
      3) Estimate amplitude A ≈ mean(|M|)
    Args:
      phases: [T, Ny, Nx] or [Ny, Nx, T]
      x: [Ny, Nx] or [Nx] grid x-coordinates
      y: [Ny, Nx] or [Ny] grid y-coordinates
      times: optional [T] timestamps
    Returns:
      dict with keys:
        - k: np.array([kx, ky])
        - A: float amplitude
        - M: complex field [Ny, Nx]
        - kgrid: dict with 'kx', 'ky' frequency grids [Ny, Nx]
    """
    M = compute_local_order_parameter_2d(phases, times)
    Ny, Nx = M.shape
    # Build dx, dy
    if x.ndim == 2:
        xs = x[0, :]
        ys = y[:, 0]
    else:
        xs = np.asarray(x)
        ys = np.asarray(y)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    # FFT and frequency mapping
    F = np.fft.fft2(M)
    Fshift = np.fft.fftshift(F)
    kx = 2.0 * math.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    ky = 2.0 * math.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    mag = np.abs(Fshift)
    peak_idx = np.unravel_index(np.argmax(mag), mag.shape)
    kx_hat = float(KX[peak_idx])
    ky_hat = float(KY[peak_idx])
    A_hat = float(np.mean(np.abs(M)))
    return {
        "k": np.array([kx_hat, ky_hat], dtype=float),
        "A": np.array(A_hat, dtype=float),
        "M": M,
        "kgrid": {"kx": KX, "ky": KY},
    }


def kernel_spectrum_fft(G_grid: np.ndarray, dx: float, dy: float) -> Dict[str, np.ndarray]:
    """
    Compute 2D Fourier spectrum Ĝ(kx, ky) on the torus for a sampled coupling kernel G(x, y).
    Returns k-grid and spectrum (fftshifted), and finite-difference gradients and Hessian components.
    Args:
      G_grid: [Ny, Nx] samples of G(x,y) on uniform grid with spacings dx, dy
    Returns:
      dict with keys:
        - Ghat: complex [Ny, Nx] (fftshifted)
        - kx, ky: [Ny, Nx] grids
        - dGhat: dict with 'dkx', 'dky' (complex [Ny,Nx])
        - H: dict with 'd2kx', 'd2ky', 'd2kxky' (complex [Ny,Nx])
    """
    Ny, Nx = G_grid.shape
    Ghat = np.fft.fftshift(np.fft.fft2(G_grid)) * dx * dy
    kx = 2.0 * math.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    ky = 2.0 * math.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    # Finite differences in k-space grid spacing
    dkx = float(kx[1] - kx[0])
    dky = float(ky[1] - ky[0])
    # Central differences with periodic wrap
    def cdiff_x(A):
        return (np.roll(A, -1, axis=1) - np.roll(A, 1, axis=1)) / (2.0 * dkx)
    def cdiff_y(A):
        return (np.roll(A, -1, axis=0) - np.roll(A, 1, axis=0)) / (2.0 * dky)
    def c2diff_x(A):
        return (np.roll(A, -1, axis=1) - 2.0 * A + np.roll(A, 1, axis=1)) / (dkx * dkx)
    def c2diff_y(A):
        return (np.roll(A, -1, axis=0) - 2.0 * A + np.roll(A, 1, axis=0)) / (dky * dky)
    dG_kx = cdiff_x(Ghat)
    dG_ky = cdiff_y(Ghat)
    d2G_kx = c2diff_x(Ghat)
    d2G_ky = c2diff_y(Ghat)
    d2G_kxky = cdiff_y(cdiff_x(Ghat))
    return {
        "Ghat": Ghat,
        "kx": KX,
        "ky": KY,
        "dGhat": {"dkx": dG_kx, "dky": dG_ky},
        "H": {"d2kx": d2G_kx, "d2ky": d2G_ky, "d2kxky": d2G_kxky},
    }


def check_long_wave_stability(
    k: np.ndarray,
    spectrum: Dict[str, np.ndarray],
    alpha: float,
    gamma: float,
) -> Dict[str, object]:
    """
    Provide a practical necessary stability check for a twisted plane wave at wavevector k.
    This implements approximate criteria inspired by the paper's long-wave conditions:
      - Existence proxy: Re(Ĝ(k)) * cos(beta) large enough vs heterogeneity (gamma)
      - Long-wave stability proxy: Ĝ has local maximum at k (Hessian negative semidefinite)
    Note: This is a necessary (not sufficient) check and uses numerical derivatives of Ĝ.
    Args:
      k: array([kx, ky])
      spectrum: dict from kernel_spectrum_fft, with Ghat, kx, ky, H
      alpha: Sakaguchi phase lag α (β = π/2 − α used in theory)
      gamma: Lorentzian half-width (heterogeneity)
    Returns:
      dict with:
        - exists_proxy: bool
        - long_wave_stable_proxy: bool
        - details: dict with local values and Hessian eigenvalues
    """
    KX = spectrum["kx"]
    KY = spectrum["ky"]
    Ghat = spectrum["Ghat"]
    H = spectrum["H"]
    # Find nearest index to k on the grid
    idx_kx = np.argmin(np.abs(KX[0, :] - k[0]))
    idx_ky = np.argmin(np.abs(KY[:, 0] - k[1]))
    Gk = Ghat[idx_ky, idx_kx]
    # Proxy existence: require Re(Ĝ(k)) * cosβ > c * gamma, with c ≈ 0 (unknown exact prefactor)
    beta = math.pi / 2.0 - float(alpha)
    exists_proxy = (np.real(Gk) * math.cos(beta)) > (1e-12 + 0.0 * gamma)
    # Hessian at k for long-wave maximum (negative semidefinite)
    Hxx = H["d2kx"][idx_ky, idx_kx]
    Hyy = H["d2ky"][idx_ky, idx_kx]
    Hxy = H["d2kxky"][idx_ky, idx_kx]
    Hmat = np.array([[np.real(Hxx), np.real(Hxy)], [np.real(Hxy), np.real(Hyy)]], dtype=float)
    evals = np.linalg.eigvalsh(Hmat)
    long_wave_stable_proxy = bool(np.max(evals) <= 0.0 + 1e-10)
    return {
        "exists_proxy": bool(exists_proxy),
        "long_wave_stable_proxy": long_wave_stable_proxy,
        "details": {
            "Ghat_at_k": Gk,
            "beta": beta,
            "Hessian_evals": evals,
            "Hessian": Hmat,
            "grid_index": (int(idx_ky), int(idx_kx)),
        },
    }


