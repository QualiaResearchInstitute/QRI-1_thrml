import math
import numpy as np
import pytest

from modules.ser_inference import (
    estimate_omega_and_gap,
    estimate_beta_and_kernel,
)


def test_estimate_omega_and_gap_regression():
    rng = np.random.default_rng(0)
    N = 256
    # Generate random complex zeta with |zeta| in (0, 1)
    mags = rng.uniform(0.0, 0.95, size=N)
    phases = rng.uniform(-math.pi, math.pi, size=N)
    zeta = mags * np.exp(1j * phases)
    # Ground-truth omega and Omega
    omega_true = 1.25
    Omega_true = 0.8
    # Generate Omega_j using SER (7)
    eta = (2.0 * (np.abs(zeta) ** 2)) / (1.0 + (np.abs(zeta) ** 2))
    Omega_j = omega_true + (Omega_true - omega_true) * eta
    # Add small noise
    Omega_j_noisy = Omega_j + 1e-4 * rng.normal(size=N)
    omega_hat, gap_hat = estimate_omega_and_gap(Omega_j_noisy, zeta)
    assert abs(omega_hat - omega_true) < 2e-3
    assert abs((Omega_true - omega_true) - gap_hat) < 2e-3


def test_estimate_beta_and_kernel_runs():
    rng = np.random.default_rng(1)
    N = 128
    # Positions uniformly on ring [-π, π]
    x = -math.pi + (2.0 * math.pi) * (np.arange(N) / N)
    # Random zeta (simulate mixture of coherent/incoherent magnitudes)
    mags = rng.uniform(0.0, 0.99, size=N)
    ph = rng.uniform(-math.pi, math.pi, size=N)
    zeta = mags * np.exp(1j * ph)
    omega = 1.0
    Omega = 0.9
    M = 6
    beta_hat, c = estimate_beta_and_kernel(x, zeta, omega=omega, Omega=Omega, M=M, n_grid=30)
    assert 0.0 <= beta_hat <= math.pi
    assert c.shape == (M + 1,)
    assert np.all(np.isfinite(c))


