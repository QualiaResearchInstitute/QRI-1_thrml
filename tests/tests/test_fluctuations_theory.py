import numpy as np
from modules.kuramoto_analytics import (
    KuramotoAnalytics,
    covariance_incoherent,
    variance_incoherent,
)


def test_incoherent_variance_and_covariance():
    ka = KuramotoAnalytics(frequency_distribution="gaussian", distribution_params={"mu": 0.0, "sigma": 1.0})
    # Check Rincoh(0) ~ 1 (definition) and Vincoh = 1 - π/4
    R0 = covariance_incoherent(0.0, g_omega=ka.g_omega)
    assert np.isclose(np.real(R0), 1.0, atol=1e-6)
    V_expected, _ = variance_incoherent()
    assert np.isclose(V_expected, 1.0 - np.pi / 4.0, atol=1e-12)


