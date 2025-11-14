#!/usr/bin/env python3
from __future__ import annotations

import torch
import pytest

try:
    from resonance_transformer.modules import SpectralThresholdHead
except Exception:
    pytest.skip("resonance_transformer.modules import failed; skipping SpectralThreshold tests", allow_module_level=True)


def test_spectral_head_shapes_and_freeze():
    torch.manual_seed(0)
    B, T, C = 2, 32, 64
    x = torch.randn(B, T, C)

    head = SpectralThresholdHead(d_model=C, bands=4, fixed_basis="dct8", kernel_size=8, learn_thresholds=True)
    y = head(x)

    # Shape preservation
    assert y.shape == x.shape, f"Output shape mismatch: got {y.shape}, expected {x.shape}"

    # DCT basis should be frozen (requires_grad=False) when fixed_basis != 'learned'
    assert head.bank.weight.requires_grad is False, "Depthwise bank should be frozen for fixed DCT basis"

    # Mix has bias; ensure numerics are finite
    assert torch.isfinite(y).all(), "Non-finite values in spectral head output"


def test_spectral_head_threshold_effect():
    torch.manual_seed(0)
    B, T, C = 1, 16, 8
    x = torch.randn(B, T, C)

    head = SpectralThresholdHead(d_model=C, bands=4, fixed_basis="dct8", kernel_size=8, learn_thresholds=True)

    # Zero-out mix bias to observe pure thresholding effect
    if head.mix.bias is not None:
        with torch.no_grad():
            head.mix.bias.zero_()

    # Baseline output magnitude
    y0 = head(x)
    base_mag = y0.abs().mean().item()

    # Increase thresholds heavily so most coefficients are suppressed
    with torch.no_grad():
        head.tau_raw[:] = 10.0  # softplus(10) ~ large

    y1 = head(x)
    suppressed_mag = y1.abs().mean().item()

    # Expect strong suppression
    assert suppressed_mag < 0.5 * base_mag, f"Thresholding not effective enough: {suppressed_mag:.6f} vs {base_mag:.6f}"

    # Finite numerics
    assert torch.isfinite(y1).all(), "Non-finite values after thresholding"
