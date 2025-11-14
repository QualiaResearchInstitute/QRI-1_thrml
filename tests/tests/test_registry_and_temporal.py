#!/usr/bin/env python3
from __future__ import annotations

import torch
import pytest

try:
    # Underscore shim dynamically loads resonance-transformer/resonance_transformer.py
    from resonance_transformer import ResonanceAttentionHead
except Exception as e:
    pytest.skip(f"resonance_transformer import failed ({e}); skipping registry/temporal tests", allow_module_level=True)


def _make_inputs(B: int = 2, T: int = 32, C: int = 32):
    torch.manual_seed(0)
    x = torch.randn(B, T, C)
    return x


def test_registry_blend_kernel_basic():
    """
    Ensure BlendKernel wiring from modules.kernels works in the production head:
      - Forward executes without error
      - Shapes are preserved
      - Outputs are finite
      - Blend weights parameter exists when using 'blend'
    """
    B, T, C = 2, 24, 32
    x = _make_inputs(B, T, C)

    head = ResonanceAttentionHead(
        d_model=C,
        d_k=C,
        d_v=C,
        n_sim_steps=5,
        use_sakaguchi=True,
        use_critical_tuning=False,   # keep simple for test speed
        use_coupling_kernel=False,   # bypass builtin kernel; use registry instead
        use_stuart_landau=True,
        use_heun=True,
        track_metrics=True,
        # Registry kernels
        use_registry_kernel=True,
        registry_kernel_type="blend",
        registry_rank=8,
        registry_temperature=1.0,
    )
    y = head(x)
    assert y.shape == x.shape, f"Output shape mismatch: got {y.shape}, expected {x.shape}"
    assert torch.isfinite(y).all(), "Non-finite values in head output with registry blend kernel"

    # Validate sub-kernel blend weights exist
    rk = getattr(head, "registry_kernel", None)
    assert rk is not None, "registry_kernel not constructed"
    assert hasattr(rk, "w_logits"), "BlendKernel should expose w_logits for mixing"
    assert rk.w_logits.numel() >= 3, "Expected at least 3 sub-kernel logits (learned, gaussian, alternating)"


def test_temporal_multiplex_learned_mix_has_grad():
    """
    Validate temporal multiplexing scaffolding:
      - Multiple streams run with distinct dt/alpha offsets
      - Learned mixing logits receive gradients
    """
    B, T, C = 2, 20, 32
    x = _make_inputs(B, T, C)

    head = ResonanceAttentionHead(
        d_model=C,
        d_k=C,
        d_v=C,
        n_sim_steps=4,
        use_sakaguchi=True,
        use_critical_tuning=False,
        use_coupling_kernel=False,
        use_stuart_landau=True,
        use_heun=True,
        track_metrics=False,
        # Use default dot-product similarity to keep test fast
        # Temporal multiplexing knobs
        use_temporal_multiplex=True,
        tm_dts=[0.01, 0.02, 0.03],
        tm_alpha_offsets=[0.0, 0.05, -0.03],
        tm_learned_mix=True,
    )

    # Learned mix logits should be present
    assert getattr(head, "tm_logits", None) is not None, "tm_logits not initialized for learned mix"
    assert head.tm_logits.requires_grad, "tm_logits should require grad"

    # Simple forward/backward to confirm gradients propagate
    y = head(x)
    loss = y.abs().mean()
    loss.backward()

    assert head.tm_logits.grad is not None, "No gradient on tm_logits after backward()"
    # Finite numerics end-to-end
    assert torch.isfinite(y).all(), "Non-finite values in temporal multiplexing output"
