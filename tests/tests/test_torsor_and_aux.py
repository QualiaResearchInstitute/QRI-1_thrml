#!/usr/bin/env python3
from __future__ import annotations

import torch
import pytest

try:
    # Import production class via package shim
    from resonance_transformer import ResonanceAttentionHead
except Exception as e:
    pytest.skip(f"resonance_transformer import failed ({e}); skipping torsor/aux tests", allow_module_level=True)


def _make_inputs(B: int = 2, T: int = 16, C: int = 32):
    torch.manual_seed(0)
    return torch.randn(B, T, C)


def test_torsor_regularizer_smoke():
    """
    Enable torsor/geometry regularizer and ensure it contributes a finite term
    to metrics['regularizers'] without crashing the forward.
    """
    B, T, C = 2, 24, 32
    x = _make_inputs(B, T, C)

    head = ResonanceAttentionHead(
        d_model=C,
        d_k=C,
        d_v=C,
        n_sim_steps=4,
        use_sakaguchi=True,
        use_critical_tuning=False,  # keep dynamics simple for test
        use_coupling_kernel=False,  # use raw dot-product similarity
        use_stuart_landau=True,
        use_heun=True,
        track_metrics=True,
        lambda_frustration=0.1,     # turn on torsor loss
    )

    y = head(x)  # forward
    assert y.shape == x.shape, f"Output shape mismatch: got {y.shape}, expected {x.shape}"
    assert torch.isfinite(y).all(), "Non-finite values in head output with torsor regularizer"

    # Check metrics
    m = getattr(head, "metrics", {}) or {}
    assert "regularizers" in m, "regularizers missing in metrics"
    reg = m["regularizers"]
    assert isinstance(reg, dict), "regularizers should be a dict"
    assert "torsor_loss" in reg, "torsor_loss not present in regularizers when lambda_frustration > 0"
    torsor = reg["torsor_loss"]
    assert torch.is_tensor(torsor), "torsor_loss should be a tensor"
    assert torch.isfinite(torsor).all(), "Non-finite torsor_loss"
    # torsor is per-batch, non-negative
    assert (torsor >= 0).all(), "torsor_loss should be non-negative"


def test_aux_interference_has_grad():
    """
    Enable auxiliary interference readout and confirm:
      - Forward executes without error
      - aux_logit receives gradients
      - Optional projection W_auxo receives gradients
    """
    B, T, C = 2, 20, 32
    x = _make_inputs(B, T, C)

    head = ResonanceAttentionHead(
        d_model=C,
        d_k=C,
        d_v=C,
        n_sim_steps=3,
        use_sakaguchi=True,
        use_critical_tuning=False,
        use_coupling_kernel=False,
        use_stuart_landau=True,
        use_heun=True,
        track_metrics=True,
        # Auxiliary interference knobs
        use_aux_interference=True,
        aux_iters=2,
        aux_step=0.05,
        aux_eps=1e-2,
        aux_mix_init=0.1,
    )

    # Ensure parameter exists and requires grad
    assert getattr(head, "aux_logit", None) is not None, "aux_logit was not initialized"
    assert head.aux_logit.requires_grad, "aux_logit should require grad when use_aux_interference=True"

    y = head(x)
    assert y.shape == x.shape, "Output shape mismatch"
    assert torch.isfinite(y).all(), "Non-finite values with auxiliary interference readout"

    loss = y.abs().mean()
    loss.backward()

    assert head.aux_logit.grad is not None, "No gradient on aux_logit after backward()"
    assert torch.isfinite(head.aux_logit.grad).all(), "Non-finite gradient on aux_logit"

    # If projection exists, it should also receive gradients
    if getattr(head, "W_auxo", None) is not None:
        assert head.W_auxo.weight.grad is not None, "No gradient on W_auxo weights"
        assert torch.isfinite(head.W_auxo.weight.grad).all(), "Non-finite gradient on W_auxo weights"

    # Optional metric emitted
    m = getattr(head, "metrics", {}) or {}
    if "aux_gamma" in m:
        _ = m["aux_gamma"]  # presence check only
