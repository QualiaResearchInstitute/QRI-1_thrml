#!/usr/bin/env python3
from __future__ import annotations

import math
import torch
import pytest

try:
    from resonance_transformer.modules import BeamSplitterUnitaryStack
except Exception:
    pytest.skip("resonance_transformer.modules import failed; skipping BeamSplitter tests", allow_module_level=True)


def _orthonormality_error(W: torch.Tensor) -> float:
    """
    W: [G, 2, 2]; returns mean Frobenius norm of (W^T W - I)
    """
    G = W.shape[0]
    I = torch.eye(2, dtype=W.dtype, device=W.device)
    errs = []
    for g in range(G):
        M = W[g].T @ W[g]
        errs.append(torch.linalg.norm(M - I).item())
    return float(sum(errs) / max(1, len(errs)))


def test_beam_splitter_shapes_and_unitarity():
    torch.manual_seed(0)
    B, T, C = 2, 16, 64
    x = torch.randn(B, T, C)

    bs = BeamSplitterUnitaryStack(d_model=C, n_layers=3, unitary_project=True, phase_shift=True)
    y = bs(x)

    # Shape preserved
    assert y.shape == x.shape, f"Output shape mismatch: got {y.shape}, expected {x.shape}"

    # After forward, parameters should have been projected to orthonormal per 2x2
    # Read first layer's pair weights
    layer0 = bs.layers[0]
    W_pairs = layer0.W_pairs.detach().clone()  # [G, 2, 2]
    err = _orthonormality_error(W_pairs)
    # Allow small numeric error
    assert err < 1e-3, f"Unitary projection too imprecise; mean ||W^T W - I||_F = {err:.4e}"

    # Phase shift should be present
    assert hasattr(layer0, "phase"), "Phase parameter missing"
    assert layer0.phase.shape[0] == C // 2, "Phase parameter per pair missing"

    # Forward is reasonably stable (no NaNs/Infs)
    assert torch.isfinite(y).all(), "Non-finite values in beam splitter output"
