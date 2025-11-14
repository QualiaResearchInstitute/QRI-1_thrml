"""
BeamSplitterUnitaryStack
- Learned linear-optics style 2x2 unitary mixes tiled over channel pairs
- Implemented as grouped 1x1 Conv1D (kernel_size=1, groups=C/2) over [B, T, C]
- Optional per-pair phase shifter applied as another 1x1 grouped conv (rotation matrix)
- ANE-friendly: Conv1x1 + elementwise sin/cos and adds; fp16/bfloat16 compatible

Notes:
- Channels C must be even (pairs of 2)
- We keep real-valued tensors and treat consecutive channels as complex pairs if desired
- Unitary retraction uses fast 2x2 polar via SVD on batched groups

API:
    BeamSplitterUnitaryStack(d_model, n_layers=2, unitary_project=True, phase_shift=True)

Input/Output:
    x: [B, T, C] float tensor (C % 2 == 0)
    returns: same shape [B, T, C]
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _project_to_unitary_2x2(W: torch.Tensor) -> torch.Tensor:
    """
    Project batched 2x2 real matrices to closest orthonormal (unitary over R^2) via polar decomposition.
    Args:
        W: [..., 2, 2] real matrices
    Returns:
        U: [..., 2, 2] with U^T U = I
    """
    # Batched SVD
    # W = U S V^T => closest orthonormal is U V^T
    U, _, Vh = torch.linalg.svd(W, full_matrices=False)
    return U @ Vh


def _pairs_to_grouped_conv1d_weight(W_pairs: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert batched per-pair 2x2 matrices to Conv1d grouped weight tensor.

    Args:
        W_pairs: [G, 2, 2] where G = C/2
    Returns:
        weight: [C_out, C_in/groups, 1] where C_out=C, groups=G, C_in/groups=2
    """
    G = W_pairs.shape[0]
    # For Conv1d grouped weight, each group has shape [out_per_group=2, in_per_group=2, 1]
    weight = torch.zeros((2 * G, 2, 1), device=device, dtype=dtype)
    # Fill group-wise kernels
    weight.view(G, 2, 2, 1)[:, :, :, 0] = W_pairs
    return weight


class BeamSplitterUnitaryLayer(nn.Module):
    """
    Single layer of beam-splitter 2x2 unitary mixes over channel pairs with optional phase shift.
    """
    def __init__(self, d_model: int, unitary_project: bool = True, phase_shift: bool = True):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for 2x2 channel pairs."
        self.d_model = d_model
        self.G = d_model // 2  # number of 2x2 groups
        self.unitary_project = bool(unitary_project)
        self.phase_shift = bool(phase_shift)

        # Unconstrained parameters for each 2x2 pair: initialize near identity
        # Param stored as [G, 2, 2]
        W0 = torch.eye(2).repeat(self.G, 1, 1)
        W0 = W0 + 0.01 * torch.randn_like(W0)
        self.W_pairs = nn.Parameter(W0)  # learned 2x2 per group

        # Optional per-pair phase φ (rotation matrix)
        if self.phase_shift:
            self.phase = nn.Parameter(torch.zeros(self.G))  # radians

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        """
        assert x.dim() == 3 and x.shape[-1] == self.d_model, "Expected x[B, T, C]"
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        # Prepare 1x1 grouped conv weight from projected 2x2s
        W = self.W_pairs
        if self.unitary_project:
            with torch.no_grad():
                # We retraction-project for stability; if needed, move to training-step hook
                W.copy_(_project_to_unitary_2x2(W.detach()))
        W_eff = W  # [G, 2, 2]

        w_conv = _pairs_to_grouped_conv1d_weight(W_eff, device=device, dtype=dtype)  # [C, 2, 1]

        # Conv1d expects [B, C, T]
        x_ct = x.transpose(1, 2)  # [B, C, T]
        # Apply grouped 1x1 conv
        y_ct = F.conv1d(x_ct, w_conv, bias=None, stride=1, padding=0, groups=self.G)

        # Optional per-pair phase shift: rotation matrix R(φ) = [[cos, -sin],[sin, cos]]
        if self.phase_shift:
            phi = self.phase  # [G]
            cos_p = torch.cos(phi).to(dtype=dtype, device=device)
            sin_p = torch.sin(phi).to(dtype=dtype, device=device)
            # Build batched rotation matrices [G, 2, 2]
            R = torch.zeros((self.G, 2, 2), device=device, dtype=dtype)
            R[:, 0, 0] = cos_p
            R[:, 0, 1] = -sin_p
            R[:, 1, 0] = sin_p
            R[:, 1, 1] = cos_p
            w_rot = _pairs_to_grouped_conv1d_weight(R, device=device, dtype=dtype)
            y_ct = F.conv1d(y_ct, w_rot, bias=None, stride=1, padding=0, groups=self.G)

        y = y_ct.transpose(1, 2)  # [B, T, C]
        return y


class BeamSplitterUnitaryStack(nn.Module):
    """
    Stack of beam-splitter layers (2x2 unitaries tiled across channels) + optional per-layer phase-shifts.
    """
    def __init__(self, d_model: int, n_layers: int = 2, unitary_project: bool = True, phase_shift: bool = True):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for 2x2 channel pairs."
        self.d_model = d_model
        self.n_layers = int(n_layers)
        self.layers = nn.ModuleList([
            BeamSplitterUnitaryLayer(d_model, unitary_project=unitary_project, phase_shift=phase_shift)
            for _ in range(self.n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] -> [B, T, C]
        """
        y = x
        for layer in self.layers:
            y = layer(y)
        return y
