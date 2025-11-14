"""
SpectralThresholdHead
- Multiresolution DCT-like depthwise conv bank + soft-thresholding per band
- ANE-friendly: Conv1d (depthwise + pointwise), abs/relus/sub/add/mul; no control flow

API:
    SpectralThresholdHead(
        d_model: int,
        bands: int = 4,
        fixed_basis: str = "dct8",   # "dct8" | "haar" | "learned"
        kernel_size: int = 8,
        learn_thresholds: bool = True,
    )

I/O:
    x: [B, T, C] -> y: [B, T, C]

Implementation details:
- Depthwise conv bank: in=C, out=C*bands, groups=C, kernel=K (default 8)
  Each input channel has 'bands' filters shaped like DCT-II modes k=0..bands-1 (subset of size K)
  We zero-pad/center-pad to kernel_size and use same padding for shape preservation
- Soft-threshold per band:
    y = sign(v) * relu(|v| - tau_b), with tau_b = softplus(tau_raw_b) >= 0
  Thresholds are per band (shared across channels); optional to extend to per-(channel,band)
- Mix-back: grouped 1x1 conv mapping C*bands -> C (groups=C)

Notes:
- If fixed_basis != "learned", the depthwise bank weights are frozen (requires_grad=False)
- Uses only elementwise ops + convs to ease Core ML (ANE) conversion
"""
from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_dct_kernel(K: int, bands: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Build DCT-II basis filters of length K for the lowest 'bands' frequencies.
    Returns tensor of shape [bands, K], normalized.
    """
    # DCT-II: X_k[n] = alpha(k) * cos( pi/K * (n + 0.5) * k ), n=0..K-1
    n = torch.arange(K, device=device, dtype=dtype).unsqueeze(0)  # [1, K]
    k = torch.arange(bands, device=device, dtype=dtype).unsqueeze(1)  # [bands, 1]
    # Avoid degenerate band > K, clamp
    k = torch.clamp(k, max=K - 1)
    alpha = torch.ones(bands, device=device, dtype=dtype) * math.sqrt(2.0 / K)
    if bands > 0:
        alpha[0] = math.sqrt(1.0 / K)  # DC normalization
    basis = alpha.unsqueeze(1) * torch.cos(math.pi / K * (n + 0.5) * k)  # [bands, K]
    # Normalize each filter to unit L2
    basis = basis / (torch.linalg.norm(basis, dim=1, keepdim=True) + 1e-12)
    return basis  # [bands, K]


def _soft_threshold(x: torch.Tensor, tau: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft-threshold with per-band tau.
    x: [B, C, bands, T]
    tau: [bands] (broadcast along B,C,T)
    """
    mag = torch.abs(x)
    sign = x / (mag + eps)
    # tau is non-negative via softplus applied at call site
    y = torch.relu(mag - tau.view(1, 1, -1, 1)) * sign
    return y


class SpectralThresholdHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        bands: int = 4,
        fixed_basis: str = "dct8",   # "dct8" | "haar" | "learned"
        kernel_size: int = 8,
        learn_thresholds: bool = True,
    ):
        super().__init__()
        assert d_model > 0 and d_model % 1 == 0
        assert bands >= 1
        assert kernel_size >= 2
        self.d_model = int(d_model)
        self.bands = int(bands)
        self.kernel_size = int(kernel_size)
        self.fixed_basis = str(fixed_basis).lower()
        self.learn_thresholds = bool(learn_thresholds)

        C = self.d_model
        BANDS = self.bands
        K = self.kernel_size

        # Depthwise conv bank: in=C, out=C*bands, groups=C
        # padding = K//2 for near "same" length
        self.pad = K // 2
        self.bank = nn.Conv1d(
            in_channels=C,
            out_channels=C * BANDS,
            kernel_size=K,
            stride=1,
            padding=self.pad,
            groups=C,
            bias=False,
        )

        # Initialize basis
        with torch.no_grad():
            if self.fixed_basis in ("dct", "dct8", "fixed"):
                # Build DCT filters
                # weight shape: [C*bands, 1, K], group by channel with bands filters each
                # We'll fill for a dummy device/dtype now; they will be moved by module.to(...)
                basis = _build_dct_kernel(K, BANDS, dtype=torch.float32, device=torch.device("cpu"))  # [bands, K]
                # Prepare weight
                w = torch.zeros((C * BANDS, 1, K), dtype=torch.float32)
                for c in range(C):
                    for b in range(BANDS):
                        w[c * BANDS + b, 0, :] = basis[b]
                self.bank.weight.copy_(w)
                # Freeze basis weights to ensure deterministic export
                self.bank.weight.requires_grad = (self.fixed_basis == "learned")
            elif self.fixed_basis == "haar":
                # Simple Haar-like low/high pairs across K; use first two bands if available
                w = torch.zeros((C * BANDS, 1, K), dtype=torch.float32)
                half = K // 2
                low = torch.cat([torch.ones(half), -torch.ones(K - half)], dim=0) / math.sqrt(K)
                high = torch.cat([torch.ones(half), torch.ones(K - half)], dim=0) / math.sqrt(K)
                low = low.to(w.dtype)
                high = high.to(w.dtype)
                for c in range(C):
                    for b in range(BANDS):
                        w[c * BANDS + b, 0, :] = low if (b % 2 == 0) else high
                self.bank.weight.copy_(w)
                self.bank.weight.requires_grad = (self.fixed_basis == "learned")
            else:
                # learned basis: random small init
                nn.init.kaiming_normal_(self.bank.weight, nonlinearity="linear")
                self.bank.weight.requires_grad = True

        # Per-band thresholds (shared across channels); parameterized via softplus
        tau_init = torch.full((BANDS,), 0.01, dtype=torch.float32)
        if self.learn_thresholds:
            self.tau_raw = nn.Parameter(tau_init)  # learned raw, passed through softplus
        else:
            self.register_buffer("tau_raw", tau_init, persistent=False)

        # Mix back: grouped 1x1 conv mapping C*bands -> C, groups=C
        self.mix = nn.Conv1d(
            in_channels=C * BANDS,
            out_channels=C,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=C,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] -> y: [B, T, C]
        """
        assert x.dim() == 3 and x.shape[-1] == self.d_model, "Expected x[B, T, C]"
        B, T, C = x.shape
        x_ct = x.transpose(1, 2)  # [B, C, T]

        # Depthwise bank
        y_bank = self.bank(x_ct)  # [B, C*bands, T_out]

        # Ensure time dimension matches input T (center-crop or pad as needed)
        L = y_bank.shape[-1]
        if L != T:
            if L > T:
                # Center-crop to T
                start = (L - T) // 2
                y_bank = y_bank[:, :, start:start + T]
            else:
                # Pad equally left/right to reach T
                pad_total = T - L
                left = pad_total // 2
                right = pad_total - left
                y_bank = F.pad(y_bank, (left, right))

        # Reshape to [B, C, bands, T]
        y_resh = y_bank.view(B, C, self.bands, T)

        # Softplus for non-negative thresholds
        tau = F.softplus(self.tau_raw)  # [bands]

        # Soft-threshold per band
        y_thr = _soft_threshold(y_resh, tau)  # [B, C, bands, T]

        # Back to [B, C*bands, T]
        y_thr_ct = y_thr.view(B, C * self.bands, T)

        # Mix back to C via grouped 1x1 conv
        y_ct = self.mix(y_thr_ct)  # [B, C, T]

        # Return [B, T, C]
        y = y_ct.transpose(1, 2)
        return y

    def freeze_basis(self) -> None:
        """Freeze depthwise bank weights (for export)."""
        self.bank.weight.requires_grad = False
