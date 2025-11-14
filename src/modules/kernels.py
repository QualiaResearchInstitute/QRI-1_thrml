"""
Coupling kernel registry for resonance transformer.

Provides various kernel types for modulating coupling strength:
- GaussianDistanceKernel: Distance-based Gaussian coupling
- AlternatingKernel: Alternating pattern coupling
- LearnedMLPKernel: Learned MLP-based coupling
- BlendKernel: Blends multiple kernels with learnable weights
"""

import torch
import torch.nn as nn
from typing import Optional, List


class GaussianDistanceKernel(nn.Module):
    """Gaussian distance-based coupling kernel."""
    
    def __init__(self, learnable: bool = True):
        super().__init__()
        if learnable:
            self.sigma = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('sigma', torch.tensor(1.0))
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Gaussian distance kernel."""
        # Stub: return identity coupling for now
        batch_size, seq_len, d_k = Q.shape
        device = Q.device
        dtype = Q.dtype
        return torch.ones(batch_size, seq_len, seq_len, device=device, dtype=dtype)


class AlternatingKernel(nn.Module):
    """Alternating pattern coupling kernel."""
    
    def __init__(self, learnable: bool = True):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('alpha', torch.tensor(0.5))
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute alternating kernel."""
        # Stub: return identity coupling for now
        batch_size, seq_len, d_k = Q.shape
        device = Q.device
        dtype = Q.dtype
        return torch.ones(batch_size, seq_len, seq_len, device=device, dtype=dtype)


class LearnedMLPKernel(nn.Module):
    """Learned MLP-based coupling kernel."""
    
    def __init__(self, d_k: int, rank: int = 32):
        super().__init__()
        self.d_k = d_k
        self.rank = rank
        # Stub: minimal MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_k * 2, rank),
            nn.ReLU(),
            nn.Linear(rank, 1)
        )
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute learned MLP kernel."""
        batch_size, seq_len, d_k = Q.shape
        # Stub: return basic QK^T scaled
        scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)
        return scores


class BlendKernel(nn.Module):
    """Blends multiple kernels with learnable weights."""
    
    def __init__(self, kernels: List[nn.Module], learnable: bool = True, temperature: float = 1.0):
        super().__init__()
        self.kernels = nn.ModuleList(kernels)
        self.temperature = temperature
        if learnable:
            self.w_logits = nn.Parameter(torch.zeros(len(kernels)))
        else:
            self.register_buffer('w_logits', torch.zeros(len(kernels)))
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Blend kernels with learned weights."""
        weights = torch.softmax(self.w_logits / self.temperature, dim=0)
        result = None
        for i, kernel in enumerate(self.kernels):
            k_out = kernel(Q, K, pos, mask)
            if result is None:
                result = weights[i] * k_out
            else:
                result = result + weights[i] * k_out
        return result

