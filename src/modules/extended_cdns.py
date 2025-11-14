"""
Extended CDNS Metrics: Additional cognitive/complexity measures.

Adds:
- Dimensionality: Effective dimensionality via PCA/eigenvalue analysis
- Integration: Information integration across system components
- Complexity: Statistical complexity measures
- Equivariance: Invariance to transformations
- Entropy: Shannon entropy of distributions
- Flow: Rate of change, temporal dynamics
- Focus: Concentration/peakiness of attention
- Arousal: Overall activation/energy level
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExtendedCDNS:
    """Extended CDNS metrics with additional cognitive measures."""
    # Original CDNS
    consonance: torch.Tensor
    dissonance: torch.Tensor
    noise: Optional[torch.Tensor] = None
    signal: Optional[torch.Tensor] = None
    
    # Extended metrics
    dimensionality: Optional[torch.Tensor] = None
    integration: Optional[torch.Tensor] = None
    complexity: Optional[torch.Tensor] = None
    equivariance: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    flow: Optional[torch.Tensor] = None
    focus: Optional[torch.Tensor] = None
    arousal: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            'consonance': self.consonance,
            'dissonance': self.dissonance,
        }
        if self.noise is not None:
            result['noise'] = self.noise
        if self.signal is not None:
            result['signal'] = self.signal
        if self.dimensionality is not None:
            result['dimensionality'] = self.dimensionality
        if self.integration is not None:
            result['integration'] = self.integration
        if self.complexity is not None:
            result['complexity'] = self.complexity
        if self.equivariance is not None:
            result['equivariance'] = self.equivariance
        if self.entropy is not None:
            result['entropy'] = self.entropy
        if self.flow is not None:
            result['flow'] = self.flow
        if self.focus is not None:
            result['focus'] = self.focus
        if self.arousal is not None:
            result['arousal'] = self.arousal
        return result


def compute_dimensionality(
    coupling_matrix: torch.Tensor,
    threshold: float = 0.01
) -> torch.Tensor:
    """
    Compute effective dimensionality via eigenvalue analysis.
    
    Uses PCA to find how many dimensions capture most variance.
    Returns normalized dimensionality (0-1 scale).
    
    Args:
        coupling_matrix: [batch, seq_len, seq_len] coupling matrix
        threshold: Fraction of variance to capture (default: 0.01 = 99%)
    
    Returns:
        Effective dimensionality [batch]
    """
    batch_size = coupling_matrix.shape[0]
    dims = []
    
    for b in range(batch_size):
        K = coupling_matrix[b]  # [seq_len, seq_len]
        
        # Symmetrize
        K_sym = (K + K.T) / 2
        
        # Compute eigenvalues
        try:
            eigenvals = torch.linalg.eigvalsh(K_sym)  # [seq_len]
            eigenvals = torch.abs(eigenvals)
            eigenvals = torch.sort(eigenvals, descending=True)[0]
            
            # Cumulative variance
            total_var = eigenvals.sum()
            cumsum = torch.cumsum(eigenvals, dim=0)
            cumsum_norm = cumsum / (total_var + 1e-10)
            
            # Find where we capture threshold fraction
            n_effective = torch.sum(cumsum_norm < (1 - threshold)).float() + 1
            n_effective = torch.clamp(n_effective, 1, len(eigenvals))
            
            # Normalize to [0, 1]
            dim_normalized = n_effective / len(eigenvals)
            dims.append(dim_normalized)
        except:
            dims.append(torch.tensor(0.5, device=coupling_matrix.device))
    
    return torch.stack(dims)


def compute_integration(
    phases: torch.Tensor,
    amplitudes: Optional[torch.Tensor] = None,
    coupling_matrix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute information integration measure.
    
    Measures how well-integrated information is across components.
    Uses mutual information or correlation-based measures.
    
    Args:
        phases: [batch, seq_len] phase values
        amplitudes: [batch, seq_len] amplitude values (optional)
        coupling_matrix: [batch, seq_len, seq_len] coupling (optional)
    
    Returns:
        Integration measure [batch] (0 = independent, 1 = fully integrated)
    """
    batch_size, seq_len = phases.shape
    
    # Use complex representation: z = r * exp(i*θ)
    if amplitudes is not None:
        z = amplitudes * torch.exp(1j * phases)  # [batch, seq_len]
    else:
        z = torch.exp(1j * phases)
    
    integrations = []
    for b in range(batch_size):
        z_b = z[b]  # [seq_len]
        z_mean = z_b.mean()
        z_centered = z_b - z_mean
        
        # Normalize
        z_norm = torch.norm(z_centered)
        if z_norm < 1e-10:
            integrations.append(torch.tensor(0.0, device=z.device))
            continue
        
        z_normalized = z_centered / z_norm
        
        # Correlation matrix (outer product)
        corr_matrix = torch.abs(torch.outer(z_normalized, z_normalized.conj()))  # [seq_len, seq_len]
        
        # Average off-diagonal correlation (integration)
        mask = ~torch.eye(seq_len, device=corr_matrix.device, dtype=torch.bool)
        off_diag = corr_matrix[mask]
        integration = off_diag.mean()
        integrations.append(integration.real)
    
    return torch.stack(integrations)


def compute_complexity(
    phases: torch.Tensor,
    amplitudes: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute statistical complexity measure.
    
    Combines entropy and structure - high complexity = high entropy + structure.
    Uses Lempel-Ziv-like complexity or permutation entropy.
    
    Args:
        phases: [batch, seq_len] phase values
        amplitudes: [batch, seq_len] amplitude values (optional)
    
    Returns:
        Complexity measure [batch] (0 = simple, 1 = complex)
    """
    batch_size, seq_len = phases.shape
    
    # Normalize phases to [0, 2π]
    phases_norm = (phases % (2 * np.pi)) / (2 * np.pi)  # [batch, seq_len]
    
    # Discretize for complexity computation
    n_bins = min(10, seq_len // 2)
    phases_discrete = (phases_norm * n_bins).long().clamp(0, n_bins - 1)
    
    complexities = []
    for b in range(batch_size):
        seq = phases_discrete[b].cpu().numpy()
        
        # Lempel-Ziv complexity (normalized)
        def lempel_ziv_complexity(s):
            """Compute normalized LZ complexity."""
            n = len(s)
            if n == 0:
                return 0.0
            
            # Convert to string for pattern matching
            s_str = ''.join(map(str, s))
            
            # Find unique patterns
            patterns = set()
            i = 0
            while i < n:
                j = i + 1
                while j <= n:
                    pattern = s_str[i:j]
                    if pattern not in patterns:
                        patterns.add(pattern)
                        i = j
                        break
                    j += 1
                else:
                    i += 1
            
            # Normalize by theoretical maximum
            max_complexity = n / np.log2(n) if n > 1 else 1
            complexity = len(patterns) / max_complexity if max_complexity > 0 else 0
            return min(complexity, 1.0)
        
        comp = lempel_ziv_complexity(seq)
        complexities.append(torch.tensor(comp, device=phases.device))
    
    return torch.stack(complexities)


def compute_equivariance(
    phases: torch.Tensor,
    coupling_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Compute equivariance measure (invariance to transformations).
    
    Measures how invariant the system is to phase shifts and rotations.
    High equivariance = system maintains structure under transformations.
    
    Args:
        phases: [batch, seq_len] phase values
        coupling_matrix: [batch, seq_len, seq_len] coupling matrix
    
    Returns:
        Equivariance measure [batch] (0 = not equivariant, 1 = fully equivariant)
    """
    batch_size = phases.shape[0]
    
    equivariances = []
    for b in range(batch_size):
        phases_b = phases[b]  # [seq_len]
        K_b = coupling_matrix[b]  # [seq_len, seq_len]
        
        # Test invariance to phase shift
        shift = torch.rand(1, device=phases.device) * 2 * np.pi
        phases_shifted = phases_b + shift
        
        # Compute order parameter before and after shift
        R_original = torch.abs(torch.mean(torch.exp(1j * phases_b)))
        R_shifted = torch.abs(torch.mean(torch.exp(1j * phases_shifted)))
        
        # Order parameter should be invariant to global phase shift
        equivariance = 1.0 - torch.abs(R_original - R_shifted)
        equivariances.append(equivariance)
    
    return torch.stack(equivariances)


def compute_entropy(
    phases: torch.Tensor,
    amplitudes: Optional[torch.Tensor] = None,
    n_bins: int = 20
) -> torch.Tensor:
    """
    Compute Shannon entropy of phase/amplitude distribution.
    
    Args:
        phases: [batch, seq_len] phase values
        amplitudes: [batch, seq_len] amplitude values (optional, for weighted entropy)
        n_bins: Number of bins for histogram
    
    Returns:
        Entropy [batch] (0 = deterministic, 1 = maximum entropy)
    """
    batch_size, seq_len = phases.shape
    
    # Normalize phases to [0, 2π]
    phases_norm = (phases % (2 * np.pi)) / (2 * np.pi)  # [batch, seq_len]
    
    entropies = []
    for b in range(batch_size):
        phases_b = phases_norm[b]  # [seq_len]
        
        if amplitudes is not None:
            weights = amplitudes[b]  # [seq_len]
            weights = weights / (weights.sum() + 1e-10)
        else:
            weights = torch.ones_like(phases_b) / seq_len
        
        # Create histogram manually with weights
        bins = torch.linspace(0, 1, n_bins + 1, device=phases.device)
        hist = torch.zeros(n_bins, device=phases.device)
        
        for i in range(seq_len):
            val = phases_b[i].item()
            weight = weights[i].item()
            # Find bin index
            bin_idx = int(np.clip(val * n_bins, 0, n_bins - 1))
            hist[bin_idx] += weight
        
        hist = hist / (hist.sum() + 1e-10)
        
        # Compute entropy
        hist_nonzero = hist[hist > 0]
        entropy = -torch.sum(hist_nonzero * torch.log(hist_nonzero + 1e-10))
        
        # Normalize by maximum entropy (log(n_bins))
        max_entropy = np.log(n_bins)
        entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        entropies.append(entropy_normalized)
    
    return torch.stack(entropies)


def compute_flow(
    phases: torch.Tensor,
    phases_prev: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute flow measure (rate of change, temporal dynamics).
    
    Measures how smoothly the system evolves over time.
    High flow = smooth, continuous dynamics.
    
    Args:
        phases: [batch, seq_len] current phase values
        phases_prev: [batch, seq_len] previous phase values (optional)
    
    Returns:
        Flow measure [batch] (0 = static, 1 = smooth flow)
    """
    if phases_prev is None:
        # If no previous state, use phase differences within sequence
        phase_diff = phases[:, 1:] - phases[:, :-1]  # [batch, seq_len-1]
        # Normalize differences
        phase_diff_norm = (phase_diff + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]
        # Flow = inverse of variance (smooth = low variance)
        flow = 1.0 / (1.0 + torch.var(phase_diff_norm, dim=1))
    else:
        # Compare with previous state
        phase_diff = phases - phases_prev
        phase_diff_norm = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        # Flow = smoothness of change
        flow = 1.0 / (1.0 + torch.mean(torch.abs(phase_diff_norm), dim=1))
    
    return torch.clamp(flow, 0, 1)


def compute_focus(
    phases: torch.Tensor,
    amplitudes: Optional[torch.Tensor] = None,
    coupling_matrix: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute focus measure (concentration/peakiness of attention).
    
    Inverse of entropy - measures how concentrated the system state is.
    High focus = concentrated, low focus = distributed.
    
    Args:
        phases: [batch, seq_len] phase values
        amplitudes: [batch, seq_len] amplitude values (optional)
        coupling_matrix: [batch, seq_len, seq_len] coupling matrix (optional)
    
    Returns:
        Focus measure [batch] (0 = distributed, 1 = focused)
    """
    # Focus is inverse of entropy
    entropy = compute_entropy(phases, amplitudes)
    focus = 1.0 - entropy
    
    # If coupling matrix available, also consider attention concentration
    if coupling_matrix is not None:
        batch_size = coupling_matrix.shape[0]
        attentions = []
        for b in range(batch_size):
            K_b = coupling_matrix[b]  # [seq_len, seq_len]
            # Row sums = attention received by each token
            attention = torch.softmax(K_b.sum(dim=0), dim=0)  # [seq_len]
            # Focus = inverse entropy of attention distribution
            attention_entropy = -torch.sum(attention * torch.log(attention + 1e-10))
            max_entropy = np.log(len(attention))
            attention_focus = 1.0 - (attention_entropy / max_entropy if max_entropy > 0 else 0)
            attentions.append(attention_focus)
        
        attention_focus = torch.stack(attentions)
        # Combine phase focus and attention focus
        focus = (focus + attention_focus) / 2
    
    return focus


def compute_arousal(
    amplitudes: torch.Tensor,
    phases: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute arousal measure (overall activation/energy level).
    
    Measures the overall "energy" or "activation" of the system.
    High arousal = high energy, low arousal = low energy.
    
    Args:
        amplitudes: [batch, seq_len] amplitude values
        phases: [batch, seq_len] phase values (optional)
    
    Returns:
        Arousal measure [batch] (0 = low arousal, 1 = high arousal)
    """
    # Normalize amplitudes to [0, 1]
    amplitudes_norm = amplitudes / (amplitudes.max(dim=1, keepdim=True)[0] + 1e-10)
    
    # Arousal = mean normalized amplitude
    arousal = amplitudes_norm.mean(dim=1)  # [batch]
    
    # If phases available, also consider phase coherence as "activation"
    if phases is not None:
        coherence = torch.abs(torch.mean(torch.exp(1j * phases), dim=1))  # [batch]
        # Combine amplitude and coherence
        arousal = (arousal + coherence) / 2
    
    return arousal


def compute_extended_cdns(
    phases: torch.Tensor,
    amplitudes: torch.Tensor,
    coupling_matrix: torch.Tensor,
    consonance: torch.Tensor,
    dissonance: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    signal: Optional[torch.Tensor] = None,
    phases_prev: Optional[torch.Tensor] = None
) -> ExtendedCDNS:
    """
    Compute extended CDNS metrics with all additional measures.
    
    Args:
        phases: [batch, seq_len] phase values
        amplitudes: [batch, seq_len] amplitude values
        coupling_matrix: [batch, seq_len, seq_len] coupling matrix
        consonance: [batch] consonance values
        dissonance: [batch] dissonance values
        noise: [batch] noise values (optional)
        signal: [batch] signal values (optional)
        phases_prev: [batch, seq_len] previous phase values (optional, for flow)
    
    Returns:
        ExtendedCDNS object with all metrics
    """
    return ExtendedCDNS(
        consonance=consonance,
        dissonance=dissonance,
        noise=noise,
        signal=signal,
        dimensionality=compute_dimensionality(coupling_matrix),
        integration=compute_integration(phases, amplitudes, coupling_matrix),
        complexity=compute_complexity(phases, amplitudes),
        equivariance=compute_equivariance(phases, coupling_matrix),
        entropy=compute_entropy(phases, amplitudes),
        flow=compute_flow(phases, phases_prev),
        focus=compute_focus(phases, amplitudes, coupling_matrix),
        arousal=compute_arousal(amplitudes, phases)
    )

