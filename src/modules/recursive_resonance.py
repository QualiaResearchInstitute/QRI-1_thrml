"""
Recursive Resonance Transformer

Implements extreme recursion phenomena:
- Aliasing/identifiability loss → phase collapse
- Renormalization → order parameter evolution
- Attractors → synchronization patterns

Probes:
- Invariance shift (MI analysis)
- Mode collapse to archetypes
- Cyclic/chaotic meta-dynamics
- Spectral gaps (Jacobian analysis)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from resonance_transformer import ResonanceAttentionHead
except ImportError:
    try:
        from resonance_transformer.resonance_transformer import ResonanceAttentionHead
    except ImportError:
        ResonanceAttentionHead = None


class AttractorDetector:
    """Detect fixed points, limit cycles, and strange attractors."""
    
    def __init__(self, tolerance: float = 1e-4, min_cycle_length: int = 3):
        self.tolerance = tolerance
        self.min_cycle_length = min_cycle_length
    
    def detect(self, phase_history: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Detect attractors in phase history.
        
        Args:
            phase_history: List of phase tensors [n_heads, batch, seq_len]
            
        Returns:
            Dictionary with attractor information
        """
        if len(phase_history) < self.min_cycle_length:
            return {'fixed_points': [], 'limit_cycles': [], 'strange_attractors': []}
        
        # Convert to numpy for analysis
        phases_array = [p.detach().cpu().numpy() for p in phase_history]
        
        # Detect fixed points
        fixed_points = self._detect_fixed_points(phases_array)
        
        # Detect limit cycles
        limit_cycles = self._detect_limit_cycles(phases_array)
        
        # Detect strange attractors
        strange_attractors = self._detect_strange_attractors(phases_array)
        
        return {
            'fixed_points': fixed_points,
            'limit_cycles': limit_cycles,
            'strange_attractors': strange_attractors,
        }
    
    def _detect_fixed_points(self, phases_array: List[np.ndarray]) -> List[int]:
        """Detect fixed points (phases that don't change)."""
        fixed_points = []
        
        for i in range(len(phases_array) - 1):
            diff = np.abs(phases_array[i+1] - phases_array[i])
            if np.all(diff < self.tolerance):
                fixed_points.append(i)
        
        return fixed_points
    
    def _detect_limit_cycles(self, phases_array: List[np.ndarray]) -> List[Dict]:
        """Detect limit cycles (periodic patterns)."""
        limit_cycles = []
        
        for period in range(self.min_cycle_length, len(phases_array) // 2):
            for start_idx in range(len(phases_array) - 2 * period):
                # Check if pattern repeats
                pattern1 = phases_array[start_idx:start_idx + period]
                pattern2 = phases_array[start_idx + period:start_idx + 2 * period]
                
                if self._patterns_match(pattern1, pattern2):
                    limit_cycles.append({
                        'start': start_idx,
                        'period': period,
                    })
                    break
        
        return limit_cycles
    
    def _detect_strange_attractors(self, phases_array: List[np.ndarray]) -> bool:
        """Detect strange attractors (bounded but non-periodic)."""
        if len(phases_array) < 10:
            return False
        
        # Check if trajectory is bounded
        phases_flat = np.concatenate([p.flatten() for p in phases_array])
        is_bounded = np.all(np.abs(phases_flat) < 2 * np.pi)
        
        # Check if non-periodic (no limit cycles found)
        limit_cycles = self._detect_limit_cycles(phases_array)
        is_non_periodic = len(limit_cycles) == 0
        
        return is_bounded and is_non_periodic
    
    def _patterns_match(self, pattern1: List[np.ndarray], pattern2: List[np.ndarray]) -> bool:
        """Check if two patterns match within tolerance."""
        if len(pattern1) != len(pattern2):
            return False
        
        for p1, p2 in zip(pattern1, pattern2):
            diff = np.abs(p1 - p2)
            if not np.all(diff < self.tolerance):
                return False
        
        return True


class RecursiveResonanceTransformer(nn.Module):
    """
    Resonance Transformer with extreme recursion support.
    
    Features:
    - Recursive self-modeling (feed output back as input)
    - Phase trajectory tracking
    - Attractor detection
    - Renormalization monitoring
    - Invariance shift tracking
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        max_recursion_depth: int = 100,
        track_phases: bool = True,
        track_attractors: bool = True,
        track_mi: bool = True,
        track_latent_dim: bool = True,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.max_recursion_depth = max_recursion_depth
        self.track_phases = track_phases
        self.track_attractors = track_attractors
        self.track_mi = track_mi
        self.track_latent_dim = track_latent_dim
        
        # Phase history
        self.phase_history: List[torch.Tensor] = []
        
        # Attractor detection
        if track_attractors:
            self.attractor_detector = AttractorDetector()
        else:
            self.attractor_detector = None
    
    def forward_recursive(
        self,
        x: torch.Tensor,
        recursion_depth: int = 10,
        convergence_threshold: float = 1e-6,
        return_full_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with recursion.
        
        Args:
            x: Input [batch, seq_len, d_model]
            recursion_depth: Maximum recursion depth
            convergence_threshold: Convergence threshold for early stopping
            return_full_history: If True, return full history
            
        Returns:
            Dictionary with output and metrics
        """
        x_initial = x.clone()
        x_current = x
        outputs = []
        metrics_history = []
        
        # Reset phase history
        self.phase_history = []
        
        for depth in range(min(recursion_depth, self.max_recursion_depth)):
            # Forward pass
            result = self.base_model(x_current)
            
            # Handle tuple return (output, metrics)
            if isinstance(result, tuple):
                x_next = result[0]
            else:
                x_next = result
            
            outputs.append(x_next.clone())
            
            # Extract metrics
            metrics = self._extract_metrics(x_initial, x_current, x_next, depth)
            metrics_history.append(metrics)
            
            # Track phases
            if self.track_phases:
                phases = self._extract_phases(x_next)
                if phases is not None:
                    self.phase_history.append(phases)
            
            # Check convergence
            if self._check_convergence(x_current, x_next, convergence_threshold):
                break
            
            # Recursive: feed output back as input
            x_current = x_next
        
        # Detect attractors
        attractors = None
        if self.track_attractors and self.attractor_detector and len(self.phase_history) > 10:
            attractors = self.attractor_detector.detect(self.phase_history)
        
        result = {
            'output': x_next,
            'recursion_depth': depth + 1,
            'metrics_history': metrics_history,
            'attractors': attractors,
        }
        
        if return_full_history:
            result['outputs'] = outputs
            result['phase_history'] = self.phase_history
        
        return result
    
    def _extract_metrics(
        self,
        x_initial: torch.Tensor,
        x_prev: torch.Tensor,
        x_next: torch.Tensor,
        depth: int,
    ) -> Dict[str, Any]:
        """Extract metrics from state transition."""
        metrics = {}
        
        # Order parameter R
        R = self._compute_order_parameter(x_next)
        metrics['R'] = R.item() if isinstance(R, torch.Tensor) else R
        
        # CDNS metrics (if available)
        cdns = self._compute_cdns(x_next)
        if cdns:
            metrics['cdns'] = cdns
        
        # Mutual Information with input
        if self.track_mi:
            mi_input = self._compute_mi(x_next, x_initial)
            metrics['mi_input'] = mi_input
        
        # Latent dimensionality
        if self.track_latent_dim:
            latent_dim = self._compute_latent_dimensionality(x_next)
            metrics['latent_dim'] = latent_dim
        
        # Phase entropy (if phases available)
        if self.phase_history:
            phase_entropy = self._compute_phase_entropy(self.phase_history[-1])
            metrics['phase_entropy'] = phase_entropy
        
        # Convergence measure
        convergence = self._compute_convergence(x_prev, x_next)
        metrics['convergence'] = convergence
        
        return metrics
    
    def _compute_order_parameter(self, x: torch.Tensor) -> torch.Tensor:
        """Compute order parameter R from activations."""
        # Use FFT to extract phases
        x_fft = torch.fft.fft(x, dim=-1)
        phases = torch.angle(x_fft)
        
        # Compute R = |(1/N) Σ exp(iθ)|
        complex_phases = torch.exp(1j * phases)
        R = torch.abs(torch.mean(complex_phases, dim=-1))
        
        return R.mean()
    
    def _compute_cdns(self, x: torch.Tensor) -> Optional[Dict[str, float]]:
        """Compute CDNS metrics if available from resonance heads."""
        # Try to extract from resonance heads
        for module in self.base_model.modules():
            if isinstance(module, ResonanceAttentionHead):
                if hasattr(module, '_last_metrics') and 'cdns' in module._last_metrics:
                    return module._last_metrics['cdns']
        
        return None
    
    def _compute_mi(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        Compute Mutual Information between two tensors.
        
        Simplified version using correlation.
        """
        # Flatten
        x1_flat = x1.flatten()
        x2_flat = x2.flatten()
        
        # Normalize
        x1_norm = (x1_flat - x1_flat.mean()) / (x1_flat.std() + 1e-8)
        x2_norm = (x2_flat - x2_flat.mean()) / (x2_flat.std() + 1e-8)
        
        # Correlation (proxy for MI)
        correlation = torch.mean(x1_norm * x2_norm)
        
        # Convert to MI approximation (simplified)
        mi = 0.5 * torch.log(1 - correlation**2 + 1e-8)
        
        return mi.item()
    
    def _compute_latent_dimensionality(self, x: torch.Tensor) -> int:
        """
        Compute effective dimensionality using PCA.
        
        Simplified version: count significant dimensions.
        """
        # Flatten to [batch*seq_len, d_model]
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Compute covariance
        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        cov = torch.mm(x_centered.T, x_centered) / (x_flat.shape[0] - 1)
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(cov).real
        
        # Sort descending
        eigenvalues, _ = torch.sort(eigenvalues, descending=True)
        
        # Find number of dimensions explaining 95% variance
        cumsum = torch.cumsum(eigenvalues, dim=0)
        total = cumsum[-1]
        threshold = 0.95 * total
        
        k = torch.sum(cumsum < threshold).item() + 1
        
        return min(k, x.shape[-1])
    
    def _compute_phase_entropy(self, phases: torch.Tensor) -> float:
        """Compute entropy of phase distribution."""
        # Discretize phases into bins
        n_bins = 32
        phases_flat = phases.flatten().detach().cpu().numpy()
        
        # Normalize to [0, 2π]
        phases_norm = (phases_flat % (2 * np.pi)) / (2 * np.pi)
        
        # Histogram
        hist, _ = np.histogram(phases_norm, bins=n_bins, range=(0, 1))
        hist = hist + 1e-8  # Avoid log(0)
        prob = hist / hist.sum()
        
        # Entropy
        entropy = -np.sum(prob * np.log(prob))
        
        return float(entropy)
    
    def _compute_convergence(
        self,
        x_prev: torch.Tensor,
        x_next: torch.Tensor,
    ) -> float:
        """Compute convergence measure (normalized difference)."""
        diff = torch.norm(x_next - x_prev)
        norm = torch.norm(x_prev)
        
        if norm < 1e-8:
            return 0.0
        
        return (diff / norm).item()
    
    def _extract_phases(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract phases from resonance heads or compute from activations."""
        phases_list = []
        
        # Try to extract from resonance heads
        for module in self.base_model.modules():
            if isinstance(module, ResonanceAttentionHead):
                if hasattr(module, '_last_phases'):
                    phases_list.append(module._last_phases)
        
        if phases_list:
            return torch.stack(phases_list, dim=0)
        else:
            # Fallback: compute phases from activations using FFT
            x_fft = torch.fft.fft(x, dim=-1)
            phases = torch.angle(x_fft)
            return phases.unsqueeze(0)  # Add head dimension
    
    def _check_convergence(
        self,
        x_prev: torch.Tensor,
        x_next: torch.Tensor,
        threshold: float,
    ) -> bool:
        """Check if state has converged."""
        convergence = self._compute_convergence(x_prev, x_next)
        return convergence < threshold


def create_recursive_resonance_model(
    base_model: nn.Module,
    max_recursion_depth: int = 100,
    **kwargs,
) -> RecursiveResonanceTransformer:
    """
    Convenience function to create a recursive resonance model.
    
    Args:
        base_model: Base model to wrap
        max_recursion_depth: Maximum recursion depth
        **kwargs: Additional arguments
        
    Returns:
        RecursiveResonanceTransformer instance
    """
    return RecursiveResonanceTransformer(
        base_model=base_model,
        max_recursion_depth=max_recursion_depth,
        **kwargs,
    )

