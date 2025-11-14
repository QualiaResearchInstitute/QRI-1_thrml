"""
Stochastic Dynamics for Oscillator Networks

Provides systematic noise injection into oscillator systems, enabling:
- Stochastic resonance: Noise-enhanced synchronization
- Robustness: System resilience to perturbations
- Realism: More biologically-plausible neural dynamics
- Lévy noise: Heavy-tailed, asymmetric noise for realistic perturbations
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, List, Any


def generate_gaussian_noise(
    shape: Tuple[int, ...],
    strength: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate Gaussian white noise.
    
    Args:
        shape: Output shape
        strength: Noise amplitude (standard deviation)
        device: Device
        dtype: Data type
    
    Returns:
        Noise tensor [shape]
    """
    return torch.randn(shape, device=device, dtype=dtype) * strength


def generate_levy_noise(
    shape: Tuple[int, ...],
    strength: float,
    alpha: float = 1.5,  # Stability parameter (1 < α ≤ 2)
    beta: float = 0.0,   # Skewness (-1 ≤ β ≤ 1)
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate Lévy-stable noise (heavy-tailed, asymmetric).
    
    For α=2, reduces to Gaussian.
    For α<2, has power-law tails.
    
    Args:
        shape: Output shape
        strength: Scale parameter
        alpha: Stability parameter (1 < α ≤ 2)
        beta: Skewness parameter (-1 ≤ β ≤ 1)
        device: Device
        dtype: Data type
    
    Returns:
        Lévy noise tensor [shape]
    """
    # Use numpy/scipy for Lévy-stable distribution, then convert to torch
    try:
        from scipy.stats import levy_stable
    except ImportError:
        # Fallback: approximate Lévy noise using power-law distribution
        # For α=1.5, use t-distribution approximation
        batch_size, seq_len = shape[:2]
        # Approximate Lévy-stable with t-distribution (heavy-tailed)
        # Degrees of freedom = 2*alpha - 1 (approximation)
        df = max(1, int(2 * alpha - 1))
        noise = np.random.standard_t(df, size=shape) * strength
        noise_tensor = torch.from_numpy(noise).to(device=device, dtype=dtype)
        return noise_tensor
    
    batch_size, seq_len = shape[:2]
    
    # Generate Lévy-stable samples
    noise_samples = []
    total_samples = np.prod(shape)
    for _ in range(total_samples):
        sample = levy_stable.rvs(alpha=alpha, beta=beta, scale=strength)
        noise_samples.append(sample)
    
    noise = np.array(noise_samples).reshape(shape)
    noise_tensor = torch.from_numpy(noise).to(device=device, dtype=dtype)
    
    return noise_tensor


def generate_colored_noise(
    shape: Tuple[int, int],
    strength: float,
    correlation: float = 0.5,  # Temporal correlation
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate colored (correlated) noise.
    
    Uses AR(1) process: x[t] = ρ*x[t-1] + ε[t]
    where ε[t] is white noise.
    
    Args:
        shape: Output shape [batch, seq_len]
        strength: Noise amplitude
        correlation: AR coefficient (0 = white, 1 = perfect correlation)
        device: Device
        dtype: Data type
    
    Returns:
        Colored noise tensor [batch, seq_len]
    """
    batch_size, seq_len = shape
    
    # White noise
    white_noise = torch.randn(shape, device=device, dtype=dtype) * strength
    
    # Apply AR(1) filter
    colored_noise = torch.zeros_like(white_noise)
    colored_noise[:, 0] = white_noise[:, 0]
    
    for t in range(1, seq_len):
        colored_noise[:, t] = (
            correlation * colored_noise[:, t-1] + 
            white_noise[:, t]
        )
    
    return colored_noise


class StochasticDynamics:
    """
    Stochastic dynamics for oscillator systems.
    
    Manages noise injection into oscillator dynamics with support for:
    - Gaussian noise (white noise)
    - Lévy noise (heavy-tailed, asymmetric)
    - Colored noise (temporally correlated)
    - Stochastic resonance detection
    """
    
    def __init__(
        self,
        noise_type: str = "gaussian",
        noise_strength: float = 0.01,
        levy_alpha: float = 1.5,
        levy_beta: float = 0.0,
        noise_correlation: float = 0.0,
        stochastic_resonance: bool = False,
    ):
        """
        Initialize stochastic dynamics.
        
        Args:
            noise_type: Type of noise ("gaussian", "levy", "colored")
            noise_strength: Noise amplitude
            levy_alpha: Lévy stability parameter (1 < α ≤ 2)
            levy_beta: Lévy skewness parameter (-1 ≤ β ≤ 1)
            noise_correlation: Temporal correlation for colored noise
            stochastic_resonance: Whether to track stochastic resonance
        """
        self.noise_type = noise_type
        self.noise_strength = noise_strength
        self.levy_alpha = levy_alpha
        self.levy_beta = levy_beta
        self.noise_correlation = noise_correlation
        self.stochastic_resonance = stochastic_resonance
        
        # For stochastic resonance: track optimal noise level
        self.optimal_noise_level: Optional[float] = None
        self.sync_vs_noise_history: List[Dict[str, float]] = []
    
    def generate_noise(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generate noise based on noise_type.
        
        Args:
            shape: Output shape
            device: Device
            dtype: Data type
        
        Returns:
            Noise tensor
        """
        if self.noise_type == "gaussian":
            return generate_gaussian_noise(shape, self.noise_strength, device, dtype)
        elif self.noise_type == "levy":
            return generate_levy_noise(
                shape, self.noise_strength, self.levy_alpha, self.levy_beta, device, dtype
            )
        elif self.noise_type == "colored":
            if len(shape) != 2:
                raise ValueError(f"Colored noise requires 2D shape [batch, seq_len], got {shape}")
            return generate_colored_noise(
                shape, self.noise_strength, self.noise_correlation, device, dtype
            )
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")
    
    def apply_noise(
        self,
        phases: torch.Tensor,
        amplitudes: Optional[torch.Tensor] = None,
        dt: float = 0.01,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply noise to oscillator dynamics.
        
        Noise is added to phase velocity (not directly to phases).
        
        Args:
            phases: Oscillator phases [batch, seq_len]
            amplitudes: Optional amplitudes [batch, seq_len]
            dt: Time step
        
        Returns:
            Tuple of (phases_noisy, amplitudes_noisy)
        """
        batch_size, seq_len = phases.shape
        device = phases.device
        dtype = phases.dtype
        
        # Generate noise
        noise_shape = (batch_size, seq_len)
        phase_noise = self.generate_noise(noise_shape, device, dtype)
        
        # Apply to phase velocity (multiply by dt for proper scaling)
        phases_noisy = phases + dt * phase_noise
        
        # Wrap phases to [0, 2π)
        phases_noisy = phases_noisy % (2 * np.pi)
        
        # Optionally add noise to amplitudes
        if amplitudes is not None:
            amp_noise = self.generate_noise(noise_shape, device, dtype)
            amplitudes_noisy = amplitudes + dt * amp_noise * 0.1  # Smaller amplitude noise
            amplitudes_noisy = torch.clamp(amplitudes_noisy, min=0.0)
        else:
            amplitudes_noisy = None
        
        return phases_noisy, amplitudes_noisy
    
    def detect_stochastic_resonance(
        self,
        order_parameter_history: List[float],
        noise_strength_history: List[float],
    ) -> Dict[str, Any]:
        """
        Detect stochastic resonance: optimal noise level that maximizes synchronization.
        
        Stochastic resonance occurs when:
        - Low noise: Weak synchronization
        - Optimal noise: Maximum synchronization (resonance peak)
        - High noise: Weak synchronization (noise overwhelms signal)
        
        Args:
            order_parameter_history: History of order parameter values
            noise_strength_history: History of noise strength values
        
        Returns:
            {
                'optimal_noise': optimal noise level,
                'resonance_strength': synchronization at optimal noise,
                'resonance_detected': whether resonance was found
            }
        """
        if len(order_parameter_history) < 3:
            return {
                'optimal_noise': None,
                'resonance_strength': None,
                'resonance_detected': False,
            }
        
        # Find noise level that maximizes order parameter
        order_array = np.array(order_parameter_history)
        noise_array = np.array(noise_strength_history)
        
        # Find peak (local maximum)
        # Simple approach: find maximum
        max_idx = np.argmax(order_array)
        optimal_noise = noise_array[max_idx]
        resonance_strength = order_array[max_idx]
        
        # Check if it's a true resonance (peak, not just endpoint)
        is_peak = False
        if max_idx > 0 and max_idx < len(order_array) - 1:
            # Check if it's higher than neighbors
            if (order_array[max_idx] > order_array[max_idx-1] and 
                order_array[max_idx] > order_array[max_idx+1]):
                is_peak = True
        
        return {
            'optimal_noise': float(optimal_noise),
            'resonance_strength': float(resonance_strength),
            'resonance_detected': is_peak,
        }

