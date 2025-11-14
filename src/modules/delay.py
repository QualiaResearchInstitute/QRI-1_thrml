"""
Delay dynamics module for temporal memory in Kuramoto oscillators.

Provides DelayLine (ring buffer) for storing past phase states and
functions for computing delayed coupling forces.
"""

import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DelayConfig:
    """Configuration for delay line ring buffer."""
    capacity: int = 10
    """Maximum number of time steps to store."""


class DelayLine:
    """
    Ring buffer for storing temporal delays of phase states.
    
    Implements a circular buffer that stores phase values at each time step,
    allowing retrieval of past states for delayed coupling calculations.
    
    Example:
        >>> delay_line = DelayLine(
        ...     (batch_size, seq_len),
        ...     DelayConfig(capacity=10),
        ...     like=phases
        ... )
        >>> delay_line.write(phases)
        >>> delayed_phases = delay_line.read(tau_steps=3)
    """
    
    def __init__(
        self,
        shape: Tuple[int, int],
        config: DelayConfig,
        like: torch.Tensor
    ):
        """
        Initialize delay line ring buffer.
        
        Args:
            shape: (batch_size, seq_len) shape of phase tensors
            config: DelayConfig with capacity setting
            like: Tensor to match device and dtype
        """
        self.batch_size, self.seq_len = shape
        self.capacity = config.capacity
        self.device = like.device
        self.dtype = like.dtype
        
        # Ring buffer: [capacity, batch_size, seq_len]
        self.buffer = torch.zeros(
            (self.capacity, self.batch_size, self.seq_len),
            device=self.device,
            dtype=self.dtype
        )
        self.write_idx = 0  # Current write position
    
    def write(self, phases: torch.Tensor):
        """
        Write current phases into the delay line.
        
        Args:
            phases: Phase tensor [batch_size, seq_len]
        """
        # Store phases at current write position
        self.buffer[self.write_idx] = phases.detach()
        # Advance write pointer (circular)
        self.write_idx = (self.write_idx + 1) % self.capacity
    
    def read(self, tau_steps: int) -> torch.Tensor:
        """
        Read phases from tau_steps ago.
        
        Args:
            tau_steps: Number of steps to look back (must be < capacity)
            
        Returns:
            Phases from tau_steps ago [batch_size, seq_len]
        """
        if tau_steps >= self.capacity:
            raise ValueError(f"tau_steps ({tau_steps}) must be < capacity ({self.capacity})")
        
        # Calculate read index (circular)
        read_idx = (self.write_idx - tau_steps - 1) % self.capacity
        return self.buffer[read_idx].clone()
    
    def read_fractional(self, tau_steps: torch.Tensor) -> torch.Tensor:
        """
        Read phases with fractional delay using linear interpolation.
        
        Args:
            tau_steps: Fractional delay steps [batch_size, seq_len] or [batch_size]
                      Values should be in [0, capacity-1]
            
        Returns:
            Interpolated phases [batch_size, seq_len]
        """
        # Clamp to valid range
        tau_clamped = torch.clamp(tau_steps, 0.0, float(self.capacity - 1))
        
        # Ensure 2D shape [batch_size, seq_len]
        if tau_clamped.dim() == 1:
            tau_clamped = tau_clamped.unsqueeze(-1).expand(-1, self.seq_len)
        
        # Split into integer and fractional parts
        tau_int = tau_clamped.floor().long()
        tau_frac = tau_clamped - tau_int.float()
        
        # Ensure tau_int is within bounds
        tau_int = torch.clamp(tau_int, 0, self.capacity - 2)
        
        # Compute read indices (circular)
        read_idx_0 = (self.write_idx - tau_int - 1) % self.capacity
        read_idx_1 = (self.write_idx - tau_int - 2) % self.capacity
        
        # Gather phases using advanced indexing
        # read_idx_0: [batch_size, seq_len] -> each element is a buffer index
        # We need buffer[read_idx_0[i, j], i, j] for all (i, j)
        batch_size, seq_len = tau_clamped.shape
        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(-1).expand(-1, seq_len)
        seq_idx = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # Gather phases at both time steps
        phases_0 = self.buffer[read_idx_0, batch_idx, seq_idx]
        phases_1 = self.buffer[read_idx_1, batch_idx, seq_idx]
        
        # Linear interpolation
        phases_interp = phases_0 + tau_frac * (phases_1 - phases_0)
        
        return phases_interp


def uniform_delayed_coupling_force(
    phases: torch.Tensor,
    delay_line: DelayLine,
    coupling_matrix: torch.Tensor,
    tau_steps: int
) -> torch.Tensor:
    """
    Compute delayed coupling force using uniform integer delay.
    
    The force is computed as:
        F_i = Σ_j K_ij * sin(θ_j(t-τ) - θ_i(t))
    
    where θ_j(t-τ) is the phase of oscillator j at time t-τ.
    
    Args:
        phases: Current phases [batch_size, seq_len]
        delay_line: DelayLine containing past phase states
        coupling_matrix: Coupling matrix [batch_size, seq_len, seq_len]
        tau_steps: Integer delay steps
        
    Returns:
        Delayed coupling force [batch_size, seq_len]
    """
    # Read delayed phases
    phases_delayed = delay_line.read(tau_steps)  # [batch_size, seq_len]
    
    # Compute phase differences: θ_j(t-τ) - θ_i(t)
    # phases_delayed: [batch_size, seq_len]
    # phases: [batch_size, seq_len]
    # We want [batch_size, seq_len, seq_len] where [i, j] = phases_delayed[j] - phases[i]
    phases_i = phases.unsqueeze(-1)  # [batch_size, seq_len, 1]
    phases_j_delayed = phases_delayed.unsqueeze(-2)  # [batch_size, 1, seq_len]
    phase_diff = phases_j_delayed - phases_i  # [batch_size, seq_len, seq_len]
    
    # Replace -inf with 0 to avoid NaN from -inf * 0 operations
    # -inf comes from masking and means no coupling (should be 0)
    coupling_matrix_safe = coupling_matrix.clone()
    coupling_matrix_safe[torch.isinf(coupling_matrix_safe) & (coupling_matrix_safe < 0)] = 0.0
    
    # Compute coupling force: Σ_j K_ij * sin(θ_j(t-τ) - θ_i(t))
    sin_diff = torch.sin(phase_diff)  # [batch_size, seq_len, seq_len]
    coupling_force = torch.sum(coupling_matrix_safe * sin_diff, dim=-1)  # [batch_size, seq_len]
    
    return coupling_force

