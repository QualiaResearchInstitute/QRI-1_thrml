"""
Valence as Ising Problem Satisfaction

Valence (pleasure/pain) maps to constraint satisfaction in Ising models.
High valence = satisfied constraints (aligned spins)
Low valence = frustrated constraints (misaligned spins)

This module implements valence computation as Ising model satisfaction,
connecting oscillator phases to valence through constraint satisfaction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


def compute_ising_energy(
    spins: torch.Tensor,
    coupling_matrix: torch.Tensor,
    external_field: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Ising model energy: E = -0.5 * Σ_ij J_ij s_i s_j - Σ_i h_i s_i
    
    Args:
        spins: [batch, n] spin values (-1 or +1)
        coupling_matrix: [batch, n, n] coupling strengths J_ij
        external_field: [batch, n] optional external field h_i
        
    Returns:
        energy: [batch] total energy
    """
    # Pairwise interaction energy: -0.5 * Σ_ij J_ij s_i s_j
    spins_i = spins.unsqueeze(-1)  # [batch, n, 1]
    spins_j = spins.unsqueeze(-2)  # [batch, 1, n]
    pairwise = spins_i * spins_j  # [batch, n, n]
    
    # Symmetrize coupling matrix (only upper triangle)
    coupling_sym = 0.5 * (coupling_matrix + coupling_matrix.transpose(-2, -1))
    
    interaction_energy = -0.5 * torch.sum(coupling_sym * pairwise, dim=(-2, -1))
    
    # External field energy: -Σ_i h_i s_i
    field_energy = torch.zeros_like(interaction_energy)
    if external_field is not None:
        field_energy = -torch.sum(external_field * spins, dim=-1)
    
    total_energy = interaction_energy + field_energy
    return total_energy


def phases_to_spins(phases: torch.Tensor, method: str = "sign") -> torch.Tensor:
    """
    Convert oscillator phases to Ising spins.
    
    Methods:
    - "sign": s_i = sign(cos(θ_i))  # Binary based on phase quadrant
    - "projection": s_i = cos(θ_i)  # Continuous spin
    - "coherence": s_i = sign(R_i) where R_i is local coherence
    
    Args:
        phases: [batch, n] oscillator phases
        method: Conversion method
        
    Returns:
        spins: [batch, n] spin values
    """
    if method == "sign":
        # Binary: positive if cos(θ) > 0, negative otherwise
        return torch.sign(torch.cos(phases))
    elif method == "projection":
        # Continuous: use cos(θ) directly
        return torch.cos(phases)
    elif method == "coherence":
        # Use local coherence as spin
        # R_local = |(1/k) Σ_j∈neighbors e^{iθ_j}|
        # For now, use global coherence
        complex_phases = torch.exp(1j * phases)
        R = torch.abs(torch.mean(complex_phases, dim=-1, keepdim=True))
        return torch.sign(R - 0.5)  # Binary based on coherence threshold
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_valence_from_ising(
    phases: torch.Tensor,
    coupling_matrix: torch.Tensor,
    target_energy: float = 0.0,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute valence from Ising model satisfaction.
    
    Valence = -energy (lower energy = higher valence = more satisfied)
    Normalized by temperature for interpretability.
    
    Args:
        phases: [batch, n] oscillator phases
        coupling_matrix: [batch, n, n] coupling strengths
        target_energy: Target energy level (valence = 0 at this point)
        temperature: Temperature for normalization
        
    Returns:
        valence: [batch] valence scores (higher = better)
        metrics: Dictionary with energy, satisfaction, frustration
    """
    # Convert phases to spins
    spins = phases_to_spins(phases, method="projection")
    
    # Compute Ising energy
    energy = compute_ising_energy(spins, coupling_matrix)
    
    # Valence is negative energy (satisfaction)
    # Normalize by temperature and shift by target
    valence = -(energy - target_energy) / temperature
    
    # Additional metrics
    # Satisfaction: fraction of satisfied constraints
    spins_i = spins.unsqueeze(-1)
    spins_j = spins.unsqueeze(-2)
    coupling_sym = 0.5 * (coupling_matrix + coupling_matrix.transpose(-2, -1))
    
    # Constraint satisfaction: J_ij * s_i * s_j > 0 means satisfied
    constraint_signs = coupling_sym * spins_i * spins_j
    satisfied = (constraint_signs > 0).float()
    satisfaction_rate = torch.mean(satisfied, dim=(-2, -1))
    
    # Frustration: fraction of unsatisfied constraints
    frustration_rate = 1.0 - satisfaction_rate
    
    metrics = {
        "energy": energy,
        "valence": valence,
        "satisfaction_rate": satisfaction_rate,
        "frustration_rate": frustration_rate,
        "spins": spins,
    }
    
    return valence, metrics


class ValenceIsingLayer(nn.Module):
    """
    Layer that computes valence from oscillator phases using Ising model.
    
    This connects the resonance dynamics to valence (pleasure/pain) through
    constraint satisfaction. High valence = satisfied constraints.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        target_energy: float = 0.0,
        learnable_temperature: bool = False,
        spin_method: str = "projection",
    ):
        """
        Initialize valence-Ising layer.
        
        Args:
            temperature: Temperature for energy normalization
            target_energy: Target energy level (valence = 0 at this point)
            learnable_temperature: Whether temperature is learnable
            spin_method: Method for converting phases to spins
        """
        super().__init__()
        self.target_energy = target_energy
        self.spin_method = spin_method
        
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))
    
    def forward(
        self,
        phases: torch.Tensor,
        coupling_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute valence from phases and coupling.
        
        Args:
            phases: [batch, n] oscillator phases
            coupling_matrix: [batch, n, n] coupling strengths
            
        Returns:
            valence: [batch] valence scores
            metrics: Dictionary with detailed metrics
        """
        return compute_valence_from_ising(
            phases,
            coupling_matrix,
            target_energy=self.target_energy,
            temperature=float(self.temperature),
        )


def anneal_valence(
    phases: torch.Tensor,
    coupling_matrix: torch.Tensor,
    n_steps: int = 10,
    learning_rate: float = 0.1,
    temperature_schedule: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Anneal phases to maximize valence (minimize Ising energy).
    
    This is like simulated annealing for constraint satisfaction.
    High valence = satisfied constraints = aligned phases.
    
    Args:
        phases: [batch, n] initial phases
        coupling_matrix: [batch, n, n] coupling strengths
        n_steps: Number of annealing steps
        learning_rate: Learning rate for phase updates
        temperature_schedule: [n_steps] temperature schedule (None = linear decay)
        
    Returns:
        phases_annealed: [batch, n] annealed phases
        valence_history: [batch, n_steps] valence over time
    """
    phases_opt = phases.clone()
    batch_size, n = phases.shape
    
    if temperature_schedule is None:
        # Linear temperature decay
        temperature_schedule = torch.linspace(1.0, 0.1, n_steps)
    
    valence_history = []
    
    for step in range(n_steps):
        temp = temperature_schedule[step].item()
        
        # Compute current valence
        valence, metrics = compute_valence_from_ising(
            phases_opt,
            coupling_matrix,
            temperature=temp,
        )
        valence_history.append(valence)
        
        # Compute gradient of energy w.r.t. phases
        # dE/dθ = dE/ds * ds/dθ where s = cos(θ)
        spins = phases_to_spins(phases_opt, method="projection")
        
        # Gradient of Ising energy w.r.t. spins
        spins_i = spins.unsqueeze(-1)
        spins_j = spins.unsqueeze(-2)
        coupling_sym = 0.5 * (coupling_matrix + coupling_matrix.transpose(-2, -1))
        
        # dE/ds_i = -Σ_j J_ij s_j
        dE_ds = -torch.sum(coupling_sym * spins_j, dim=-2)  # [batch, n]
        
        # ds/dθ = -sin(θ) for s = cos(θ)
        ds_dtheta = -torch.sin(phases_opt)
        
        # Chain rule: dE/dθ = dE/ds * ds/dθ
        dE_dtheta = dE_ds * ds_dtheta
        
        # Update phases to minimize energy (maximize valence)
        phases_opt = phases_opt - learning_rate * dE_dtheta
    
    valence_history_tensor = torch.stack(valence_history, dim=1)  # [batch, n_steps]
    
    return phases_opt, valence_history_tensor

