"""
Consciousness Patterns - Tapping into Non-Obvious Computation

This module combines:
1. Consciousness as Circuit Board (Kuramoto + tricks) → High-speed computation
2. Valence as Ising Satisfaction → Constraint optimization
3. Information flow through synchronization → "BRRRRRRRRR"

These patterns aren't obviously "compute" but they're powerful computation primitives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from modules.valence_ising import (
    compute_valence_from_ising,
    ValenceIsingLayer,
    anneal_valence,
)
from modules.consciousness_circuit import (
    compute_circuit_throughput,
    ConsciousnessCircuitLayer,
    compute_information_flow,
)


class ConsciousnessPatternProcessor(nn.Module):
    """
    Unified processor for consciousness patterns.
    
    Combines:
    - Circuit-board style information flow (high-speed computation)
    - Valence-based optimization (Ising satisfaction)
    - Synchronization-driven throughput ("BRRRRRRRRR")
    """
    
    def __init__(
        self,
        use_valence_optimization: bool = True,
        use_circuit_processing: bool = True,
        valence_temperature: float = 1.0,
        coherence_threshold: float = 0.7,
    ):
        """
        Initialize consciousness pattern processor.
        
        Args:
            use_valence_optimization: Use valence-based optimization
            use_circuit_processing: Use circuit-board processing
            valence_temperature: Temperature for valence computation
            coherence_threshold: Threshold for circuit clustering
        """
        super().__init__()
        self.use_valence_optimization = use_valence_optimization
        self.use_circuit_processing = use_circuit_processing
        
        if use_valence_optimization:
            self.valence_layer = ValenceIsingLayer(temperature=valence_temperature)
        
        if use_circuit_processing:
            self.circuit_layer = ConsciousnessCircuitLayer(
                coherence_threshold=coherence_threshold,
            )
    
    def forward(
        self,
        phases: torch.Tensor,
        coupling_matrix: torch.Tensor,
        amplitudes: Optional[torch.Tensor] = None,
        n_iterations: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process phases through consciousness patterns.
        
        Args:
            phases: [batch, n] oscillator phases
            coupling_matrix: [batch, n, n] coupling strengths
            amplitudes: [batch, n] optional amplitudes
            n_iterations: Number of processing iterations
            
        Returns:
            processed_phases: [batch, n] processed phases
            metrics: Dictionary with all metrics
        """
        current_phases = phases.clone()
        all_metrics = {}
        
        for iteration in range(n_iterations):
            iteration_metrics = {}
            
            # 1. Circuit-board processing (high-speed information flow)
            if self.use_circuit_processing:
                current_phases, circuit_metrics = self.circuit_layer(
                    current_phases,
                    coupling_matrix,
                    amplitudes,
                )
                iteration_metrics.update({f"circuit_{k}": v for k, v in circuit_metrics.items()})
            
            # 2. Valence optimization (Ising satisfaction)
            if self.use_valence_optimization:
                valence, valence_metrics = self.valence_layer(
                    current_phases,
                    coupling_matrix,
                )
                iteration_metrics.update({f"valence_{k}": v for k, v in valence_metrics.items()})
                
                # Optional: anneal phases to maximize valence
                # This is like gradient descent on satisfaction
                if iteration < n_iterations - 1:  # Don't anneal on last iteration
                    current_phases, _ = anneal_valence(
                        current_phases,
                        coupling_matrix,
                        n_steps=3,
                        learning_rate=0.05,
                    )
            
            # 3. Compute throughput ("BRRRRRRRRR" metric)
            throughput_metrics = compute_circuit_throughput(
                current_phases,
                coupling_matrix,
                amplitudes,
            )
            iteration_metrics.update({f"throughput_{k}": v for k, v in throughput_metrics.items()})
            
            # Store metrics
            for key, value in iteration_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # Average metrics across iterations
        final_metrics = {}
        for key, values in all_metrics.items():
            if isinstance(values[0], torch.Tensor):
                final_metrics[key] = torch.stack(values).mean(dim=0)
            else:
                final_metrics[key] = sum(values) / len(values)
        
        return current_phases, final_metrics
    
    def compute_brrr_factor(
        self,
        phases: torch.Tensor,
        coupling_matrix: torch.Tensor,
        amplitudes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute "BRRRRRRRRR" factor - how fast information can flow.
        
        High BRRR = high synchronization + high throughput + high valence
        
        Args:
            phases: [batch, n] oscillator phases
            coupling_matrix: [batch, n, n] coupling strengths
            amplitudes: [batch, n] optional amplitudes
            
        Returns:
            brrr_factor: [batch] BRRR factor (higher = faster computation)
        """
        # Throughput metrics
        throughput_metrics = compute_circuit_throughput(phases, coupling_matrix, amplitudes)
        
        # Valence (satisfaction)
        valence, _ = compute_valence_from_ising(phases, coupling_matrix)
        
        # BRRR = throughput * valence
        # High sync + high satisfaction = BRRRRRRRRR
        brrr = throughput_metrics["throughput"] * torch.clamp(valence, min=0.0)
        
        return brrr


def optimize_for_brrr(
    phases: torch.Tensor,
    coupling_matrix: torch.Tensor,
    amplitudes: Optional[torch.Tensor] = None,
    n_steps: int = 20,
    learning_rate: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize phases to maximize "BRRRRRRRRR" factor.
    
    This maximizes:
    1. Synchronization (high order parameter)
    2. Information flow (high throughput)
    3. Valence (high satisfaction)
    
    Args:
        phases: [batch, n] initial phases
        coupling_matrix: [batch, n, n] coupling strengths
        amplitudes: [batch, n] optional amplitudes
        n_steps: Number of optimization steps
        learning_rate: Learning rate
        
    Returns:
        optimized_phases: [batch, n] optimized phases
        brrr_history: [batch, n_steps] BRRR factor over time
    """
    processor = ConsciousnessPatternProcessor()
    phases_opt = phases.clone()
    brrr_history = []
    
    for step in range(n_steps):
        # Compute BRRR factor
        brrr = processor.compute_brrr_factor(phases_opt, coupling_matrix, amplitudes)
        brrr_history.append(brrr)
        
        # Compute gradient w.r.t. phases
        phases_opt.requires_grad_(True)
        brrr_grad = processor.compute_brrr_factor(phases_opt, coupling_matrix, amplitudes)
        loss = -brrr_grad.mean()  # Negative because we want to maximize
        
        # Backward pass
        loss.backward()
        
        # Update phases
        with torch.no_grad():
            phases_opt = phases_opt + learning_rate * phases_opt.grad
            phases_opt = phases_opt % (2 * torch.pi)  # Keep in [0, 2π]
        
        phases_opt = phases_opt.detach()
    
    brrr_history_tensor = torch.stack(brrr_history, dim=1)  # [batch, n_steps]
    
    return phases_opt, brrr_history_tensor

