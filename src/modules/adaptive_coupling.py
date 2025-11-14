"""
Adaptive/Learning Coupling During Simulation

Provides coupling adaptation during simulation based on:
- Synchronization feedback: Adjust coupling based on current synchronization state
- Performance metrics: Adapt based on task performance
- Dynamic optimization: Real-time optimization of coupling patterns
- Gradient-based adaptation: Use gradients to optimize coupling
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable


class AdaptiveCoupling:
    """
    Adaptive coupling that adjusts during simulation.
    
    Provides feedback-based adaptation of coupling strength based on
    synchronization metrics or other feedback signals.
    """
    
    def __init__(
        self,
        adaptation_rate: float = 0.01,
        adaptation_signal: str = "order_parameter",
        adaptation_target: float = 0.6,
        min_coupling: float = 0.0,
        max_coupling: float = 10.0,
    ):
        """
        Initialize adaptive coupling.
        
        Args:
            adaptation_rate: Learning rate for adaptation
            adaptation_signal: Signal to optimize ("order_parameter", "dissonance", etc.)
            adaptation_target: Target value for adaptation signal
            min_coupling: Minimum coupling strength
            max_coupling: Maximum coupling strength
        """
        self.adaptation_rate = adaptation_rate
        self.adaptation_signal = adaptation_signal
        self.adaptation_target = adaptation_target
        self.min_coupling = min_coupling
        self.max_coupling = max_coupling
    
    def adapt_coupling_feedback(
        self,
        coupling_matrix: torch.Tensor,
        current_metric: float,
        target_metric: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Adapt coupling based on feedback signal.
        
        Simple proportional control:
        ΔK = -α * (current - target)
        
        Args:
            coupling_matrix: Current coupling [batch, seq_len, seq_len]
            current_metric: Current value of adaptation signal
            target_metric: Target value (uses self.adaptation_target if None)
        
        Returns:
            Adapted coupling matrix
        """
        if target_metric is None:
            target_metric = self.adaptation_target
        
        # Compute error
        error = current_metric - target_metric
        
        # Adapt coupling: reduce if too high, increase if too low
        # For order parameter: if R < target, increase coupling
        #                     if R > target, decrease coupling
        adaptation_factor = 1.0 - self.adaptation_rate * error
        
        # Apply adaptation
        adapted_coupling = coupling_matrix * adaptation_factor
        
        # Clamp to valid range
        adapted_coupling = torch.clamp(
            adapted_coupling,
            min=self.min_coupling,
            max=self.max_coupling
        )
        
        return adapted_coupling
    
    def adapt_coupling_local(
        self,
        coupling_matrix: torch.Tensor,
        phases: torch.Tensor,
        local_order_params: torch.Tensor,  # [batch, seq_len] local R
        target_local_R: float = 0.6,
    ) -> torch.Tensor:
        """
        Adapt coupling locally based on local synchronization.
        
        Each oscillator adapts its coupling based on local order parameter.
        
        Args:
            coupling_matrix: Current coupling [batch, seq_len, seq_len]
            phases: Current phases [batch, seq_len]
            local_order_params: Local order parameter for each oscillator
            target_local_R: Target local order parameter
        
        Returns:
            Locally adapted coupling matrix
        """
        batch_size, seq_len = phases.shape
        
        # Compute local adaptation factors
        local_errors = local_order_params - target_local_R
        local_factors = 1.0 - self.adaptation_rate * local_errors.unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Apply local adaptation to coupling
        adapted_coupling = coupling_matrix * local_factors
        
        # Clamp
        adapted_coupling = torch.clamp(
            adapted_coupling,
            min=self.min_coupling,
            max=self.max_coupling
        )
        
        return adapted_coupling


def adapt_coupling_gradient(
    coupling_matrix: torch.Tensor,
    phases: torch.Tensor,
    target_metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    target_value: float = 0.6,
    learning_rate: float = 0.01,
    n_iterations: int = 5,
    min_coupling: float = 0.0,
    max_coupling: float = 10.0,
) -> torch.Tensor:
    """
    Adapt coupling using gradient descent to optimize target metric.
    
    Minimizes: L = (metric - target_value)²
    
    Args:
        coupling_matrix: Current coupling [batch, seq_len, seq_len]
        phases: Current phases [batch, seq_len]
        target_metric_fn: Function that computes metric given coupling and phases
        target_value: Target metric value
        learning_rate: Gradient descent learning rate
        n_iterations: Number of gradient steps
        min_coupling: Minimum coupling strength
        max_coupling: Maximum coupling strength
    
    Returns:
        Optimized coupling matrix
    """
    coupling = coupling_matrix.clone()
    coupling.requires_grad_(True)
    
    for iteration in range(n_iterations):
        # Compute metric
        metric = target_metric_fn(coupling, phases)
        
        # Compute loss
        loss = (metric - target_value) ** 2
        
        # Compute gradient
        loss.backward()
        
        # Update coupling
        with torch.no_grad():
            coupling -= learning_rate * coupling.grad
            coupling.grad.zero_()
            
            # Clamp to valid range
            coupling.clamp_(min=min_coupling, max=max_coupling)
    
    return coupling.detach()


def compute_order_parameter_from_coupling(
    coupling_matrix: torch.Tensor,
    phases: torch.Tensor,
    n_sim_steps: int = 5,
    dt: float = 0.01,
) -> torch.Tensor:
    """
    Compute order parameter after simulating with given coupling.
    
    Used as target metric function for gradient-based adaptation.
    
    Args:
        coupling_matrix: Coupling matrix [batch, seq_len, seq_len]
        phases: Initial phases [batch, seq_len]
        n_sim_steps: Number of simulation steps
        dt: Time step
    
    Returns:
        Order parameter [batch]
    """
    # Simple simulation to compute R
    phases_sim = phases.clone()
    
    for _ in range(n_sim_steps):
        # Compute coupling force
        phase_diff = phases_sim.unsqueeze(-1) - phases_sim.unsqueeze(-2)
        sin_coupling = torch.sin(phase_diff)
        coupling_force = (coupling_matrix * sin_coupling).sum(dim=-1)
        
        # Update phases
        phases_sim = (phases_sim + dt * coupling_force) % (2 * np.pi)
    
    # Compute order parameter
    complex_phases = torch.exp(1j * phases_sim)
    order_param = torch.abs(complex_phases.mean(dim=-1))
    
    return order_param.mean()


class PerformanceAdaptiveCoupling:
    """
    Adaptive coupling based on task performance.
    
    Adapts coupling strength based on performance metrics (loss, accuracy, etc.)
    rather than synchronization metrics.
    """
    
    def __init__(
        self,
        performance_metric: str = "loss",
        adaptation_strategy: str = "gradient",
        adaptation_rate: float = 0.01,
    ):
        """
        Initialize performance-adaptive coupling.
        
        Args:
            performance_metric: Performance metric to optimize ("loss", "accuracy", "perplexity")
            adaptation_strategy: Adaptation strategy ("gradient", "feedback")
            adaptation_rate: Learning rate for adaptation
        """
        self.performance_metric = performance_metric
        self.adaptation_strategy = adaptation_strategy
        self.adaptation_rate = adaptation_rate
        self.performance_history: List[float] = []
    
    def adapt_based_on_performance(
        self,
        coupling_matrix: torch.Tensor,
        current_performance: float,
        performance_history: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Adapt coupling based on task performance.
        
        If performance improves, maintain/strengthen coupling.
        If performance degrades, adjust coupling.
        
        Args:
            coupling_matrix: Current coupling [batch, seq_len, seq_len]
            current_performance: Current performance metric value
            performance_history: History of performance (for trend analysis)
        
        Returns:
            Adapted coupling matrix
        """
        if performance_history is None:
            performance_history = self.performance_history
        
        if len(performance_history) < 2:
            # Not enough history, use simple adaptation
            return coupling_matrix
        
        # Compute performance trend
        recent_performance = performance_history[-5:] if len(performance_history) >= 5 else performance_history
        performance_trend = np.mean(np.diff(recent_performance))
        
        # Adapt based on trend
        # For loss: negative trend = improvement, positive trend = degradation
        # For accuracy: positive trend = improvement, negative trend = degradation
        if self.performance_metric == "loss":
            # Lower is better
            if performance_trend < 0:
                # Performance improving: maintain or slightly strengthen coupling
                adaptation_factor = 1.0 + self.adaptation_rate * 0.1
            elif performance_trend > 0:
                # Performance degrading: adjust coupling
                adaptation_factor = 1.0 - self.adaptation_rate * 0.5
            else:
                # Performance stable: maintain coupling
                adaptation_factor = 1.0
        else:
            # For accuracy/perplexity: higher is better (or lower for perplexity)
            if performance_trend > 0:
                # Performance improving
                adaptation_factor = 1.0 + self.adaptation_rate * 0.1
            elif performance_trend < 0:
                # Performance degrading
                adaptation_factor = 1.0 - self.adaptation_rate * 0.5
            else:
                # Performance stable
                adaptation_factor = 1.0
        
        adapted_coupling = coupling_matrix * adaptation_factor
        adapted_coupling = torch.clamp(adapted_coupling, min=0.0, max=10.0)
        
        return adapted_coupling
    
    def update_performance_history(self, performance: float):
        """Update performance history."""
        self.performance_history.append(performance)
        # Limit history size
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

