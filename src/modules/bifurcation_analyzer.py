"""
Bifurcation and Phase Transition Analyzer

Provides explicit detection and analysis of:
- Bifurcation types: Hopf, pitchfork, saddle-node
- Phase transitions: Order ↔ disorder
- Critical point identification
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any


class BifurcationAnalyzer:
    """
    Bifurcation and phase transition analyzer.
    
    Detects various types of bifurcations and tracks phase transitions
    in dynamical systems based on order parameter history.
    """
    
    def __init__(self):
        self.bifurcation_history = []
    
    def detect_hopf_bifurcation(
        self,
        phases: torch.Tensor,
        coupling_strength: float,
        order_parameter: float,
        order_parameter_history: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Detect Hopf bifurcation: transition to oscillatory behavior.
        
        Hopf bifurcation occurs when:
        - Fixed point loses stability
        - Oscillatory behavior emerges
        - Order parameter starts oscillating
        
        Args:
            phases: Current phase values [batch, seq_len]
            coupling_strength: Current coupling strength
            order_parameter: Current order parameter value
            order_parameter_history: History of order parameter values
            
        Returns:
            {
                'detected': bool,
                'type': 'hopf',
                'critical_strength': float,
                'direction': 'forward' or 'backward',
                'oscillation_strength': float,
            }
        """
        # Check for oscillatory behavior
        if order_parameter_history is None or len(order_parameter_history) < 10:
            return {'detected': False}
        
        # Compute variance of order parameter (oscillation indicator)
        R_var = np.var(order_parameter_history[-10:])
        R_mean = np.mean(order_parameter_history[-10:])
        
        # Hopf bifurcation: high variance + mean away from 0 or 1
        is_oscillatory = R_var > 0.01 and 0.2 < R_mean < 0.8
        
        if is_oscillatory:
            # Estimate critical coupling strength
            # Find where oscillation started
            R_array = np.array(order_parameter_history)
            R_diff = np.abs(np.diff(R_array))
            oscillation_start = np.where(R_diff > 0.05)[0]
            
            if len(oscillation_start) > 0:
                critical_idx = oscillation_start[0]
                # Would need coupling_strength_history to get exact value
                return {
                    'detected': True,
                    'type': 'hopf',
                    'critical_strength': coupling_strength,  # Approximate
                    'direction': 'forward',  # Assume forward
                    'oscillation_strength': float(R_var),
                }
        
        return {'detected': False}
    
    def detect_pitchfork_bifurcation(
        self,
        phases: torch.Tensor,
        coupling_strength: float,
        order_parameter: float,
        order_parameter_history: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Detect pitchfork bifurcation: symmetry breaking.
        
        Pitchfork bifurcation occurs when:
        - Symmetric state becomes unstable
        - System breaks into multiple branches
        - Order parameter jumps discontinuously
        
        Args:
            phases: Current phase values [batch, seq_len]
            coupling_strength: Current coupling strength
            order_parameter: Current order parameter value
            order_parameter_history: History of order parameter values
            
        Returns:
            {
                'detected': bool,
                'type': 'pitchfork',
                'critical_strength': float,
                'direction': 'forward' or 'backward',
                'jump_magnitude': float,
            }
        """
        if order_parameter_history is None or len(order_parameter_history) < 5:
            return {'detected': False}
        
        # Check for discontinuous jump (symmetry breaking)
        R_array = np.array(order_parameter_history)
        R_diff = np.abs(np.diff(R_array))
        
        # Large jump indicates pitchfork bifurcation
        large_jump = np.any(R_diff > 0.3)
        
        if large_jump:
            jump_idx = np.argmax(R_diff)
            return {
                'detected': True,
                'type': 'pitchfork',
                'critical_strength': coupling_strength,  # Approximate
                'direction': 'forward',
                'jump_magnitude': float(R_diff[jump_idx]),
            }
        
        return {'detected': False}
    
    def detect_saddle_node_bifurcation(
        self,
        phases: torch.Tensor,
        coupling_strength: float,
        order_parameter: float,
        order_parameter_history: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Detect saddle-node bifurcation: collision of fixed points.
        
        Saddle-node bifurcation occurs when:
        - Two fixed points collide and annihilate
        - System transitions from bistable to monostable
        - Order parameter shows rapid change
        
        Args:
            phases: Current phase values [batch, seq_len]
            coupling_strength: Current coupling strength
            order_parameter: Current order parameter value
            order_parameter_history: History of order parameter values
            
        Returns:
            {
                'detected': bool,
                'type': 'saddle_node',
                'critical_strength': float,
                'direction': 'forward' or 'backward',
            }
        """
        if order_parameter_history is None or len(order_parameter_history) < 5:
            return {'detected': False}
        
        # Check for rapid change (collision)
        R_array = np.array(order_parameter_history)
        R_diff = np.abs(np.diff(R_array))
        
        # Rapid change indicates collision
        rapid_change = np.any(R_diff > 0.2) and np.std(R_array) < 0.1
        
        if rapid_change:
            return {
                'detected': True,
                'type': 'saddle_node',
                'critical_strength': coupling_strength,
                'direction': 'forward',
            }
        
        return {'detected': False}
    
    def detect_bifurcations(
        self,
        phases: torch.Tensor,
        coupling_strength: float,
        order_parameter: float,
        order_parameter_history: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Detect all types of bifurcations.
        
        Args:
            phases: Current phase values [batch, seq_len]
            coupling_strength: Current coupling strength
            order_parameter: Current order parameter value
            order_parameter_history: History of order parameter values
            
        Returns:
            {
                'hopf': {...},
                'pitchfork': {...},
                'saddle_node': {...},
                'any_detected': bool,
            }
        """
        hopf = self.detect_hopf_bifurcation(
            phases, coupling_strength, order_parameter, order_parameter_history
        )
        pitchfork = self.detect_pitchfork_bifurcation(
            phases, coupling_strength, order_parameter, order_parameter_history
        )
        saddle_node = self.detect_saddle_node_bifurcation(
            phases, coupling_strength, order_parameter, order_parameter_history
        )
        
        any_detected = (
            hopf.get('detected', False) or
            pitchfork.get('detected', False) or
            saddle_node.get('detected', False)
        )
        
        return {
            'hopf': hopf,
            'pitchfork': pitchfork,
            'saddle_node': saddle_node,
            'any_detected': any_detected,
        }
    
    def track_phase_transition(
        self,
        order_parameter_history: List[float],
        coupling_strength_history: List[float],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Track phase transitions: order ↔ disorder.
        
        Phase transitions:
        - Order → Disorder: R decreases below threshold
        - Disorder → Order: R increases above threshold
        
        Args:
            order_parameter_history: History of order parameter values
            coupling_strength_history: History of coupling strength values
            threshold: Order parameter threshold for phase transition
            
        Returns:
            {
                'transitions': list of transition events,
                'current_state': 'order' or 'disorder',
                'transition_count': int,
            }
        """
        if len(order_parameter_history) < 2:
            return {'transitions': [], 'current_state': None, 'transition_count': 0}
        
        R_array = np.array(order_parameter_history)
        transitions = []
        current_state = None
        
        # Determine initial state
        if R_array[0] > threshold:
            current_state = 'order'
        else:
            current_state = 'disorder'
        
        # Track transitions
        for i in range(1, len(R_array)):
            prev_R = R_array[i-1]
            curr_R = R_array[i]
            
            # Order → Disorder
            if prev_R > threshold and curr_R <= threshold:
                transitions.append({
                    'type': 'order_to_disorder',
                    'index': i,
                    'coupling_strength': coupling_strength_history[i] if i < len(coupling_strength_history) else None,
                    'R_before': float(prev_R),
                    'R_after': float(curr_R),
                })
                current_state = 'disorder'
            
            # Disorder → Order
            elif prev_R <= threshold and curr_R > threshold:
                transitions.append({
                    'type': 'disorder_to_order',
                    'index': i,
                    'coupling_strength': coupling_strength_history[i] if i < len(coupling_strength_history) else None,
                    'R_before': float(prev_R),
                    'R_after': float(curr_R),
                })
                current_state = 'order'
        
        return {
            'transitions': transitions,
            'current_state': current_state,
            'transition_count': len(transitions),
        }
    
    def identify_critical_point(
        self,
        coupling_strength_range: Tuple[float, float],
        order_parameter_fn: callable,
        n_points: int = 50,
        threshold: float = 0.5,
    ) -> Dict[str, Optional[float]]:
        """
        Identify critical coupling strength K_c where phase transition occurs.
        
        Sweeps coupling strength and finds where order parameter crosses threshold.
        
        Args:
            coupling_strength_range: (min, max) range to sweep
            order_parameter_fn: Function that computes R given coupling strength
            n_points: Number of points to sample
            threshold: Order parameter threshold for transition
            
        Returns:
            {
                'critical_strength': K_c or None,
                'critical_R': R at critical point or None,
                'transition_type': 'forward' or 'backward' or None,
            }
        """
        K_min, K_max = coupling_strength_range
        K_values = np.linspace(K_min, K_max, n_points)
        R_values = []
        
        # Compute order parameter for each coupling strength
        for K in K_values:
            try:
                R = order_parameter_fn(K)
                R_values.append(float(R))
            except Exception:
                R_values.append(0.0)
        
        R_array = np.array(R_values)
        
        # Find where R crosses threshold
        above_threshold = R_array > threshold
        below_threshold = R_array <= threshold
        
        # Find transition point
        transition_indices = np.where(np.diff(above_threshold.astype(int)) != 0)[0]
        
        if len(transition_indices) > 0:
            # Use first transition
            trans_idx = transition_indices[0]
            K_c = K_values[trans_idx]
            R_c = R_array[trans_idx]
            
            # Determine direction
            if trans_idx + 1 < len(R_array) and R_array[trans_idx] < R_array[trans_idx + 1]:
                direction = 'forward'  # Increasing R
            else:
                direction = 'backward'  # Decreasing R
            
            return {
                'critical_strength': float(K_c),
                'critical_R': float(R_c),
                'transition_type': direction,
            }
        else:
            # No clear transition found
            return {
                'critical_strength': None,
                'critical_R': None,
                'transition_type': None,
            }

