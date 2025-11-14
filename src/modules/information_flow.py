"""
Information Flow & Propagation Analysis

Provides quantitative metrics for:
- Transfer entropy: Directed information transfer between oscillators
- Granger causality: Statistical causality between time series
- Information propagation: How information spreads through network
- Information bottleneck: Identifying bottlenecks in information flow
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import entropy


def compute_transfer_entropy(
    phases_i: torch.Tensor,  # [batch, time] source oscillator
    phases_j: torch.Tensor,   # [batch, time] target oscillator
    k: int = 1,              # History length
    l: int = 1,              # Lag
    n_bins: int = 10,        # Discretization bins
) -> torch.Tensor:
    """
    Compute transfer entropy T(i → j): information transfer from i to j.
    
    Transfer entropy measures how much information about future of j
    is contained in past of i, beyond what's in past of j.
    
    T(i → j) = H(j_future | j_past) - H(j_future | j_past, i_past)
    
    Args:
        phases_i: Source oscillator phases [batch, time]
        phases_j: Target oscillator phases [batch, time]
        k: History length for j
        l: Lag for i
        n_bins: Number of bins for discretization
    
    Returns:
        Transfer entropy [batch]
    """
    batch_size, time_steps = phases_i.shape
    
    if time_steps < k + l + 1:
        return torch.zeros(batch_size, device=phases_i.device)
    
    transfer_entropies = []
    
    for b in range(batch_size):
        # Extract time series
        x_i = phases_i[b].detach().cpu().numpy()
        x_j = phases_j[b].detach().cpu().numpy()
        
        # Discretize (bin phases)
        x_i_binned = np.digitize(x_i, bins=np.linspace(0, 2*np.pi, n_bins+1)) - 1
        x_j_binned = np.digitize(x_j, bins=np.linspace(0, 2*np.pi, n_bins+1)) - 1
        
        # Compute transfer entropy
        # T(i → j) = H(j_future | j_past) - H(j_future | j_past, i_past)
        
        # Prepare sequences
        j_future = x_j_binned[k+l:]
        j_past = np.array([x_j_binned[i:i+k] for i in range(len(x_j_binned)-k-l)])
        i_past = np.array([x_i_binned[i:i+l] for i in range(len(x_i_binned)-k-l)])
        
        if len(j_future) == 0:
            transfer_entropies.append(0.0)
            continue
        
        # Compute conditional entropies
        # H(j_future | j_past)
        j_past_str = [tuple(p) for p in j_past]
        j_future_j_past = {}
        for idx, past in enumerate(j_past_str):
            future = j_future[idx]
            if past not in j_future_j_past:
                j_future_j_past[past] = []
            j_future_j_past[past].append(future)
        
        h_j_future_given_j_past = 0.0
        for past, futures in j_future_j_past.items():
            p_past = len(futures) / len(j_future)
            h_cond = entropy(np.bincount(futures), base=2) if len(set(futures)) > 1 else 0.0
            h_j_future_given_j_past += p_past * h_cond
        
        # H(j_future | j_past, i_past)
        joint_past_str = [(tuple(j_p), tuple(i_p)) for j_p, i_p in zip(j_past, i_past)]
        j_future_joint_past = {}
        for idx, joint_past in enumerate(joint_past_str):
            future = j_future[idx]
            if joint_past not in j_future_joint_past:
                j_future_joint_past[joint_past] = []
            j_future_joint_past[joint_past].append(future)
        
        h_j_future_given_joint_past = 0.0
        for joint_past, futures in j_future_joint_past.items():
            p_joint_past = len(futures) / len(j_future)
            h_cond = entropy(np.bincount(futures), base=2) if len(set(futures)) > 1 else 0.0
            h_j_future_given_joint_past += p_joint_past * h_cond
        
        # Transfer entropy
        T = h_j_future_given_j_past - h_j_future_given_joint_past
        transfer_entropies.append(max(0.0, T))  # Non-negative
    
    return torch.tensor(transfer_entropies, device=phases_i.device)


def compute_transfer_entropy_matrix(
    phases_history: List[torch.Tensor],  # List of [batch, seq_len] phase snapshots
    k: int = 1,
    l: int = 1,
    n_bins: int = 10,
) -> torch.Tensor:
    """
    Compute transfer entropy matrix for all oscillator pairs.
    
    Args:
        phases_history: List of phase snapshots over time
        k: History length
        l: Lag
        n_bins: Discretization bins
    
    Returns:
        Transfer entropy matrix [batch, seq_len, seq_len]
    """
    if len(phases_history) < k + l + 1:
        # Not enough history
        batch_size, seq_len = phases_history[0].shape
        device = phases_history[0].device
        return torch.zeros(batch_size, seq_len, seq_len, device=device)
    
    # Stack into time series
    phases_series = torch.stack(phases_history, dim=2)  # [batch, seq_len, time]
    batch_size, seq_len, time_steps = phases_series.shape
    
    transfer_matrix = torch.zeros(batch_size, seq_len, seq_len, device=phases_series.device)
    
    # Compute transfer entropy for each pair
    for i in range(seq_len):
        for j in range(seq_len):
            if i == j:
                continue  # No self-transfer
            
            phases_i = phases_series[:, i, :]  # [batch, time]
            phases_j = phases_series[:, j, :]  # [batch, time]
            
            T_ij = compute_transfer_entropy(phases_i, phases_j, k, l, n_bins)
            transfer_matrix[:, i, j] = T_ij
    
    return transfer_matrix


def compute_granger_causality(
    phases_i: torch.Tensor,  # [batch, time] source
    phases_j: torch.Tensor,   # [batch, time] target
    max_lag: int = 5,
) -> torch.Tensor:
    """
    Compute Granger causality: does i help predict j?
    
    Uses F-test to compare:
    - Model 1: j[t] = f(j[t-1], j[t-2], ...)
    - Model 2: j[t] = f(j[t-1], j[t-2], ..., i[t-1], i[t-2], ...)
    
    If Model 2 significantly better → i Granger-causes j
    
    Args:
        phases_i: Source phases [batch, time]
        phases_j: Target phases [batch, time]
        max_lag: Maximum lag to test
    
    Returns:
        Granger causality statistic [batch] (higher = stronger causality)
    """
    batch_size, time_steps = phases_i.shape
    
    if time_steps < 2 * max_lag + 1:
        return torch.zeros(batch_size, device=phases_i.device)
    
    causality_stats = []
    
    for b in range(batch_size):
        x_i = phases_i[b].detach().cpu().numpy()
        x_j = phases_j[b].detach().cpu().numpy()
        
        # Simple approach: compare prediction errors
        # Model 1: predict j from j's past
        errors_model1 = []
        for t in range(max_lag, time_steps):
            # Simple linear prediction
            j_past = x_j[t-max_lag:t]
            j_pred = np.mean(j_past)  # Simple mean predictor
            error = (x_j[t] - j_pred) ** 2
            errors_model1.append(error)
        
        # Model 2: predict j from j's past + i's past
        errors_model2 = []
        for t in range(max_lag, time_steps):
            j_past = x_j[t-max_lag:t]
            i_past = x_i[t-max_lag:t]
            # Combined predictor
            j_pred = 0.5 * np.mean(j_past) + 0.5 * np.mean(i_past)
            error = (x_j[t] - j_pred) ** 2
            errors_model2.append(error)
        
        # Granger causality: improvement in prediction
        mse1 = np.mean(errors_model1)
        mse2 = np.mean(errors_model2)
        
        if mse1 > 0:
            gc_stat = (mse1 - mse2) / mse1  # Relative improvement
        else:
            gc_stat = 0.0
        
        causality_stats.append(max(0.0, gc_stat))
    
    return torch.tensor(causality_stats, device=phases_i.device)


def compute_granger_causality_matrix(
    phases_history: List[torch.Tensor],
    max_lag: int = 5,
) -> torch.Tensor:
    """
    Compute Granger causality matrix for all oscillator pairs.
    
    Args:
        phases_history: List of phase snapshots over time
        max_lag: Maximum lag to test
    
    Returns:
        Granger causality matrix [batch, seq_len, seq_len]
    """
    if len(phases_history) < 2 * max_lag + 1:
        batch_size, seq_len = phases_history[0].shape
        device = phases_history[0].device
        return torch.zeros(batch_size, seq_len, seq_len, device=device)
    
    # Stack into time series
    phases_series = torch.stack(phases_history, dim=2)  # [batch, seq_len, time]
    batch_size, seq_len, time_steps = phases_series.shape
    
    causality_matrix = torch.zeros(batch_size, seq_len, seq_len, device=phases_series.device)
    
    # Compute Granger causality for each pair
    for i in range(seq_len):
        for j in range(seq_len):
            if i == j:
                continue  # No self-causality
            
            phases_i = phases_series[:, i, :]  # [batch, time]
            phases_j = phases_series[:, j, :]  # [batch, time]
            
            GC_ij = compute_granger_causality(phases_i, phases_j, max_lag)
            causality_matrix[:, i, j] = GC_ij
    
    return causality_matrix


def track_information_propagation(
    phases_history: List[torch.Tensor],
    source_oscillator: int,
    transfer_entropy_matrix: torch.Tensor,
    propagation_steps: int = 10,
) -> Dict[str, Any]:
    """
    Track how information propagates from source oscillator.
    
    Uses transfer entropy to identify propagation paths.
    
    Args:
        phases_history: Phase snapshots over time
        source_oscillator: Source oscillator index
        transfer_entropy_matrix: [batch, seq_len, seq_len] transfer entropy
        propagation_steps: Number of steps to track
    
    Returns:
        {
            'propagation_paths': list of paths,
            'propagation_times': time to reach each oscillator,
            'propagation_strength': strength at each oscillator,
        }
    """
    batch_size, seq_len, _ = transfer_entropy_matrix.shape
    
    # Threshold for significant information transfer
    threshold = transfer_entropy_matrix.mean() * 0.5
    
    propagation_results = []
    
    for b in range(batch_size):
        T = transfer_entropy_matrix[b]  # [seq_len, seq_len]
        
        # Find propagation paths using graph search
        visited = {source_oscillator}
        propagation_times = {source_oscillator: 0}
        propagation_strength = {source_oscillator: 1.0}
        
        current_wavefront = [source_oscillator]
        
        for step in range(propagation_steps):
            next_wavefront = []
            
            for node in current_wavefront:
                # Find neighbors with significant transfer entropy
                neighbors = torch.where(T[node] > threshold)[0].tolist()
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        propagation_times[neighbor] = step + 1
                        propagation_strength[neighbor] = float(T[node, neighbor])
                        next_wavefront.append(neighbor)
            
            if not next_wavefront:
                break  # No more propagation
            
            current_wavefront = next_wavefront
        
        propagation_results.append({
            'propagation_times': propagation_times,
            'propagation_strength': propagation_strength,
            'reached_oscillators': list(visited),
        })
    
    return {
        'propagation_results': propagation_results,
        'source': source_oscillator,
    }


def identify_information_bottlenecks(
    transfer_entropy_matrix: torch.Tensor,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Identify information bottlenecks: oscillators with low information flow.
    
    Bottlenecks:
    - Low incoming information (low in-degree)
    - Low outgoing information (low out-degree)
    - Low total information flow
    
    Args:
        transfer_entropy_matrix: [batch, seq_len, seq_len] transfer entropy
        threshold: Threshold for low information flow
    
    Returns:
        {
            'bottlenecks': list of bottleneck oscillator indices,
            'bottleneck_scores': scores for each oscillator,
        }
    """
    batch_size, seq_len, _ = transfer_entropy_matrix.shape
    
    bottleneck_results = []
    
    for b in range(batch_size):
        T = transfer_entropy_matrix[b]  # [seq_len, seq_len]
        
        # Compute in-degree and out-degree (information flow)
        in_degree = T.sum(dim=0)  # Incoming information
        out_degree = T.sum(dim=1)  # Outgoing information
        total_degree = in_degree + out_degree
        
        # Identify bottlenecks: low total information flow
        bottleneck_mask = total_degree < threshold
        bottlenecks = torch.where(bottleneck_mask)[0].tolist()
        
        bottleneck_scores = {
            i: {
                'in_flow': float(in_degree[i]),
                'out_flow': float(out_degree[i]),
                'total_flow': float(total_degree[i]),
            }
            for i in range(seq_len)
        }
        
        bottleneck_results.append({
            'bottlenecks': bottlenecks,
            'bottleneck_scores': bottleneck_scores,
        })
    
    return {
        'bottleneck_results': bottleneck_results,
        'threshold': threshold,
    }


class InformationFlowAnalyzer:
    """
    Information flow analyzer for oscillator systems.
    
    Provides quantitative metrics for information transfer, causality,
    propagation, and bottleneck identification.
    """
    
    def __init__(
        self,
        transfer_entropy_k: int = 1,
        transfer_entropy_l: int = 1,
        transfer_entropy_bins: int = 10,
        granger_max_lag: int = 5,
        propagation_steps: int = 10,
        bottleneck_threshold: float = 0.1,
    ):
        """
        Initialize information flow analyzer.
        
        Args:
            transfer_entropy_k: History length for transfer entropy
            transfer_entropy_l: Lag for transfer entropy
            transfer_entropy_bins: Discretization bins for transfer entropy
            granger_max_lag: Maximum lag for Granger causality
            propagation_steps: Number of steps for propagation tracking
            bottleneck_threshold: Threshold for bottleneck identification
        """
        self.transfer_entropy_k = transfer_entropy_k
        self.transfer_entropy_l = transfer_entropy_l
        self.transfer_entropy_bins = transfer_entropy_bins
        self.granger_max_lag = granger_max_lag
        self.propagation_steps = propagation_steps
        self.bottleneck_threshold = bottleneck_threshold
    
    def compute_information_flow(
        self,
        phases_history: List[torch.Tensor],
        compute_transfer_entropy: bool = True,
        compute_granger: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute information flow metrics.
        
        Args:
            phases_history: List of phase snapshots over time
            compute_transfer_entropy: Whether to compute transfer entropy
            compute_granger: Whether to compute Granger causality
        
        Returns:
            Dictionary with information flow matrices
        """
        results = {}
        
        if compute_transfer_entropy:
            transfer_matrix = compute_transfer_entropy_matrix(
                phases_history,
                k=self.transfer_entropy_k,
                l=self.transfer_entropy_l,
                n_bins=self.transfer_entropy_bins,
            )
            results['transfer_entropy'] = transfer_matrix
        
        if compute_granger:
            granger_matrix = compute_granger_causality_matrix(
                phases_history,
                max_lag=self.granger_max_lag,
            )
            results['granger_causality'] = granger_matrix
        
        return results
    
    def track_propagation(
        self,
        phases_history: List[torch.Tensor],
        source_oscillator: int,
        transfer_entropy_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Track information propagation from source oscillator.
        
        Args:
            phases_history: List of phase snapshots over time
            source_oscillator: Source oscillator index
            transfer_entropy_matrix: Optional pre-computed transfer entropy matrix
        
        Returns:
            Propagation tracking results
        """
        if transfer_entropy_matrix is None:
            transfer_matrix = compute_transfer_entropy_matrix(
                phases_history,
                k=self.transfer_entropy_k,
                l=self.transfer_entropy_l,
                n_bins=self.transfer_entropy_bins,
            )
        else:
            transfer_matrix = transfer_entropy_matrix
        
        return track_information_propagation(
            phases_history,
            source_oscillator,
            transfer_matrix,
            propagation_steps=self.propagation_steps,
        )
    
    def identify_bottlenecks(
        self,
        transfer_entropy_matrix: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Identify information bottlenecks.
        
        Args:
            transfer_entropy_matrix: Transfer entropy matrix [batch, seq_len, seq_len]
        
        Returns:
            Bottleneck identification results
        """
        return identify_information_bottlenecks(
            transfer_entropy_matrix,
            threshold=self.bottleneck_threshold,
        )

