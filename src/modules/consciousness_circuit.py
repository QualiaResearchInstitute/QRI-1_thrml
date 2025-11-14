"""
Consciousness as Circuit Board - High-Speed Information Processing

Consciousness can be seen as a circuit board where information flows
through synchronized oscillators. When oscillators sync, information
can "go BRRRRRRRRR" - high-speed, efficient computation.

This module implements circuit-board style patterns:
- Information routing through phase coherence
- High-speed computation via synchronization
- Circuit-like information flow patterns
- Parallel processing through oscillator clusters
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
import math


def compute_information_flow(
    phases: torch.Tensor,
    coupling_matrix: torch.Tensor,
    amplitudes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute information flow through oscillator network.
    
    Information flows along paths of high phase coherence.
    When oscillators sync, information can propagate rapidly.
    
    Args:
        phases: [batch, n] oscillator phases
        coupling_matrix: [batch, n, n] coupling strengths
        amplitudes: [batch, n] optional amplitudes
        
    Returns:
        flow: [batch, n, n] information flow matrix
    """
    batch_size, n = phases.shape
    
    # Phase differences
    phases_i = phases.unsqueeze(-1)  # [batch, n, 1]
    phases_j = phases.unsqueeze(-2)  # [batch, 1, n]
    phase_diff = phases_j - phases_i  # [batch, n, n]
    
    # Coherence: cos(Δθ) measures alignment
    coherence = torch.cos(phase_diff)  # [-1, 1]
    
    # Information flow = coupling * coherence * amplitude_product
    if amplitudes is not None:
        amp_i = amplitudes.unsqueeze(-1)
        amp_j = amplitudes.unsqueeze(-2)
        amp_product = amp_i * amp_j
    else:
        amp_product = torch.ones_like(coherence)
    
    # Flow is stronger when:
    # 1. High coupling (strong connection)
    # 2. High coherence (aligned phases)
    # 3. High amplitude (strong signal)
    flow = coupling_matrix * coherence * amp_product
    
    return flow


def compute_circuit_paths(
    phases: torch.Tensor,
    coupling_matrix: torch.Tensor,
    source: int,
    target: int,
    max_path_length: int = 5,
) -> Tuple[List[List[int]], torch.Tensor]:
    """
    Find circuit paths from source to target through oscillator network.
    
    Paths follow high coherence routes - information flows along
    synchronized oscillator chains.
    
    Args:
        phases: [batch, n] oscillator phases
        coupling_matrix: [batch, n, n] coupling strengths
        source: Source oscillator index
        target: Target oscillator index
        max_path_length: Maximum path length to consider
        
    Returns:
        paths: List of paths (each path is list of oscillator indices)
        path_strengths: [batch, n_paths] path strengths
    """
    _, n = phases.shape
    
    # Compute coherence matrix
    phases_i = phases.unsqueeze(-1)
    phases_j = phases.unsqueeze(-2)
    coherence = torch.cos(phases_j - phases_i)
    
    # Path strength = product of coupling * coherence along path
    # Use BFS to find paths
    paths = []
    path_strengths = []
    
    # Simple implementation: find direct and 2-hop paths
    # Full implementation would use graph algorithms
    
    # Direct path
    if coupling_matrix[0, source, target] > 0:
        paths.append([source, target])
        strength = coupling_matrix[:, source, target] * coherence[:, source, target]
        path_strengths.append(strength)
    
    # 2-hop paths
    for intermediate in range(n):
        if intermediate == source or intermediate == target:
            continue
        
        if (coupling_matrix[0, source, intermediate] > 0 and
            coupling_matrix[0, intermediate, target] > 0):
            paths.append([source, intermediate, target])
            strength = (
                coupling_matrix[:, source, intermediate] * coherence[:, source, intermediate] *
                coupling_matrix[:, intermediate, target] * coherence[:, intermediate, target]
            )
            path_strengths.append(strength)
    
    batch_size = phases.shape[0]
    if path_strengths:
        path_strengths_tensor = torch.stack(path_strengths, dim=1)  # [batch, n_paths]
    else:
        path_strengths_tensor = torch.zeros(batch_size, 0, device=phases.device)
    
    return paths, path_strengths_tensor


def compute_parallel_clusters(
    phases: torch.Tensor,
    coupling_matrix: torch.Tensor,
    coherence_threshold: float = 0.7,
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    Identify parallel processing clusters based on phase coherence.
    
    When oscillators synchronize, they form clusters that can process
    information in parallel - "going BRRRRRRRRR" through synchronized groups.
    
    Args:
        phases: [batch, n] oscillator phases
        coupling_matrix: [batch, n, n] coupling strengths
        coherence_threshold: Minimum coherence for cluster membership
        
    Returns:
        cluster_mask: [batch, n, n_clusters] membership probabilities
        clusters: List of cluster indices
    """
    batch_size, n = phases.shape
    
    # Compute pairwise coherence
    phases_i = phases.unsqueeze(-1)
    phases_j = phases.unsqueeze(-2)
    coherence = torch.cos(phases_j - phases_i)
    
    # Average coherence matrix (across batch)
    coherence_avg = coherence.mean(dim=0)  # [n, n]
    
    # Find clusters using simple thresholding
    # In practice, use graph clustering algorithms
    clusters = []
    assigned = set()
    
    for i in range(n):
        if i in assigned:
            continue
        
        # Find oscillators highly coherent with i
        cluster = [i]
        assigned.add(i)
        
        for j in range(n):
            if j in assigned:
                continue
            
            if coherence_avg[i, j] > coherence_threshold:
                cluster.append(j)
                assigned.add(j)
        
        if len(cluster) > 1:  # Only keep clusters with >1 oscillator
            clusters.append(cluster)
    
    # Create membership mask
    n_clusters = len(clusters)
    cluster_mask = torch.zeros(batch_size, n, n_clusters, device=phases.device)
    
    for c_idx, cluster in enumerate(clusters):
        for osc_idx in cluster:
            cluster_mask[:, osc_idx, c_idx] = 1.0
    
    return cluster_mask, clusters


def compute_circuit_throughput(
    phases: torch.Tensor,
    coupling_matrix: torch.Tensor,
    amplitudes: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute circuit throughput - how fast information can flow.
    
    High synchronization = high throughput = "BRRRRRRRRR"
    
    Args:
        phases: [batch, n] oscillator phases
        coupling_matrix: [batch, n, n] coupling strengths
        amplitudes: [batch, n] optional amplitudes
        
    Returns:
        metrics: Dictionary with throughput metrics
    """
    batch_size, n = phases.shape
    
    # Order parameter (global synchronization)
    complex_phases = torch.exp(1j * phases)
    if amplitudes is not None:
        weighted_phases = amplitudes * complex_phases
    else:
        weighted_phases = complex_phases
    
    R = torch.abs(torch.mean(weighted_phases, dim=-1))  # [batch]
    
    # Information flow rate
    flow = compute_information_flow(phases, coupling_matrix, amplitudes)
    flow_rate = torch.mean(torch.abs(flow), dim=(-2, -1))  # [batch]
    
    # Parallel processing capacity (number of independent clusters)
    _, clusters = compute_parallel_clusters(phases, coupling_matrix)
    n_clusters = len(clusters)
    parallel_capacity = torch.tensor(n_clusters, device=phases.device).expand(batch_size)
    
    # Throughput = R * flow_rate * parallel_capacity
    # High sync + high flow + many clusters = BRRRRRRRRR
    throughput = R * flow_rate * parallel_capacity.float()
    
    metrics = {
        "order_parameter": R,
        "flow_rate": flow_rate,
        "parallel_capacity": parallel_capacity.float(),
        "throughput": throughput,
        "n_clusters": n_clusters,
    }
    
    return metrics


class ConsciousnessCircuitLayer(nn.Module):
    """
    Layer that implements circuit-board style information processing.
    
    When oscillators synchronize, information flows rapidly through
    the network - "going BRRRRRRRRR" through synchronized paths.
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.7,
        track_throughput: bool = True,
    ):
        """
        Initialize consciousness circuit layer.
        
        Args:
            coherence_threshold: Threshold for cluster identification
            track_throughput: Whether to track throughput metrics
        """
        super().__init__()
        self.coherence_threshold = coherence_threshold
        self.track_throughput = track_throughput
    
    def forward(
        self,
        phases: torch.Tensor,
        coupling_matrix: torch.Tensor,
        amplitudes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process information through circuit-board pattern.
        
        Args:
            phases: [batch, n] oscillator phases
            coupling_matrix: [batch, n, n] coupling strengths
            amplitudes: [batch, n] optional amplitudes
            
        Returns:
            processed_phases: [batch, n] phases after circuit processing
            metrics: Dictionary with circuit metrics
        """
        # Compute information flow
        flow = compute_information_flow(phases, coupling_matrix, amplitudes)
        
        # Flow influences phase dynamics (information propagation)
        # dθ/dt += Σ_j flow_ij * sin(θ_j - θ_i)
        phases_i = phases.unsqueeze(-1)
        phases_j = phases.unsqueeze(-2)
        phase_diff = phases_j - phases_i
        
        flow_force = torch.sum(flow * torch.sin(phase_diff), dim=-2)
        
        # Update phases (small step)
        processed_phases = phases + 0.01 * flow_force
        
        metrics = {}
        if self.track_throughput:
            throughput_metrics = compute_circuit_throughput(
                phases,
                coupling_matrix,
                amplitudes,
            )
            metrics.update(throughput_metrics)
        
        metrics["flow"] = flow
        metrics["flow_force"] = flow_force
        
        return processed_phases, metrics


def visualize_parallel_clusters(
    phases: torch.Tensor,
    coupling_matrix: torch.Tensor,
    coherence_threshold: float = 0.7,
    edge_threshold: float = 0.05,
    save_path: Optional[str] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """
    Visualize the identified oscillator clusters as a graph.
    
    Each node represents an oscillator and is colored by cluster membership,
    while edges encode the average coherence-weighted coupling strength.
    
    Args:
        phases: [batch, n] oscillator phases
        coupling_matrix: [batch, n, n] coupling strengths
        coherence_threshold: Threshold passed to compute_parallel_clusters
        edge_threshold: Minimum absolute edge weight to draw
        save_path: Optional path to save the figure
        ax: Optional matplotlib axis to draw on
        title: Optional plot title
        
    Returns:
        fig, ax: Matplotlib figure and axis containing the visualization
    
    Example:
        >>> fig, _ = visualize_parallel_clusters(
        ...     phases,
        ...     coupling_matrix,
        ...     coherence_threshold=0.65,
        ...     save_path="clusters.png",
        ... )
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib is required for visualize_parallel_clusters"
        ) from exc
    
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "networkx is required for visualize_parallel_clusters"
        ) from exc
    
    _, n = phases.shape
    
    _, clusters = compute_parallel_clusters(
        phases, coupling_matrix, coherence_threshold
    )
    
    # Average coherence-weighted coupling strength
    phases_i = phases.unsqueeze(-1)
    phases_j = phases.unsqueeze(-2)
    coherence = torch.cos(phases_j - phases_i)
    weighted_coupling = (coupling_matrix * coherence).mean(dim=0)
    weighted_coupling = weighted_coupling.detach().cpu()
    
    # Build graph
    G = nx.Graph()
    cluster_lookup = {osc: c_idx for c_idx, cluster in enumerate(clusters) for osc in cluster}
    
    for osc_idx in range(n):
        G.add_node(
            osc_idx,
            cluster=cluster_lookup.get(osc_idx, -1),
        )
    
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(weighted_coupling[i, j].item())
            if abs(weight) < edge_threshold:
                continue
            G.add_edge(i, j, weight=weight)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    if len(G.edges) == 0:
        ax.text(0.5, 0.5, "No edges above threshold", ha="center", va="center")
        ax.set_axis_off()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig, ax
    
    pos = nx.spring_layout(G, weight="weight", seed=42)
    
    cmap = plt.cm.get_cmap("tab10", max(1, len(clusters)))
    node_colors = []
    for node in G.nodes:
        cluster_id = G.nodes[node]["cluster"]
        if cluster_id == -1:
            node_colors.append("#BBBBBB")
        else:
            node_colors.append(cmap(cluster_id % cmap.N))
    
    edge_weights = [abs(G[u][v]["weight"]) for u, v in G.edges]
    max_weight = max(edge_weights)
    edge_widths = [1.0 + 4.0 * (w / max_weight) for w in edge_weights]
    
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color="#999999",
        alpha=0.85,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=450,
        linewidths=1.5,
        edgecolors="#222222",
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color="#111111")
    
    ax.set_axis_off()
    ax.set_title(title or "Consciousness Circuit Clusters")
    
    if clusters:
        handles: List[Line2D] = []
        labels: List[str] = []
        for idx, cluster in enumerate(clusters):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=8,
                    markerfacecolor=cmap(idx % cmap.N),
                    markeredgecolor="#222222",
                )
            )
            labels.append(f"Cluster {idx}: {cluster}")
        ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    return fig, ax
