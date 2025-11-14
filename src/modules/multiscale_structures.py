"""
Multi-Scale/Hierarchical Network Structures

Provides hierarchical coupling structures:
- Hierarchical coupling: Local + global coupling explicitly
- Scale-free topology: Power-law degree distributions
- Nested synchronization: Synchronization at multiple scales
- Multi-scale network structures: Different structures at different scales
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


def construct_hierarchical_coupling(
    seq_len: int,
    local_scale: int = 5,
    global_scale: int = 20,
    local_weight: float = 0.7,
    global_weight: float = 0.3,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct hierarchical coupling: local + global.
    
    Local coupling: Strong coupling within neighborhoods
    Global coupling: Weak coupling across long distances
    
    Args:
        seq_len: Sequence length
        local_scale: Size of local neighborhood
        global_scale: Range of global connections
        local_weight: Weight for local coupling
        global_weight: Weight for global coupling
        device: Device
    
    Returns:
        (local_coupling, global_coupling) [seq_len, seq_len]
    """
    # Initialize coupling matrices
    local_coupling = torch.zeros(seq_len, seq_len, device=device)
    global_coupling = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(seq_len):
        for j in range(seq_len):
            if i == j:
                continue
            
            distance = abs(i - j)
            
            # Local coupling: strong within neighborhood
            if distance <= local_scale:
                local_coupling[i, j] = local_weight * np.exp(-distance / local_scale)
            
            # Global coupling: weak across long distances
            if distance > local_scale and distance <= global_scale:
                global_coupling[i, j] = global_weight * np.exp(-distance / global_scale)
    
    return local_coupling, global_coupling


def combine_hierarchical_coupling(
    local_coupling: torch.Tensor,
    global_coupling: torch.Tensor,
    base_coupling: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Combine local and global coupling.
    
    Args:
        local_coupling: Local coupling matrix [seq_len, seq_len]
        global_coupling: Global coupling matrix [seq_len, seq_len]
        base_coupling: Base coupling to modulate (optional)
    
    Returns:
        Combined coupling matrix [seq_len, seq_len] or [batch, seq_len, seq_len]
    """
    if base_coupling is not None:
        # Modulate base coupling with hierarchical structure
        hierarchical_mask = local_coupling + global_coupling
        
        # Handle batch dimension
        if base_coupling.dim() == 3:
            # [batch, seq_len, seq_len]
            hierarchical_mask = hierarchical_mask.unsqueeze(0).expand_as(base_coupling)
        
        combined = base_coupling * hierarchical_mask
    else:
        # Direct combination
        combined = local_coupling + global_coupling
    
    return combined


def generate_scale_free_topology(
    seq_len: int,
    hub_fraction: float = 0.1,
    power_law_exponent: float = 2.5,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate scale-free (hub-and-spoke) topology.
    
    Uses Barabási-Albert model: preferential attachment.
    
    Args:
        seq_len: Number of nodes
        hub_fraction: Fraction of nodes that are hubs
        power_law_exponent: Power-law exponent for degree distribution
        device: Device
        seed: Random seed for reproducibility
    
    Returns:
        Adjacency matrix [seq_len, seq_len]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with small number of nodes
    n_hubs = max(1, int(seq_len * hub_fraction))
    n_spokes = seq_len - n_hubs
    
    # Create hub-and-spoke structure
    adjacency = torch.zeros(seq_len, seq_len, device=device)
    
    # Hubs are highly connected
    hub_indices = list(range(n_hubs))
    
    # Connect hubs to each other (rich club)
    for i in hub_indices:
        for j in hub_indices:
            if i != j:
                adjacency[i, j] = 1.0
    
    # Connect spokes to hubs (preferential attachment)
    if n_spokes > 0:
        spoke_indices = list(range(n_hubs, seq_len))
        
        for spoke in spoke_indices:
            # Connect to hubs with probability proportional to hub degree
            hub_degrees = adjacency[hub_indices].sum(dim=1)
            total_degree = hub_degrees.sum().item()
            
            if total_degree > 0:
                hub_probs = hub_degrees.float() / total_degree
                hub_probs_np = hub_probs.cpu().numpy()
            else:
                # Uniform if no connections yet
                hub_probs_np = np.ones(n_hubs) / n_hubs
            
            # Connect to multiple hubs
            n_connections = np.random.poisson(lam=3)  # Average connections per spoke
            n_connections = min(max(1, n_connections), n_hubs)  # At least 1, at most n_hubs
            
            selected_hubs = np.random.choice(
                n_hubs,
                size=n_connections,
                replace=False,
                p=hub_probs_np
            )
            
            for hub_idx in selected_hubs:
                adjacency[spoke, hub_idx] = 1.0
                adjacency[hub_idx, spoke] = 1.0  # Undirected
    
    return adjacency


def apply_scale_free_topology_to_coupling(
    coupling_matrix: torch.Tensor,
    scale_free_adjacency: torch.Tensor,
) -> torch.Tensor:
    """
    Apply scale-free topology to coupling matrix.
    
    Only keep coupling where topology has connections.
    
    Args:
        coupling_matrix: Base coupling [batch, seq_len, seq_len] or [seq_len, seq_len]
        scale_free_adjacency: Scale-free adjacency [seq_len, seq_len]
    
    Returns:
        Topology-filtered coupling [batch, seq_len, seq_len] or [seq_len, seq_len]
    """
    # Expand adjacency to batch dimension if needed
    if coupling_matrix.dim() == 3:
        batch_size = coupling_matrix.shape[0]
        adjacency_expanded = scale_free_adjacency.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        adjacency_expanded = scale_free_adjacency
    
    # Apply topology mask
    filtered_coupling = coupling_matrix * adjacency_expanded
    
    return filtered_coupling


def compute_nested_synchronization(
    phases: torch.Tensor,
    local_coupling: torch.Tensor,
    global_coupling: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute synchronization at multiple scales.
    
    Local synchronization: within neighborhoods
    Global synchronization: across network
    
    Args:
        phases: Current phases [batch, seq_len]
        local_coupling: Local coupling [seq_len, seq_len]
        global_coupling: Global coupling [seq_len, seq_len]
    
    Returns:
        {
            'local_order_param': R_local [batch],
            'global_order_param': R_global [batch],
            'nested_order_param': Combined R [batch],
        }
    """
    batch_size, seq_len = phases.shape
    
    # Compute local order parameter (within neighborhoods)
    local_R = []
    for b in range(batch_size):
        phases_b = phases[b]  # [seq_len]
        
        # Find local clusters (connected components in local coupling)
        local_clusters = []
        visited = set()
        
        for i in range(seq_len):
            if i in visited:
                continue
            
            # Find neighborhood
            neighbors = torch.where(local_coupling[i] > 0.1)[0].tolist()
            cluster = [i] + neighbors
            local_clusters.append(cluster)
            visited.update(cluster)
        
        # Compute order parameter for each local cluster
        local_Rs = []
        for cluster in local_clusters:
            if len(cluster) > 1:
                cluster_phases = phases_b[cluster]
                complex_phases = torch.exp(1j * cluster_phases)
                R_cluster = torch.abs(complex_phases.mean())
                local_Rs.append(R_cluster.item())
        
        local_R.append(np.mean(local_Rs) if local_Rs else 0.0)
    
    # Compute global order parameter (across network)
    complex_phases = torch.exp(1j * phases)  # [batch, seq_len]
    global_R = torch.abs(complex_phases.mean(dim=-1))  # [batch]
    
    # Nested order parameter (weighted combination)
    local_R_tensor = torch.tensor(local_R, device=phases.device, dtype=phases.dtype)
    nested_R = 0.7 * local_R_tensor + 0.3 * global_R
    
    return {
        'local_order_param': local_R_tensor,
        'global_order_param': global_R,
        'nested_order_param': nested_R,
    }


class MultiScaleStructures:
    """
    Multi-scale/hierarchical network structures manager.
    
    Provides hierarchical coupling, scale-free topology, and nested synchronization.
    """
    
    def __init__(
        self,
        use_hierarchical_coupling: bool = False,
        local_scale: int = 5,
        global_scale: int = 20,
        local_weight: float = 0.7,
        global_weight: float = 0.3,
        use_scale_free_topology: bool = False,
        hub_fraction: float = 0.1,
        power_law_exponent: float = 2.5,
    ):
        """
        Initialize multi-scale structures.
        
        Args:
            use_hierarchical_coupling: Enable hierarchical coupling
            local_scale: Size of local neighborhood
            global_scale: Range of global connections
            local_weight: Weight for local coupling
            global_weight: Weight for global coupling
            use_scale_free_topology: Enable scale-free topology
            hub_fraction: Fraction of nodes that are hubs
            power_law_exponent: Power-law exponent for degree distribution
        """
        self.use_hierarchical_coupling = use_hierarchical_coupling
        self.local_scale = local_scale
        self.global_scale = global_scale
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.use_scale_free_topology = use_scale_free_topology
        self.hub_fraction = hub_fraction
        self.power_law_exponent = power_law_exponent
        
        # Cache for pre-computed structures
        self.local_coupling_cache = {}
        self.global_coupling_cache = {}
        self.scale_free_adjacency_cache = {}
    
    def get_hierarchical_coupling(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get or compute hierarchical coupling.
        
        Args:
            seq_len: Sequence length
            device: Device
        
        Returns:
            (local_coupling, global_coupling) or (None, None) if disabled
        """
        if not self.use_hierarchical_coupling:
            return None, None
        
        cache_key = seq_len
        if cache_key not in self.local_coupling_cache:
            local_coupling, global_coupling = construct_hierarchical_coupling(
                seq_len,
                self.local_scale,
                self.global_scale,
                self.local_weight,
                self.global_weight,
                device=device,
            )
            self.local_coupling_cache[cache_key] = local_coupling
            self.global_coupling_cache[cache_key] = global_coupling
        
        return self.local_coupling_cache[cache_key], self.global_coupling_cache[cache_key]
    
    def get_scale_free_topology(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Get or compute scale-free topology.
        
        Args:
            seq_len: Sequence length
            device: Device
            seed: Random seed
        
        Returns:
            Scale-free adjacency matrix or None if disabled
        """
        if not self.use_scale_free_topology:
            return None
        
        cache_key = (seq_len, seed)
        if cache_key not in self.scale_free_adjacency_cache:
            adjacency = generate_scale_free_topology(
                seq_len,
                self.hub_fraction,
                self.power_law_exponent,
                device=device,
                seed=seed,
            )
            self.scale_free_adjacency_cache[cache_key] = adjacency
        
        return self.scale_free_adjacency_cache[cache_key]
    
    def apply_to_coupling(
        self,
        coupling_matrix: torch.Tensor,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Apply multi-scale structures to coupling matrix.
        
        Args:
            coupling_matrix: Base coupling [batch, seq_len, seq_len]
            seq_len: Sequence length
            device: Device
        
        Returns:
            Modified coupling matrix
        """
        result = coupling_matrix.clone()
        
        # Apply hierarchical coupling
        if self.use_hierarchical_coupling:
            local_coupling, global_coupling = self.get_hierarchical_coupling(seq_len, device)
            if local_coupling is not None and global_coupling is not None:
                result = combine_hierarchical_coupling(
                    local_coupling,
                    global_coupling,
                    base_coupling=result,
                )
        
        # Apply scale-free topology
        if self.use_scale_free_topology:
            adjacency = self.get_scale_free_topology(seq_len, device)
            if adjacency is not None:
                result = apply_scale_free_topology_to_coupling(result, adjacency)
        
        return result
    
    def compute_nested_sync(
        self,
        phases: torch.Tensor,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Compute nested synchronization if hierarchical coupling is enabled.
        
        Args:
            phases: Current phases [batch, seq_len]
            seq_len: Sequence length
            device: Device
        
        Returns:
            Nested synchronization metrics or None if disabled
        """
        if not self.use_hierarchical_coupling:
            return None
        
        local_coupling, global_coupling = self.get_hierarchical_coupling(seq_len, device)
        if local_coupling is None or global_coupling is None:
            return None
        
        return compute_nested_synchronization(phases, local_coupling, global_coupling)

