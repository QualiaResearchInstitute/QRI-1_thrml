"""
CRV Attribution Graph Builder: Builds attribution graphs from oscillator dynamics.

Constructs directed graphs representing information flow through oscillator features,
similar to CRV paper but adapted for resonance transformer's oscillator dynamics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
import networkx as nx

from modules.crv_oscillator_features import OscillatorFeatureExtractor


class OscillatorAttributionGraph:
    """
    Builds attribution graphs from oscillator dynamics.
    
    Nodes: oscillator features (phase clusters, amplitude peaks, coupling hubs)
    Edges: information flow between features (via coupling matrix, attention paths)
    """
    
    def __init__(
        self,
        feature_extractor: Optional[OscillatorFeatureExtractor] = None,
        attribution_threshold: float = 0.1,
        max_nodes: int = 4096,
    ):
        """
        Initialize attribution graph builder.
        
        Args:
            feature_extractor: OscillatorFeatureExtractor instance
            attribution_threshold: Minimum attribution score to include edge
            max_nodes: Maximum number of nodes to include in graph
        """
        self.feature_extractor = feature_extractor or OscillatorFeatureExtractor()
        self.attribution_threshold = attribution_threshold
        self.max_nodes = max_nodes
    
    def build_graph(
        self,
        oscillator_features: Dict[str, Any],
        logits: torch.Tensor,
        step_idx: int = 0,
        batch_idx: int = 0,
    ) -> nx.DiGraph:
        """
        Build directed graph representing information flow.
        
        Args:
            oscillator_features: Dictionary with phases, amplitudes, coupling, CDNS
            logits: Output logits [batch, seq_len, vocab_size]
            step_idx: Step index in reasoning chain
            batch_idx: Batch index
        
        Returns:
            NetworkX directed graph with:
            - Nodes: input tokens, oscillator features, output logits
            - Edges: weighted by influence/attribution
        """
        G = nx.DiGraph()
        
        # Extract discrete feature nodes
        features = self.feature_extractor.extract_features(
            phases=oscillator_features.get('phases'),
            amplitudes=oscillator_features.get('amplitudes'),
            coupling_matrix=oscillator_features.get('coupling_matrix'),
            cdns=oscillator_features.get('cdns'),
        )
        
        feature_nodes = self.feature_extractor.get_feature_nodes(features, batch_idx=batch_idx)
        
        # Add input token nodes
        seq_len = oscillator_features.get('phases', torch.zeros(1, 128)).shape[-2] if isinstance(oscillator_features.get('phases'), torch.Tensor) else 128
        for i in range(seq_len):
            G.add_node(f"token_{i}", node_type="input_token", position=i)
        
        # Add oscillator feature nodes
        for node_data in feature_nodes:
            node_id = f"feature_{node_data['node_id']}"
            G.add_node(node_id, **node_data)
        
        # Add output logit nodes (top-k logits)
        if logits is not None:
            logits_batch = logits[batch_idx] if logits.dim() > 2 else logits
            if logits_batch.dim() == 2:  # [seq_len, vocab_size]
                # Use final token's logits
                final_logits = logits_batch[-1]
                top_k = min(10, len(final_logits))
                top_probs, top_indices = torch.topk(torch.softmax(final_logits, dim=-1), k=top_k)
                
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    node_id = f"logit_{idx.item()}"
                    G.add_node(node_id, node_type="output_logit", vocab_idx=int(idx.item()), probability=float(prob.item()))
        
        # Add edges based on coupling matrix
        coupling_matrix = oscillator_features.get('coupling_matrix')
        if coupling_matrix is not None:
            self._add_coupling_edges(G, coupling_matrix, batch_idx, feature_nodes)
        
        # Add edges from features to logits (simplified attribution)
        if logits is not None and feature_nodes:
            self._add_feature_to_logit_edges(G, feature_nodes, logits, batch_idx)
        
        # Add edges from tokens to features (based on position)
        self._add_token_to_feature_edges(G, feature_nodes, seq_len)
        
        # Prune graph to most influential nodes
        G = self._prune_graph(G, max_nodes=self.max_nodes)
        
        return G
    
    def _add_coupling_edges(
        self,
        G: nx.DiGraph,
        coupling_matrix: torch.Tensor,
        batch_idx: int,
        feature_nodes: List[Dict[str, Any]],
    ) -> None:
        """
        Add edges based on coupling matrix between oscillator features.
        
        Args:
            G: Graph to add edges to
            coupling_matrix: Coupling matrix [batch, seq_len, seq_len] or [batch, heads, seq_len, seq_len]
            batch_idx: Batch index
            feature_nodes: List of feature node dictionaries
        """
        # Handle different shapes
        if coupling_matrix.dim() == 4:  # [batch, heads, seq_len, seq_len]
            coupling_matrix = coupling_matrix[batch_idx].mean(dim=0)  # Average over heads
        elif coupling_matrix.dim() == 3:  # [batch, seq_len, seq_len]
            coupling_matrix = coupling_matrix[batch_idx]
        else:
            coupling_matrix = coupling_matrix
        
        coupling_np = coupling_matrix.detach().cpu().numpy()
        seq_len = coupling_np.shape[0]
        
        # Add edges between features at different positions
        for node_data in feature_nodes:
            if 'position' in node_data:
                pos_i = node_data['position']
                node_id_i = f"feature_{node_data['node_id']}"
                
                # Find other features at different positions
                for other_node in feature_nodes:
                    if other_node['node_id'] != node_data['node_id']:
                        if 'position' in other_node:
                            pos_j = other_node['position']
                            if 0 <= pos_i < seq_len and 0 <= pos_j < seq_len:
                                weight = float(coupling_np[pos_i, pos_j])
                                if weight > self.attribution_threshold:
                                    node_id_j = f"feature_{other_node['node_id']}"
                                    G.add_edge(node_id_i, node_id_j, weight=weight, edge_type="coupling")
            
            elif 'positions' in node_data:
                # Phase cluster spans multiple positions
                positions = node_data['positions']
                node_id_i = f"feature_{node_data['node_id']}"
                
                for pos_i in positions:
                    for other_node in feature_nodes:
                        if other_node['node_id'] != node_data['node_id']:
                            if 'position' in other_node:
                                pos_j = other_node['position']
                                if 0 <= pos_i < seq_len and 0 <= pos_j < seq_len:
                                    weight = float(coupling_np[pos_i, pos_j])
                                    if weight > self.attribution_threshold:
                                        node_id_j = f"feature_{other_node['node_id']}"
                                        G.add_edge(node_id_i, node_id_j, weight=weight, edge_type="coupling")
    
    def _add_feature_to_logit_edges(
        self,
        G: nx.DiGraph,
        feature_nodes: List[Dict[str, Any]],
        logits: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """
        Add edges from oscillator features to output logits.
        
        Uses simplified attribution: features near final token position influence final logits.
        
        Args:
            G: Graph to add edges to
            feature_nodes: List of feature node dictionaries
            logits: Output logits [batch, seq_len, vocab_size]
            batch_idx: Batch index
        """
        logits_batch = logits[batch_idx] if logits.dim() > 2 else logits
        if logits_batch.dim() == 2:  # [seq_len, vocab_size]
            seq_len = logits_batch.shape[0]
            final_pos = seq_len - 1
            
            # Get top logit nodes
            final_logits = logits_batch[-1]
            top_k = min(10, len(final_logits))
            top_probs, top_indices = torch.topk(torch.softmax(final_logits, dim=-1), k=top_k)
            
            # Connect features near final position to logits
            for node_data in feature_nodes:
                node_id = f"feature_{node_data['node_id']}"
                
                # Check if feature is near final position
                is_near_final = False
                if 'position' in node_data:
                    pos = node_data['position']
                    is_near_final = abs(pos - final_pos) <= 3  # Within 3 tokens
                elif 'positions' in node_data:
                    positions = node_data['positions']
                    is_near_final = any(abs(pos - final_pos) <= 3 for pos in positions)
                
                if is_near_final:
                    # Connect to top logits with weights based on feature importance
                    feature_importance = self._compute_feature_importance(node_data)
                    
                    for prob, idx in zip(top_probs, top_indices):
                        logit_node_id = f"logit_{idx.item()}"
                        if G.has_node(logit_node_id):
                            # Weight by feature importance and logit probability
                            weight = float(feature_importance * prob.item())
                            if weight > self.attribution_threshold * 0.1:  # Lower threshold for feature->logit
                                G.add_edge(node_id, logit_node_id, weight=weight, edge_type="feature_to_logit")
    
    def _add_token_to_feature_edges(
        self,
        G: nx.DiGraph,
        feature_nodes: List[Dict[str, Any]],
        seq_len: int,
    ) -> None:
        """
        Add edges from input tokens to oscillator features.
        
        Args:
            G: Graph to add edges to
            feature_nodes: List of feature node dictionaries
            seq_len: Sequence length
        """
        for node_data in feature_nodes:
            node_id = f"feature_{node_data['node_id']}"
            
            # Connect tokens to features based on position
            if 'position' in node_data:
                pos = node_data['position']
                if 0 <= pos < seq_len:
                    token_node_id = f"token_{pos}"
                    if G.has_node(token_node_id):
                        G.add_edge(token_node_id, node_id, weight=1.0, edge_type="token_to_feature")
            
            elif 'positions' in node_data:
                positions = node_data['positions']
                for pos in positions:
                    if 0 <= pos < seq_len:
                        token_node_id = f"token_{pos}"
                        if G.has_node(token_node_id):
                            G.add_edge(token_node_id, node_id, weight=1.0 / len(positions), edge_type="token_to_feature")
    
    def _compute_feature_importance(self, node_data: Dict[str, Any]) -> float:
        """
        Compute importance score for a feature node.
        
        Args:
            node_data: Feature node dictionary
        
        Returns:
            Importance score [0, 1]
        """
        importance = 0.5  # Default
        
        # Phase cluster: importance based on coherence
        if node_data.get('node_type') == 'phase_cluster':
            coherence = node_data.get('coherence', 0.5)
            size = node_data.get('size', 1)
            importance = float(coherence * (1.0 + 0.1 * size))  # Boost for larger clusters
        
        # Amplitude peak: high importance
        elif node_data.get('node_type') == 'amplitude_peak':
            importance = 0.8
        
        # Coupling hub: high importance
        elif node_data.get('node_type') == 'coupling_hub':
            importance = 0.9
        
        # CDNS feature: importance based on value
        elif node_data.get('node_type') == 'cdns_feature':
            value = abs(node_data.get('value', 0.0))
            importance = float(min(1.0, value))
        
        return min(1.0, max(0.0, importance))
    
    def _prune_graph(self, G: nx.DiGraph, max_nodes: int = 4096) -> nx.DiGraph:
        """
        Prune graph to most influential nodes.
        
        Keeps nodes with highest total influence (sum of incoming + outgoing edge weights).
        
        Args:
            G: Graph to prune
            max_nodes: Maximum number of nodes to keep
        
        Returns:
            Pruned graph
        """
        if len(G.nodes()) <= max_nodes:
            return G
        
        # Compute node influence scores
        node_influence = {}
        for node in G.nodes():
            in_weight = sum(data.get('weight', 0.0) for _, _, data in G.in_edges(node, data=True))
            out_weight = sum(data.get('weight', 0.0) for _, _, data in G.out_edges(node, data=True))
            node_influence[node] = in_weight + out_weight
        
        # Keep top nodes
        sorted_nodes = sorted(node_influence.items(), key=lambda x: x[1], reverse=True)
        top_nodes = set(node for node, _ in sorted_nodes[:max_nodes])
        
        # Create subgraph with top nodes
        G_pruned = G.subgraph(top_nodes).copy()
        
        return G_pruned
    
    def compute_attribution_paths(
        self,
        G: nx.DiGraph,
        source_nodes: Optional[List[str]] = None,
        target_nodes: Optional[List[str]] = None,
        max_path_length: int = 5,
    ) -> List[Tuple[List[str], float]]:
        """
        Compute attribution paths from source to target nodes.
        
        Args:
            G: Attribution graph
            source_nodes: Source node IDs (default: input tokens)
            target_nodes: Target node IDs (default: output logits)
            max_path_length: Maximum path length
        
        Returns:
            List of (path, total_weight) tuples
        """
        if source_nodes is None:
            source_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'input_token']
        
        if target_nodes is None:
            target_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'output_logit']
        
        paths = []
        for source in source_nodes:
            for target in target_nodes:
                try:
                    # Find shortest paths
                    all_paths = list(nx.all_simple_paths(G, source, target, cutoff=max_path_length))
                    for path in all_paths:
                        # Compute path weight (product of edge weights)
                        path_weight = 1.0
                        for i in range(len(path) - 1):
                            edge_data = G.get_edge_data(path[i], path[i + 1], {})
                            edge_weight = edge_data.get('weight', 0.0)
                            path_weight *= edge_weight
                        paths.append((path, path_weight))
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by path weight
        paths.sort(key=lambda x: x[1], reverse=True)
        
        return paths

