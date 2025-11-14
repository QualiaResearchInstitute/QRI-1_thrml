"""
CRV Graph Feature Extractor: Extracts structural features from attribution graphs.

Extracts three-level feature hierarchy:
1. Global graph statistics
2. Node influence and activation statistics
3. Topological and path-based features
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import networkx as nx


class GraphFeatureExtractor:
    """
    Extracts structural features from attribution graphs.
    
    Similar to CRV paper but adapted for oscillator graphs.
    """
    
    def __init__(self, prune_threshold: float = 0.8):
        """
        Initialize feature extractor.
        
        Args:
            prune_threshold: Threshold for pruning graph (keep nodes accounting for threshold% of total influence)
        """
        self.prune_threshold = prune_threshold
    
    def extract_features(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Extract comprehensive feature set from attribution graph.
        
        Args:
            graph: NetworkX directed graph
        
        Returns:
            Dictionary with feature names and values
        """
        features = {}
        
        # Prune graph to most influential nodes
        graph_pruned = self._prune_by_influence(graph, threshold=self.prune_threshold)
        
        # 1. Global Graph Statistics
        features.update(self._extract_global_stats(graph_pruned))
        
        # 2. Node Influence & Activation Statistics
        features.update(self._extract_node_stats(graph_pruned))
        
        # 3. Topological & Path-Based Features
        features.update(self._extract_topological_features(graph_pruned))
        
        return features
    
    def _prune_by_influence(
        self,
        graph: nx.DiGraph,
        threshold: float = 0.8,
    ) -> nx.DiGraph:
        """
        Prune graph to nodes accounting for threshold% of total influence.
        
        Args:
            graph: Graph to prune
            threshold: Cumulative influence threshold (0-1)
        
        Returns:
            Pruned graph
        """
        # Compute total influence per node
        node_influence = {}
        for node in graph.nodes():
            in_weight = sum(data.get('weight', 0.0) for _, _, data in graph.in_edges(node, data=True))
            out_weight = sum(data.get('weight', 0.0) for _, _, data in graph.out_edges(node, data=True))
            node_influence[node] = in_weight + out_weight
        
        # Sort by influence
        sorted_nodes = sorted(node_influence.items(), key=lambda x: x[1], reverse=True)
        
        # Compute cumulative influence
        total_influence = sum(node_influence.values())
        if total_influence == 0:
            return graph
        
        cumulative = 0.0
        keep_nodes = set()
        for node, influence in sorted_nodes:
            keep_nodes.add(node)
            cumulative += influence
            if cumulative / total_influence >= threshold:
                break
        
        # Always keep input tokens and output logits
        for node in graph.nodes():
            node_type = graph.nodes[node].get('node_type', '')
            if node_type in ['input_token', 'output_logit']:
                keep_nodes.add(node)
        
        return graph.subgraph(keep_nodes).copy()
    
    def _extract_global_stats(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Extract global graph statistics.
        
        Returns:
            Dictionary with global features
        """
        features = {}
        
        # Node counts
        features['n_nodes'] = float(graph.number_of_nodes())
        features['n_edges'] = float(graph.number_of_edges())
        
        # Count nodes by type
        node_types = {}
        for node in graph.nodes():
            node_type = graph.nodes[node].get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        features['n_input_tokens'] = float(node_types.get('input_token', 0))
        features['n_oscillator_features'] = float(node_types.get('phase_cluster', 0) + 
                                                   node_types.get('amplitude_peak', 0) + 
                                                   node_types.get('coupling_hub', 0))
        features['n_output_logits'] = float(node_types.get('output_logit', 0))
        
        # Logit statistics (if available)
        logit_probs = []
        for node in graph.nodes():
            if graph.nodes[node].get('node_type') == 'output_logit':
                prob = graph.nodes[node].get('probability', 0.0)
                logit_probs.append(prob)
        
        if logit_probs:
            logit_probs = np.array(logit_probs)
            features['top_logit_prob'] = float(np.max(logit_probs))
            features['logit_entropy'] = float(-np.sum(logit_probs * np.log(logit_probs + 1e-10)))
        else:
            features['top_logit_prob'] = 0.0
            features['logit_entropy'] = 0.0
        
        return features
    
    def _extract_node_stats(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Extract node influence and activation statistics.
        
        Returns:
            Dictionary with node-level features
        """
        features = {}
        
        # Compute influence scores for all nodes
        influence_scores = []
        for node in graph.nodes():
            in_weight = sum(data.get('weight', 0.0) for _, _, data in graph.in_edges(node, data=True))
            out_weight = sum(data.get('weight', 0.0) for _, _, data in graph.out_edges(node, data=True))
            total_influence = in_weight + out_weight
            influence_scores.append(total_influence)
        
        if influence_scores:
            influence_scores = np.array(influence_scores)
            features['mean_node_influence'] = float(np.mean(influence_scores))
            features['max_node_influence'] = float(np.max(influence_scores))
            features['std_node_influence'] = float(np.std(influence_scores))
        else:
            features['mean_node_influence'] = 0.0
            features['max_node_influence'] = 0.0
            features['std_node_influence'] = 0.0
        
        # Extract oscillator feature statistics
        phase_coherences = []
        feature_sizes = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data.get('node_type', '')
            
            if node_type == 'phase_cluster':
                coherence = node_data.get('coherence', 0.0)
                size = node_data.get('size', 0)
                phase_coherences.append(coherence)
                feature_sizes.append(size)
            elif node_type in ['amplitude_peak', 'coupling_hub']:
                feature_sizes.append(1)
        
        if phase_coherences:
            phase_coherences = np.array(phase_coherences)
            features['mean_phase_coherence'] = float(np.mean(phase_coherences))
            features['max_phase_coherence'] = float(np.max(phase_coherences))
            features['std_phase_coherence'] = float(np.std(phase_coherences))
        else:
            features['mean_phase_coherence'] = 0.0
            features['max_phase_coherence'] = 0.0
            features['std_phase_coherence'] = 0.0
        
        if feature_sizes:
            feature_sizes = np.array(feature_sizes)
            features['mean_feature_size'] = float(np.mean(feature_sizes))
            features['max_feature_size'] = float(np.max(feature_sizes))
        else:
            features['mean_feature_size'] = 0.0
            features['max_feature_size'] = 0.0
        
        # Layer-wise feature histogram (simplified: by node type)
        # In full implementation, would track actual layer information
        features['n_phase_clusters'] = float(sum(1 for n in graph.nodes() if graph.nodes[n].get('node_type') == 'phase_cluster'))
        features['n_amplitude_peaks'] = float(sum(1 for n in graph.nodes() if graph.nodes[n].get('node_type') == 'amplitude_peak'))
        features['n_coupling_hubs'] = float(sum(1 for n in graph.nodes() if graph.nodes[n].get('node_type') == 'coupling_hub'))
        
        return features
    
    def _extract_topological_features(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Extract topological and path-based features.
        
        Returns:
            Dictionary with topological features
        """
        features = {}
        
        # Edge statistics
        edge_weights = []
        for _, _, data in graph.edges(data=True):
            weight = data.get('weight', 0.0)
            edge_weights.append(weight)
        
        if edge_weights:
            edge_weights = np.array(edge_weights)
            features['mean_edge_weight'] = float(np.mean(edge_weights))
            features['max_edge_weight'] = float(np.max(edge_weights))
            features['std_edge_weight'] = float(np.std(edge_weights))
            features['total_edge_weight'] = float(np.sum(edge_weights))
        else:
            features['mean_edge_weight'] = 0.0
            features['max_edge_weight'] = 0.0
            features['std_edge_weight'] = 0.0
            features['total_edge_weight'] = 0.0
        
        # Graph density
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        if n_nodes > 1:
            max_edges = n_nodes * (n_nodes - 1)
            features['graph_density'] = float(n_edges / max_edges) if max_edges > 0 else 0.0
        else:
            features['graph_density'] = 0.0
        
        # Centrality measures
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(graph)
            if degree_centrality:
                deg_values = list(degree_centrality.values())
                features['mean_degree_centrality'] = float(np.mean(deg_values))
                features['max_degree_centrality'] = float(np.max(deg_values))
            else:
                features['mean_degree_centrality'] = 0.0
                features['max_degree_centrality'] = 0.0
            
            # Betweenness centrality (weighted)
            try:
                betweenness = nx.betweenness_centrality(graph, weight='weight')
                if betweenness:
                    betw_values = list(betweenness.values())
                    features['mean_betweenness_centrality'] = float(np.mean(betw_values))
                    features['max_betweenness_centrality'] = float(np.max(betw_values))
                else:
                    features['mean_betweenness_centrality'] = 0.0
                    features['max_betweenness_centrality'] = 0.0
            except Exception:
                features['mean_betweenness_centrality'] = 0.0
                features['max_betweenness_centrality'] = 0.0
        except Exception:
            features['mean_degree_centrality'] = 0.0
            features['max_degree_centrality'] = 0.0
            features['mean_betweenness_centrality'] = 0.0
            features['max_betweenness_centrality'] = 0.0
        
        # Connectivity
        try:
            # Weakly connected components
            if graph.is_directed():
                undirected = graph.to_undirected()
            else:
                undirected = graph
            
            components = list(nx.connected_components(undirected))
            features['n_connected_components'] = float(len(components))
            
            # Average shortest path length (in largest component)
            if components:
                largest_component = max(components, key=len)
                if len(largest_component) > 1:
                    subgraph = undirected.subgraph(largest_component)
                    try:
                        avg_path_length = nx.average_shortest_path_length(subgraph)
                        features['avg_shortest_path_length'] = float(avg_path_length)
                    except (nx.NetworkXError, nx.NetworkXNotImplemented):
                        features['avg_shortest_path_length'] = 0.0
                else:
                    features['avg_shortest_path_length'] = 0.0
            else:
                features['avg_shortest_path_length'] = 0.0
            
            # Shortest path from input tokens to output logits
            input_tokens = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'input_token']
            output_logits = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'output_logit']
            
            if input_tokens and output_logits:
                path_lengths = []
                for token in input_tokens:
                    for logit in output_logits:
                        try:
                            if graph.is_directed():
                                path_length = nx.shortest_path_length(graph, token, logit)
                            else:
                                path_length = nx.shortest_path_length(undirected, token, logit)
                            path_lengths.append(path_length)
                        except nx.NetworkXNoPath:
                            pass
                
                if path_lengths:
                    features['min_token_to_logit_path'] = float(np.min(path_lengths))
                    features['mean_token_to_logit_path'] = float(np.mean(path_lengths))
                else:
                    features['min_token_to_logit_path'] = 0.0
                    features['mean_token_to_logit_path'] = 0.0
            else:
                features['min_token_to_logit_path'] = 0.0
                features['mean_token_to_logit_path'] = 0.0
        except Exception:
            features['n_connected_components'] = 0.0
            features['avg_shortest_path_length'] = 0.0
            features['min_token_to_logit_path'] = 0.0
            features['mean_token_to_logit_path'] = 0.0
        
        return features

