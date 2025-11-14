"""
Enhanced Swarm Coordinator with Additional Networks

Integrates all the networks from swarm_networks.py into the SwarmCoordinator.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
from modules.recursive_swarm import SwarmCoordinator, SwarmConfig, TinyRecursiveModel
from modules.swarm_networks import (
    InterSwarmCommunicationNetwork,
    CrossScaleAttentionNetwork,
    HierarchicalAggregationNetwork,
    SwarmRoutingNetwork,
    RecursiveDynamicsTrackingNetwork,
    MetaCoordinatorNetwork,
)


class EnhancedSwarmCoordinator(SwarmCoordinator):
    """
    Enhanced Swarm Coordinator with additional networks.
    
    Adds:
    - Inter-swarm communication network
    - Hierarchical aggregation network
    - Swarm routing network
    - Recursive dynamics tracking
    """
    
    def __init__(
        self,
        config: SwarmConfig,
        vocab_size: int,
        use_enhanced_networks: bool = True,
    ):
        # Initialize base SwarmCoordinator
        super().__init__(config, vocab_size)
        
        self.use_enhanced_networks = use_enhanced_networks
        
        if use_enhanced_networks:
            # Replace simple communication with attention-based
            self.inter_swarm_comm = InterSwarmCommunicationNetwork(
                d_model=config.swarm_d_model,
                n_heads=config.swarm_n_heads,
                num_swarm_members=config.num_swarm_models,
            )
            
            # Hierarchical aggregation
            self.hierarchical_agg = HierarchicalAggregationNetwork(
                d_model=config.swarm_d_model,
                n_heads=config.swarm_n_heads,
            )
            
            # Learned routing
            self.routing_network = SwarmRoutingNetwork(
                d_model=config.swarm_d_model,
                num_swarm_members=config.num_swarm_models,
            )
            
            # Dynamics tracking
            self.dynamics_tracker = RecursiveDynamicsTrackingNetwork(
                d_model=config.swarm_d_model,
            )
            
            # Meta-coordinator (optional)
            self.meta_coordinator = MetaCoordinatorNetwork(
                d_model=config.swarm_d_model,
                num_swarm_members=config.num_swarm_models,
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        num_iterations: int = None,
        return_dynamics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced forward pass with additional networks.
        """
        if num_iterations is None:
            num_iterations = self.config.num_recursive_iterations
        
        batch_size, seq_len = input_ids.shape
        
        # Chunk sequence
        chunks = self.chunk_sequence(input_ids)
        num_chunks = len(chunks)
        
        # Get chunk features for routing (if using enhanced networks)
        if self.use_enhanced_networks:
            # Encode chunks for routing
            chunk_features = []
            for chunk in chunks:
                # Simple encoding: use embeddings
                chunk_embeds = self.swarm[0].token_embedding(chunk).mean(dim=1)  # [batch, d_model]
                chunk_features.append(chunk_embeds)
            
            chunk_features_tensor = torch.stack(chunk_features, dim=1)  # [batch, num_chunks, d_model]
            
            # Learned routing
            routing_probs = self.routing_network(chunk_features_tensor)  # [batch, num_chunks, num_members]
            
            # Sample assignments from routing probabilities
            chunk_assignments = routing_probs.argmax(dim=-1)  # [batch, num_chunks]
        else:
            # Fallback to round-robin
            chunk_assignments = [i % self.config.num_swarm_models for i in range(num_chunks)]
        
        # Process chunks
        all_outputs = []
        all_communications = []
        state_histories = []  # For dynamics tracking
        
        for chunk_idx, chunk in enumerate(chunks):
            # Get swarm member assignment
            if self.use_enhanced_networks:
                swarm_idx = chunk_assignments[0, chunk_idx].item()  # Use first batch
            else:
                swarm_idx = chunk_assignments[chunk_idx]
            
            tiny_model = self.swarm[swarm_idx]
            
            # Get context from other swarm members
            context = None
            if len(all_communications) > 0:
                if self.use_enhanced_networks:
                    # Use inter-swarm communication network
                    # Stack communications: [num_comms, batch, d_model]
                    prev_comms_list = all_communications[-self.config.num_swarm_models:]
                    if len(prev_comms_list) > 0:
                        comms_tensor = torch.stack(prev_comms_list, dim=0)  # [num_comms, batch, d_model]
                        # Transpose to [batch, num_comms, d_model]
                        comms_tensor = comms_tensor.transpose(0, 1)
                        
                        enhanced_comms = self.inter_swarm_comm(comms_tensor)
                        context = enhanced_comms.mean(dim=1)  # [batch, d_model]
                else:
                    # Simple averaging
                    prev_comms = torch.stack(all_communications[-self.config.num_swarm_models:], dim=0)
                    # [num_comms, batch, d_model] -> [batch, d_model]
                    context = prev_comms.mean(dim=0)
            
            # Forward through tiny model
            logits, communication = tiny_model(
                chunk,
                context=context,
                num_iterations=num_iterations,
            )
            
            all_outputs.append(logits)
            all_communications.append(communication)
            
            # Track state for dynamics (store communication vectors)
            if return_dynamics:
                state_histories.append(communication)
        
        # Track recursive dynamics
        dynamics_metrics = {}
        if return_dynamics and len(state_histories) >= 2:
            dynamics_metrics = self.dynamics_tracker(state_histories)
        
        # Aggregate outputs using hierarchical aggregation
        chunk_outputs = torch.stack(all_outputs, dim=1)  # [batch, num_chunks, chunk_size, vocab_size]
        
        if self.use_enhanced_networks:
            # Use hierarchical aggregation
            # Get communication vectors as features
            communications_tensor = torch.stack(all_communications, dim=1)  # [batch, num_chunks, d_model]
            
            # Expand communications to match chunk structure
            # Each chunk gets its communication vector repeated
            chunk_features_list = []
            for chunk_idx in range(num_chunks):
                comm_vec = communications_tensor[:, chunk_idx, :]  # [batch, d_model]
                # Expand to chunk_size
                comm_expanded = comm_vec.unsqueeze(1).expand(-1, self.config.chunk_size, -1)
                chunk_features_list.append(comm_expanded)
            
            chunk_features_agg = torch.stack(chunk_features_list, dim=1)  # [batch, num_chunks, chunk_size, d_model]
            
            # Use hierarchical aggregation
            aggregated_features = self.hierarchical_agg(chunk_features_agg)
            
            # Project back to vocab_size
            aggregated = self._features_to_logits(aggregated_features, seq_len)
        else:
            # Fallback to simple aggregation
            aggregated = self._aggregate_chunks(chunk_outputs, seq_len)
        
        # Compute swarm metrics
        communications_tensor = torch.stack(all_communications, dim=1)  # [batch, num_chunks, d_model]
        swarm_coherence = self._compute_swarm_coherence(communications_tensor)
        
        metrics = {
            'num_chunks': num_chunks,
            'swarm_coherence': swarm_coherence,
            'communications': communications_tensor,
        }
        
        if return_dynamics:
            metrics.update(dynamics_metrics)
        
        return aggregated, metrics
    
    def _features_to_logits(
        self,
        features: torch.Tensor,
        target_seq_len: int,
    ) -> torch.Tensor:
        """
        Convert aggregated features back to logits.
        """
        # Simple projection (in practice, use learned projection)
        vocab_size = self.vocab_size
        
        # Project features to vocab_size
        if features.size(1) < target_seq_len:
            # Pad
            padding = torch.zeros(
                features.size(0),
                target_seq_len - features.size(1),
                features.size(2),
                device=features.device,
            )
            features = torch.cat([features, padding], dim=1)
        elif features.size(1) > target_seq_len:
            # Truncate
            features = features[:, :target_seq_len, :]
        
        # Project to vocab_size (simplified - should use shared embeddings)
        # For now, use a simple linear layer
        if not hasattr(self, 'feature_to_logits'):
            self.feature_to_logits = nn.Linear(self.config.swarm_d_model, vocab_size).to(features.device)
        
        logits = self.feature_to_logits(features)
        
        return logits


def create_enhanced_swarm_coordinator(
    config: SwarmConfig,
    vocab_size: int,
    use_enhanced: bool = True,
) -> EnhancedSwarmCoordinator:
    """
    Factory function to create enhanced swarm coordinator.
    """
    return EnhancedSwarmCoordinator(
        config=config,
        vocab_size=vocab_size,
        use_enhanced_networks=use_enhanced,
    )

