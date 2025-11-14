"""
Additional Networks for Swarm-Augmented System

Networks to add for enhanced swarm coordination and hierarchical reasoning:
1. Inter-Swarm Communication Network (better than simple averaging)
2. Cross-Scale Attention Network (swarm ↔ main model)
3. Meta-Coordinator Network (orchestrates swarm)
4. Hierarchical Aggregation Network (multi-level fusion)
5. Recursive Dynamics Tracking Network (monitors swarm dynamics)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
import math


class InterSwarmCommunicationNetwork(nn.Module):
    """
    Learnable communication network between swarm members.
    
    Replaces simple averaging with attention-based communication.
    Each swarm member can attend to other members' communications.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        num_swarm_members: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_swarm_members = num_swarm_members
        
        # Multi-head attention for inter-swarm communication
        self.communication_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        
        # Learnable swarm member embeddings (for positional awareness)
        self.swarm_embeddings = nn.Embedding(num_swarm_members, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        communications: torch.Tensor,
        swarm_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process inter-swarm communications.
        
        Args:
            communications: [batch, num_members, d_model] communication vectors
            swarm_indices: [num_members] indices of swarm members
        
        Returns:
            enhanced_communications: [batch, num_members, d_model]
        """
        batch_size, num_members, d_model = communications.shape
        
        # Add swarm member embeddings
        if swarm_indices is None:
            swarm_indices = torch.arange(num_members, device=communications.device)
        
        member_embeds = self.swarm_embeddings(swarm_indices)  # [num_members, d_model]
        member_embeds = member_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add embeddings to communications
        x = communications + member_embeds
        
        # Self-attention: each member attends to all members
        enhanced, _ = self.communication_attention(x, x, x)
        
        # Residual + norm
        enhanced = self.norm(enhanced + x)
        
        # Project
        output = self.output_proj(enhanced)
        
        return output


class CrossScaleAttentionNetwork(nn.Module):
    """
    Attention network connecting swarm (local) and main model (global).
    
    Enables bidirectional information flow:
    - Swarm → Main: Local insights inform global understanding
    - Main → Swarm: Global context guides local processing
    """
    
    def __init__(
        self,
        main_d_model: int,
        swarm_d_model: int,
        n_heads: int = 8,
    ):
        super().__init__()
        self.main_d_model = main_d_model
        self.swarm_d_model = swarm_d_model
        
        # Project to common dimension
        common_dim = min(main_d_model, swarm_d_model)
        self.main_proj = nn.Linear(main_d_model, common_dim)
        self.swarm_proj = nn.Linear(swarm_d_model, common_dim)
        
        # Cross-attention: swarm attends to main
        self.swarm_to_main = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        
        # Cross-attention: main attends to swarm
        self.main_to_swarm = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        
        # Output projections
        self.main_output = nn.Linear(common_dim, main_d_model)
        self.swarm_output = nn.Linear(common_dim, swarm_d_model)
        
        self.norm_main = nn.LayerNorm(main_d_model)
        self.norm_swarm = nn.LayerNorm(swarm_d_model)
    
    def forward(
        self,
        main_features: torch.Tensor,  # [batch, seq_len, main_d_model]
        swarm_features: torch.Tensor,  # [batch, num_chunks, swarm_d_model]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-scale attention between main and swarm.
        
        Returns:
            enhanced_main: [batch, seq_len, main_d_model]
            enhanced_swarm: [batch, num_chunks, swarm_d_model]
        """
        # Project to common dimension
        main_proj = self.main_proj(main_features)
        swarm_proj = self.swarm_proj(swarm_features)
        
        # Swarm attends to main (local → global)
        swarm_enhanced, _ = self.swarm_to_main(
            query=swarm_proj,
            key=main_proj,
            value=main_proj,
        )
        
        # Main attends to swarm (global → local)
        main_enhanced, _ = self.main_to_swarm(
            query=main_proj,
            key=swarm_proj,
            value=swarm_proj,
        )
        
        # Project back and residual
        main_out = self.norm_main(main_features + self.main_output(main_enhanced))
        swarm_out = self.norm_swarm(swarm_features + self.swarm_output(swarm_enhanced))
        
        return main_out, swarm_out


class MetaCoordinatorNetwork(nn.Module):
    """
    Meta-network that orchestrates the swarm.
    
    Learns to:
    - Allocate chunks optimally
    - Coordinate swarm member activities
    - Balance load across swarm
    - Detect when swarm members should communicate
    """
    
    def __init__(
        self,
        d_model: int,
        num_swarm_members: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_swarm_members = num_swarm_members
        
        # Encoder: processes sequence to get coordination signal
        self.encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Allocation network: decides chunk-to-member assignment
        self.allocation_network = nn.Sequential(
            nn.Linear(hidden_dim + d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_swarm_members),
            nn.Softmax(dim=-1),  # Probability distribution over members
        )
        
        # Coordination signal: guides swarm behavior
        self.coordination_signal = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.Tanh(),  # Bounded coordination signal
        )
        
        # Load balancer: tracks and balances workload
        self.load_tracker = nn.Parameter(torch.zeros(num_swarm_members))
    
    def forward(
        self,
        sequence_features: torch.Tensor,  # [batch, seq_len, d_model]
        chunk_features: List[torch.Tensor],  # List of [batch, chunk_size, d_model]
    ) -> Dict[str, torch.Tensor]:
        """
        Coordinate swarm based on input.
        
        Returns:
            allocation: [batch, num_chunks, num_members] allocation probabilities
            coordination: [batch, d_model] coordination signal
            load_balance: [num_members] current load
        """
        batch_size, seq_len, d_model = sequence_features.shape
        
        # Encode sequence
        seq_encoded = sequence_features.mean(dim=1)  # [batch, d_model]
        encoded = self.encoder(seq_encoded)  # [batch, hidden_dim]
        
        # Generate coordination signal
        coordination = self.coordination_signal(encoded)  # [batch, d_model]
        
        # Allocate chunks to members
        allocations = []
        for chunk_feat in chunk_features:
            chunk_mean = chunk_feat.mean(dim=1)  # [batch, d_model]
            # Concatenate encoded + chunk features
            combined = torch.cat([encoded, chunk_mean], dim=-1)  # [batch, hidden_dim + d_model]
            allocation = self.allocation_network(combined)  # [batch, num_members]
            allocations.append(allocation)
        
        allocations = torch.stack(allocations, dim=1)  # [batch, num_chunks, num_members]
        
        # Update load tracker
        load_balance = allocations.sum(dim=1).mean(dim=0)  # [num_members]
        
        return {
            'allocation': allocations,
            'coordination': coordination,
            'load_balance': load_balance,
        }


class HierarchicalAggregationNetwork(nn.Module):
    """
    Multi-level aggregation network for combining swarm outputs.
    
    Uses hierarchical attention to aggregate:
    - Chunk-level: Within each chunk
    - Swarm-level: Across swarm members
    - Global-level: Full sequence
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        num_levels: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        
        # Level 1: Chunk-level attention
        self.chunk_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        
        # Level 2: Swarm-level attention
        self.swarm_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        
        # Level 3: Global attention
        self.global_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_levels)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_levels)
        ])
    
    def forward(
        self,
        chunk_outputs: torch.Tensor,  # [batch, num_chunks, chunk_size, d_model]
        global_features: Optional[torch.Tensor] = None,  # [batch, seq_len, d_model]
    ) -> torch.Tensor:
        """
        Hierarchical aggregation of swarm outputs.
        
        Returns:
            aggregated: [batch, seq_len, d_model]
        """
        batch_size, num_chunks, chunk_size, d_model = chunk_outputs.shape
        
        # Level 1: Chunk-level (within each chunk)
        chunk_outputs_flat = chunk_outputs.view(batch_size * num_chunks, chunk_size, d_model)
        chunk_agg, _ = self.chunk_attention(chunk_outputs_flat, chunk_outputs_flat, chunk_outputs_flat)
        chunk_agg = self.norms[0](chunk_agg + chunk_outputs_flat)
        chunk_agg = self.fusion_layers[0](chunk_agg)
        
        # Reshape: [batch, num_chunks, d_model] (mean over chunk)
        chunk_reprs = chunk_agg.mean(dim=1).view(batch_size, num_chunks, d_model)
        
        # Level 2: Swarm-level (across chunks)
        swarm_agg, _ = self.swarm_attention(chunk_reprs, chunk_reprs, chunk_reprs)
        swarm_agg = self.norms[1](swarm_agg + chunk_reprs)
        swarm_agg = self.fusion_layers[1](swarm_agg)
        
        # Expand back to sequence length
        # For now, simple expansion (could use learned upsampling)
        seq_len = num_chunks * chunk_size
        expanded = swarm_agg.repeat_interleave(chunk_size, dim=1)  # [batch, seq_len, d_model]
        expanded = expanded[:, :seq_len, :]
        
        # Level 3: Global attention (if global features provided)
        if global_features is not None:
            global_agg, _ = self.global_attention(
                query=expanded,
                key=global_features,
                value=global_features,
            )
            global_agg = self.norms[2](global_agg + expanded)
            global_agg = self.fusion_layers[2](global_agg)
            return global_agg
        
        return expanded


class RecursiveDynamicsTrackingNetwork(nn.Module):
    """
    Network that tracks recursive dynamics across swarm members.
    
    Monitors:
    - Aliasing/renormalization per chunk
    - Attractor formation
    - Spectral properties
    - Meta-dynamics
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Encoder for state sequences
        self.encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Dynamics predictor
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Current + previous
            nn.GELU(),
            nn.Linear(hidden_dim, 4),  # 4 dynamics indicators
            nn.Sigmoid(),
        )
        
        # Output: aliasing, renormalization, attractor, chaos
        self.dynamics_head = nn.Linear(hidden_dim, d_model)
    
    def forward(
        self,
        state_sequence: List[torch.Tensor],  # List of [batch, d_model]
    ) -> Dict[str, torch.Tensor]:
        """
        Track recursive dynamics in state sequence.
        
        Returns:
            dynamics_metrics: Dictionary with dynamics indicators
        """
        if len(state_sequence) < 2:
            return {
                'aliasing': torch.tensor(0.0),
                'renormalization': torch.tensor(0.0),
                'attractor': torch.tensor(0.0),
                'chaos': torch.tensor(0.0),
            }
        
        # Encode states
        states = torch.stack(state_sequence)  # [num_states, batch, d_model]
        states_encoded = self.encoder(states)  # [num_states, batch, hidden_dim]
        
        # Compute dynamics indicators
        current = states_encoded[-1]  # [batch, hidden_dim]
        previous = states_encoded[-2]  # [batch, hidden_dim]
        
        combined = torch.cat([current, previous], dim=-1)  # [batch, hidden_dim * 2]
        dynamics = self.dynamics_predictor(combined)  # [batch, 4]
        
        return {
            'aliasing': dynamics[:, 0],
            'renormalization': dynamics[:, 1],
            'attractor': dynamics[:, 2],
            'chaos': dynamics[:, 3],
        }


class SwarmRoutingNetwork(nn.Module):
    """
    Learned routing network for optimal chunk-to-member assignment.
    
    Replaces round-robin with learned routing based on:
    - Chunk content
    - Member capabilities
    - Current load
    """
    
    def __init__(
        self,
        d_model: int,
        num_swarm_members: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_swarm_members = num_swarm_members
        
        # Chunk encoder
        self.chunk_encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Member capability embeddings
        self.member_embeddings = nn.Embedding(num_swarm_members, hidden_dim)
        
        # Routing network
        self.routing = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # chunk + member
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Routing score
        )
        
        # Load tracker
        self.load_tracker = nn.Parameter(torch.zeros(num_swarm_members))
    
    def forward(
        self,
        chunk_features: torch.Tensor,  # [batch, num_chunks, d_model]
    ) -> torch.Tensor:
        """
        Route chunks to swarm members.
        
        Returns:
            routing_scores: [batch, num_chunks, num_members] routing scores
        """
        batch_size, num_chunks, d_model = chunk_features.shape
        
        # Encode chunks
        chunk_encoded = self.chunk_encoder(chunk_features)  # [batch, num_chunks, hidden_dim]
        
        # Get member capabilities
        member_caps = self.member_embeddings.weight  # [num_members, hidden_dim]
        
        # Compute routing scores
        routing_scores = []
        for chunk_idx in range(num_chunks):
            chunk_enc = chunk_encoded[:, chunk_idx, :]  # [batch, hidden_dim]
            chunk_scores = []
            
            for member_idx in range(self.num_swarm_members):
                member_cap = member_caps[member_idx]  # [hidden_dim]
                # Expand for batch
                member_cap_expanded = member_cap.unsqueeze(0).expand(batch_size, -1)
                
                # Combine chunk + member
                combined = torch.cat([chunk_enc, member_cap_expanded], dim=-1)
                score = self.routing(combined)  # [batch, 1]
                
                # Adjust by load
                load_penalty = self.load_tracker[member_idx]
                score = score - load_penalty
                
                chunk_scores.append(score)
            
            routing_scores.append(torch.cat(chunk_scores, dim=-1))  # [batch, num_members]
        
        routing_scores = torch.stack(routing_scores, dim=1)  # [batch, num_chunks, num_members]
        
        # Softmax over members
        routing_probs = torch.softmax(routing_scores, dim=-1)
        
        return routing_probs

