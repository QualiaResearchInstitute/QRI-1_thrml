"""
Recursive Swarm: A swarm of tiny recursive models for hierarchical reasoning

Augments the main Resonance Transformer with a swarm of tiny recursive models
that can:
- Process sub-sequences in parallel
- Provide recursive refinement
- Enable meta-level reasoning
- Create hierarchical information flow

Each tiny model is a minimal recursive Transformer that can:
- Process small chunks (e.g., 8-16 tokens)
- Iteratively refine its understanding
- Communicate with other swarm members
- Feed insights back to the main model
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import math


@dataclass
class SwarmConfig:
    """Configuration for recursive swarm."""
    num_swarm_models: int = 8  # Number of tiny models in swarm
    swarm_d_model: int = 64  # Dimension for tiny models (much smaller)
    swarm_n_layers: int = 2  # Few layers for tiny models
    swarm_n_heads: int = 4  # Few heads
    chunk_size: int = 16  # Tokens per chunk
    overlap: int = 4  # Overlap between chunks
    num_recursive_iterations: int = 3  # Recursive refinement steps
    communication_dim: int = 32  # Dimension for inter-swarm communication
    use_attention_pooling: bool = True  # Pool swarm outputs with attention


class TinyRecursiveModel(nn.Module):
    """
    Tiny recursive Transformer model for swarm.
    
    Processes small chunks and refines them recursively.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 32,
        n_sim_steps: int = 5,  # Fewer steps for speed
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Import Resonance Transformer components
        import importlib.util
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        rt_file = project_root / "resonance_transformer.py"
        spec = importlib.util.spec_from_file_location("resonance_transformer", rt_file)
        rt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt_module)
        
        ResonanceTransformerBlock = rt_module.ResonanceTransformerBlock
        
        # Tiny transformer blocks
        self.layers = nn.ModuleList([
            ResonanceTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=2 * d_model,  # Smaller FFN
                n_sim_steps=n_sim_steps,
                dropout=0.1,
            )
            for _ in range(n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(0.1)
        
        # Communication vector (for swarm coordination)
        self.communication_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        num_iterations: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional recursive refinement.
        
        Args:
            input_ids: [batch, seq_len] token IDs
            context: [batch, context_dim] optional context from other swarm members
            num_iterations: Number of recursive refinement steps
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            communication: [batch, d_model] communication vector for swarm
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Add context if provided (broadcast to sequence)
        if context is not None:
            context_expanded = context.unsqueeze(1).expand(-1, seq_len, -1)
            x = x + context_expanded
        
        # Recursive refinement
        for iteration in range(num_iterations):
            # Pass through layers
            for layer in self.layers:
                x = layer(x, mask=None)
            
            # If multiple iterations, use output as input for next iteration
            if iteration < num_iterations - 1:
                # Project back through embeddings (soft feedback)
                x = self.token_embedding(input_ids) + 0.5 * x
        
        # Final output
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Communication vector (pooled representation)
        communication = self.communication_proj(x.mean(dim=1))  # [batch, d_model]
        
        return logits, communication


class SwarmCoordinator(nn.Module):
    """
    Coordinates a swarm of tiny recursive models.
    
    Manages:
    - Chunking input into sub-sequences
    - Distributing chunks to swarm members
    - Inter-swarm communication
    - Aggregating outputs
    """
    
    def __init__(self, config: SwarmConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Create swarm of tiny models
        self.swarm = nn.ModuleList([
            TinyRecursiveModel(
                vocab_size=vocab_size,
                d_model=config.swarm_d_model,
                n_layers=config.swarm_n_layers,
                n_heads=config.swarm_n_heads,
                max_seq_len=config.chunk_size + config.overlap,
                n_sim_steps=5,
            )
            for _ in range(config.num_swarm_models)
        ])
        
        # Communication network (allows swarm members to share information)
        self.communication_network = nn.ModuleList([
            nn.Linear(config.swarm_d_model, config.communication_dim)
            for _ in range(config.num_swarm_models)
        ])
        
        # Attention-based aggregation
        if config.use_attention_pooling:
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=config.swarm_d_model,
                num_heads=config.swarm_n_heads,
                batch_first=True,
            )
        else:
            self.attention_pool = None
        
        # Output projection
        self.output_proj = nn.Linear(config.swarm_d_model, config.swarm_d_model)
    
    def chunk_sequence(
        self,
        input_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Chunk sequence into overlapping sub-sequences.
        
        Returns:
            List of chunk tensors [batch, chunk_size]
        """
        batch_size, seq_len = input_ids.shape
        chunks = []
        
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        stride = chunk_size - overlap
        
        for start in range(0, seq_len, stride):
            end = min(start + chunk_size, seq_len)
            chunk = input_ids[:, start:end]
            
            # Pad if needed
            if chunk.size(1) < chunk_size:
                padding = torch.zeros(
                    batch_size,
                    chunk_size - chunk.size(1),
                    dtype=chunk.dtype,
                    device=chunk.device,
                )
                chunk = torch.cat([chunk, padding], dim=1)
            
            chunks.append(chunk)
        
        return chunks
    
    def forward(
        self,
        input_ids: torch.Tensor,
        num_iterations: int = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process input through swarm.
        
        Args:
            input_ids: [batch, seq_len] input tokens
            num_iterations: Recursive refinement iterations
        
        Returns:
            aggregated_output: [batch, seq_len, vocab_size] aggregated logits
            metrics: Dictionary with swarm metrics
        """
        if num_iterations is None:
            num_iterations = self.config.num_recursive_iterations
        
        batch_size, seq_len = input_ids.shape
        
        # Chunk sequence
        chunks = self.chunk_sequence(input_ids)
        num_chunks = len(chunks)
        
        # Distribute chunks to swarm (round-robin)
        chunk_assignments = [i % self.config.num_swarm_models for i in range(num_chunks)]
        
        # Process chunks in parallel (simulated - in practice could be truly parallel)
        all_outputs = []
        all_communications = []
        
        # First pass: process all chunks
        for chunk_idx, chunk in enumerate(chunks):
            swarm_idx = chunk_assignments[chunk_idx]
            tiny_model = self.swarm[swarm_idx]
            
            # Get context from other swarm members
            context = None
            if len(all_communications) > 0:
                # Average previous communications (simple communication)
                prev_comms = torch.stack(all_communications[-self.config.num_swarm_models:])
                context = prev_comms.mean(dim=0)
            
            # Forward through tiny model
            logits, communication = tiny_model(
                chunk,
                context=context,
                num_iterations=num_iterations,
            )
            
            all_outputs.append(logits)
            all_communications.append(communication)
        
        # Optional: Second pass with full swarm context (recursive refinement)
        # This allows swarm members to refine based on all other members' outputs
        if num_iterations > 1 and len(all_communications) >= self.config.num_swarm_models:
            # Re-process with full swarm context
            all_outputs_refined = []
            all_communications_refined = []
            
            # Create global swarm context
            global_context = torch.stack(all_communications).mean(dim=0)  # [batch, d_model]
            
            for chunk_idx, chunk in enumerate(chunks):
                swarm_idx = chunk_assignments[chunk_idx]
                tiny_model = self.swarm[swarm_idx]
                
                # Use global context
                logits, communication = tiny_model(
                    chunk,
                    context=global_context,
                    num_iterations=1,  # One more iteration with global context
                )
                
                all_outputs_refined.append(logits)
                all_communications_refined.append(communication)
            
            # Use refined outputs
            all_outputs = all_outputs_refined
            all_communications = all_communications_refined
        
        # Aggregate outputs
        # Stack all chunk outputs
        chunk_outputs = torch.stack(all_outputs, dim=1)  # [batch, num_chunks, chunk_size, vocab_size]
        
        # Reshape and aggregate
        # For now, simple concatenation (could use attention pooling)
        aggregated = self._aggregate_chunks(chunk_outputs, seq_len)
        
        # Compute swarm metrics
        communications_tensor = torch.stack(all_communications, dim=1)  # [batch, num_chunks, d_model]
        swarm_coherence = self._compute_swarm_coherence(communications_tensor)
        
        metrics = {
            'num_chunks': num_chunks,
            'swarm_coherence': swarm_coherence,
            'communications': communications_tensor,
        }
        
        return aggregated, metrics
    
    def _aggregate_chunks(
        self,
        chunk_outputs: torch.Tensor,
        target_seq_len: int,
    ) -> torch.Tensor:
        """
        Aggregate chunk outputs into full sequence.
        
        Args:
            chunk_outputs: [batch, num_chunks, chunk_size, vocab_size]
            target_seq_len: Target sequence length
        
        Returns:
            [batch, target_seq_len, vocab_size]
        """
        batch_size, num_chunks, chunk_size, vocab_size = chunk_outputs.shape
        
        # Reshape: [batch, num_chunks * chunk_size, vocab_size]
        flat_outputs = chunk_outputs.view(batch_size, -1, vocab_size)
        
        # Take first target_seq_len tokens
        aggregated = flat_outputs[:, :target_seq_len, :]
        
        # If we have overlap, average overlapping regions
        # For simplicity, just take first target_seq_len
        if aggregated.size(1) < target_seq_len:
            # Pad if needed
            padding = torch.zeros(
                batch_size,
                target_seq_len - aggregated.size(1),
                vocab_size,
                device=aggregated.device,
            )
            aggregated = torch.cat([aggregated, padding], dim=1)
        
        return aggregated
    
    def _compute_swarm_coherence(self, communications: torch.Tensor) -> torch.Tensor:
        """
        Compute coherence across swarm members.
        
        Args:
            communications: [batch, num_chunks, d_model]
        
        Returns:
            coherence: [batch] coherence score
        """
        if communications.size(1) < 2:
            return torch.tensor(1.0, device=communications.device).expand(communications.size(0))
        
        # Compute pairwise similarity
        comms_normalized = torch.nn.functional.normalize(communications, p=2, dim=-1)
        similarity_matrix = torch.bmm(comms_normalized, comms_normalized.transpose(1, 2))
        
        # Coherence = mean off-diagonal similarity
        batch_size = similarity_matrix.size(0)
        mask = ~torch.eye(similarity_matrix.size(1), dtype=torch.bool, device=similarity_matrix.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        coherence = similarity_matrix[mask].view(batch_size, -1).mean(dim=1)
        
        return coherence


class SwarmAugmentedResonanceTransformer(nn.Module):
    """
    Resonance Transformer augmented with recursive swarm.
    
    Combines:
    - Main Resonance Transformer (large, processes full sequence)
    - Swarm of tiny recursive models (process chunks in parallel)
    - Hierarchical information flow
    """
    
    def __init__(
        self,
        vocab_size: int,
        main_d_model: int = 512,
        main_n_layers: int = 6,
        main_n_heads: int = 8,
        main_n_sim_steps: int = 15,
        swarm_config: Optional[SwarmConfig] = None,
        swarm_weight: float = 0.3,  # Weight for swarm output vs main model
    ):
        super().__init__()
        
        # Main Resonance Transformer
        import importlib.util
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        rt_file = project_root / "resonance_transformer.py"
        spec = importlib.util.spec_from_file_location("resonance_transformer", rt_file)
        rt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt_module)
        
        self.main_model = rt_module.ResonanceTransformer(
            vocab_size=vocab_size,
            d_model=main_d_model,
            n_layers=main_n_layers,
            n_heads=main_n_heads,
            n_sim_steps=main_n_sim_steps,
        )
        
        # Swarm coordinator
        if swarm_config is None:
            swarm_config = SwarmConfig()
        self.swarm_config = swarm_config
        self.swarm_weight = swarm_weight
        
        self.swarm_coordinator = SwarmCoordinator(swarm_config, vocab_size)
        
        # Fusion layer (combines main and swarm outputs)
        self.fusion = nn.Sequential(
            nn.Linear(main_d_model + swarm_config.swarm_d_model, main_d_model),
            nn.GELU(),
            nn.Linear(main_d_model, vocab_size),
        )
        
        # Or simpler: weighted combination
        self.use_weighted_fusion = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        use_swarm: bool = True,
        swarm_iterations: int = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through main model and swarm.
        
        Args:
            input_ids: [batch, seq_len] input tokens
            use_swarm: Whether to use swarm augmentation
            swarm_iterations: Recursive iterations for swarm
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            metrics: Dictionary with main and swarm metrics
        """
        # Main model forward
        main_logits = self.main_model(input_ids)  # [batch, seq_len, vocab_size]
        
        metrics = {
            'main_logits': main_logits,
        }
        
        if not use_swarm:
            return main_logits, metrics
        
        # Swarm forward
        swarm_logits, swarm_metrics = self.swarm_coordinator(
            input_ids,
            num_iterations=swarm_iterations,
        )
        
        metrics.update({
            'swarm_logits': swarm_logits,
            'swarm_coherence': swarm_metrics['swarm_coherence'],
            'num_chunks': swarm_metrics['num_chunks'],
        })
        
        # Fuse outputs
        if self.use_weighted_fusion:
            # Simple weighted combination
            fused_logits = (
                (1.0 - self.swarm_weight) * main_logits +
                self.swarm_weight * swarm_logits
            )
        else:
            # Learnable fusion (requires matching dimensions)
            # Project both to same space and fuse
            main_proj = main_logits  # Already vocab_size
            swarm_proj = swarm_logits  # Already vocab_size
            fused_logits = (
                (1.0 - self.swarm_weight) * main_proj +
                self.swarm_weight * swarm_proj
            )
        
        return fused_logits, metrics


def create_swarm_augmented_model(
    vocab_size: int,
    main_d_model: int = 512,
    main_n_layers: int = 6,
    main_n_heads: int = 8,
    num_swarm_models: int = 8,
    swarm_d_model: int = 64,
    swarm_weight: float = 0.3,
) -> SwarmAugmentedResonanceTransformer:
    """
    Create a swarm-augmented Resonance Transformer.
    
    Args:
        vocab_size: Vocabulary size
        main_d_model: Main model dimension
        main_n_layers: Main model layers
        main_n_heads: Main model heads
        num_swarm_models: Number of tiny models in swarm
        swarm_d_model: Dimension for tiny models
        swarm_weight: Weight for swarm output (0-1)
    
    Returns:
        SwarmAugmentedResonanceTransformer
    """
    swarm_config = SwarmConfig(
        num_swarm_models=num_swarm_models,
        swarm_d_model=swarm_d_model,
        swarm_n_layers=2,
        swarm_n_heads=4,
        chunk_size=16,
        overlap=4,
        num_recursive_iterations=3,
    )
    
    model = SwarmAugmentedResonanceTransformer(
        vocab_size=vocab_size,
        main_d_model=main_d_model,
        main_n_layers=main_n_layers,
        main_n_heads=main_n_heads,
        swarm_config=swarm_config,
        swarm_weight=swarm_weight,
    )
    
    return model

