"""
Custom Model Optimized for Autonomous Learning

A lightweight transformer-like model designed specifically for:
- Fast inference (< 0.1s per forward pass)
- Resonance dynamics learning
- Ising-like criticality maintenance
- Minimal memory footprint
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from resonance_transformer import ResonanceAttentionHead


class MinimalResonanceLayer(nn.Module):
    """
    Minimal transformer layer with resonance attention.
    Optimized for speed and learning.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        resonance_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        d_ff = d_ff or (d_model * 2)
        
        # Resonance attention heads
        resonance_kwargs = resonance_kwargs or {}
        # Start with higher coupling for Ising critical point (R ≈ 0.6)
        initial_coupling = resonance_kwargs.get('initial_coupling', 3.0)  # Higher than default 1.0
        
        # Try to import PID for full dragon mode
        try:
            from spare_parts.controllers.pid_autopilot import PIDAutopilot, PIDConfig
            PID_AVAILABLE = True
        except ImportError:
            try:
                from spare_parts.controllers import PIDAutopilot, PIDConfig
                PID_AVAILABLE = True
            except ImportError:
                PID_AVAILABLE = False
                PIDAutopilot = None
                PIDConfig = None
        
        # Full dragon mode: enable all features
        use_full_dragon = resonance_kwargs.get('use_full_dragon', True)
        target_R = resonance_kwargs.get('target_R', 0.6)
        
        self.attention_heads = nn.ModuleList([
            ResonanceAttentionHead(
                d_model=self.head_dim,
                d_k=self.head_dim,
                d_v=self.head_dim,
                n_sim_steps=resonance_kwargs.get('n_sim_steps', 5 if use_full_dragon else 3),  # More steps for full dragon
                dt=resonance_kwargs.get('dt', 0.02 if use_full_dragon else 0.03),
                coupling_strength=initial_coupling,
                use_sakaguchi=resonance_kwargs.get('use_sakaguchi', True),
                use_stuart_landau=resonance_kwargs.get('use_stuart_landau', True),
                track_cdns=resonance_kwargs.get('track_cdns', True),
                use_coupling_kernel=resonance_kwargs.get('use_coupling_kernel', use_full_dragon),  # Enable for full dragon
                use_critical_tuning=use_full_dragon,  # Critical coupling tuner
                hybrid_readout=resonance_kwargs.get('hybrid_readout', True),
                hybrid_mix_init=resonance_kwargs.get('hybrid_mix_init', 0.3 if use_full_dragon else 0.5),  # More resonance
                use_extended_cdns=False,  # Keep disabled for speed
                use_spectral_gating=use_full_dragon,  # Enable spectral gating
                spectral_num_bands=resonance_kwargs.get('spectral_num_bands', 4),
                telemetry=resonance_kwargs.get('telemetry', False),  # Can enable for debugging
                target_R=target_R,
                lambda_criticality=resonance_kwargs.get('lambda_criticality', 0.01),
                # PID Autopilot - the ultimate criticality controller
                use_pid_autopilot=use_full_dragon and PID_AVAILABLE,
                pid_config=PIDConfig(target_R=float(target_R)) if (use_full_dragon and PID_AVAILABLE and PIDConfig is not None) else None,
                # Adaptive coupling - automatic adjustment
                use_adaptive_coupling=use_full_dragon,
                adaptation_rate=resonance_kwargs.get('adaptation_rate', 0.05),
                adaptation_signal='order_parameter',
                adaptation_target=target_R,
                # Use Heun integrator for better accuracy
                use_heun=use_full_dragon,
            )
            for _ in range(n_heads)
        ])
        
        # Initialize _K_runtime to initial_coupling for all heads
        for head in self.attention_heads:
            if hasattr(head, '_K_runtime'):
                head._K_runtime = float(initial_coupling)
        
        # Minimal feed-forward (single layer for speed)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass optimized for speed.
        """
        # Attention (resonance)
        x_norm = self.ln1(x)
        
        # Split into heads
        batch_size, seq_len, _ = x_norm.shape
        x_heads = x_norm.view(batch_size, seq_len, self.n_heads, self.head_dim)
        x_heads = x_heads.transpose(1, 2)  # [B, H, T, D_h]
        
        # Process each head
        outputs = []
        for head_idx, head in enumerate(self.attention_heads):
            head_input = x_heads[:, head_idx, :, :]  # Take this head's slice
            # Ensure dimensions match
            if head_input.shape[-1] != self.head_dim:
                head_input = head_input[:, :, :self.head_dim]
            output = head(head_input, mask=mask)
            outputs.append(output)
        
        # Concatenate heads
        x_attn = torch.stack(outputs, dim=1)  # [B, H, T, D_h]
        x_attn = x_attn.transpose(1, 2).contiguous()  # [B, T, H, D_h]
        x_attn = x_attn.view(batch_size, seq_len, self.d_model)
        
        # Residual + dropout
        x = x + self.dropout(x_attn)
        
        # Feed-forward
        x = x + self.dropout(self.ff(self.ln2(x)))
        
        return x


class AutonomousLearningModel(nn.Module):
    """
    Custom model optimized for autonomous learning.
    
    Features:
    - Minimal layers (1-2 layers)
    - Small hidden dimension (64-128)
    - Resonance attention only
    - Fast inference
    - Designed for dynamics learning
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 64,
        n_layers: int = 1,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        max_seq_len: int = 32,
        dropout: float = 0.0,
        resonance_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # Token embeddings (learnable)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings (learnable, not sinusoidal for speed)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Resonance layers
        self.layers = nn.ModuleList([
            MinimalResonanceLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                resonance_kwargs=resonance_kwargs,
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fast forward pass optimized for learning.
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        tok_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)
        x = tok_emb + pos_emb
        
        # Through layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def get_resonance_heads(self):
        """Get all resonance attention heads for metric extraction."""
        heads = []
        for layer in self.layers:
            heads.extend(layer.attention_heads)
        return heads


def create_autonomous_learner(
    vocab_size: int = 1000,
    d_model: int = 64,
    n_layers: int = 1,
    n_heads: int = 4,
    device: Optional[torch.device] = None,
    resonance_kwargs: Optional[Dict] = None,
) -> AutonomousLearningModel:
    """
    Factory function to create optimized autonomous learning model.
    
    Defaults optimized for speed:
    - Small model (64 dims, 1 layer, 4 heads)
    - Minimal resonance steps (3)
    - Fast inference (< 0.1s)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    model = AutonomousLearningModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        resonance_kwargs=resonance_kwargs,
    )
    
    model.to(device)
    model.eval()
    
    return model

