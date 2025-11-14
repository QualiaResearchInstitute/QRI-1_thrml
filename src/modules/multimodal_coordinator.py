"""
Multi-Modal Coordinator: Resonance, Text, Video, Audio

Coordinates all four modalities with:
- Parallel processing
- Cross-modal attention
- Unified fusion
- Swarm integration
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
from modules.multimodal_networks import (
    VideoResonanceNetwork,
    AudioResonanceNetwork,
    TextResonanceNetwork,
    MultiModalFusionNetwork,
)
from modules.recursive_swarm import SwarmCoordinator, SwarmConfig


class MultiModalCoordinator(nn.Module):
    """
    Coordinates resonance, text, video, and audio processing.
    
    Integrates with swarm system for hierarchical reasoning.
    """
    
    def __init__(
        self,
        # Text
        vocab_size: int,
        text_d_model: int = 512,
        text_n_layers: int = 6,
        text_n_heads: int = 8,
        # Video
        video_d_model: int = 512,
        video_n_layers: int = 6,
        video_n_heads: int = 8,
        video_patch_size: int = 16,
        video_image_size: int = 224,
        # Audio
        audio_d_model: int = 512,
        audio_n_layers: int = 6,
        audio_n_heads: int = 8,
        audio_sample_rate: int = 16000,
        # Fusion
        fusion_d_model: int = 512,
        # Swarm (optional)
        use_swarm: bool = True,
        swarm_config: Optional[SwarmConfig] = None,
    ):
        super().__init__()
        
        # Common dimension
        self.d_model = fusion_d_model
        
        # Text network
        self.text_network = TextResonanceNetwork(
            vocab_size=vocab_size,
            d_model=text_d_model,
            n_layers=text_n_layers,
            n_heads=text_n_heads,
        )
        
        # Video network
        self.video_network = VideoResonanceNetwork(
            d_model=video_d_model,
            n_layers=video_n_layers,
            n_heads=video_n_heads,
            patch_size=video_patch_size,
            image_size=video_image_size,
        )
        
        # Audio network
        self.audio_network = AudioResonanceNetwork(
            d_model=audio_d_model,
            n_layers=audio_n_layers,
            n_heads=audio_n_heads,
            sample_rate=audio_sample_rate,
        )
        
        # Projection layers to common dimension
        self.text_proj = nn.Linear(text_d_model, fusion_d_model)
        self.video_proj = nn.Linear(video_d_model, fusion_d_model)
        self.audio_proj = nn.Linear(audio_d_model, fusion_d_model)
        
        # Resonance features (from main resonance transformer)
        self.resonance_proj = nn.Linear(fusion_d_model, fusion_d_model)  # Identity if same dim
        
        # Fusion network
        self.fusion_network = MultiModalFusionNetwork(
            d_model=fusion_d_model,
            n_heads=text_n_heads,
            num_modalities=4,
        )
        
        # Swarm coordinator (for text processing)
        self.use_swarm = use_swarm
        if use_swarm and swarm_config is not None:
            from modules.enhanced_swarm_coordinator import create_enhanced_swarm_coordinator
            self.swarm_coordinator = create_enhanced_swarm_coordinator(
                config=swarm_config,
                vocab_size=vocab_size,
                use_enhanced=True,
            )
        else:
            self.swarm_coordinator = None
    
    def forward(
        self,
        # Inputs
        text_input: Optional[torch.Tensor] = None,  # [batch, seq_len] token IDs
        video_input: Optional[torch.Tensor] = None,  # [batch, frames, 3, H, W]
        audio_input: Optional[torch.Tensor] = None,  # [batch, samples] or [batch, n_mels, time]
        resonance_features: Optional[torch.Tensor] = None,  # [batch, d_model]
        # Options
        return_modality_features: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process multi-modal inputs.
        
        Returns:
            fused_output: [batch, d_model]
            metrics: Dictionary with all modality metrics
        """
        metrics = {}
        
        # Process text
        text_features = None
        if text_input is not None:
            if self.use_swarm and self.swarm_coordinator is not None:
                # Use swarm for text processing
                swarm_logits, swarm_metrics = self.swarm_coordinator(text_input)
                # Get features from swarm (simplified - would need to extract activations)
                # Use embedding weights to project logits to features
                embedding_weight = self.text_network.model.token_embedding.weight  # [vocab_size, d_model]
                # Project: logits @ embedding_weight
                text_features = torch.matmul(swarm_logits.mean(dim=1), embedding_weight)  # [batch, d_model]
            else:
                # Use standard text network
                text_features, text_metrics = self.text_network(text_input)
            metrics['text'] = swarm_metrics if (self.use_swarm and self.swarm_coordinator is not None) else text_metrics
        else:
            # Zero features if no text
            text_features = torch.zeros(
                (1, self.d_model),
                device=next(self.parameters()).device
            )
        
        # Process video
        video_features = None
        if video_input is not None:
            video_features, video_metrics = self.video_network(video_input)
            metrics['video'] = video_metrics
        else:
            video_features = torch.zeros(
                (1, self.d_model),
                device=next(self.parameters()).device
            )
        
        # Process audio
        audio_features = None
        if audio_input is not None:
            audio_features, audio_metrics = self.audio_network(audio_input)
            metrics['audio'] = audio_metrics
        else:
            audio_features = torch.zeros(
                (1, self.d_model),
                device=next(self.parameters()).device
            )
        
        # Resonance features (from main model or default)
        if resonance_features is None:
            resonance_features = torch.zeros(
                (1, self.d_model),
                device=next(self.parameters()).device
            )
        
        # Project to common dimension
        batch_size = max(
            text_features.size(0) if text_features is not None else 1,
            video_features.size(0) if video_features is not None else 1,
            audio_features.size(0) if audio_features is not None else 1,
            resonance_features.size(0) if resonance_features is not None else 1,
        )
        
        # Ensure same batch size
        if text_features.size(0) != batch_size:
            text_features = text_features.expand(batch_size, -1)
        if video_features.size(0) != batch_size:
            video_features = video_features.expand(batch_size, -1)
        if audio_features.size(0) != batch_size:
            audio_features = audio_features.expand(batch_size, -1)
        if resonance_features.size(0) != batch_size:
            resonance_features = resonance_features.expand(batch_size, -1)
        
        # Project
        text_proj = self.text_proj(text_features)
        video_proj = self.video_proj(video_features)
        audio_proj = self.audio_proj(audio_features)
        resonance_proj = self.resonance_proj(resonance_features)
        
        # Fuse
        fused, fusion_metrics = self.fusion_network(
            resonance_proj,
            text_proj,
            video_proj,
            audio_proj,
        )
        
        metrics['fusion'] = fusion_metrics
        
        if return_modality_features:
            metrics['modality_features'] = {
                'text': text_proj,
                'video': video_proj,
                'audio': audio_proj,
                'resonance': resonance_proj,
            }
        
        return fused, metrics


def create_multimodal_coordinator(
    vocab_size: int,
    use_swarm: bool = True,
    **kwargs
) -> MultiModalCoordinator:
    """
    Factory function to create multi-modal coordinator.
    """
    swarm_config = None
    if use_swarm:
        swarm_config = SwarmConfig(
            num_swarm_models=8,
            swarm_d_model=64,
            chunk_size=16,
            overlap=4,
        )
    
    return MultiModalCoordinator(
        vocab_size=vocab_size,
        use_swarm=use_swarm,
        swarm_config=swarm_config,
        **kwargs
    )

