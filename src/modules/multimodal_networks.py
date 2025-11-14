"""
Multi-Modal Networks: Resonance, Text, Video, and Audio

Networks for processing different modalities with resonance attention:
1. Resonance Network (existing - Kuramoto oscillators)
2. Text Network (existing - token-based)
3. Video Network (NEW - frame sequences)
4. Audio Network (NEW - waveform/spectrogram)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
import math


class VideoResonanceNetwork(nn.Module):
    """
    Video processing network using resonance attention.
    
    Processes video frames with:
    - Temporal resonance (across frames)
    - Spatial resonance (within frames)
    - Frame-level Kuramoto dynamics
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        n_sim_steps: int = 15,
        patch_size: int = 16,
        image_size: int = 224,
        max_frames: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.image_size = image_size
        self.max_frames = max_frames
        
        # Import Resonance Transformer components
        import importlib.util
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        rt_file = project_root / "resonance_transformer.py"
        spec = importlib.util.spec_from_file_location("resonance_transformer", rt_file)
        rt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt_module)
        
        ResonanceTransformerBlock = rt_module.ResonanceTransformerBlock
        
        # Frame embedding: convert frames to patches
        num_patches_per_frame = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3, d_model, kernel_size=patch_size, stride=patch_size
        )
        
        # Frame position embeddings
        self.frame_pos_embedding = nn.Embedding(max_frames, d_model)
        
        # Patch position embeddings (within frame)
        self.patch_pos_embedding = nn.Embedding(num_patches_per_frame, d_model)
        
        # Temporal resonance layers (across frames)
        self.temporal_layers = nn.ModuleList([
            ResonanceTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=4 * d_model,
                n_sim_steps=n_sim_steps,
                dropout=0.1,
            )
            for _ in range(n_layers)
        ])
        
        # Spatial resonance layers (within frames)
        self.spatial_layers = nn.ModuleList([
            ResonanceTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=4 * d_model,
                n_sim_steps=n_sim_steps,
                dropout=0.1,
            )
            for _ in range(n_layers // 2)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        video: torch.Tensor,  # [batch, frames, channels, height, width]
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process video with resonance attention.
        
        Args:
            video: [batch, frames, 3, H, W] video tensor
            return_features: Whether to return intermediate features
        
        Returns:
            features: [batch, frames, num_patches, d_model] or [batch, frames, d_model]
            metrics: Dictionary with resonance metrics
        """
        batch_size, num_frames, channels, height, width = video.shape
        
        # Process each frame
        frame_features = []
        for frame_idx in range(num_frames):
            frame = video[:, frame_idx, :, :, :]  # [batch, 3, H, W]
            
            # Patch embedding
            patches = self.patch_embedding(frame)  # [batch, d_model, H', W']
            patches = patches.flatten(2).transpose(1, 2)  # [batch, num_patches, d_model]
            
            # Add patch position embeddings
            patch_positions = torch.arange(patches.size(1), device=patches.device)
            patch_pos_embeds = self.patch_pos_embedding(patch_positions)
            patches = patches + patch_pos_embeds.unsqueeze(0)
            
            # Spatial resonance (within frame)
            spatial_out = patches
            for layer in self.spatial_layers:
                spatial_out = layer(spatial_out)
            
            # Pool to frame-level representation
            frame_repr = spatial_out.mean(dim=1)  # [batch, d_model]
            frame_features.append(frame_repr)
        
        # Stack frames
        frame_features = torch.stack(frame_features, dim=1)  # [batch, num_frames, d_model]
        
        # Add frame position embeddings
        frame_positions = torch.arange(num_frames, device=frame_features.device)
        frame_pos_embeds = self.frame_pos_embedding(frame_positions)
        frame_features = frame_features + frame_pos_embeds.unsqueeze(0)
        
        # Temporal resonance (across frames)
        temporal_out = frame_features
        for layer in self.temporal_layers:
            temporal_out = layer(temporal_out)
        
        # Output
        output = self.norm(temporal_out)
        output = self.output_proj(output)
        
        metrics = {
            'num_frames': num_frames,
            'frame_features': frame_features,
        }
        
        if return_features:
            return output, metrics
        
        # Pool to sequence-level
        sequence_repr = output.mean(dim=1)  # [batch, d_model]
        
        return sequence_repr, metrics


class AudioResonanceNetwork(nn.Module):
    """
    Audio processing network using resonance attention.
    
    Processes audio with:
    - Frequency-domain resonance (spectral)
    - Time-domain resonance (temporal)
    - Mel-spectrogram or raw waveform
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        n_sim_steps: int = 15,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 160,
        use_spectrogram: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.use_spectrogram = use_spectrogram
        
        # Import Resonance Transformer components
        import importlib.util
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        rt_file = project_root / "resonance_transformer.py"
        spec = importlib.util.spec_from_file_location("resonance_transformer", rt_file)
        rt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt_module)
        
        ResonanceTransformerBlock = rt_module.ResonanceTransformerBlock
        
        # Shared encoder (works for both waveform and spectrogram)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, d_model // 4, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model // 4, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=7, stride=2, padding=3),
        )
        
        # Spectrogram-specific projection (if needed)
        if use_spectrogram:
            self.spectrogram_proj = nn.Linear(n_mels, d_model)
        else:
            self.spectrogram_proj = None
        
        # Frequency position embeddings (for spectrogram)
        self.freq_pos_embedding = nn.Embedding(n_mels, d_model)
        
        # Time position embeddings (dynamically sized)
        self.max_time_steps = 10000  # Large enough for most audio
        self.time_pos_embedding = nn.Embedding(self.max_time_steps, d_model)
        
        # Frequency-domain resonance layers
        self.freq_layers = nn.ModuleList([
            ResonanceTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=4 * d_model,
                n_sim_steps=n_sim_steps,
                dropout=0.1,
            )
            for _ in range(n_layers // 2)
        ])
        
        # Temporal resonance layers
        self.temporal_layers = nn.ModuleList([
            ResonanceTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=4 * d_model,
                n_sim_steps=n_sim_steps,
                dropout=0.1,
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        audio: torch.Tensor,  # [batch, samples] or [batch, n_mels, time] for spectrogram
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process audio with resonance attention.
        
        Args:
            audio: [batch, samples] waveform or [batch, n_mels, time] spectrogram
            return_features: Whether to return intermediate features
        
        Returns:
            features: [batch, time_steps, d_model] or [batch, d_model]
            metrics: Dictionary with resonance metrics
        """
        # Handle input format
        if audio.dim() == 2:
            # Waveform: [batch, samples] -> [batch, 1, samples]
            audio = audio.unsqueeze(1)
            audio_features = self.audio_encoder(audio)  # [batch, d_model, time']
        elif audio.dim() == 3:
            if audio.size(1) == self.n_mels and self.use_spectrogram:
                # Spectrogram: [batch, n_mels, time] -> project to d_model
                audio_features = audio.transpose(1, 2)  # [batch, time, n_mels]
                if self.spectrogram_proj is not None:
                    audio_features = self.spectrogram_proj(audio_features)  # [batch, time, d_model]
                else:
                    # Fallback: use frequency embeddings
                    audio_features = nn.functional.linear(
                        audio_features,
                        self.freq_pos_embedding.weight.T
                    )  # [batch, time, d_model]
                # Transpose to [batch, d_model, time] for consistency
                audio_features = audio_features.transpose(1, 2)
            elif audio.size(1) == 1:
                # Single channel waveform: [batch, 1, samples]
                audio_features = self.audio_encoder(audio)  # [batch, d_model, time']
            else:
                # Multi-channel, use first channel
                audio_features = self.audio_encoder(audio[:, 0:1, :])  # [batch, d_model, time']
        else:
            raise ValueError(f"Unexpected audio input shape: {audio.shape}")
        
        # Transpose: [batch, time, d_model]
        audio_features = audio_features.transpose(1, 2)
        
        # Add position embeddings
        time_steps = audio_features.size(1)
        time_positions = torch.arange(time_steps, device=audio_features.device)
        # Clamp to valid range
        time_positions = torch.clamp(time_positions, 0, self.max_time_steps - 1)
        time_pos_embeds = self.time_pos_embedding(time_positions)
        audio_features = audio_features + time_pos_embeds.unsqueeze(0)
        
        # Frequency-domain resonance (if spectrogram)
        if self.use_spectrogram and audio.dim() == 3 and audio.size(1) == self.n_mels:
            # Process frequency dimension
            freq_out = audio_features
            for layer in self.freq_layers:
                freq_out = layer(freq_out)
            audio_features = freq_out
        
        # Temporal resonance
        temporal_out = audio_features
        for layer in self.temporal_layers:
            temporal_out = layer(temporal_out)
        
        # Output
        output = self.norm(temporal_out)
        output = self.output_proj(output)
        
        metrics = {
            'time_steps': time_steps,
            'audio_features': audio_features,
        }
        
        if return_features:
            return output, metrics
        
        # Pool to sequence-level
        sequence_repr = output.mean(dim=1)  # [batch, d_model]
        
        return sequence_repr, metrics


class TextResonanceNetwork(nn.Module):
    """
    Text processing network using resonance attention.
    
    Wrapper around existing Resonance Transformer for text.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        n_sim_steps: int = 15,
        max_seq_len: int = 512,
    ):
        super().__init__()
        
        # Import Resonance Transformer
        import importlib.util
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        rt_file = project_root / "resonance_transformer.py"
        spec = importlib.util.spec_from_file_location("resonance_transformer", rt_file)
        rt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt_module)
        
        ResonanceTransformer = rt_module.ResonanceTransformer
        
        self.model = ResonanceTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_sim_steps=n_sim_steps,
            max_seq_len=max_seq_len,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process text with resonance attention.
        
        Args:
            input_ids: [batch, seq_len] token IDs
            return_features: Whether to return intermediate features
        
        Returns:
            logits: [batch, seq_len, vocab_size] or features
            metrics: Dictionary with resonance metrics
        """
        logits = self.model(input_ids)
        
        metrics = {
            'seq_len': input_ids.size(1),
        }
        
        if return_features:
            # Return last hidden state (simplified - would need to modify model)
            return logits, metrics
        
        # Pool to sequence-level
        sequence_repr = logits.mean(dim=1)  # [batch, vocab_size]
        # Project to d_model (simplified)
        if not hasattr(self, 'proj'):
            self.proj = nn.Linear(logits.size(-1), self.model.d_model).to(logits.device)
        sequence_repr = self.proj(sequence_repr)
        
        return sequence_repr, metrics


class MultiModalFusionNetwork(nn.Module):
    """
    Fuses outputs from resonance, text, video, and audio networks.
    
    Uses cross-modal attention and learned fusion.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        num_modalities: int = 4,  # resonance, text, video, audio
    ):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        
        # Modality embeddings
        self.modality_embeddings = nn.Embedding(num_modalities, d_model)
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        resonance_features: torch.Tensor,  # [batch, d_model]
        text_features: torch.Tensor,  # [batch, d_model]
        video_features: torch.Tensor,  # [batch, d_model]
        audio_features: torch.Tensor,  # [batch, d_model]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fuse multi-modal features.
        
        Returns:
            fused: [batch, d_model]
            metrics: Dictionary with fusion metrics
        """
        batch_size = resonance_features.size(0)
        
        # Stack modalities: [batch, num_modalities, d_model]
        modalities = torch.stack([
            resonance_features,
            text_features,
            video_features,
            audio_features,
        ], dim=1)
        
        # Add modality embeddings
        modality_ids = torch.arange(self.num_modalities, device=modalities.device)
        modality_embeds = self.modality_embeddings(modality_ids)
        modalities = modalities + modality_embeds.unsqueeze(0)
        
        # Cross-modal attention
        fused, attention_weights = self.cross_modal_attention(
            modalities, modalities, modalities
        )
        
        # Residual
        fused = self.norm(fused + modalities)
        
        # Flatten and fuse
        fused_flat = fused.view(batch_size, -1)  # [batch, num_modalities * d_model]
        fused = self.fusion(fused_flat)  # [batch, d_model]
        
        # Output
        output = self.output_proj(fused)
        
        metrics = {
            'attention_weights': attention_weights,
            'modality_features': fused,
        }
        
        return output, metrics

