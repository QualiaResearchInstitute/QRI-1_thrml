"""
Enhanced Multi-Modal Networks with Advanced Features

Integrates spectral filters, gating, complex representations, and multi-scale biases
into the multi-modal networks (text, video, audio).
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
from modules.multimodal_networks import (
    VideoResonanceNetwork,
    AudioResonanceNetwork,
    TextResonanceNetwork,
)
from modules.advanced_multimodal_features import EnhancedMultimodalFeatures


class EnhancedVideoResonanceNetwork(VideoResonanceNetwork):
    """
    Enhanced video network with advanced features.
    """
    
    def __init__(
        self,
        *args,
        use_spectral: bool = True,
        use_gating: bool = True,
        use_complex: bool = True,
        use_multiscale: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Add enhanced features
        self.enhanced_features = EnhancedMultimodalFeatures(
            d_model=self.d_model,
            use_spectral=use_spectral,
            use_gating=use_gating,
            use_complex=use_complex,
            use_multiscale=use_multiscale,
        )
    
    def forward(
        self,
        video: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced forward with advanced features."""
        # Get base features
        features, base_metrics = super().forward(video, return_features=True)
        
        # Apply enhanced features
        enhanced_features, enhanced_metrics = self.enhanced_features(features)
        
        # Combine metrics
        all_metrics = {**base_metrics, **enhanced_metrics}
        
        if return_features:
            return enhanced_features, all_metrics
        
        # Pool to sequence-level
        sequence_repr = enhanced_features.mean(dim=1)  # [batch, d_model]
        
        return sequence_repr, all_metrics


class EnhancedAudioResonanceNetwork(AudioResonanceNetwork):
    """
    Enhanced audio network with advanced features.
    """
    
    def __init__(
        self,
        *args,
        use_spectral: bool = True,
        use_gating: bool = True,
        use_complex: bool = True,
        use_multiscale: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Add enhanced features
        self.enhanced_features = EnhancedMultimodalFeatures(
            d_model=self.d_model,
            use_spectral=use_spectral,
            use_gating=use_gating,
            use_complex=use_complex,
            use_multiscale=use_multiscale,
        )
    
    def forward(
        self,
        audio: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced forward with advanced features."""
        # Get base features
        features, base_metrics = super().forward(audio, return_features=True)
        
        # Apply enhanced features
        enhanced_features, enhanced_metrics = self.enhanced_features(features)
        
        # Combine metrics
        all_metrics = {**base_metrics, **enhanced_metrics}
        
        if return_features:
            return enhanced_features, all_metrics
        
        # Pool to sequence-level
        sequence_repr = enhanced_features.mean(dim=1)  # [batch, d_model]
        
        return sequence_repr, all_metrics


class EnhancedTextResonanceNetwork(TextResonanceNetwork):
    """
    Enhanced text network with advanced features.
    """
    
    def __init__(
        self,
        *args,
        use_spectral: bool = True,
        use_gating: bool = True,
        use_complex: bool = True,
        use_multiscale: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Add enhanced features
        self.enhanced_features = EnhancedMultimodalFeatures(
            d_model=self.model.d_model,
            use_spectral=use_spectral,
            use_gating=use_gating,
            use_complex=use_complex,
            use_multiscale=use_multiscale,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced forward with advanced features."""
        # Get base logits
        logits = self.model(input_ids)
        
        # Extract features (simplified - would need to modify model to get activations)
        # For now, use logits as proxy
        if not hasattr(self, 'logit_to_feature'):
            self.logit_to_feature = nn.Linear(logits.size(-1), self.model.d_model).to(logits.device)
        
        features = self.logit_to_feature(logits)  # [batch, seq_len, d_model]
        
        # Apply enhanced features
        enhanced_features, enhanced_metrics = self.enhanced_features(features)
        
        # Combine metrics
        all_metrics = {
            'seq_len': input_ids.size(1),
            **enhanced_metrics,
        }
        
        if return_features:
            return enhanced_features, all_metrics
        
        # Pool to sequence-level
        sequence_repr = enhanced_features.mean(dim=1)  # [batch, d_model]
        
        return sequence_repr, all_metrics

