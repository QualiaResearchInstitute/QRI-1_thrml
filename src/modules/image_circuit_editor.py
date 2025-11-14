"""
Image Editing via Consciousness Circuit Frequency Bands

Maps image features to consciousness circuit channels and allows editing
by manipulating frequency bands in the circuit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from modules.consciousness_circuit import (
    compute_information_flow,
    compute_circuit_throughput,
    ConsciousnessCircuitLayer,
)
from modules.frequency_domain_analysis import (
    compute_fft_spectrum,
    compute_spectral_coherence,
    FrequencyDomainAnalyzer,
)


class ImageCircuitMapper:
    """
    Maps image features to consciousness circuit channels.
    
    Each image patch/feature becomes an oscillator in the circuit.
    Frequency bands correspond to different image characteristics:
    - Low frequencies: Global structure, colors
    - Mid frequencies: Textures, patterns
    - High frequencies: Fine details, edges
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        patch_size: int = 16,
        n_frequency_bands: int = 4,
    ):
        self.patch_size = patch_size
        self.n_frequency_bands = n_frequency_bands
        
        if TRANSFORMERS_AVAILABLE:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            # Load model to extract features
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
        else:
            self.processor = None
            self.model = None
    
    def extract_image_features(
        self,
        image: Image.Image,
        return_patches: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract image features and map to circuit channels.
        
        Args:
            image: PIL Image
            return_patches: Whether to return patch information
        
        Returns:
            features: [batch, n_patches, d_model] feature tensor
            patches: Optional patch information
        """
        if self.model is None:
            # Fallback: use simple patch extraction
            return self._extract_patches_simple(image, return_patches)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Extract features from vision model
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get patch embeddings (before attention)
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state  # [batch, n_patches+1, d_model]
                # Remove CLS token
                features = features[:, 1:, :]
            elif hasattr(outputs, 'hidden_states'):
                # Use intermediate layer
                features = outputs.hidden_states[-2][:, 1:, :]
            else:
                # Fallback
                features = outputs.pooler_output.unsqueeze(1)
        
        patches = None
        if return_patches:
            # Extract patch positions
            h, w = image.size[1], image.size[0]
            n_patches_h = h // self.patch_size
            n_patches_w = w // self.patch_size
            patches = {
                'n_patches_h': n_patches_h,
                'n_patches_w': n_patches_w,
                'patch_size': self.patch_size,
            }
        
        return features, patches
    
    def _extract_patches_simple(
        self,
        image: Image.Image,
        return_patches: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Simple patch extraction without vision model."""
        img_array = np.array(image.convert('RGB'))
        h, w = img_array.shape[:2]
        
        n_patches_h = h // self.patch_size
        n_patches_w = w // self.patch_size
        
        patches = []
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = img_array[
                    i*self.patch_size:(i+1)*self.patch_size,
                    j*self.patch_size:(j+1)*self.patch_size
                ]
                # Flatten and normalize
                patch_flat = patch.flatten().astype(np.float32) / 255.0
                patches.append(patch_flat)
        
        features = torch.tensor(np.array(patches)).unsqueeze(0)  # [1, n_patches, patch_size^2*3]
        
        patches_info = None
        if return_patches:
            patches_info = {
                'n_patches_h': n_patches_h,
                'n_patches_w': n_patches_w,
                'patch_size': self.patch_size,
            }
        
        return features, patches_info
    
    def features_to_circuit_state(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert image features to circuit state (phases, amplitudes, coupling).
        
        Args:
            features: [batch, n_patches, d_model] feature tensor
        
        Returns:
            phases: [batch, n_patches] oscillator phases
            amplitudes: [batch, n_patches] oscillator amplitudes
            coupling_matrix: [batch, n_patches, n_patches] coupling strengths
        """
        batch_size, n_patches, d_model = features.shape
        
        # Phases: from feature magnitude/phase decomposition
        # Use PCA-like projection to get phase
        feature_norm = torch.norm(features, dim=-1)  # [batch, n_patches]
        feature_mean = features.mean(dim=-1, keepdim=True)  # [batch, n_patches, 1]
        feature_centered = features - feature_mean
        
        # Phase from principal component
        if d_model > 1:
            # Use first two principal components for phase
            U, S, V = torch.svd(feature_centered[0])  # [n_patches, d_model]
            phase_components = U[:, :2]  # [n_patches, 2]
            phases = torch.atan2(phase_components[:, 1], phase_components[:, 0])  # [n_patches]
            phases = phases.unsqueeze(0).expand(batch_size, -1)  # [batch, n_patches]
        else:
            phases = torch.atan2(feature_centered[0, :, 0], feature_norm[0]).unsqueeze(0)
        
        # Amplitudes: from feature magnitude
        amplitudes = feature_norm / (feature_norm.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        # Coupling matrix: from feature similarity
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)
        # Cosine similarity = coupling strength
        coupling_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))  # [batch, n_patches, n_patches]
        
        # Add spatial coupling (neighbors are more coupled)
        # This creates a grid-like coupling structure
        if hasattr(self, '_spatial_coupling'):
            coupling_matrix = coupling_matrix + self._spatial_coupling.to(coupling_matrix.device)
        
        # Make symmetric and ensure positive
        coupling_matrix = (coupling_matrix + coupling_matrix.transpose(-2, -1)) / 2
        coupling_matrix = torch.clamp(coupling_matrix, min=0.0)
        
        return phases, amplitudes, coupling_matrix
    
    def compute_frequency_bands(
        self,
        features: torch.Tensor,
        n_bands: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute frequency bands from image features.
        
        Args:
            features: [batch, n_patches, d_model] feature tensor
            n_bands: Number of frequency bands (default: self.n_frequency_bands)
        
        Returns:
            Dictionary with frequency band information
        """
        n_bands = n_bands or self.n_frequency_bands
        
        # Convert features to frequency domain
        # Treat each patch as a time sample, compute FFT across patches
        features_fft = torch.fft.rfft(features, dim=1)  # [batch, n_freq, d_model]
        n_freq = features_fft.shape[1]
        
        # Compute power spectrum
        power = torch.abs(features_fft) ** 2  # [batch, n_freq, d_model]
        power_total = power.sum(dim=-1)  # [batch, n_freq]
        
        # Divide into frequency bands
        band_size = n_freq // n_bands
        bands = {}
        
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < n_bands - 1 else n_freq
            
            band_power = power_total[:, start_idx:end_idx].sum(dim=1)  # [batch]
            band_features = features_fft[:, start_idx:end_idx, :]  # [batch, band_size, d_model]
            
            bands[f'band_{i}'] = {
                'power': band_power,
                'features': band_features,
                'indices': (start_idx, end_idx),
            }
        
        return bands


class FrequencyBandEditor:
    """
    Edit images by manipulating frequency bands in the consciousness circuit.
    """
    
    def __init__(
        self,
        mapper: ImageCircuitMapper,
        n_frequency_bands: int = 4,
    ):
        self.mapper = mapper
        self.n_frequency_bands = n_frequency_bands
        self.circuit_layer = ConsciousnessCircuitLayer(
            coherence_threshold=0.7,
            track_throughput=True,
        )
        self.frequency_analyzer = FrequencyDomainAnalyzer()
    
    def edit_image(
        self,
        image: Image.Image,
        band_weights: Optional[Dict[int, float]] = None,
        band_filters: Optional[Dict[int, float]] = None,
        n_circuit_iterations: int = 5,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Edit image by manipulating frequency bands.
        
        Args:
            image: Input PIL Image
            band_weights: Dict mapping band_idx -> weight (amplify/attenuate band)
            band_filters: Dict mapping band_idx -> filter_strength (0-1, 1=pass, 0=block)
            n_circuit_iterations: Number of circuit processing iterations
        
        Returns:
            edited_image: Edited PIL Image
            metrics: Dictionary with editing metrics
        """
        # Extract features
        features, patches_info = self.mapper.extract_image_features(image, return_patches=True)
        
        if patches_info is None:
            raise ValueError("Could not extract patches from image")
        
        # Convert to circuit state
        phases, amplitudes, coupling_matrix = self.mapper.features_to_circuit_state(features)
        
        # Compute frequency bands
        frequency_bands = self.mapper.compute_frequency_bands(features, n_bands=self.n_frequency_bands)
        
        # Apply band editing
        if band_weights or band_filters:
            features_edited = self._apply_band_editing(
                features,
                frequency_bands,
                band_weights,
                band_filters,
            )
        else:
            features_edited = features
        
        # Process through circuit
        phases_edited = phases.clone()
        for _ in range(n_circuit_iterations):
            phases_edited, circuit_metrics = self.circuit_layer(
                phases_edited,
                coupling_matrix,
                amplitudes,
            )
        
        # Reconstruct image from edited features
        edited_image = self._reconstruct_image(
            features_edited,
            phases_edited,
            patches_info,
            image.size,
        )
        
        # Compute metrics
        metrics = {
            'circuit_metrics': circuit_metrics,
            'frequency_bands': {
                k: {'power': float(v['power'].mean().item())}
                for k, v in frequency_bands.items()
            },
            'throughput': float(circuit_metrics.get('throughput', torch.tensor(0.0)).mean().item())
            if isinstance(circuit_metrics.get('throughput'), torch.Tensor) else 0.0,
        }
        
        return edited_image, metrics
    
    def _apply_band_editing(
        self,
        features: torch.Tensor,
        frequency_bands: Dict[str, Any],
        band_weights: Optional[Dict[int, float]],
        band_filters: Optional[Dict[int, float]],
    ) -> torch.Tensor:
        """Apply frequency band editing to features."""
        # Convert to frequency domain
        features_fft = torch.fft.rfft(features, dim=1)  # [batch, n_freq, d_model]
        
        # Apply band weights/filters
        for band_idx in range(self.n_frequency_bands):
            band_key = f'band_{band_idx}'
            if band_key not in frequency_bands:
                continue
            
            start_idx, end_idx = frequency_bands[band_key]['indices']
            
            # Apply weight
            if band_weights and band_idx in band_weights:
                weight = band_weights[band_idx]
                features_fft[:, start_idx:end_idx, :] *= weight
            
            # Apply filter
            if band_filters and band_idx in band_filters:
                filter_strength = band_filters[band_idx]
                features_fft[:, start_idx:end_idx, :] *= filter_strength
        
        # Convert back to spatial domain
        features_edited = torch.fft.irfft(features_fft, n=features.shape[1], dim=1)
        
        return features_edited
    
    def _reconstruct_image(
        self,
        features: torch.Tensor,
        phases: torch.Tensor,
        patches_info: Dict,
        original_size: Tuple[int, int],
    ) -> Image.Image:
        """Reconstruct image from edited features."""
        # For now, use simple reconstruction
        # In practice, you'd use a decoder or feature inversion
        
        batch_size, n_patches, d_model = features.shape
        n_patches_h = patches_info['n_patches_h']
        n_patches_w = patches_info['n_patches_w']
        patch_size = patches_info['patch_size']
        
        # Simple reconstruction: use feature magnitude as intensity
        # This is a placeholder - real reconstruction would use a decoder
        feature_magnitude = torch.norm(features, dim=-1)  # [batch, n_patches]
        feature_magnitude = feature_magnitude[0].cpu().numpy()  # [n_patches]
        
        # Reshape to image grid
        magnitude_grid = feature_magnitude.reshape(n_patches_h, n_patches_w)
        
        # Normalize
        magnitude_grid = (magnitude_grid - magnitude_grid.min()) / (magnitude_grid.max() - magnitude_grid.min() + 1e-8)
        magnitude_grid = (magnitude_grid * 255).astype(np.uint8)
        
        # Upsample to original size
        h, w = original_size[1], original_size[0]
        try:
            from scipy.ndimage import zoom
            scale_h = h / (n_patches_h * patch_size)
            scale_w = w / (n_patches_w * patch_size)
            magnitude_upsampled = zoom(magnitude_grid, (scale_h, scale_w), order=1)
        except ImportError:
            # Fallback: use torch interpolation
            magnitude_tensor = torch.from_numpy(magnitude_grid).float().unsqueeze(0).unsqueeze(0)
            magnitude_upsampled_t = F.interpolate(
                magnitude_tensor,
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            )
            magnitude_upsampled = magnitude_upsampled_t[0, 0].cpu().numpy()
        
        # Convert to PIL Image (grayscale for now)
        img_array = np.clip(magnitude_upsampled, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L').convert('RGB')
        
        return img


def create_band_presets() -> Dict[str, Dict[int, float]]:
    """Create preset frequency band editing configurations."""
    return {
        'enhance_low': {0: 1.5, 1: 1.0, 2: 0.8, 3: 0.5},  # Enhance global structure
        'enhance_mid': {0: 0.8, 1: 1.5, 2: 1.5, 3: 0.8},  # Enhance textures
        'enhance_high': {0: 0.5, 1: 0.8, 2: 1.0, 3: 1.5},  # Enhance details
        'smooth': {0: 1.0, 1: 0.7, 2: 0.3, 3: 0.1},  # Smooth (reduce high freq)
        'sharpen': {0: 0.8, 1: 1.0, 2: 1.3, 3: 1.5},  # Sharpen (enhance high freq)
        'low_pass': {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0},  # Low-pass filter
        'high_pass': {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0},  # High-pass filter
    }

