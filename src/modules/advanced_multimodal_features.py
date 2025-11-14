"""
Advanced Multi-Modal Features

Integrates:
1. Spectral Filters - Frequency-domain filtering
2. Gating Mechanisms - Learned gating for information flow
3. Complex Representations - Complex-valued processing
4. Multi-Scale Structural Biases - Hierarchical multi-scale processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import math


class SpectralFilterBank(nn.Module):
    """
    Spectral filter bank for frequency-domain filtering.
    
    Uses DCT/FFT-based filters to decompose signals into frequency bands.
    """
    
    def __init__(
        self,
        d_model: int,
        num_bands: int = 8,
        filter_type: str = "dct",  # "dct", "fft", "learned"
        learnable: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_bands = num_bands
        self.filter_type = filter_type
        
        if filter_type == "dct":
            # DCT-based filters
            self.filters = nn.ModuleList([
                nn.Conv1d(d_model, d_model, kernel_size=8, padding=4, groups=d_model)
                for _ in range(num_bands)
            ])
            # Initialize with DCT basis
            self._init_dct_filters()
        elif filter_type == "fft":
            # FFT-based (learned in frequency domain)
            self.freq_weights = nn.Parameter(torch.randn(num_bands, d_model))
        elif filter_type == "learned":
            # Fully learned filters
            self.filters = nn.ModuleList([
                nn.Conv1d(d_model, d_model, kernel_size=8, padding=4, groups=d_model)
                for _ in range(num_bands)
            ])
        
        # Band mixing
        self.band_mixer = nn.Conv1d(
            d_model * num_bands,
            d_model,
            kernel_size=1,
            groups=d_model,
        )
        
        self.learnable = learnable
        if not learnable and filter_type != "learned":
            for param in self.parameters():
                param.requires_grad = False
    
    def _init_dct_filters(self):
        """Initialize filters with DCT basis."""
        for i, filter_layer in enumerate(self.filters):
            with torch.no_grad():
                # DCT-II basis
                kernel_size = filter_layer.kernel_size[0]
                n = torch.arange(kernel_size, dtype=torch.float32)
                k = i
                basis = torch.cos(math.pi / kernel_size * (n + 0.5) * k)
                basis = basis / (torch.norm(basis) + 1e-8)
                
                # Set weights (assuming depthwise conv)
                filter_layer.weight.data = basis.view(1, 1, -1).repeat(
                    self.d_model, 1, 1
                )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply spectral filter bank.
        
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            filtered: [batch, seq_len, d_model]
            band_energies: [batch, num_bands] band energy per band
        """
        batch_size, seq_len, d_model = x.shape
        
        # Transpose for conv: [batch, d_model, seq_len]
        x_conv = x.transpose(1, 2)
        
        if self.filter_type == "fft":
            # FFT-based filtering
            x_fft = torch.fft.rfft(x_conv, dim=-1)  # [batch, d_model, freq_bins]
            freq_bins = x_fft.size(-1)
            
            # Apply band weights
            band_outputs = []
            for band_idx in range(self.num_bands):
                weights = self.freq_weights[band_idx]  # [d_model]
                # Apply weights in frequency domain
                weighted_fft = x_fft * weights.unsqueeze(-1)
                band_signal = torch.fft.irfft(weighted_fft, n=seq_len, dim=-1)
                band_outputs.append(band_signal)
            
            # Stack bands
            bands = torch.stack(band_outputs, dim=1)  # [batch, num_bands, d_model, seq_len]
            bands = bands.view(batch_size, self.num_bands * d_model, seq_len)
        else:
            # Convolution-based filtering
            band_outputs = []
            for filter_layer in self.filters:
                band_signal = filter_layer(x_conv)  # [batch, d_model, seq_len]
                band_outputs.append(band_signal)
            
            # Stack bands
            bands = torch.cat(band_outputs, dim=1)  # [batch, num_bands * d_model, seq_len]
        
        # Mix bands
        mixed = self.band_mixer(bands)  # [batch, d_model, seq_len]
        
        # Transpose back: [batch, seq_len, d_model]
        filtered = mixed.transpose(1, 2)
        
        # Compute band energies
        if self.filter_type == "fft":
            band_energies = torch.abs(bands).mean(dim=(2, 3))  # [batch, num_bands]
        else:
            band_energies = torch.stack([
                torch.abs(band).mean(dim=(1, 2))  # [batch]
                for band in band_outputs
            ], dim=1)  # [batch, num_bands]
        
        return filtered, band_energies


class AdaptiveGatingMechanism(nn.Module):
    """
    Adaptive gating mechanism for controlling information flow.
    
    Uses multiple gating strategies:
    - Input gating (what to let in)
    - Output gating (what to let out)
    - Forget gating (what to forget)
    - Modulation gating (how to modulate)
    """
    
    def __init__(
        self,
        d_model: int,
        gate_type: str = "multi",  # "simple", "multi", "attention"
        num_gates: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.gate_type = gate_type
        self.num_gates = num_gates
        
        if gate_type == "multi":
            # Multiple specialized gates
            self.input_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            self.output_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            self.forget_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            self.modulation_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
            )
        elif gate_type == "attention":
            # Attention-based gating
            self.gate_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=8,
                batch_first=True,
            )
            self.gate_proj = nn.Linear(d_model, d_model)
        else:
            # Simple gate
            self.gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply gating mechanism.
        
        Args:
            x: [batch, seq_len, d_model] input
            context: [batch, seq_len, d_model] optional context
        
        Returns:
            gated: [batch, seq_len, d_model]
            gate_metrics: Dictionary with gate activations
        """
        if self.gate_type == "multi":
            # Multi-gate mechanism
            input_gate = self.input_gate(x)
            output_gate = self.output_gate(x)
            forget_gate = self.forget_gate(x)
            modulation = self.modulation_gate(x)
            
            # Apply gates
            gated = input_gate * x  # What to let in
            gated = forget_gate * gated  # What to forget
            gated = gated + modulation  # Modulate
            gated = output_gate * gated  # What to let out
            
            gate_metrics = {
                'input_gate': input_gate.mean().item(),
                'output_gate': output_gate.mean().item(),
                'forget_gate': forget_gate.mean().item(),
                'modulation': modulation.mean().item(),
            }
        elif self.gate_type == "attention":
            # Attention-based gating
            if context is None:
                context = x
            
            gate_weights, _ = self.gate_attention(x, context, context)
            gate_proj = self.gate_proj(gate_weights)
            gated = x * torch.sigmoid(gate_proj)
            
            gate_metrics = {
                'attention_gate': gate_weights.mean().item(),
            }
        else:
            # Simple gate
            gate = self.gate(x)
            gated = gate * x
            
            gate_metrics = {
                'gate': gate.mean().item(),
            }
        
        return gated, gate_metrics


class ComplexRepresentationLayer(nn.Module):
    """
    Complex-valued representation layer.
    
    Processes signals in complex domain for richer representations.
    """
    
    def __init__(
        self,
        d_model: int,
        use_complex: bool = True,
        complex_operations: str = "polar",  # "polar", "cartesian", "both"
    ):
        super().__init__()
        self.d_model = d_model
        self.use_complex = use_complex
        self.complex_operations = complex_operations
        
        if use_complex:
            # Complex-valued projections
            self.complex_proj_real = nn.Linear(d_model, d_model)
            self.complex_proj_imag = nn.Linear(d_model, d_model)
            
            # Complex operations
            if complex_operations in ["polar", "both"]:
                self.magnitude_proj = nn.Linear(d_model, d_model)
                self.phase_proj = nn.Linear(d_model, d_model)
            
            # Output projection
            self.output_proj = nn.Linear(d_model * 2, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process with complex representation.
        
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            output: [batch, seq_len, d_model]
            complex_metrics: Dictionary with complex metrics
        """
        if not self.use_complex:
            return x, {}
        
        # Real and imaginary parts
        real_part = self.complex_proj_real(x)
        imag_part = self.complex_proj_imag(x)
        
        # Complex representation: z = real + i*imag
        complex_tensor = torch.complex(real_part, imag_part)
        
        # Compute complex metrics
        magnitude = torch.abs(complex_tensor)  # [batch, seq_len, d_model]
        phase = torch.angle(complex_tensor)  # [batch, seq_len, d_model]
        
        complex_metrics = {
            'magnitude_mean': magnitude.mean().item(),
            'phase_mean': phase.mean().item(),
            'complex_energy': (magnitude ** 2).mean().item(),
        }
        
        # Process based on operation type
        if self.complex_operations == "polar":
            # Use polar representation
            magnitude_proc = self.magnitude_proj(magnitude)
            phase_proc = self.phase_proj(phase)
            # Convert back to complex
            complex_output = magnitude_proc * torch.exp(1j * phase_proc)
        elif self.complex_operations == "cartesian":
            # Use cartesian representation
            complex_output = complex_tensor
        else:  # both
            # Combine both
            magnitude_proc = self.magnitude_proj(magnitude)
            phase_proc = self.phase_proj(phase)
            complex_polar = magnitude_proc * torch.exp(1j * phase_proc)
            complex_output = (complex_tensor + complex_polar) / 2
        
        # Extract real and imaginary parts
        real_output = complex_output.real
        imag_output = complex_output.imag
        
        # Concatenate and project
        combined = torch.cat([real_output, imag_output], dim=-1)  # [batch, seq_len, 2*d_model]
        output = self.output_proj(combined)  # [batch, seq_len, d_model]
        
        return output, complex_metrics


class MultiScaleStructuralBias(nn.Module):
    """
    Multi-scale structural bias for hierarchical processing.
    
    Applies structural biases at multiple scales:
    - Local scale (neighborhood)
    - Medium scale (intermediate)
    - Global scale (long-range)
    """
    
    def __init__(
        self,
        d_model: int,
        num_scales: int = 3,
        scale_types: List[str] = None,  # ["local", "medium", "global"]
        use_hierarchical: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        
        if scale_types is None:
            scale_types = ["local", "medium", "global"]
        self.scale_types = scale_types[:num_scales]
        
        # Scale-specific processing
        self.scale_processors = nn.ModuleDict()
        for scale_type in self.scale_types:
            if scale_type == "local":
                # Local: small kernel
                self.scale_processors[scale_type] = nn.Conv1d(
                    d_model, d_model, kernel_size=3, padding=1, groups=d_model
                )
            elif scale_type == "medium":
                # Medium: medium kernel
                self.scale_processors[scale_type] = nn.Conv1d(
                    d_model, d_model, kernel_size=7, padding=3, groups=d_model
                )
            elif scale_type == "global":
                # Global: attention-based
                self.scale_processors[scale_type] = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=8,
                    batch_first=True,
                )
        
        # Hierarchical fusion
        self.use_hierarchical = use_hierarchical
        if use_hierarchical:
            self.hierarchical_fusion = nn.Sequential(
                nn.Linear(d_model * num_scales, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
        else:
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply multi-scale structural bias.
        
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            output: [batch, seq_len, d_model]
            scale_metrics: Dictionary with scale metrics
        """
        batch_size, seq_len, d_model = x.shape
        
        scale_outputs = []
        scale_metrics = {}
        
        for scale_type in self.scale_types:
            processor = self.scale_processors[scale_type]
            
            if scale_type == "global":
                # Attention-based global processing
                scale_out, _ = processor(x, x, x)
            else:
                # Convolution-based local/medium processing
                x_conv = x.transpose(1, 2)  # [batch, d_model, seq_len]
                scale_out_conv = processor(x_conv)
                scale_out = scale_out_conv.transpose(1, 2)  # [batch, seq_len, d_model]
            
            scale_outputs.append(scale_out)
            scale_metrics[f'{scale_type}_energy'] = scale_out.abs().mean().item()
        
        # Fuse scales
        if self.use_hierarchical:
            # Hierarchical fusion
            scales_concat = torch.cat(scale_outputs, dim=-1)  # [batch, seq_len, num_scales * d_model]
            output = self.hierarchical_fusion(scales_concat)
        else:
            # Weighted combination
            weights = torch.softmax(self.scale_weights, dim=0)
            output = sum(
                w * scale_out
                for w, scale_out in zip(weights, scale_outputs)
            )
        
        scale_metrics['num_scales'] = len(self.scale_types)
        
        return output, scale_metrics


class EnhancedMultimodalFeatures(nn.Module):
    """
    Enhanced multi-modal features with all advanced mechanisms.
    
    Integrates:
    - Spectral filters
    - Gating mechanisms
    - Complex representations
    - Multi-scale structural biases
    """
    
    def __init__(
        self,
        d_model: int,
        use_spectral: bool = True,
        use_gating: bool = True,
        use_complex: bool = True,
        use_multiscale: bool = True,
        spectral_bands: int = 8,
        gate_type: str = "multi",
        complex_operations: str = "polar",
        num_scales: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Spectral filters
        self.use_spectral = use_spectral
        if use_spectral:
            self.spectral_filter = SpectralFilterBank(
                d_model=d_model,
                num_bands=spectral_bands,
            )
        
        # Gating mechanisms
        self.use_gating = use_gating
        if use_gating:
            self.gating = AdaptiveGatingMechanism(
                d_model=d_model,
                gate_type=gate_type,
            )
        
        # Complex representations
        self.use_complex = use_complex
        if use_complex:
            self.complex_layer = ComplexRepresentationLayer(
                d_model=d_model,
                use_complex=True,
                complex_operations=complex_operations,
            )
        
        # Multi-scale structural biases
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.multiscale_bias = MultiScaleStructuralBias(
                d_model=d_model,
                num_scales=num_scales,
            )
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply all enhanced features.
        
        Args:
            x: [batch, seq_len, d_model]
            context: Optional context for gating
        
        Returns:
            output: [batch, seq_len, d_model]
            metrics: Dictionary with all metrics
        """
        metrics = {}
        output = x
        
        # 1. Spectral filtering
        if self.use_spectral:
            output, band_energies = self.spectral_filter(output)
            metrics['spectral'] = {
                'band_energies': band_energies,
            }
        
        # 2. Gating
        if self.use_gating:
            output, gate_metrics = self.gating(output, context=context)
            metrics['gating'] = gate_metrics
        
        # 3. Complex representation
        if self.use_complex:
            output, complex_metrics = self.complex_layer(output)
            metrics['complex'] = complex_metrics
        
        # 4. Multi-scale structural bias
        if self.use_multiscale:
            output, scale_metrics = self.multiscale_bias(output)
            metrics['multiscale'] = scale_metrics
        
        # Normalize
        output = self.norm(output)
        
        return output, metrics

