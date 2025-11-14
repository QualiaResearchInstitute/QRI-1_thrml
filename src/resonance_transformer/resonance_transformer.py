"""
Resonance Transformer: Kuramoto-based Attention Mechanism

Replaces static QK^T matrix multiplication with dynamic, physical simulation.
The entire "attention head" becomes a Kuramoto oscillator system.

Enhanced with:
- Critical coupling tuning (edge of bifurcation)
- Kuramoto-Sakaguchi variant (phase lag)
- Order parameter tracking
- Coupling kernels (distance-based modulation)
- Metastability monitoring
- Stuart-Landau dynamics (phase + amplitude)
- Heun integrator (RK2)
- CDNS metrics with gradients
"""

import contextlib
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
import math

# Ensure project root is on sys.path so top-level packages (e.g., spare_parts/) are importable
import sys
import pathlib as _rt_pathlib
_rt_file_path = _rt_pathlib.Path(__file__).resolve()
_project_root = _rt_file_path.parent.parent
_module_dir = _rt_file_path.parent / "modules"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

# Provide import alias so 'from resonance_transformer.export_coreml ...' works when this file is a module.
try:
    import importlib as _rt_importlib
    _export_coreml = _rt_importlib.import_module("scripts.export_coreml")
    sys.modules[__name__ + ".export_coreml"] = _export_coreml  # type: ignore
except Exception:
    pass

# ANE-friendly constraint path modules (try absolute import first, then relative)
try:
    from resonance_transformer.modules.beam_splitter import BeamSplitterUnitaryStack
    from resonance_transformer.modules.spectral_threshold import SpectralThresholdHead
except ImportError:
    # Fallback: try direct import from modules directory
    try:
        from modules.beam_splitter import BeamSplitterUnitaryStack
        from modules.spectral_threshold import SpectralThresholdHead
    except ImportError:
        # Last resort: create stubs
        class BeamSplitterUnitaryStack:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, x): return x
        class SpectralThresholdHead:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, x): return x

from modules.delay import DelayLine, DelayConfig, uniform_delayed_coupling_force
# Robust import: try modules.telemetry first, fallback to spare_parts.telemetry if needed
try:
    from modules.telemetry import TelemetryWriter, TelemetryConfig
except ImportError:
    try:
        from spare_parts.telemetry import TelemetryWriter, TelemetryConfig  # type: ignore
    except ImportError:
        # Fallback: create stub classes if telemetry is unavailable
        from typing import Optional, Dict, Any
        from dataclasses import dataclass
        
        @dataclass
        class TelemetryConfig:  # type: ignore
            enabled: bool = True
            output_path: Optional[str] = None
        
        class TelemetryWriter:  # type: ignore
            def __init__(self, config: Optional[TelemetryConfig] = None):
                self.config = config or TelemetryConfig()
                self.enabled = self.config.enabled
            def emit(self, data: Dict[str, Any]) -> None:
                pass
            def trace_operation(self, *args, **kwargs):
                return contextlib.nullcontext()
            def observe_device_headroom(self, *args, **kwargs):
                return None

# Robust import: handle module/package name collision (spare_parts/controllers.py vs spare_parts/controllers/)
try:
    from spare_parts.controllers.pid_autopilot import PIDAutopilot, PIDConfig  # preferred (package submodule)
except Exception:
    try:
        # Fallback if controllers is a module exporting these symbols
        from spare_parts.controllers import PIDAutopilot, PIDConfig  # type: ignore
    except Exception:
        PIDAutopilot = None  # type: ignore
        PIDConfig = None     # type: ignore
from modules.cdns_torch import compute_cdns_torch
try:
    from modules.extended_cdns import compute_extended_cdns, ExtendedCDNS
    EXTENDED_CDNS_AVAILABLE = True
except ImportError:
    EXTENDED_CDNS_AVAILABLE = False
    ExtendedCDNS = None
from modules.device_manager import get_device_manager, get_optimal_device
try:
    from modules.unified_device_orchestrator import get_orchestrator
    UNIFIED_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    UNIFIED_ORCHESTRATOR_AVAILABLE = False
    get_orchestrator = None
from modules.kernels import (
    GaussianDistanceKernel as SP_GaussianDistanceKernel,
    AlternatingKernel as SP_AlternatingKernel,
    LearnedMLPKernel as SP_LearnedMLPKernel,
    BlendKernel as SP_BlendKernel,
)

# MSF analyzer (optional)
try:
    from modules.msf_analysis import MSFAnalyzer  # type: ignore
    MSF_AVAILABLE = True
except Exception:
    MSFAnalyzer = None  # type: ignore
    MSF_AVAILABLE = False

# Bifurcation analyzer (optional)
try:
    from modules.bifurcation_analyzer import BifurcationAnalyzer  # type: ignore
    BIFURCATION_AVAILABLE = True
except Exception:
    BifurcationAnalyzer = None  # type: ignore
    BIFURCATION_AVAILABLE = False

# Stochastic dynamics (optional)
try:
    from modules.stochastic_dynamics import StochasticDynamics  # type: ignore
    STOCHASTIC_DYNAMICS_AVAILABLE = True
except Exception:
    StochasticDynamics = None  # type: ignore
    STOCHASTIC_DYNAMICS_AVAILABLE = False

# Information flow analyzer (optional)
try:
    from modules.information_flow import InformationFlowAnalyzer  # type: ignore
    INFORMATION_FLOW_AVAILABLE = True
except Exception:
    InformationFlowAnalyzer = None  # type: ignore
    INFORMATION_FLOW_AVAILABLE = False

# Adaptive coupling (optional)
try:
    from modules.adaptive_coupling import (
        AdaptiveCoupling, 
        PerformanceAdaptiveCoupling,
        adapt_coupling_gradient,
        compute_order_parameter_from_coupling,
    )  # type: ignore
    ADAPTIVE_COUPLING_AVAILABLE = True
except Exception:
    AdaptiveCoupling = None  # type: ignore
    PerformanceAdaptiveCoupling = None  # type: ignore
    adapt_coupling_gradient = None  # type: ignore
    compute_order_parameter_from_coupling = None  # type: ignore
    ADAPTIVE_COUPLING_AVAILABLE = False

# Frequency domain analysis (optional)
try:
    from modules.frequency_domain_analysis import (
        FrequencyDomainAnalyzer,
        compute_fft_spectrum,
        compute_spectral_coherence,
        compute_frequency_dependent_coupling,
        detect_multi_frequency_synchronization,
    )  # type: ignore
    FREQUENCY_DOMAIN_AVAILABLE = True
except Exception:
    FrequencyDomainAnalyzer = None  # type: ignore
    compute_fft_spectrum = None  # type: ignore
    compute_spectral_coherence = None  # type: ignore
    compute_frequency_dependent_coupling = None  # type: ignore
    detect_multi_frequency_synchronization = None  # type: ignore
    FREQUENCY_DOMAIN_AVAILABLE = False

# Multi-scale structures (optional)
try:
    from modules.multiscale_structures import (
        MultiScaleStructures,
        construct_hierarchical_coupling,
        combine_hierarchical_coupling,
        generate_scale_free_topology,
        apply_scale_free_topology_to_coupling,
        compute_nested_synchronization,
    )  # type: ignore
    MULTISCALE_AVAILABLE = True
except Exception:
    MultiScaleStructures = None  # type: ignore
    construct_hierarchical_coupling = None  # type: ignore
    combine_hierarchical_coupling = None  # type: ignore
    generate_scale_free_topology = None  # type: ignore
    apply_scale_free_topology_to_coupling = None  # type: ignore
    compute_nested_synchronization = None  # type: ignore
    MULTISCALE_AVAILABLE = False


# -----------------------------
# Stuart-Landau RHS (PyTorch version)
# -----------------------------

def stuart_landau_rhs_torch(
    phases: torch.Tensor,
    amplitudes: torch.Tensor,
    gains: torch.Tensor,
    coupling_matrix: torch.Tensor,
    phase_lag: Optional[torch.Tensor] = None,
    harmonics: Optional[torch.Tensor] = None,
    kappa_harm: Optional[torch.Tensor] = None,
    alpha_harm: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the coupled Stuart-Landau phase + amplitude derivatives (PyTorch).
    
    Phase: θ̇_i = Σ_j r_i r_j K_ij sin(θ_j - θ_i - α)
    Amplitude: ṙ_i = (α_i - r_i²)r_i - Σ_j r_j K_ij cos(θ_i - θ_j)
    """
    # Replace -inf with 0 to avoid NaN from -inf * 0 operations
    # -inf comes from masking and means no coupling (should be 0)
    coupling_matrix_safe = coupling_matrix.clone()
    coupling_matrix_safe[torch.isinf(coupling_matrix_safe) & (coupling_matrix_safe < 0)] = 0.0
    
    theta_i = phases.unsqueeze(-1)  # [batch, seq_len, 1]
    theta_j = phases.unsqueeze(-2)  # [batch, 1, seq_len]
    phase_diff = theta_j - theta_i  # [batch, seq_len, seq_len]
    
    # Apply phase lag if provided (Kuramoto-Sakaguchi)
    if phase_lag is not None:
        phase_diff = phase_diff - phase_lag
    
    sin_diff = torch.sin(phase_diff)
    cos_diff = torch.cos(-phase_diff)  # cos(θ_i - θ_j)
    
    # Phase force: Σ_j r_i r_j K_ij sin(θ_j - θ_i - α)
    amp_outer = amplitudes.unsqueeze(-1) * amplitudes.unsqueeze(-2)  # [batch, seq_len, seq_len]
    phase_force = torch.sum(amp_outer * coupling_matrix_safe * sin_diff, dim=-1)  # [batch, seq_len]

    # Optional harmonic coupling terms: add Σ_m κ_m Σ_j r_i r_j K_ij sin(m(θ_j - θ_i - α_m))
    if (harmonics is not None) and (kappa_harm is not None) and (alpha_harm is not None):
        try:
            n_h = int(harmonics.shape[0])
        except Exception:
            n_h = 0
        if n_h > 0:
            for idx in range(n_h):
                m_val = harmonics[idx]
                a_val = alpha_harm[idx]
                k_val = kappa_harm[idx]
                # m(θ_j - θ_i - α_m)
                sin_m = torch.sin(m_val * (phase_diff - a_val))
                phase_force = phase_force + k_val * torch.sum(amp_outer * coupling_matrix_safe * sin_m, dim=-1)

    # Amplitude force: (α_i - r_i²)r_i - Σ_j r_j K_ij cos(θ_i - θ_j)
    stuart_landau_term = (gains - amplitudes**2) * amplitudes
    hamiltonian_grad = torch.sum(amplitudes.unsqueeze(-2) * coupling_matrix_safe * cos_diff, dim=-1)
    amplitude_force = stuart_landau_term - hamiltonian_grad
    
    return phase_force, amplitude_force


def soft_clamp_torch(amplitudes: torch.Tensor, clamp_max: Optional[float] = None) -> torch.Tensor:
    """Softly clamp amplitudes to avoid runaway growth."""
    if clamp_max is None or clamp_max <= 0:
        return amplitudes
    scale = max(clamp_max, 1e-6)
    return scale * torch.tanh(amplitudes / scale)


def heun_step_torch(
    phases: torch.Tensor,
    amplitudes: torch.Tensor,
    gains: torch.Tensor,
    coupling_matrix: torch.Tensor,
    natural_frequencies: torch.Tensor,
    dt: float,
    phase_lag: Optional[torch.Tensor] = None,
    clamp_max: Optional[float] = 1.0,
    harmonics: Optional[torch.Tensor] = None,
    kappa_harm: Optional[torch.Tensor] = None,
    alpha_harm: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Heun's method (RK2) for Stuart-Landau dynamics.
    More accurate than Euler, especially for larger dt.
    """
    def rhs(ph: torch.Tensor, amp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dphase, damp = stuart_landau_rhs_torch(
            ph, amp, gains, coupling_matrix, phase_lag,
            harmonics=harmonics, kappa_harm=kappa_harm, alpha_harm=alpha_harm
        )
        # Add natural frequencies to phase derivative
        dphase = dphase + natural_frequencies
        return dphase, damp
    
    # k1 step
    k1_phase, k1_amp = rhs(phases, amplitudes)
    pred_phases = (phases + dt * k1_phase) % (2 * np.pi)
    pred_amplitudes = amplitudes + dt * k1_amp
    pred_amplitudes = soft_clamp_torch(pred_amplitudes, clamp_max)
    pred_amplitudes = torch.clamp(pred_amplitudes, min=0.0)
    
    # k2 step
    k2_phase, k2_amp = rhs(pred_phases, pred_amplitudes)
    
    # Combine
    new_phases = (phases + dt * 0.5 * (k1_phase + k2_phase)) % (2 * np.pi)
    new_amplitudes = amplitudes + dt * 0.5 * (k1_amp + k2_amp)
    new_amplitudes = soft_clamp_torch(new_amplitudes, clamp_max)
    new_amplitudes = torch.clamp(new_amplitudes, min=0.0)
    
    return new_phases, new_amplitudes


# -----------------------------
# Time-Reversible Integration
# -----------------------------

def compute_optimal_window_length(
    coupling_strength: float,
    order_parameter: float,
    dt: float,
) -> int:
    """
    Heuristic rule for optimal forward-backward window length.
    
    Based on paper's heuristics:
    - Larger window for stronger coupling
    - Smaller window near criticality (R ≈ 0.5-0.7)
    - Adjust based on dt
    
    Args:
        coupling_strength: Average coupling strength
        order_parameter: Current order parameter R
        dt: Time step
        
    Returns:
        Optimal window length (number of steps)
    """
    # Base window
    base_window = 5
    
    # Adjust for coupling strength
    coupling_factor = 1.0 + coupling_strength * 2.0
    
    # Adjust for criticality (smaller near critical point)
    if 0.4 < order_parameter < 0.8:
        criticality_factor = 0.8  # Smaller window near criticality
    else:
        criticality_factor = 1.0
    
    # Adjust for dt (smaller dt needs more steps)
    dt_factor = 1.0 / (dt * 100)  # Normalize to dt=0.01
    
    window = int(base_window * coupling_factor * criticality_factor * dt_factor)
    return max(3, min(window, 15))  # Clamp to reasonable range


# -----------------------------
# CDNS Metrics (PyTorch version)
# -----------------------------

def grad_order_parameter_torch(phases: torch.Tensor, amplitudes: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Analytic gradient ∂C/∂θ_k = (1/N) sin(θ_k - phi) weighted by amplitudes.
    
    Args:
        phases: [batch, seq_len] phases
        amplitudes: [batch, seq_len] optional amplitudes
        
    Returns:
        grad_C: [batch, seq_len] gradient
    """
    batch_size, seq_len = phases.shape
    
    if amplitudes is None:
        amplitudes = torch.ones_like(phases) / seq_len
    
    # Compute order parameter phase phi
    complex_phases = amplitudes * torch.exp(1j * phases)  # [batch, seq_len]
    R_complex = torch.sum(complex_phases, dim=-1)  # [batch]
    phi = torch.angle(R_complex)  # [batch]
    
    # Gradient: (1/N) sin(θ_k - phi) weighted by amplitudes
    phase_diff = phases - phi.unsqueeze(-1)  # [batch, seq_len]
    grad = (1.0 / seq_len) * torch.sin(phase_diff) * amplitudes
    
    return grad


def xy_energy_torch(phases: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    XY energy: 0.5 Σ_ij w_ij (1 - cos(θ_i - θ_j)).
    
    Args:
        phases: [batch, seq_len] phases
        weights: [batch, seq_len, seq_len] optional coupling weights
        
    Returns:
        energy: [batch] energy per batch
    """
    if weights is None:
        return torch.zeros(phases.shape[0], device=phases.device)
    
    # Sanitize weights to avoid propagating -inf (masked), +inf, or NaN into metrics
    try:
        weights_safe = weights.clone()
    except Exception:
        weights_safe = weights
    # Treat any infinite or NaN weights as zero contribution
    weights_safe = torch.nan_to_num(weights_safe, nan=0.0, posinf=0.0, neginf=0.0)

    theta_i = phases.unsqueeze(-1)  # [batch, seq_len, 1]
    theta_j = phases.unsqueeze(-2)  # [batch, 1, seq_len]
    phase_diff = theta_j - theta_i  # [batch, seq_len, seq_len]
    
    energy = 0.5 * torch.sum(weights_safe * (1.0 - torch.cos(phase_diff)), dim=(-2, -1))
    return energy


def xy_energy_grad_torch(phases: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Gradient of XY energy w.r.t. phases.
    
    Args:
        phases: [batch, seq_len] phases
        weights: [batch, seq_len, seq_len] optional coupling weights
        
    Returns:
        grad: [batch, seq_len] gradient
    """
    if weights is None:
        return torch.zeros_like(phases)
    
    # Sanitize weights to avoid propagating -inf (masked), +inf, or NaN into gradients
    try:
        weights_safe = weights.clone()
    except Exception:
        weights_safe = weights
    weights_safe = torch.nan_to_num(weights_safe, nan=0.0, posinf=0.0, neginf=0.0)

    theta_i = phases.unsqueeze(-1)  # [batch, seq_len, 1]
    theta_j = phases.unsqueeze(-2)  # [batch, 1, seq_len]
    phase_diff = theta_j - theta_i  # [batch, seq_len, seq_len]
    
    # Gradient: Σ_j w_ij sin(θ_i - θ_j)
    grad = torch.sum(weights_safe * torch.sin(phase_diff), dim=-1)  # [batch, seq_len]
    return grad


class CouplingKernel(nn.Module):
    """
    Coupling Kernel for modulating coupling strength based on distance or learned patterns.
    
    Supports:
    - Distance-based kernels (Gaussian, exponential decay)
    - Learned kernels (MLP-based)
    - Global resonant mode tuning
    """
    
    def __init__(self, d_k: int, kernel_type: str = "learned", 
                 use_distance: bool = True, max_seq_len: int = 512,
                 use_spectral_gating: bool = False,
                 spectral_num_bands: int = 3,
                 spectral_learnable: bool = True,
                 spectral_basis: str = "positional",
                 use_unified_orchestrator: bool = True):
        """
        Initialize Coupling Kernel.
        
        Args:
            d_k: Key dimension
            kernel_type: "learned", "gaussian", or "exponential"
            use_distance: Whether to incorporate positional distance
            max_seq_len: Maximum sequence length for distance computation
            use_unified_orchestrator: Use unified device orchestrator for transparent device placement
        """
        super().__init__()
        self.d_k = d_k
        self.kernel_type = kernel_type
        self.use_distance = use_distance
        self.max_seq_len = max_seq_len
        self.use_spectral_gating = use_spectral_gating
        self.spectral_num_bands = spectral_num_bands
        self.spectral_learnable = spectral_learnable
        self.spectral_basis = spectral_basis  # "positional" | "coupling"
        self.spectral_cache = None  # optional: to be attached externally
        self.use_unified_orchestrator = use_unified_orchestrator and UNIFIED_ORCHESTRATOR_AVAILABLE
        self._orchestrator = None
        if self.use_unified_orchestrator:
            try:
                self._orchestrator = get_orchestrator()
            except Exception:
                self.use_unified_orchestrator = False
        
        if kernel_type == "learned":
            # Learned kernel: MLP that takes key similarities and outputs coupling modulation
            self.kernel_mlp = nn.Sequential(
                nn.Linear(1, d_k // 2),
                nn.GELU(),
                nn.Linear(d_k // 2, 1),
                nn.Sigmoid()  # Output in [0, 1] for modulation
            )
        elif kernel_type == "gaussian":
            # Gaussian kernel parameters
            self.sigma = nn.Parameter(torch.tensor(10.0))
        elif kernel_type == "exponential":
            # Exponential decay parameter
            self.decay_rate = nn.Parameter(torch.tensor(0.1))
        
        # Spectral gating band weights (initialized to identity behavior)
        if self.use_spectral_gating:
            init = torch.ones(self.spectral_num_bands, dtype=torch.float32)
            if self.spectral_learnable:
                self.register_parameter("band_weights", nn.Parameter(init))
            else:
                self.register_buffer("band_weights", init, persistent=False)

    def attach_spectral_cache(self, cache) -> None:
        """
        Attach a SpectralCache providing positional eigenpairs for buckets.
        """
        self.spectral_cache = cache
    
    def compute_distance_matrix(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute positional distance matrix."""
        positions = torch.arange(seq_len, device=device).float()
        dist_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        return dist_matrix
    
    def forward(self, K_ij: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply coupling kernel modulation.
        
        Args:
            K_ij: Base coupling matrix [batch, seq_len, seq_len]
            seq_len: Sequence length
            
        Returns:
            Modulated coupling matrix [batch, seq_len, seq_len]
        """
        batch_size = K_ij.shape[0]
        
        if self.use_distance:
            dist_matrix = self.compute_distance_matrix(seq_len, K_ij.device)
            dist_matrix = dist_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            dist_matrix = None
        
        if self.kernel_type == "learned":
            # Use learned MLP to modulate based on similarity
            # Input: normalized similarity scores
            # Process in flattened form to reduce memory usage
            # For very large sequences, process in chunks to avoid OOM
            original_shape = K_ij.shape
            batch_size, seq_len = original_shape[0], original_shape[1]
            
            # Determine optimal device for this operation
            device_manager = get_device_manager()
            tensor_size_bytes = batch_size * seq_len * seq_len * K_ij.element_size()
            
            # Check if we should use CPU for this large operation
            use_cpu_for_mlp = device_manager.should_use_cpu_for_large_operation(
                (batch_size, seq_len, seq_len),
                dtype=K_ij.dtype,
                num_tensors=2,  # K_ij + similarity_norm
            )
            
            # Use chunked processing for sequences > 1M elements to reduce memory pressure
            chunk_size = 1024 * 1024  # Process ~1M elements at a time
            total_elements = batch_size * seq_len * seq_len
            
            if total_elements > chunk_size or use_cpu_for_mlp:
                # Chunked processing for large sequences or when CPU is preferred
                modulation_parts = []
                original_device = K_ij.device
                
                # Process each batch item separately to reduce memory
                for b in range(batch_size):
                    batch_K = K_ij[b]  # [seq_len, seq_len]
                    
                    # Move to CPU for MLP if needed, then back
                    if use_cpu_for_mlp:
                        batch_K_cpu = batch_K.cpu()
                        batch_similarity = torch.sigmoid(batch_K_cpu).flatten().unsqueeze(-1)  # [seq_len * seq_len, 1]
                    else:
                        batch_similarity = torch.sigmoid(batch_K).flatten().unsqueeze(-1)  # [seq_len * seq_len, 1]
                    
                    batch_modulation_flat = []
                    
                    # Process in chunks
                    for i in range(0, batch_similarity.shape[0], chunk_size):
                        chunk = batch_similarity[i:i + chunk_size]
                        chunk_mod = self.kernel_mlp(chunk).squeeze(-1)
                        batch_modulation_flat.append(chunk_mod)
                    
                    batch_modulation = torch.cat(batch_modulation_flat, dim=0)
                    
                    # Move back to original device if we used CPU
                    if use_cpu_for_mlp:
                        batch_modulation = batch_modulation.to(original_device)
                    
                    modulation_parts.append(batch_modulation.view(seq_len, seq_len))
                
                modulation = torch.stack(modulation_parts, dim=0)  # [batch, seq_len, seq_len]
            else:
                # Standard processing for smaller sequences
                similarity_norm = torch.sigmoid(K_ij).flatten(0, 2).unsqueeze(-1)  # [batch * seq_len * seq_len, 1]
                modulation_flat = self.kernel_mlp(similarity_norm).squeeze(-1)  # [batch * seq_len * seq_len]
                modulation = modulation_flat.view(original_shape)  # [batch, seq_len, seq_len]
            
            if dist_matrix is not None:
                # Combine with distance-based decay
                distance_modulation = torch.exp(-dist_matrix / self.max_seq_len)
                modulation = modulation * distance_modulation
            
            K_mod = K_ij * modulation
        
        elif self.kernel_type == "gaussian":
            if dist_matrix is not None:
                gaussian_mod = torch.exp(-0.5 * (dist_matrix / self.sigma) ** 2)
                K_mod = K_ij * gaussian_mod
            else:
                K_mod = K_ij
        
        elif self.kernel_type == "exponential":
            if dist_matrix is not None:
                exp_mod = torch.exp(-self.decay_rate * dist_matrix)
                K_mod = K_ij * exp_mod
            else:
                K_mod = K_ij
        
        else:
            K_mod = K_ij
        
        # Optional spectral gating (torch-native)
        if self.use_spectral_gating and seq_len > 0 and self.spectral_num_bands > 0:
            K_mod = self._apply_spectral_gating(K_mod)
        
        return K_mod

    def _apply_spectral_gating(self, K_batched: torch.Tensor) -> torch.Tensor:
        """
        Gate K via Laplacian eigen-bands with learnable per-band weights.
        For each batch b:
            L = D - K
            (λ, V) = eigh(L)
            Split modes into bands; K_recon = Σ_b w_b (P_b K P_b), where P_b = V_b V_b^T
        """
        batch_size, n, _ = K_batched.shape
        device = K_batched.device
        dtype = K_batched.dtype
        # Clamp num bands to n
        num_bands = int(max(1, min(self.spectral_num_bands, n)))
        # Initialize weights (ensuring tensor on device)
        if hasattr(self, "band_weights"):
            bw = self.band_weights
        else:
            bw = torch.ones(num_bands, device=device, dtype=dtype)
        # If buffer/parameter lives on different device (rare), move a copy
        bw = bw.to(device=device, dtype=dtype)
        # Build edges for band splits by index; replaced if cache is used
        edges = torch.linspace(0, n, steps=num_bands + 1, device=device, dtype=torch.int64).to(torch.int64)
        out_list = []
        for b in range(batch_size):
            K = K_batched[b]
            # Symmetrize to avoid numerical drift
            K = 0.5 * (K + K.T)
            # Basis
            if (self.spectral_basis == "positional") and (self.spectral_cache is not None):
                # Use cached positional eigen-basis (ring Laplacian)
                spec = self.spectral_cache.get(n, device=device, dtype=dtype)
                if spec is not None:
                    V = spec.V  # [n, n]
                    band_edges = spec.band_edges
                else:
                    # Fallback to coupling-derived basis
                    spec = None
                    band_edges = None
            else:
                spec = None
                band_edges = None
            if spec is None:
                # Build coupling-derived Laplacian basis
                d = torch.sum(K, dim=1)
                L = torch.diag(d) - K
                try:
                    lams, V = torch.linalg.eigh(L)
                except Exception:
                    L_cpu = L.detach().to('cpu')
                    lams_cpu, V_cpu = torch.linalg.eigh(L_cpu)
                    lams = lams_cpu.to(device=device, dtype=dtype)
                    V = V_cpu.to(device=device, dtype=dtype)
                # Sort ascending
                idx = torch.argsort(lams)
                V = V[:, idx]
                # Create default uniform band edges
                band_edges = [(int(edges[i].item()), int(edges[i + 1].item())) for i in range(num_bands)]
            # Reconstruct with band weights
            K_rec = torch.zeros_like(K)
            for band in range(num_bands):
                if band_edges is not None:
                    lo, hi = band_edges[band]
                else:
                    lo = int(edges[band].item())
                    hi = int(edges[band + 1].item())
                if hi <= lo:
                    continue
                Vb = V[:, lo:hi]  # (n, k_b)
                # P_b = Vb Vb^T
                Pb = Vb @ Vb.T
                # K_b = P_b K P_b
                Kb = Pb @ K @ Pb
                K_rec = K_rec + bw[band] * Kb
            # Re-symmetrize
            K_rec = 0.5 * (K_rec + K_rec.T)
            out_list.append(K_rec)
        return torch.stack(out_list, dim=0)


def hodge_decompose_coupling(
    coupling_matrix: torch.Tensor,
    return_harmonic_dim: bool = False,
    harmonic_threshold: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """
    Decompose coupling matrix into Hodge components (exact, coexact, harmonic).
    
    The Hodge decomposition splits edge flows (coupling interactions) into three
    orthogonal components:
    1. Exact component (gradient-like): Fast decay, damped by heat smoothing
    2. Coexact component (curl-like): Fast decay, damped by heat smoothing
    3. Harmonic component (topology-anchored): Lives in kernel of Laplacian,
       corresponds to cycles in graph, provides long-term memory
    
    Args:
        coupling_matrix: [batch, seq_len, seq_len] coupling matrix
        return_harmonic_dim: If True, return dimension of harmonic space
        harmonic_threshold: Threshold for identifying zero eigenvalues (harmonic modes)
    
    Returns:
        {
            'exact': exact component (gradient-like) [batch, seq_len, seq_len],
            'coexact': coexact component (curl-like) [batch, seq_len, seq_len],
            'harmonic': harmonic component (cycles, long memory) [batch, seq_len, seq_len],
            'harmonic_dim': (optional) dimension of harmonic space [batch]
        }
    """
    batch_size, n, _ = coupling_matrix.shape
    device = coupling_matrix.device
    dtype = coupling_matrix.dtype
    
    # Validate input
    if torch.any(torch.isnan(coupling_matrix)):
        nan_count = torch.isnan(coupling_matrix).sum().item()
        raise ValueError(f"hodge_decompose_coupling: coupling_matrix contains NaN. NaN count: {nan_count}")
    
    # Check for +inf (not allowed, -inf from masking is OK)
    if torch.any(torch.isinf(coupling_matrix)):
        inf_mask = torch.isinf(coupling_matrix)
        inf_values = coupling_matrix[inf_mask]
        pos_inf = inf_values > 0
        if torch.any(pos_inf):
            raise ValueError(f"hodge_decompose_coupling: coupling_matrix contains +Inf. -inf from masking is OK")
    
    # Compute graph Laplacian: L = D - W
    # For undirected graphs, we use symmetric coupling matrix
    # Make symmetric if needed (take average of K and K^T)
    K_sym = 0.5 * (coupling_matrix + coupling_matrix.transpose(-2, -1))
    
    # Compute degree matrix: D_ii = sum_j K_ij
    degree = K_sym.sum(dim=-1)  # [batch, seq_len]
    
    # Avoid division by zero for isolated nodes
    degree = torch.clamp(degree, min=1e-8)
    
    # Degree matrix: D = diag(degree)
    D = torch.diag_embed(degree)  # [batch, seq_len, seq_len]
    
    # Laplacian: L = D - K
    L = D - K_sym  # [batch, seq_len, seq_len]
    
    # Eigen-decomposition of Laplacian
    # Note: torch.linalg.eigh works on last two dimensions
    # We need to handle batch dimension
    eigenvals_list = []
    eigenvecs_list = []
    
    for b in range(batch_size):
        L_b = L[b]  # [seq_len, seq_len]
        
        # Ensure symmetric for numerical stability
        L_b = 0.5 * (L_b + L_b.T)
        
        # Eigen-decomposition
        eigenvals_b, eigenvecs_b = torch.linalg.eigh(L_b)
        # eigenvals_b: [seq_len], eigenvecs_b: [seq_len, seq_len]
        
        eigenvals_list.append(eigenvals_b)
        eigenvecs_list.append(eigenvecs_b)
    
    eigenvals = torch.stack(eigenvals_list, dim=0)  # [batch, seq_len]
    eigenvecs = torch.stack(eigenvecs_list, dim=0)  # [batch, seq_len, seq_len]
    
    # Harmonic = kernel of Laplacian (zero eigenvalues)
    harmonic_mask = torch.abs(eigenvals) < harmonic_threshold  # [batch, seq_len]
    harmonic_dim = harmonic_mask.sum(dim=-1)  # [batch]
    
    # Project coupling onto harmonic modes
    # For each batch, compute harmonic projection matrix
    harmonic_component_list = []
    exact_component_list = []
    
    for b in range(batch_size):
        V_b = eigenvecs[b]  # [seq_len, seq_len]
        mask_b = harmonic_mask[b]  # [seq_len]
        K_b = K_sym[b]  # [seq_len, seq_len]
        
        # Harmonic projection: P_harm = V_harm V_harm^T
        V_harm = V_b[:, mask_b]  # [seq_len, num_harmonic]
        if V_harm.shape[1] > 0:
            P_harm = V_harm @ V_harm.T  # [seq_len, seq_len]
            # Project coupling onto harmonic subspace
            K_harm = P_harm @ K_b @ P_harm.T
        else:
            # No harmonic modes
            K_harm = torch.zeros_like(K_b)
        
        # Exact + coexact = complement (simplified: treat as single component)
        # In full Hodge theory, we'd separate exact and coexact, but for now
        # we treat non-harmonic as "standard" component
        K_exact = K_b - K_harm
        
        harmonic_component_list.append(K_harm)
        exact_component_list.append(K_exact)
    
    harmonic_component = torch.stack(harmonic_component_list, dim=0)  # [batch, seq_len, seq_len]
    exact_component = torch.stack(exact_component_list, dim=0)  # [batch, seq_len, seq_len]
    
    # Coexact component: For now, set to zero (can refine later with full Hodge theory)
    coexact_component = torch.zeros_like(coupling_matrix)
    
    # Validate outputs
    if torch.any(torch.isnan(harmonic_component)) or torch.any(torch.isnan(exact_component)):
        raise ValueError("hodge_decompose_coupling: Output contains NaN")
    
    result = {
        'exact': exact_component,
        'coexact': coexact_component,
        'harmonic': harmonic_component,
    }
    
    if return_harmonic_dim:
        result['harmonic_dim'] = harmonic_dim
    
    return result


class CriticalCouplingTuner(nn.Module):
    """
    Automatically tunes coupling strength near critical value K_c (edge of bifurcation).
    
    The critical coupling strength K_c is where synchronization emerges. Operating
    near this point maximizes computational capacity and complexity.
    """
    
    def __init__(self, initial_K: float = 1.0, adaptive: bool = True):
        """
        Initialize Critical Coupling Tuner.
        
        Args:
            initial_K: Initial coupling strength estimate
            adaptive: Whether to adaptively tune K based on order parameter
        """
        super().__init__()
        self.adaptive = adaptive
        
        # Learnable critical coupling estimate
        # K_c ≈ 2 / (π * g(0)) for uniform frequency distribution
        # We learn a multiplier around this estimate
        self.K_c_multiplier = nn.Parameter(torch.tensor(1.0))
        self.base_K_c = initial_K
    
    def estimate_critical_coupling(self, natural_frequencies: torch.Tensor) -> torch.Tensor:
        """
        Estimate critical coupling strength K_c from natural frequency distribution.
        
        For uniform distribution: K_c ≈ 2 / (π * g(0))
        We use a simplified estimate based on frequency spread.
        """
        # Compute frequency spread (standard deviation)
        freq_std = torch.std(natural_frequencies, dim=-1, keepdim=True)  # [batch, 1]
        
        # Estimate K_c: larger spread requires stronger coupling
        # Simplified: K_c ∝ std(ω)
        K_c_estimate = self.base_K_c * (1.0 + freq_std) * self.K_c_multiplier
        
        return K_c_estimate
    
    def forward(self, coupling_matrix: torch.Tensor, 
                natural_frequencies: torch.Tensor,
                order_parameter: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Tune coupling matrix to operate near critical point.
        
        Args:
            coupling_matrix: Base coupling matrix [batch, seq_len, seq_len]
            natural_frequencies: Natural frequencies [batch, seq_len]
            order_parameter: Current order parameter R (if adaptive)
            
        Returns:
            Tuned coupling matrix
        """
        # Validate inputs - allow -inf from masking, but not NaN or +inf
        if torch.any(torch.isnan(coupling_matrix)):
            nan_count = torch.isnan(coupling_matrix).sum().item()
            raise ValueError(f"CriticalTuner.forward: coupling_matrix contains NaN on input. NaN count: {nan_count}")
        
        # Check for +inf (not allowed, -inf from masking is OK)
        if torch.any(torch.isinf(coupling_matrix)):
            inf_mask = torch.isinf(coupling_matrix)
            inf_values = coupling_matrix[inf_mask]
            # Check if any inf values are positive (not -inf)
            pos_inf = inf_values > 0
            if torch.any(pos_inf):
                raise ValueError(f"CriticalTuner.forward: coupling_matrix contains +Inf on input. -inf from masking is OK, but +inf is not allowed")
        if torch.any(torch.isnan(natural_frequencies)) or torch.any(torch.isinf(natural_frequencies)):
            raise ValueError(f"CriticalTuner.forward: natural_frequencies contains NaN or Inf")
        
        if self.adaptive and order_parameter is not None:
            # Validate order_parameter
            if torch.any(torch.isnan(order_parameter)) or torch.any(torch.isinf(order_parameter)):
                raise ValueError(f"CriticalTuner.forward: order_parameter contains NaN or Inf")
            
            # Adaptive tuning: adjust coupling to keep R near critical point (R ≈ 0.5-0.7)
            target_R = 0.6  # Target near criticality
            R_error = target_R - order_parameter.mean()
            
            # Validate R_error
            if torch.isnan(R_error) or torch.isinf(R_error):
                raise ValueError(f"CriticalTuner.forward: R_error is NaN or Inf (value={R_error})")
            
            # Adjust coupling strength based on order parameter
            # If R too low, increase coupling; if too high, decrease
            adjustment = 1.0 + 0.1 * R_error  # Small adjustments
            
            # Validate adjustment
            if torch.isnan(adjustment) or torch.isinf(adjustment):
                raise ValueError(f"CriticalTuner.forward: adjustment is NaN or Inf (value={adjustment})")
            
            coupling_matrix = coupling_matrix * adjustment.clamp(0.5, 2.0)
            
            # Validate after adjustment - allow -inf from masking, but not NaN or +inf
            if torch.any(torch.isnan(coupling_matrix)):
                nan_count = torch.isnan(coupling_matrix).sum().item()
                raise ValueError(f"CriticalTuner.forward: coupling_matrix contains NaN after adaptive adjustment. NaN count: {nan_count}")
            
            # Check for +inf (not allowed, -inf from masking is OK)
            if torch.any(torch.isinf(coupling_matrix)):
                inf_mask = torch.isinf(coupling_matrix)
                inf_values = coupling_matrix[inf_mask]
                pos_inf = inf_values > 0
                if torch.any(pos_inf):
                    raise ValueError(f"CriticalTuner.forward: coupling_matrix contains +Inf after adaptive adjustment. -inf from masking is OK")
        else:
            # Static tuning: scale to estimated K_c
            K_c = self.estimate_critical_coupling(natural_frequencies)
            
            # Validate K_c
            if torch.isnan(K_c) or torch.isinf(K_c) or K_c <= 0:
                raise ValueError(f"CriticalTuner.forward: K_c is invalid (value={K_c})")
            
            # Scale coupling matrix to operate near K_c
            current_max = coupling_matrix.abs().max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            
            # Validate current_max
            if torch.any(torch.isnan(current_max)) or torch.any(torch.isinf(current_max)):
                raise ValueError(f"CriticalTuner.forward: current_max is NaN or Inf")
            
            scale_factor = K_c / (current_max + 1e-8)
            
            # Validate scale_factor
            if torch.any(torch.isnan(scale_factor)) or torch.any(torch.isinf(scale_factor)):
                raise ValueError(f"CriticalTuner.forward: scale_factor is NaN or Inf")
            
            coupling_matrix = coupling_matrix * scale_factor.clamp(0.1, 10.0)
            
            # Validate after scaling - allow -inf from masking, but not NaN or +inf
            if torch.any(torch.isnan(coupling_matrix)):
                nan_count = torch.isnan(coupling_matrix).sum().item()
                raise ValueError(f"CriticalTuner.forward: coupling_matrix contains NaN after static scaling. NaN count: {nan_count}")
            
            # Check for +inf (not allowed, -inf from masking is OK)
            if torch.any(torch.isinf(coupling_matrix)):
                inf_mask = torch.isinf(coupling_matrix)
                inf_values = coupling_matrix[inf_mask]
                pos_inf = inf_values > 0
                if torch.any(pos_inf):
                    raise ValueError(f"CriticalTuner.forward: coupling_matrix contains +Inf after static scaling. -inf from masking is OK")
        
        return coupling_matrix


class ResonanceAttentionHead(nn.Module):
    """
    Enhanced Resonance Attention Head using Kuramoto-Sakaguchi dynamics.
    
    Features:
    - Critical coupling tuning (edge of bifurcation)
    - Kuramoto-Sakaguchi variant (phase lag)
    - Order parameter tracking
    - Coupling kernels
    - Metastability monitoring
    """
    
    def _init_telemetry_writer(self, telemetry):
        if isinstance(telemetry, TelemetryWriter):
            return telemetry
        if isinstance(telemetry, TelemetryConfig):
            return TelemetryWriter(telemetry)
        if isinstance(telemetry, dict):
            try:
                cfg = TelemetryConfig(**telemetry)
            except TypeError:
                cfg = TelemetryConfig()
            return TelemetryWriter(cfg)
        if telemetry:
            return TelemetryWriter(TelemetryConfig())
        return None

    def _trace_operation(self, op_name: str, *, device: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        metadata = metadata or {}
        if self.telemetry_writer is None:
            return contextlib.nullcontext()
        return self.telemetry_writer.trace_operation(op_name, device=device, metadata=metadata)

    def __init__(self, d_model: int, d_k: int = None, d_v: int = None,
                 n_sim_steps: int = 15, dt: float = 0.01, 
                 coupling_strength: float = 1.0,
                 use_sakaguchi: bool = True,
                 use_critical_tuning: bool = True,
                 use_coupling_kernel: bool = True,
                 kernel_type: str = "learned",
                 track_metrics: bool = True,
                 store_visualization_traces: bool = False,
                 visualization_history_limit: int = 64,
                 use_stuart_landau: bool = True,
                 use_heun: bool = True,
                 track_cdns: bool = True,
                 # Delay dynamics
                 use_delays: bool = False,
                 tau_steps: int = 0,
                 delay_gain: float = 1.0,
                 use_learnable_delays: bool = False,
                 # Spectral gating knobs
                 use_spectral_gating: bool = True,
                 spectral_num_bands: int = 4,
                 spectral_learnable: bool = True,
                 # Regularizer knobs
                 lambda_dissonance: float = 0.0,
                 lambda_criticality: float = 0.0,
                 target_R: float = 0.6,
                 lambda_frustration: float = 0.0,
                 # Order-parameter safety (PhaseGPT-inspired)
                 use_order_param_safety: bool = True,
                 R_min: float = 0.3,
                 R_max: float = 0.85,
                 R_clamp_gain: float = 0.5,
                 # PID / CDNS / Hybrid knobs
                 use_pid_autopilot: bool = False,
                 pid_config: Optional[PIDConfig] = None,
                 use_cdns_full: bool = False,
                 cdns_top_m: int = 0,
                 use_extended_cdns: bool = False,
                 lambda_noise: float = 0.0,
                 lambda_signal: float = 0.0,
                 hybrid_readout: bool = False,
                 hybrid_mix_init: float = 0.2,
                 # Auxiliary interference readout
                 use_aux_interference: bool = False,
                 aux_iters: int = 3,
                 aux_step: float = 0.1,
                 aux_eps: float = 1e-2,
                 aux_mix_init: float = 0.05,
                 # Registry kernels
                 use_registry_kernel: bool = False,
                 registry_kernel_type: str = "blend",
                 registry_rank: int = 32,
                 registry_temperature: float = 1.0,
                 # Temporal multiplexing
                 use_temporal_multiplex: bool = False,
                 tm_dts=None,
                 tm_alpha_offsets=None,
                 tm_learned_mix: bool = True,
                 # Harmonic coupling
                 use_harmonics: bool = False,
                 harmonics=None,
                 kappa_harm=None,
                 alpha_harm=None,
                 # Telemetry
                 telemetry: bool = False,
                 # Analytical theory
                 use_analytical_variance: bool = False,
                 analytical_freq_dist: str = "gaussian",
                 analytical_dist_params: Optional[Dict] = None,
                 # MSF integration
                 use_msf_regularizer: bool = False,
                 lambda_msf: float = 0.0,
                 msf_target: str = "critical",
                 msf_n_clusters: int = 8,
                 use_msf_autotune: bool = False,
                 msf_autotune_safety: float = 0.9,
                 msf_eval_every: int = 5,
                 # Hodge decomposition
                 use_hodge_decomposition: bool = False,
                 harmonic_leak_rate: float = 0.001,
                 standard_leak_rate: float = 0.1,
                 # Bifurcation analysis
                 detect_bifurcations: bool = False,
                 track_phase_transitions: bool = False,
                 phase_transition_threshold: float = 0.5,
                 # Time-reversible synchronization
                 use_time_reversible: bool = False,
                 tr_window_length: Optional[int] = None,  # Auto-compute if None
                 tr_adaptive_steps: bool = False,
                 tr_convergence_threshold: float = 1e-5,
                 tr_tolerance: float = 1e-6,
                 # ReLU coupling / Phase 7: Electrical/Memristive coupling
                 coupling_type: str = "sine",  # "sine" (default), "relu", "hybrid", "electrical", or "memristive"
                 relu_weight: float = 0.5,  # For hybrid coupling
                 # Stochastic dynamics
                 use_stochastic_dynamics: bool = False,
                 noise_type: str = "gaussian",  # "gaussian", "levy", "colored"
                 noise_strength: float = 0.01,
                 levy_alpha: float = 1.5,  # Stability parameter (1 < α ≤ 2)
                 levy_beta: float = 0.0,   # Skewness (-1 ≤ β ≤ 1)
                 noise_correlation: float = 0.0,  # For colored noise
                 stochastic_resonance: bool = False,  # Track stochastic resonance
                 # Dual-timescale coupling (NMDA-AMPA)
                 use_dual_timescale: bool = False,
                 fast_tau: float = 0.005,  # Fast decay (~5ms, AMPA-like)
                 slow_tau: float = 0.1,    # Slow decay (~100ms, NMDA-like)
                 fast_coupling_strength: float = 1.0,
                 slow_coupling_strength: float = 0.5,
                 use_voltage_gating: bool = False,  # Voltage-dependent slow coupling gating
                 # Chimera death detection
                 detect_chimera_death: bool = False,
                 amplitude_death_threshold: float = 0.01,
                 coherence_sync_threshold: float = 0.7,
                 # Lag synchronization detection
                 detect_lag_synchronization: bool = False,
                 lag_similarity_threshold: float = 0.1,
                 max_lag: int = 10,
                 # Information flow & propagation
                 track_information_flow: bool = False,
                 info_flow_history_length: int = 20,
                 transfer_entropy_k: int = 1,
                 transfer_entropy_l: int = 1,
                 transfer_entropy_bins: int = 10,
                 granger_max_lag: int = 5,
                 propagation_steps: int = 10,
                 bottleneck_threshold: float = 0.1,
                 # Adaptive coupling
                 use_adaptive_coupling: bool = False,
                 adaptation_rate: float = 0.01,
                 adaptation_signal: str = "order_parameter",
                 adaptation_target: float = 0.6,
                 use_gradient_adaptation: bool = False,
                 use_performance_adaptation: bool = False,
                 performance_metric: str = "loss",
                 use_local_adaptation: bool = False,
                 # Frequency domain analysis
                 analyze_frequency_domain: bool = False,
                 frequency_analysis_history_length: int = 50,
                 use_frequency_dependent_coupling: bool = False,
                 frequency_bands: Optional[List[float]] = None,
                 coupling_per_band: Optional[List[float]] = None,
                 frequency_range: Optional[Tuple[float, float]] = None,
                 # Multi-scale structures
                 use_hierarchical_coupling: bool = False,
                 local_scale: int = 5,
                 global_scale: int = 20,
                 local_weight: float = 0.7,
                 global_weight: float = 0.3,
                 use_scale_free_topology: bool = False,
                 hub_fraction: float = 0.1,
                 power_law_exponent: float = 2.5,
                 # Structural biasing (Graphormer-inspired)
                 use_structural_bias: bool = False,
                 structural_bias_gain: float = 0.1,
                 structural_bias_decay: float = 0.5,
                 structural_bias_threshold: float = 0.5):
        """
        Initialize Enhanced Resonance Attention Head.
        
        Args:
            d_model: Model dimension
            d_k: Key dimension (default: d_model)
            d_v: Value dimension (default: d_model)
            n_sim_steps: Number of Kuramoto simulation steps
            dt: Time step for integration
            coupling_strength: Base coupling strength multiplier
            use_sakaguchi: Use Kuramoto-Sakaguchi variant (phase lag)
            use_critical_tuning: Enable critical coupling tuning
            use_coupling_kernel: Enable coupling kernel modulation
            kernel_type: "learned", "gaussian", or "exponential"
            track_metrics: Track metastability and order parameter metrics
            store_visualization_traces: Cache per-step phases/order parameters for visualization tooling
            visualization_history_limit: Max number of simulation steps to cache for visualization
            use_stuart_landau: Use full Stuart-Landau dynamics (amplitude + phase)
            use_heun: Use Heun integrator (RK2) instead of Euler
            track_cdns: Track CDNS metrics (Consonance, Dissonance, Noise, Signal)
            use_analytical_variance: Enable analytical variance computation from theory
            analytical_freq_dist: Frequency distribution for analytical theory ("gaussian" or "lorentzian")
            analytical_dist_params: Parameters for frequency distribution
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model
        self.n_sim_steps = n_sim_steps
        self.dt = dt
        self.coupling_strength = coupling_strength
        self.use_sakaguchi = use_sakaguchi
        self.use_critical_tuning = use_critical_tuning
        self.use_coupling_kernel = use_coupling_kernel
        self.track_metrics = track_metrics
        self.store_visualization_traces = bool(store_visualization_traces)
        self.visualization_history_limit = int(max(1, visualization_history_limit))
        self._last_phases: Optional[torch.Tensor] = None
        self._last_phase_history: Optional[torch.Tensor] = None
        self._last_order_parameter_history: Optional[torch.Tensor] = None
        self._last_attention_scores: Optional[torch.Tensor] = None
        self._last_attention_distribution: Optional[torch.Tensor] = None
        self._last_coupling_matrix: Optional[torch.Tensor] = None
        self._last_amplitudes: Optional[torch.Tensor] = None
        self.use_stuart_landau = use_stuart_landau
        self.use_heun = use_heun
        self.track_cdns = track_cdns
        self.lambda_dissonance = float(lambda_dissonance)
        self.lambda_criticality = float(lambda_criticality)
        self.target_R = float(target_R)
        self.lambda_frustration = float(lambda_frustration)
        self.use_order_param_safety = bool(use_order_param_safety)
        # Clamp bounds to valid [0, 1] range and ensure min < max
        R_min_clamped = max(0.0, min(1.0, float(R_min)))
        R_max_clamped = max(R_min_clamped + 1e-3, min(1.0, float(R_max)))
        self.R_min = R_min_clamped
        self.R_max = R_max_clamped
        self.R_clamp_gain = float(max(0.0, R_clamp_gain))
        # Structural bias configuration
        self.use_structural_bias = bool(use_structural_bias)
        self.structural_bias_gain = float(structural_bias_gain)
        # Decay in [0, 1); values close to 1.0 favour long memory, 0 disables smoothing
        self.structural_bias_decay = float(min(max(structural_bias_decay, 0.0), 0.999))
        # Threshold in [-1, 1] (cosine similarity domain)
        self.structural_bias_threshold = float(max(-1.0, min(1.0, structural_bias_threshold)))
        
        # Analytical theory (optional)
        self.use_analytical_variance = bool(use_analytical_variance)
        if self.use_analytical_variance:
            try:
                from resonance_transformer.modules.kuramoto_analytics import KuramotoAnalytics
            except ImportError:
                try:
                    from modules.kuramoto_analytics import KuramotoAnalytics
                except ImportError:
                    KuramotoAnalytics = None
                    self.use_analytical_variance = False
            
            if self.use_analytical_variance and KuramotoAnalytics is not None:
                self.analytics = KuramotoAnalytics(
                    frequency_distribution=analytical_freq_dist,
                    distribution_params=analytical_dist_params
                )
            else:
                self.analytics = None
                self.use_analytical_variance = False
        else:
            self.analytics = None

        # MSF integration knobs
        self.use_msf_regularizer = bool(use_msf_regularizer)
        self.lambda_msf = float(lambda_msf)
        self.msf_target = str(msf_target)
        self.msf_n_clusters = int(msf_n_clusters)
        self.use_msf_autotune = bool(use_msf_autotune)
        self.msf_autotune_safety = float(msf_autotune_safety)
        self.msf_eval_every = int(msf_eval_every)
        
        # Hodge decomposition knobs
        self.use_hodge_decomposition = bool(use_hodge_decomposition)
        self.harmonic_leak_rate = float(harmonic_leak_rate)
        self.standard_leak_rate = float(standard_leak_rate)
        
        # Bifurcation analysis knobs
        self.detect_bifurcations = bool(detect_bifurcations) and BIFURCATION_AVAILABLE
        self.track_phase_transitions = bool(track_phase_transitions) and BIFURCATION_AVAILABLE
        
        # Time-reversible synchronization
        self.use_time_reversible = bool(use_time_reversible)
        self.tr_window_length = tr_window_length  # None means auto-compute
        self.tr_adaptive_steps = bool(tr_adaptive_steps)
        self.tr_convergence_threshold = float(tr_convergence_threshold)
        self.tr_tolerance = float(tr_tolerance)
        
        # Time-reversible requires Heun integrator
        if self.use_time_reversible and not self.use_heun:
            import warnings
            warnings.warn("Time-reversible synchronization requires Heun integrator. Enabling use_heun=True.")
            self.use_heun = True
        
        # ReLU coupling / Phase 7: Electrical/Memristive coupling
        self.coupling_type = str(coupling_type).lower()
        valid_coupling_types = ["sine", "relu", "hybrid", "electrical", "memristive"]
        if self.coupling_type not in valid_coupling_types:
            import warnings
            warnings.warn(f"Unknown coupling_type '{coupling_type}', defaulting to 'sine'. Valid types: {valid_coupling_types}")
            self.coupling_type = "sine"
        self.relu_weight = float(relu_weight)
        if self.relu_weight < 0.0 or self.relu_weight > 1.0:
            import warnings
            warnings.warn(f"relu_weight must be in [0, 1], clamping {self.relu_weight} to [0, 1]")
            self.relu_weight = max(0.0, min(1.0, self.relu_weight))
        
        self.phase_transition_threshold = float(phase_transition_threshold)
        
        if self.detect_bifurcations or self.track_phase_transitions:
            if BifurcationAnalyzer is not None:
                self.bifurcation_analyzer = BifurcationAnalyzer()
                self.order_parameter_history = []
                self.coupling_strength_history = []
                self.history_max_length = 100  # Limit history size
            else:
                self.bifurcation_analyzer = None
                self.detect_bifurcations = False
                self.track_phase_transitions = False
        else:
            self.bifurcation_analyzer = None
        
        # Stochastic dynamics knobs
        self.use_stochastic_dynamics = bool(use_stochastic_dynamics) and STOCHASTIC_DYNAMICS_AVAILABLE
        if self.use_stochastic_dynamics:
            if StochasticDynamics is not None:
                self.stochastic_dynamics = StochasticDynamics(
                    noise_type=noise_type,
                    noise_strength=noise_strength,
                    levy_alpha=levy_alpha,
                    levy_beta=levy_beta,
                    noise_correlation=noise_correlation,
                    stochastic_resonance=stochastic_resonance,
                )
            else:
                self.stochastic_dynamics = None
                self.use_stochastic_dynamics = False
        else:
            self.stochastic_dynamics = None
        
        # Dual-timescale coupling knobs
        self.use_dual_timescale = bool(use_dual_timescale)
        self.fast_tau = float(fast_tau)
        self.slow_tau = float(slow_tau)
        self.fast_coupling_strength = float(fast_coupling_strength)
        self.slow_coupling_strength = float(slow_coupling_strength)
        self.use_voltage_gating = bool(use_voltage_gating)
        
        # Memory buffer for slow coupling (initialized per batch in forward)
        self.slow_memory = None  # Will be [batch, seq_len, seq_len] when used
        # Memory buffer for fast coupling smoothing (optional)
        self.fast_memory = None  # Will be [batch, seq_len] when used
        
        # Chimera death detection knobs
        self.detect_chimera_death = bool(detect_chimera_death)
        self.amplitude_death_threshold = float(amplitude_death_threshold)
        self.coherence_sync_threshold = float(coherence_sync_threshold)
        
        # Chimera death requires Stuart-Landau (needs amplitudes)
        if self.detect_chimera_death and not self.use_stuart_landau:
            import warnings
            warnings.warn("Chimera death detection requires Stuart-Landau dynamics (use_stuart_landau=True). Disabling detect_chimera_death.")
            self.detect_chimera_death = False
        
        # Lag synchronization detection knobs
        self.detect_lag_synchronization = bool(detect_lag_synchronization)
        self.lag_similarity_threshold = float(lag_similarity_threshold)
        self.max_lag = int(max_lag)
        
        # Information flow & propagation knobs
        self.track_information_flow = bool(track_information_flow) and INFORMATION_FLOW_AVAILABLE
        if self.track_information_flow:
            if InformationFlowAnalyzer is not None:
                self.info_flow_analyzer = InformationFlowAnalyzer(
                    transfer_entropy_k=transfer_entropy_k,
                    transfer_entropy_l=transfer_entropy_l,
                    transfer_entropy_bins=transfer_entropy_bins,
                    granger_max_lag=granger_max_lag,
                    propagation_steps=propagation_steps,
                    bottleneck_threshold=bottleneck_threshold,
                )
                self.info_flow_history_length = int(info_flow_history_length)
                self.phases_history = []  # Will store phase snapshots
            else:
                self.info_flow_analyzer = None
                self.track_information_flow = False
        else:
            self.info_flow_analyzer = None
            self.phases_history = []
        
        # Adaptive coupling knobs
        self.use_adaptive_coupling = bool(use_adaptive_coupling) and ADAPTIVE_COUPLING_AVAILABLE
        self.adaptation_rate = float(adaptation_rate)
        self.adaptation_signal = str(adaptation_signal)
        self.adaptation_target = float(adaptation_target)
        self.use_gradient_adaptation = bool(use_gradient_adaptation)
        self.use_performance_adaptation = bool(use_performance_adaptation)
        self.use_local_adaptation = bool(use_local_adaptation)
        self.performance_metric = str(performance_metric)
        
        if self.use_adaptive_coupling:
            if AdaptiveCoupling is not None:
                self.adaptive_coupling = AdaptiveCoupling(
                    adaptation_rate=adaptation_rate,
                    adaptation_signal=adaptation_signal,
                    adaptation_target=adaptation_target,
                )
            else:
                self.adaptive_coupling = None
                self.use_adaptive_coupling = False
        
        if self.use_performance_adaptation:
            if PerformanceAdaptiveCoupling is not None:
                self.performance_adaptive = PerformanceAdaptiveCoupling(
                    performance_metric=performance_metric,
                    adaptation_rate=adaptation_rate,
                )
            else:
                self.performance_adaptive = None
                self.use_performance_adaptation = False
        
        # Frequency domain analysis knobs
        self.analyze_frequency_domain = bool(analyze_frequency_domain) and FREQUENCY_DOMAIN_AVAILABLE
        self.frequency_analysis_history_length = int(frequency_analysis_history_length)
        self.use_frequency_dependent_coupling = bool(use_frequency_dependent_coupling) and FREQUENCY_DOMAIN_AVAILABLE
        self.frequency_bands = frequency_bands or [0.1, 1.0, 10.0]
        self.coupling_per_band = coupling_per_band or [1.0, 0.5, 0.1]
        self.frequency_range = frequency_range
        
        if self.analyze_frequency_domain or self.use_frequency_dependent_coupling:
            if FrequencyDomainAnalyzer is not None:
                self.frequency_analyzer = FrequencyDomainAnalyzer(
                    history_length=self.frequency_analysis_history_length,
                    dt=self.dt,
                    frequency_range=self.frequency_range,
                )
                # Share phase history with information flow if both are enabled
                if not hasattr(self, 'phases_history'):
                    self.phases_history_freq = []
            else:
                self.frequency_analyzer = None
                self.analyze_frequency_domain = False
                self.use_frequency_dependent_coupling = False
        else:
            self.frequency_analyzer = None
            if not hasattr(self, 'phases_history'):
                self.phases_history_freq = []
        
        # Multi-scale structures knobs
        self.use_hierarchical_coupling = bool(use_hierarchical_coupling) and MULTISCALE_AVAILABLE
        self.local_scale = int(local_scale)
        self.global_scale = int(global_scale)
        self.local_weight = float(local_weight)
        self.global_weight = float(global_weight)
        self.use_scale_free_topology = bool(use_scale_free_topology) and MULTISCALE_AVAILABLE
        self.hub_fraction = float(hub_fraction)
        self.power_law_exponent = float(power_law_exponent)
        
        if self.use_hierarchical_coupling or self.use_scale_free_topology:
            if MultiScaleStructures is not None:
                self.multiscale_structures = MultiScaleStructures(
                    use_hierarchical_coupling=self.use_hierarchical_coupling,
                    local_scale=self.local_scale,
                    global_scale=self.global_scale,
                    local_weight=self.local_weight,
                    global_weight=self.global_weight,
                    use_scale_free_topology=self.use_scale_free_topology,
                    hub_fraction=self.hub_fraction,
                    power_law_exponent=self.power_law_exponent,
                )
            else:
                self.multiscale_structures = None
                self.use_hierarchical_coupling = False
                self.use_scale_free_topology = False
        else:
            self.multiscale_structures = None
        
        # MLPs to generate oscillator parameters
        self.W_q = nn.Linear(d_model, self.d_k)  # Query → driving frequency
        self.W_k = nn.Linear(d_model, self.d_k)  # Key → natural frequency
        self.W_v = nn.Linear(d_model, self.d_v)  # Value → amplitude
        self.W_o = nn.Linear(self.d_v, d_model)  # Output projection
        
        # Phase lag parameter for Kuramoto-Sakaguchi (learnable)
        if use_sakaguchi:
            self.phase_lag = nn.Parameter(torch.tensor(0.0))  # α in sin(θ_j - θ_i - α)
        
        # Gain parameter for Stuart-Landau dynamics (learnable excitability)
        if use_stuart_landau:
            self.gain_mlp = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Tanh()  # Gains typically in [-1, 1] range
            )
        
        # Coupling kernel
        if use_coupling_kernel:
            self.coupling_kernel = CouplingKernel(
                d_k=self.d_k,
                kernel_type=kernel_type,
                use_distance=True,
                use_spectral_gating=use_spectral_gating,
                spectral_num_bands=spectral_num_bands,
                spectral_learnable=spectral_learnable
            )
        
        # Critical coupling tuner
        if use_critical_tuning:
            self.critical_tuner = CriticalCouplingTuner(
                initial_K=coupling_strength,
                adaptive=True
            )
        
        # Scale factor
        self.scale = 1.0 / np.sqrt(self.d_k)
        
        # Delay dynamics
        self.use_delays = bool(use_delays)
        self.tau_steps = int(tau_steps)
        self.delay_gain = float(delay_gain)
        self.use_learnable_delays = bool(use_learnable_delays)
        if self.use_learnable_delays:
            # Per-token delay predictor τ_j = softplus(w^T K_j)/dt (converted to steps)
            self.tau_proj = nn.Linear(self.d_k, 1)

        # Telemetry
        self.telemetry_writer = self._init_telemetry_writer(telemetry)

        # Runtime coupling and PID/autopilot
        self.use_pid_autopilot = bool(use_pid_autopilot)
        self._K_runtime: float = float(coupling_strength)
        self.alpha_offset: float = 0.0
        # Only initialize PID if available and requested
        if self.use_pid_autopilot and PIDAutopilot is not None and PIDConfig is not None:
            try:
                self.pid = PIDAutopilot(pid_config or PIDConfig(target_R=float(target_R)))
            except Exception:
                # Fallback: disable PID if initialization fails
                self.use_pid_autopilot = False
                self.pid = None
        else:
            self.pid = None
            if self.use_pid_autopilot:
                # PID was requested but not available - disable it
                self.use_pid_autopilot = False

        # CDNS full metrics/loss knobs
        self.use_cdns_full = bool(use_cdns_full)
        self.cdns_top_m = int(cdns_top_m)
        self.use_extended_cdns = bool(use_extended_cdns) and EXTENDED_CDNS_AVAILABLE
        self.lambda_noise = float(lambda_noise)
        self.lambda_signal = float(lambda_signal)

        # Hybrid readout (resonant + QK^T)
        self.hybrid_readout = bool(hybrid_readout)
        if self.hybrid_readout:
            mix = float(hybrid_mix_init)
            mix = min(max(mix, 1e-3), 1.0 - 1e-3)
            self.hybrid_logit = nn.Parameter(torch.tensor(np.log(mix / (1.0 - mix)), dtype=torch.float32))
        else:
            self.hybrid_logit = None  # type: ignore

        # Auxiliary interference readout (path-integral-inspired)
        self.use_aux_interference = bool(use_aux_interference)
        self.aux_iters = int(aux_iters)
        self.aux_step = float(aux_step)
        self.aux_eps = float(aux_eps)
        if self.use_aux_interference:
            mix = float(aux_mix_init)
            mix = min(max(mix, 1e-4), 1.0 - 1e-4)
            self.aux_logit = nn.Parameter(torch.tensor(np.log(mix / (1.0 - mix)), dtype=torch.float32))
            self.W_auxo = nn.Linear(self.d_v, self.d_model)
        else:
            self.aux_logit = None  # type: ignore
            self.W_auxo = None  # type: ignore

        # Registry coupling kernel (optional, from modules.kernels)
        self.use_registry_kernel = bool(use_registry_kernel)
        if self.use_registry_kernel:
            try:
                if registry_kernel_type == "blend":
                    _kernels = [
                        SP_LearnedMLPKernel(self.d_k, rank=int(registry_rank)),
                        SP_GaussianDistanceKernel(learnable=True),
                        SP_AlternatingKernel(learnable=True),
                    ]
                    self.registry_kernel = SP_BlendKernel(_kernels, learnable=True, temperature=float(registry_temperature))
                elif registry_kernel_type == "learned":
                    self.registry_kernel = SP_LearnedMLPKernel(self.d_k, rank=int(registry_rank))
                elif registry_kernel_type == "gaussian":
                    self.registry_kernel = SP_GaussianDistanceKernel(learnable=True)
                elif registry_kernel_type == "alternating":
                    self.registry_kernel = SP_AlternatingKernel(learnable=True)
                else:
                    self.registry_kernel = SP_LearnedMLPKernel(self.d_k, rank=int(registry_rank))
            except Exception:
                # Fallback gracefully if registry wiring fails
                self.registry_kernel = None
                self.use_registry_kernel = False
        else:
            self.registry_kernel = None

        # Temporal multiplexing
        self.use_temporal_multiplex = bool(use_temporal_multiplex)
        self.tm_dts = [float(x) for x in (tm_dts or [])]
        self.tm_alpha_offsets = [float(x) for x in (tm_alpha_offsets or [])]
        if self.use_temporal_multiplex and len(self.tm_dts) > 1 and bool(tm_learned_mix):
            self.tm_logits = nn.Parameter(torch.zeros(len(self.tm_dts)))
        else:
            self.tm_logits = None

        # Harmonic coupling (optional)
        self.use_harmonics = bool(use_harmonics)
        # Default to simple m=2 harmonic with small κ if not provided
        if self.use_harmonics:
            h_list = harmonics if (harmonics is not None and len(harmonics) > 0) else [2.0]
            k_list = kappa_harm if (kappa_harm is not None and len(kappa_harm) > 0) else [0.05] * len(h_list)
            a_list = alpha_harm if (alpha_harm is not None and len(alpha_harm) > 0) else [0.0] * len(h_list)
            # Fixed integer-ish m as buffer; κ and α learnable
            self.register_buffer("harmonics_buf", torch.tensor([float(x) for x in h_list], dtype=torch.float32))
            self.kappa_harm = nn.Parameter(torch.tensor([float(x) for x in k_list], dtype=torch.float32))
            self.alpha_harm = nn.Parameter(torch.tensor([float(x) for x in a_list], dtype=torch.float32))
        else:
            self.harmonics_buf = None  # type: ignore
            self.kappa_harm = None  # type: ignore
            self.alpha_harm = None  # type: ignore

        # Metrics storage (for monitoring)
        self.metrics = {}
    
    def compute_coupling_matrix(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Compute coupling matrix K_ij from embeddings.
        
        If use_registry_kernel=True, delegates to modules.kernels,
        otherwise uses dot-product with optional in-head kernel modulation.
        """
        batch_size, seq_len, _ = K.shape

        # Produce base K_ij
        if getattr(self, "use_registry_kernel", False) and (getattr(self, "registry_kernel", None) is not None):
            try:
                K_ij = self.registry_kernel(Q=Q, K=K, pos=None, mask=None)  # [batch, seq_len, seq_len]
                # Validate after registry_kernel
                if torch.any(torch.isnan(K_ij)) or torch.any(torch.isinf(K_ij)):
                    raise ValueError(f"compute_coupling_matrix: K_ij contains NaN or Inf after registry_kernel")
            except Exception:
                # Fallback to dot-product if registry fails
                K_ij = torch.bmm(K, K.transpose(1, 2))
                # Validate after fallback
                if torch.any(torch.isnan(K_ij)) or torch.any(torch.isinf(K_ij)):
                    raise ValueError(f"compute_coupling_matrix: K_ij contains NaN or Inf after fallback bmm")
        else:
            # Default dot-product similarity
            K_ij = torch.bmm(K, K.transpose(1, 2))  # [batch, seq_len, seq_len]
            # Validate after bmm
            if torch.any(torch.isnan(K_ij)) or torch.any(torch.isinf(K_ij)):
                raise ValueError(f"compute_coupling_matrix: K_ij contains NaN or Inf after bmm")
        
        # Scale and apply base/runtime coupling strength
        scale_K = self._K_runtime if hasattr(self, "_K_runtime") else self.coupling_strength
        
        # Validate scale parameters
        if torch.isnan(torch.tensor(self.scale)) or torch.isinf(torch.tensor(self.scale)):
            raise ValueError(f"compute_coupling_matrix: self.scale is NaN or Inf (value={self.scale})")
        if torch.isnan(torch.tensor(scale_K)) or torch.isinf(torch.tensor(scale_K)):
            raise ValueError(f"compute_coupling_matrix: scale_K is NaN or Inf (value={scale_K})")
        
        K_ij = K_ij * self.scale * scale_K
        
        # Validate after scaling
        if torch.any(torch.isnan(K_ij)) or torch.any(torch.isinf(K_ij)):
            raise ValueError(f"compute_coupling_matrix: K_ij contains NaN or Inf after scaling")
        
        # Apply in-head coupling kernel modulation only for builtin path
        if self.use_coupling_kernel and not getattr(self, "use_registry_kernel", False):
            K_ij = self.coupling_kernel(K_ij, seq_len)
            # Validate after coupling_kernel
            if torch.any(torch.isnan(K_ij)) or torch.any(torch.isinf(K_ij)):
                raise ValueError(f"compute_coupling_matrix: K_ij contains NaN or Inf after coupling_kernel")
        
        # Apply Hodge decomposition with differential damping if enabled
        if self.use_hodge_decomposition:
            # Decompose coupling into Hodge components
            hodge = hodge_decompose_coupling(K_ij, return_harmonic_dim=False)
            
            # Apply differential leak rates (damping)
            # Harmonic component: tiny leak rate (preserves long-term memory)
            harmonic_K = hodge['harmonic'] * (1.0 - self.harmonic_leak_rate * self.dt)
            
            # Standard component (exact + coexact): normal leak rate (fast decay)
            standard_K = hodge['exact'] * (1.0 - self.standard_leak_rate * self.dt)
            # Note: coexact is currently zero, but included for completeness
            coexact_K = hodge['coexact'] * (1.0 - self.standard_leak_rate * self.dt)
            
            # Combine components
            K_ij = harmonic_K + standard_K + coexact_K
            
            # Validate after Hodge decomposition
            if torch.any(torch.isnan(K_ij)) or torch.any(torch.isinf(K_ij)):
                raise ValueError(f"compute_coupling_matrix: K_ij contains NaN or Inf after Hodge decomposition")
        
        return K_ij
    
    def compute_order_parameter(self, phases: torch.Tensor, 
                               amplitudes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute synchronization order parameter R.
        
        R = |(1/N) Σ_j a_j exp(iθ_j)| where a_j are amplitudes.
        
        R ∈ [0, 1]:
        - R ≈ 0: No synchronization (phases random)
        - R ≈ 1: Full synchronization (all phases aligned)
        - R ≈ 0.5-0.7: Critical point (metastable, optimal for computation)
        
        Args:
            phases: Oscillator phases [batch, seq_len]
            amplitudes: Optional amplitudes [batch, seq_len]
            
        Returns:
            Order parameter R [batch]
        """
        batch_size, seq_len = phases.shape
        
        if amplitudes is None:
            amplitudes = torch.ones_like(phases) / seq_len
        
        # Convert phases to complex numbers
        complex_phases = amplitudes * torch.exp(1j * phases)  # [batch, seq_len]
        
        # Sum over oscillators
        R_complex = torch.sum(complex_phases, dim=-1)  # [batch]
        
        # Take magnitude
        R = torch.abs(R_complex)  # [batch]
        
        return R
    
    def compute_metastability_metrics(self, phases_history: torch.Tensor,
                                     order_params_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute metastability metrics to track system state near criticality.
        
        Metrics:
        - Phase coherence variance: Measures stability of synchronization
        - Order parameter variance: Tracks fluctuations near critical point
        - Phase velocity variance: Measures dynamic complexity
        
        Args:
            phases_history: Phase history [batch, n_steps, seq_len]
            order_params_history: Order parameter history [batch, n_steps]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Order parameter variance (high variance = metastable)
        metrics['order_param_variance'] = torch.var(order_params_history, dim=1)
        
        # Phase coherence: variance of phase differences
        if phases_history.shape[1] > 1:
            phase_diffs = phases_history[:, 1:] - phases_history[:, :-1]  # [batch, n_steps-1, seq_len]
            phase_coherence = torch.mean(torch.cos(phase_diffs), dim=-1)  # [batch, n_steps-1]
            metrics['phase_coherence_variance'] = torch.var(phase_coherence, dim=1)
            metrics['mean_phase_coherence'] = torch.mean(phase_coherence, dim=1)
        else:
            metrics['phase_coherence_variance'] = torch.zeros(phases_history.shape[0], device=phases_history.device)
            metrics['mean_phase_coherence'] = torch.ones(phases_history.shape[0], device=phases_history.device)
        
        # Criticality index: how close to critical point (R ≈ 0.6)
        target_R = 0.6
        R_distance = torch.abs(order_params_history - target_R)
        metrics['criticality_index'] = 1.0 / (1.0 + R_distance.mean(dim=1))  # Higher = closer to critical
        
        return metrics
    
    def _phase_cluster_assignments(self, phases: torch.Tensor, K: Optional[int] = None) -> torch.Tensor:
        """
        Fast angular binning into K clusters.
        Args:
            phases: [batch, seq_len] in radians
            K: number of clusters (defaults to self.msf_n_clusters)
        Returns:
            [batch, seq_len] long tensor with cluster ids in [0, K-1]
        """
        if phases.dim() != 2:
            phases = phases.view(phases.shape[0], -1)
        Kc = int(K or max(1, self.msf_n_clusters))
        Kc = int(max(1, Kc))
        # Normalize to [0, 2π) then to [0,1)
        norm = torch.remainder(phases, 2.0 * np.pi) / (2.0 * np.pi)
        ids = torch.clamp(torch.floor(norm * Kc), 0, Kc - 1).to(torch.long)
        return ids

    def _compute_msf_signals(self, coupling_matrix: torch.Tensor, phases: torch.Tensor) -> Optional[Dict]:
        """
        Compute MSF signals for current state. Returns a dict or None on failure/unavailable.
        """
        if not MSF_AVAILABLE:
            return None
        try:
            B, N = phases.shape
            if N <= 1:
                return None
            # Build assignments by angular bins
            assignments = self._phase_cluster_assignments(phases, self.msf_n_clusters)  # [B, N]
            # Analyzer expects [B,N,N] K and [B,N] assignments
            analyzer = MSFAnalyzer()
            # Effective alpha (scalar)
            alpha_eff = 0.0
            if getattr(self, "use_sakaguchi", False) and hasattr(self, "phase_lag"):
                try:
                    alpha_eff = float(self.phase_lag.detach().item())
                except Exception:
                    alpha_eff = 0.0
            try:
                alpha_eff = float(alpha_eff + float(getattr(self, "alpha_offset", 0.0)))
            except Exception:
                pass
            dt_eff = float(self.dt)
            # Use sigma=1.0 so magnitude resides in K itself
            msf_out = analyzer.compute_transverse_lyapunov_exponents(
                coupling_matrix=coupling_matrix,
                cluster_assignments=assignments,
                dt=dt_eff,
                sigma=1.0,
                alpha=alpha_eff,
            )
            regimes = analyzer.predict_dynamical_regime(msf_out['Lambda'])
            sigma_crit = analyzer.predict_critical_coupling(
                coupling_matrix=coupling_matrix,
                cluster_assignments=assignments,
                dt=dt_eff,
                alpha=alpha_eff,
            )
            return {
                'Lambda': msf_out['Lambda'],
                'lambda_max': msf_out['lambda_max'],
                'stable_flags': msf_out['stable_flags'],
                'regimes': regimes,
                'sigma_crit': sigma_crit,
                'gamma': msf_out.get('gamma', None),
                'cos_alpha': msf_out.get('cos_alpha', None),
                'alpha_used': alpha_eff,
                'dt_used': dt_eff,
            }
        except Exception:
            return None

    def _msf_regularizer_loss(self, msf: Dict, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build per-batch MSF loss tensor based on target regime.
        """
        if not isinstance(msf, dict) or ("Lambda" not in msf):
            return torch.zeros(batch, device=device, dtype=dtype)
        Lambda_list = msf.get("Lambda", [])
        # Build per-batch losses
        losses = []
        for b in range(batch):
            lam_dict = Lambda_list[b] if (b < len(Lambda_list)) else {}
            vals = list(lam_dict.values())
            if len(vals) == 0:
                losses.append(0.0)
                continue
            if self.msf_target == "sync":
                # penalize positive Λ (unstable transverse)
                l = sum([max(0.0, float(v)) for v in vals]) / len(vals)
            elif self.msf_target == "async":
                # penalize negative Λ
                l = sum([max(0.0, -float(v)) for v in vals]) / len(vals)
            elif self.msf_target == "chimera":
                neg_frac = sum([1 for v in vals if float(v) < 0.0]) / len(vals)
                l = abs(neg_frac - 0.5)  # prefer mixture
            else:  # "critical": push toward boundary
                l = sum([abs(float(v)) for v in vals]) / len(vals)
            losses.append(float(l))
        return torch.tensor(losses, device=device, dtype=dtype)

    def _time_reversible_integrate(
        self,
        phases: torch.Tensor,
        amplitudes: Optional[torch.Tensor],
        coupling_matrix: torch.Tensor,
        natural_frequencies: torch.Tensor,
        dt: float,
        n_steps: int,
        gains: torch.Tensor,
        phase_lag: Optional[torch.Tensor] = None,
        harmonics: Optional[torch.Tensor] = None,
        kappa_harm: Optional[torch.Tensor] = None,
        alpha_harm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Time-reversible integration using forward-backward iteration.
        
        Algorithm:
        1. Run forward n_steps steps
        2. Run backward n_steps steps (reverse time)
        3. Check reversibility error
        4. Return forward result with metrics
        
        Args:
            phases: Initial phases [batch, seq_len]
            amplitudes: Initial amplitudes [batch, seq_len] or None
            coupling_matrix: Coupling matrix [batch, seq_len, seq_len]
            natural_frequencies: Natural frequencies [batch, seq_len]
            dt: Time step
            n_steps: Number of forward/backward steps
            gains: Coupling gains [batch, seq_len, seq_len]
            phase_lag: Optional phase lag for Sakaguchi [batch, seq_len, seq_len]
            harmonics: Optional harmonics tensor
            kappa_harm: Optional harmonics coupling
            alpha_harm: Optional harmonics phase
            
        Returns:
            phases_final: Final phases after reversible integration
            amplitudes_final: Final amplitudes (or None)
            metrics: Dictionary with reversibility metrics
        """
        batch_size, seq_len = phases.shape
        device = phases.device
        dtype = phases.dtype
        
        # Store initial state
        phases_init = phases.clone()
        amplitudes_init = amplitudes.clone() if amplitudes is not None else None
        
        # Forward integration
        phases_fwd = phases.clone()
        amplitudes_fwd = amplitudes.clone() if amplitudes is not None else None
        
        for step in range(n_steps):
            if self.use_stuart_landau and self.use_heun:
                phases_fwd, amplitudes_fwd = heun_step_torch(
                    phases_fwd, amplitudes_fwd, gains, coupling_matrix,
                    natural_frequencies, dt, phase_lag=phase_lag,
                    clamp_max=1.0,
                    harmonics=harmonics, kappa_harm=kappa_harm, alpha_harm=alpha_harm
                )
            else:
                # Fallback to Euler if not using Heun
                # This is a simplified version - full implementation would need Euler step
                raise NotImplementedError("Time-reversible integration requires Heun integrator")
        
        # Backward integration (reverse time: dt -> -dt)
        phases_bwd = phases_fwd.clone()
        amplitudes_bwd = amplitudes_fwd.clone() if amplitudes_fwd is not None else None
        
        for step in range(n_steps):
            if self.use_stuart_landau and self.use_heun:
                phases_bwd, amplitudes_bwd = heun_step_torch(
                    phases_bwd, amplitudes_bwd, gains, coupling_matrix,
                    natural_frequencies, -dt, phase_lag=phase_lag,  # Negative dt!
                    clamp_max=1.0,
                    harmonics=harmonics, kappa_harm=kappa_harm, alpha_harm=alpha_harm
                )
            else:
                raise NotImplementedError("Time-reversible integration requires Heun integrator")
        
        # Reversibility error
        reversibility_error = torch.norm(phases_bwd - phases_init, dim=-1)  # [batch]
        if amplitudes_init is not None and amplitudes_bwd is not None:
            amp_error = torch.norm(amplitudes_bwd - amplitudes_init, dim=-1)  # [batch]
            reversibility_error = reversibility_error + amp_error
        
        metrics = {
            'reversibility_error': reversibility_error.mean().item(),
            'reversibility_error_per_batch': reversibility_error,  # [batch]
            'n_iterations': 1,
        }
        
        # Warn if error is large
        if reversibility_error.mean() > 1e-3:
            metrics['warning'] = 'Large reversibility error'
        
        return phases_fwd, amplitudes_fwd, metrics

    def _time_reversible_adaptive(
        self,
        phases: torch.Tensor,
        amplitudes: Optional[torch.Tensor],
        coupling_matrix: torch.Tensor,
        natural_frequencies: torch.Tensor,
        dt: float,
        max_steps: int,
        convergence_threshold: float,
        gains: torch.Tensor,
        phase_lag: Optional[torch.Tensor] = None,
        harmonics: Optional[torch.Tensor] = None,
        kappa_harm: Optional[torch.Tensor] = None,
        alpha_harm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Adaptive time-reversible integration with convergence detection.
        
        Stops when reversibility error is below threshold or max_steps reached.
        
        Args:
            phases: Initial phases [batch, seq_len]
            amplitudes: Initial amplitudes [batch, seq_len] or None
            coupling_matrix: Coupling matrix [batch, seq_len, seq_len]
            natural_frequencies: Natural frequencies [batch, seq_len]
            dt: Time step
            max_steps: Maximum number of steps
            convergence_threshold: Threshold for convergence
            gains: Coupling gains [batch, seq_len, seq_len]
            phase_lag: Optional phase lag for Sakaguchi
            harmonics: Optional harmonics tensor
            kappa_harm: Optional harmonics coupling
            alpha_harm: Optional harmonics phase
            
        Returns:
            phases_final: Final phases
            amplitudes_final: Final amplitudes (or None)
            metrics: Dictionary with convergence metrics
        """
        phases_current = phases.clone()
        amplitudes_current = amplitudes.clone() if amplitudes is not None else None
        
        for n_steps in range(1, max_steps + 1):
            # Forward-backward iteration
            phases_fwd, amplitudes_fwd, tr_metrics = self._time_reversible_integrate(
                phases_current, amplitudes_current, coupling_matrix,
                natural_frequencies, dt, n_steps, gains,
                phase_lag=phase_lag,
                harmonics=harmonics, kappa_harm=kappa_harm, alpha_harm=alpha_harm
            )
            
            # Check convergence (using change in phases)
            phase_change = torch.norm(phases_fwd - phases_current, dim=-1).mean()
            
            if phase_change < convergence_threshold:
                # Converged!
                return phases_fwd, amplitudes_fwd, {
                    'reversibility_error': tr_metrics['reversibility_error'],
                    'n_steps_used': n_steps,
                    'converged': True,
                    'phase_change': phase_change.item(),
                }
            
            phases_current = phases_fwd
            amplitudes_current = amplitudes_fwd
        
        # Max steps reached
        return phases_current, amplitudes_current, {
            'reversibility_error': tr_metrics.get('reversibility_error', float('inf')),
            'n_steps_used': max_steps,
            'converged': False,
            'phase_change': phase_change.item() if 'phase_change' in locals() else float('inf'),
        }

    
    def _compute_coupling_force(
        self,
        phases: torch.Tensor,  # [batch, seq_len]
        coupling_matrix: torch.Tensor,  # [batch, seq_len, seq_len]
        phase_lag: Optional[torch.Tensor] = None,
        amplitudes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute coupling force matrix based on coupling_type.
        
        Dispatches to appropriate coupling computation:
        - "sine": Standard Kuramoto-Sakaguchi coupling
        - "relu": ReLU coupling (asymmetric)
        - "hybrid": Mix of sine and ReLU
        - "electrical": Direct diffusive coupling (Phase 7)
        - "memristive": Field coupling via memristor/flux (Phase 7)
        
        Args:
            phases: Current phase values [batch, seq_len]
            coupling_matrix: Coupling strength matrix [batch, seq_len, seq_len]
            phase_lag: Optional phase lag (for Sakaguchi)
            amplitudes: Optional amplitude values (needed for memristive coupling)
            
        Returns:
            coupling_force_matrix: [batch, seq_len, seq_len] coupling force matrix
                                  (to be weighted by amplitudes and summed)
        """
        # Phase differences: θ_j - θ_i
        phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)  # [batch, seq_len, seq_len]
        
        # Apply phase lag if using Sakaguchi
        if phase_lag is not None:
            if phase_lag.dim() == 0:
                phase_diff_shifted = phase_diff - phase_lag
            else:
                phase_diff_shifted = phase_diff - phase_lag.unsqueeze(-1)
        else:
            phase_diff_shifted = phase_diff
        
        if self.coupling_type == "sine":
            # Standard sine coupling (Kuramoto-Sakaguchi)
            sin_coupling = torch.sin(phase_diff_shifted)
            coupling_force_matrix = coupling_matrix * sin_coupling
            return coupling_force_matrix
        
        elif self.coupling_type == "relu":
            # ReLU coupling: only positive contributions
            sin_coupling = torch.sin(phase_diff_shifted)
            relu_coupling = torch.relu(sin_coupling)
            coupling_force_matrix = coupling_matrix * relu_coupling
            return coupling_force_matrix
        
        elif self.coupling_type == "hybrid":
            # Hybrid coupling: mix of sine and ReLU
            sin_coupling = torch.sin(phase_diff_shifted)
            relu_coupling = torch.relu(sin_coupling)
            hybrid_coupling = (1.0 - self.relu_weight) * sin_coupling + self.relu_weight * relu_coupling
            coupling_force_matrix = coupling_matrix * hybrid_coupling
            return coupling_force_matrix
        
        elif self.coupling_type == "electrical":
            # Electrical coupling: direct diffusive coupling
            return self.compute_electrical_coupling_force(
                phases, coupling_matrix, phase_lag
            )
        
        elif self.coupling_type == "memristive":
            # Memristive (field) coupling: coupling via memristor/flux variable
            if amplitudes is None:
                amplitudes = torch.ones_like(phases)  # Default amplitude
            return self.compute_memristive_coupling_force(
                phases, amplitudes, coupling_matrix, phase_lag
            )
        
        else:
            raise ValueError(f"Unknown coupling_type: {self.coupling_type}")

    def compute_fast_coupling(
        self,
        phases: torch.Tensor,  # [batch, seq_len]
        coupling_matrix: torch.Tensor,  # [batch, seq_len, seq_len]
        phase_lag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fast coupling (AMPA-like): immediate response, no memory.
        
        Fast coupling responds immediately to current phase differences.
        No temporal memory - purely instantaneous.
        
        Args:
            phases: Current phase values [batch, seq_len]
            coupling_matrix: Coupling strength matrix [batch, seq_len, seq_len]
            phase_lag: Optional phase lag (for Sakaguchi)
            
        Returns:
            fast_coupling_force: [batch, seq_len] coupling force
        """
        # Phase differences: θ_j - θ_i
        phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)  # [batch, seq_len, seq_len]
        
        # Apply phase lag if using Sakaguchi
        if phase_lag is not None:
            if phase_lag.dim() == 0:
                phase_diff_shifted = phase_diff - phase_lag
            else:
                phase_diff_shifted = phase_diff - phase_lag.unsqueeze(-1)
        else:
            phase_diff_shifted = phase_diff
        
        # Fast coupling: immediate response (sine coupling)
        sin_coupling = torch.sin(phase_diff_shifted)
        fast_force_matrix = coupling_matrix * sin_coupling
        
        # Sum over oscillators (to be weighted by amplitudes later)
        fast_force = fast_force_matrix.sum(dim=-1)  # [batch, seq_len]
        
        return fast_force

    def compute_slow_coupling(
        self,
        phases: torch.Tensor,  # [batch, seq_len]
        coupling_matrix: torch.Tensor,  # [batch, seq_len, seq_len]
        tau_slow: float,
        dt: float,
        memory_buffer: Optional[torch.Tensor] = None,
        phase_lag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Slow coupling (NMDA-like): delayed, sustained response with memory.
        
        Slow coupling has:
        - Temporal memory (exponential decay with tau_slow)
        - Sustained response (long decay time)
        
        Args:
            phases: Current phase values [batch, seq_len]
            coupling_matrix: Coupling strength matrix [batch, seq_len, seq_len]
            tau_slow: Slow decay time constant
            dt: Time step
            memory_buffer: Previous slow coupling state [batch, seq_len, seq_len]
            phase_lag: Optional phase lag (for Sakaguchi)
        
        Returns:
            slow_coupling_force: [batch, seq_len] coupling force
            updated_memory_buffer: [batch, seq_len, seq_len] updated memory
        """
        batch_size, seq_len = phases.shape
        
        # Phase differences: θ_j - θ_i
        phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)  # [batch, seq_len, seq_len]
        
        # Apply phase lag if using Sakaguchi
        if phase_lag is not None:
            if phase_lag.dim() == 0:
                phase_diff_shifted = phase_diff - phase_lag
            else:
                phase_diff_shifted = phase_diff - phase_lag.unsqueeze(-1)
        else:
            phase_diff_shifted = phase_diff
        
        # Current phase coupling
        sin_coupling = torch.sin(phase_diff_shifted)
        current_coupling = coupling_matrix * sin_coupling  # [batch, seq_len, seq_len]
        
        # Update memory buffer with exponential decay
        if memory_buffer is None:
            memory_buffer = current_coupling.clone()
        else:
            # Exponential decay: memory = memory * exp(-dt/tau) + current * (1 - exp(-dt/tau))
            decay_factor = math.exp(-dt / max(tau_slow, 1e-6))
            memory_buffer = memory_buffer * decay_factor + current_coupling * (1 - decay_factor)
        
        # Slow coupling force from memory (sum over oscillators)
        slow_force = memory_buffer.sum(dim=-1)  # [batch, seq_len]
        
        return slow_force, memory_buffer

    def compute_dual_timescale_coupling(
        self,
        phases: torch.Tensor,  # [batch, seq_len]
        coupling_matrix: torch.Tensor,  # [batch, seq_len, seq_len]
        dt: float,
        slow_memory: Optional[torch.Tensor] = None,
        phase_lag: Optional[torch.Tensor] = None,
        amplitudes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combined fast + slow coupling (dual-timescale).
        
        Args:
            phases: Current phase values [batch, seq_len]
            coupling_matrix: Coupling strength matrix [batch, seq_len, seq_len]
            dt: Time step
            slow_memory: Previous slow coupling memory buffer [batch, seq_len, seq_len]
            phase_lag: Optional phase lag (for Sakaguchi)
            amplitudes: Optional amplitudes for voltage-dependent gating [batch, seq_len]
        
        Returns:
            total_coupling_force: [batch, seq_len] combined coupling force
            updated_slow_memory: [batch, seq_len, seq_len] updated slow memory buffer
        """
        # Fast coupling (immediate) with optional smoothing via fast_tau
        fast_raw = self.compute_fast_coupling(phases, coupling_matrix, phase_lag)
        if getattr(self, "fast_tau", None) is not None and self.fast_tau > 0.0:
            decay_fast = math.exp(-dt / max(self.fast_tau, 1e-6))
            if self.fast_memory is None or (self.fast_memory.shape != fast_raw.shape):
                self.fast_memory = fast_raw.clone()
            else:
                self.fast_memory = self.fast_memory * decay_fast + fast_raw * (1.0 - decay_fast)
            fast_force = self.fast_memory
        else:
            fast_force = fast_raw
        
        # Slow coupling (sustained)
        slow_force, updated_memory = self.compute_slow_coupling(
            phases, coupling_matrix, self.slow_tau, dt, slow_memory, phase_lag
        )
        # Persist slow memory for external access if needed
        self.slow_memory = updated_memory
        
        # Optional voltage-dependent gating for slow coupling
        if self.use_voltage_gating and amplitudes is not None:
            # Voltage gate: sigmoid((amplitude - threshold) * scale)
            # Low amplitude: blocked (Mg²⁺ block), High amplitude: unblocked
            mg_block_threshold = 0.5
            voltage_gate = torch.sigmoid((amplitudes - mg_block_threshold) * 10.0)
            slow_force = slow_force * voltage_gate
        
        # Combine fast and slow coupling
        total_force = self.fast_coupling_strength * fast_force + self.slow_coupling_strength * slow_force
        
        return total_force, updated_memory

    def compute_mean_phase_difference(
        self,
        phases: torch.Tensor,  # [batch, seq_len]
        coupling_matrix: torch.Tensor,  # [batch, seq_len, seq_len]
    ) -> torch.Tensor:
        """
        Compute mean phase difference (MPD) between coupled oscillators.
        
        MPD = mean(|θ_i - θ_j|) for coupled pairs (i,j)
        
        MPD reveals subtle synchronization regimes:
        - MPD ≈ 0: Perfect synchrony
        - MPD ≈ π: Anti-phase synchrony
        - MPD random: Asynchronous
        
        Args:
            phases: Phase values [batch, seq_len]
            coupling_matrix: Coupling matrix [batch, seq_len, seq_len]
        
        Returns:
            mpd: [batch] mean phase difference
        """
        batch_size, seq_len = phases.shape
        
        # Phase differences for all pairs
        phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)  # [batch, seq_len, seq_len]
        
        # Mask to only coupled pairs
        coupled_mask = coupling_matrix > 1e-6  # [batch, seq_len, seq_len]
        
        # Wrap to [-π, π]
        phase_diff_wrapped = torch.atan2(
            torch.sin(phase_diff),
            torch.cos(phase_diff)
        )
        
        # Mean phase difference (absolute value) for coupled pairs only
        mpd_abs = torch.abs(phase_diff_wrapped) * coupled_mask.float()
        coupled_count = coupled_mask.sum(dim=(-2, -1)).clamp(min=1)
        mpd = mpd_abs.sum(dim=(-2, -1)) / coupled_count  # [batch]
        
        return mpd
    
    def detect_steady_state(
        self,
        amplitudes: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Detect oscillators in steady state (amplitude → 0).
        
        Args:
            amplitudes: [batch, seq_len] amplitude values
            threshold: Amplitude threshold for "dead" state (default: self.amplitude_death_threshold)
        
        Returns:
            is_dead: [batch, seq_len] boolean mask indicating dead oscillators
        """
        if threshold is None:
            threshold = self.amplitude_death_threshold
        return amplitudes < threshold
    
    def detect_chimera_death(
        self,
        phases: torch.Tensor,
        amplitudes: torch.Tensor,
        coupling_matrix: torch.Tensor,
        amplitude_threshold: Optional[float] = None,
        coherence_threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Detect chimera death: synchronized cluster + dead oscillators.
        
        Chimera death occurs when some oscillators synchronize while others
        reach steady state (amplitude → 0). This provides natural sparse attention.
        
        Args:
            phases: [batch, seq_len] phase values
            amplitudes: [batch, seq_len] amplitude values
            coupling_matrix: [batch, seq_len, seq_len] coupling matrix
            amplitude_threshold: Amplitude threshold for death (default: self.amplitude_death_threshold)
            coherence_threshold: Coherence threshold for synchronization (default: self.coherence_sync_threshold)
        
        Returns:
            Dictionary with:
            - 'is_dead': [batch, seq_len] dead oscillator mask
            - 'is_synchronized': [batch, seq_len] synchronized oscillator mask
            - 'chimera_death_ratio': [batch] fraction in chimera death state
            - 'death_ratio': [batch] fraction of dead oscillators
            - 'sync_ratio': [batch] fraction of synchronized oscillators
            - 'local_coherence': [batch, seq_len] local coherence values
        """
        if amplitude_threshold is None:
            amplitude_threshold = self.amplitude_death_threshold
        if coherence_threshold is None:
            coherence_threshold = self.coherence_sync_threshold
        
        batch_size, seq_len = phases.shape
        device = phases.device
        dtype = phases.dtype
        
        # Detect dead oscillators (steady state)
        is_dead = self.detect_steady_state(amplitudes, amplitude_threshold)
        
        # Detect synchronized oscillators (via phase coherence)
        # Compute local order parameter for each oscillator
        phase_coords = torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1)  # [batch, seq_len, 2]
        
        # Local coherence: coherence with neighbors (coupled oscillators)
        local_coherence = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        
        for i in range(seq_len):
            # Neighbors (coupled oscillators) - use coupling matrix
            neighbors = coupling_matrix[:, i] > 1e-6  # [batch, seq_len]
            
            # Coherence with neighbors
            neighbor_phases = phase_coords * neighbors.unsqueeze(-1)  # [batch, seq_len, 2]
            neighbor_sum = neighbor_phases.sum(dim=1)  # [batch, 2]
            neighbor_count = neighbors.sum(dim=-1, keepdim=True).clamp(min=1)  # [batch, 1]
            neighbor_mean = neighbor_sum / neighbor_count  # [batch, 2]
            
            # Local coherence = norm of mean phase vector
            local_coherence[:, i] = torch.norm(neighbor_mean, dim=-1)  # [batch]
        
        is_synchronized = local_coherence > coherence_threshold
        
        # Chimera death: synchronized + dead coexist
        # Synchronized but not dead = chimera death cluster
        chimera_death_mask = is_synchronized & (~is_dead)  # [batch, seq_len]
        
        # Ratios
        death_ratio = is_dead.float().mean(dim=-1)  # [batch]
        sync_ratio = is_synchronized.float().mean(dim=-1)  # [batch]
        chimera_death_ratio = chimera_death_mask.float().mean(dim=-1)  # [batch]
        
        return {
            'is_dead': is_dead,
            'is_synchronized': is_synchronized,
            'chimera_death_ratio': chimera_death_ratio,
            'death_ratio': death_ratio,
            'sync_ratio': sync_ratio,
            'local_coherence': local_coherence,
        }
    
    def analyze_higher_order_coupling(
        self,
        coupling_matrix: torch.Tensor,
        phases: torch.Tensor,
        amplitudes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze 1-simplex (pairwise) vs. 2-simplex (triplet) coupling.
        
        For now, approximate 2-simplex via harmonic coupling terms if available.
        This helps understand how higher-order coupling affects chimera death.
        
        Args:
            coupling_matrix: [batch, seq_len, seq_len] coupling matrix (1-simplex)
            phases: [batch, seq_len] phase values
            amplitudes: [batch, seq_len] amplitude values
        
        Returns:
            Dictionary with:
            - 'pairwise_strength': [batch] 1-simplex coupling strength
            - 'triplet_strength': [batch] 2-simplex coupling strength (harmonic, if available)
            - 'chimera_death_ratio': [batch] fraction in chimera death state
        """
        batch_size = coupling_matrix.shape[0]
        device = coupling_matrix.device
        dtype = coupling_matrix.dtype
        
        # Pairwise (1-simplex) coupling strength
        pairwise_strength = coupling_matrix.abs().mean(dim=(-2, -1))  # [batch]
        
        # Triplet (2-simplex) coupling approximated via harmonic terms
        # If using harmonics, estimate triplet strength
        # For now, use harmonic coupling if available (check if harmonics are used)
        # This is a placeholder - full implementation would analyze harmonic coupling terms
        triplet_strength = torch.zeros(batch_size, device=device, dtype=dtype)  # Placeholder
        
        # Chimera death detection
        cd_metrics = self.detect_chimera_death(phases, amplitudes, coupling_matrix)
        
        return {
            'pairwise_strength': pairwise_strength,
            'triplet_strength': triplet_strength,
            'chimera_death_ratio': cd_metrics['chimera_death_ratio'],
            'death_ratio': cd_metrics['death_ratio'],
            'sync_ratio': cd_metrics['sync_ratio'],
        }
    
    def compute_similarity_function(
        self,
        phases: torch.Tensor,  # [batch, seq_len]
        max_lag: int = 10,
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute similarity function S(τ) for lag synchronization detection.
        
        S(τ) = <[x_i(t) - x_j(t - τ)]²> / sqrt(<x_i²(t)><x_j²(t)>)
        
        For phases: use circular distance (wrapped phase differences).
        
        Args:
            phases: Phase values [batch, seq_len]
            max_lag: Maximum lag to test
            normalize: Whether to normalize similarity function
        
        Returns:
            Dictionary with:
            - 'similarity': [batch, seq_len, seq_len, max_lag] similarity values
            - 'optimal_lag': [batch, seq_len, seq_len] optimal lag τ
            - 'min_similarity': [batch, seq_len, seq_len] minimum S(τ)
            - 'lag_sync_quality': [batch] overall lag sync quality
        """
        batch_size, n = phases.shape
        device = phases.device
        dtype = phases.dtype
        
        # Initialize similarity tensor
        similarity = torch.zeros(batch_size, n, n, max_lag, device=device, dtype=dtype)
        
        # For each lag τ
        for tau in range(max_lag):
            # Shift phases by lag (circular shift)
            phases_shifted = torch.roll(phases, shifts=tau, dims=-1)
            
            # Compute squared differences: [x_i(t) - x_j(t - τ)]²
            # For phases, use circular distance (wrapped to [-π, π])
            phase_diff = phases.unsqueeze(-2) - phases_shifted.unsqueeze(-3)  # [batch, n, n]
            
            # Wrap to [-π, π] using atan2
            phase_diff_wrapped = torch.atan2(
                torch.sin(phase_diff),
                torch.cos(phase_diff)
            )
            squared_diff = phase_diff_wrapped ** 2  # [batch, n, n]
            
            # Normalize by variance if requested
            if normalize:
                # Compute variance for normalization
                phase_var_i = phases.var(dim=-1, keepdim=True).unsqueeze(-1)  # [batch, n, 1]
                phase_var_j = phases_shifted.var(dim=-1, keepdim=True).unsqueeze(-2)  # [batch, 1, n]
                normalization = torch.sqrt(phase_var_i * phase_var_j).clamp(min=1e-6)
                similarity[:, :, :, tau] = squared_diff / normalization
            else:
                similarity[:, :, :, tau] = squared_diff
        
        # Find optimal lag (minimum similarity)
        min_similarity, optimal_lag = similarity.min(dim=-1)  # [batch, n, n]
        
        # Lag synchronization quality: average minimum similarity (lower is better)
        lag_sync_quality = min_similarity.mean(dim=(-2, -1))  # [batch]
        
        return {
            'similarity': similarity,
            'optimal_lag': optimal_lag,
            'min_similarity': min_similarity,
            'lag_sync_quality': lag_sync_quality,
        }
    
    def detect_lag_synchronization(
        self,
        phases: torch.Tensor,
        coupling_matrix: torch.Tensor,
        similarity_threshold: float = 0.1,
        max_lag: int = 10,
    ) -> Dict[str, Any]:
        """
        Detect lag synchronization between coupled oscillators.
        
        Lag synchronization: x_i(t) ≈ x_j(t - τ) for some lag τ
        
        Args:
            phases: Phase values [batch, seq_len]
            coupling_matrix: Coupling matrix [batch, seq_len, seq_len]
            similarity_threshold: Threshold for lag synchronization (lower = better sync)
            max_lag: Maximum lag to test
        
        Returns:
            Dictionary with:
            - 'is_lag_sync': [batch, seq_len, seq_len] boolean mask
            - 'optimal_lag': [batch, seq_len, seq_len] optimal lag τ
            - 'lag_sync_quality': [batch] overall quality
            - 'lag_sync_pairs': list of lag-synchronized pairs per batch
            - 'min_similarity': [batch, seq_len, seq_len] minimum S(τ)
        """
        # Compute similarity function
        sim_metrics = self.compute_similarity_function(phases, max_lag)
        
        # Detect lag synchronization: low similarity + coupling exists
        coupled_mask = coupling_matrix.abs() > 1e-6  # [batch, seq_len, seq_len]
        is_lag_sync = (sim_metrics['min_similarity'] < similarity_threshold) & coupled_mask
        
        # Extract lag-synchronized pairs
        lag_sync_pairs = []
        batch_size, n, _ = is_lag_sync.shape
        for b in range(batch_size):
            pairs_b = []
            for i in range(n):
                for j in range(i+1, n):
                    if is_lag_sync[b, i, j]:
                        pairs_b.append((i, j, sim_metrics['optimal_lag'][b, i, j].item()))
            lag_sync_pairs.append(pairs_b)
        
        return {
            'is_lag_sync': is_lag_sync,
            'optimal_lag': sim_metrics['optimal_lag'],
            'lag_sync_quality': sim_metrics['lag_sync_quality'],
            'lag_sync_pairs': lag_sync_pairs,
            'min_similarity': sim_metrics['min_similarity'],
        }
    
    def detect_complete_synchronization(
        self,
        phases: torch.Tensor,
        coupling_matrix: torch.Tensor,
        sync_threshold: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Detect complete synchronization (no lag, phases are identical).
        
        Complete synchronization: x_i(t) ≈ x_j(t) for all i, j
        
        Args:
            phases: Phase values [batch, seq_len]
            coupling_matrix: Coupling matrix [batch, seq_len, seq_len]
            sync_threshold: Threshold for complete synchronization (phase difference)
        
        Returns:
            Dictionary with:
            - 'is_complete_sync': [batch, seq_len, seq_len] boolean mask
            - 'sync_quality': [batch] overall complete sync quality
            - 'phase_diff': [batch, seq_len, seq_len] phase differences
        """
        # Phase differences: θ_j - θ_i
        phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)  # [batch, seq_len, seq_len]
        
        # Wrap to [-π, π]
        phase_diff_wrapped = torch.atan2(
            torch.sin(phase_diff),
            torch.cos(phase_diff)
        )
        
        # Complete synchronization: phase difference near zero
        phase_diff_abs = torch.abs(phase_diff_wrapped)
        coupled_mask = coupling_matrix.abs() > 1e-6  # [batch, seq_len, seq_len]
        is_complete_sync = (phase_diff_abs < sync_threshold) & coupled_mask
        
        # Complete sync quality: average phase difference (lower is better)
        sync_quality = phase_diff_abs.mean(dim=(-2, -1))  # [batch]
        
        return {
            'is_complete_sync': is_complete_sync,
            'sync_quality': sync_quality,
            'phase_diff': phase_diff_wrapped,
        }
    
    def compute_electrical_coupling_force(
        self,
        phases: torch.Tensor,
        coupling_matrix: torch.Tensor,
        phase_lag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Electrical coupling: direct diffusive coupling.
        
        Electrical coupling achieves complete synchronization over wide coupling strength range.
        Transitions from lag synchronization → complete synchronization.
        
        F_i = Σ_j K_ij * sin(θ_j - θ_i - α)
        
        For phases: uses standard Kuramoto-Sakaguchi coupling.
        
        Args:
            phases: Current phase values [batch, seq_len]
            coupling_matrix: Coupling strength matrix [batch, seq_len, seq_len]
            phase_lag: Optional phase lag α (for Sakaguchi)
        
        Returns:
            coupling_force_matrix: [batch, seq_len, seq_len] coupling force matrix
        """
        # Phase differences: θ_j - θ_i
        phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)  # [batch, seq_len, seq_len]
        
        # Apply phase lag if using Sakaguchi
        if phase_lag is not None:
            if phase_lag.dim() == 0:
                phase_diff_shifted = phase_diff - phase_lag
            else:
                phase_diff_shifted = phase_diff - phase_lag.unsqueeze(-1)
        else:
            phase_diff_shifted = phase_diff
        
        # Electrical coupling: standard sine coupling
        sin_coupling = torch.sin(phase_diff_shifted)
        coupling_force_matrix = coupling_matrix * sin_coupling
        
        return coupling_force_matrix
    
    def compute_memristive_coupling_force(
        self,
        phases: torch.Tensor,
        amplitudes: torch.Tensor,
        coupling_matrix: torch.Tensor,
        phase_lag: Optional[torch.Tensor] = None,
        flux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Memristive (field) coupling: coupling via memristor/flux variable.
        
        Memristive coupling achieves lag synchronization but NOT complete synchronization.
        Supports lag synchronization over broader range. Introduces changes in attractor shapes.
        Can produce anti-phase synchronization.
        
        Coupling is modulated by flux/amplitude variable.
        
        Args:
            phases: Current phase values [batch, seq_len]
            amplitudes: Amplitude values [batch, seq_len] (used as proxy for flux if flux not provided)
            coupling_matrix: Coupling strength matrix [batch, seq_len, seq_len]
            phase_lag: Optional phase lag α (for Sakaguchi)
            flux: Optional flux variable [batch, seq_len] (if None, uses amplitudes)
        
        Returns:
            coupling_force_matrix: [batch, seq_len, seq_len] coupling force matrix
        """
        # Phase differences: θ_j - θ_i
        phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)  # [batch, seq_len, seq_len]
        
        # Apply phase lag if using Sakaguchi
        if phase_lag is not None:
            if phase_lag.dim() == 0:
                phase_diff_shifted = phase_diff - phase_lag
            else:
                phase_diff_shifted = phase_diff - phase_lag.unsqueeze(-1)
        else:
            phase_diff_shifted = phase_diff
        
        # Memristive coupling: phase coupling modulated by flux
        sin_coupling = torch.sin(phase_diff_shifted)
        
        # Flux modulation: use flux if provided, otherwise use amplitudes
        if flux is None:
            flux = amplitudes  # Use amplitude as proxy for flux
        
        # Flux modulation: coupling strength modulated by flux product
        # Higher flux → stronger coupling modulation
        flux_modulation = flux.unsqueeze(-1) * flux.unsqueeze(-2)  # [batch, seq_len, seq_len]
        
        # Memristive coupling: coupling modulated by flux
        coupling_force_matrix = coupling_matrix * sin_coupling * flux_modulation
        
        return coupling_force_matrix

    def kuramoto_simulation(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                           coupling_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Run enhanced Kuramoto-Sakaguchi simulation to compute attention.
        
        Args:
            Q: Query embeddings [batch, seq_len, d_k]
            K: Key embeddings [batch, seq_len, d_k]
            V: Value embeddings [batch, seq_len, d_v]
            coupling_matrix: K_ij coupling matrix [batch, seq_len, seq_len]
            
        Returns:
            (attention_scores, final_phases, metrics) where:
            - attention_scores: [batch, seq_len, seq_len] phase coherence
            - final_phases: [batch, seq_len] final Key oscillator phases
            - metrics: Dictionary of metastability and order parameter metrics
        """
        # Validate inputs for NaN/Inf
        if torch.any(torch.isnan(Q)) or torch.any(torch.isinf(Q)):
            raise ValueError(f"kuramoto_simulation: Q contains NaN or Inf")
        if torch.any(torch.isnan(K)) or torch.any(torch.isinf(K)):
            raise ValueError(f"kuramoto_simulation: K contains NaN or Inf")
        if torch.any(torch.isnan(V)) or torch.any(torch.isinf(V)):
            raise ValueError(f"kuramoto_simulation: V contains NaN or Inf")
        # Check for NaN (never allowed)
        if torch.any(torch.isnan(coupling_matrix)):
            nan_count = torch.isnan(coupling_matrix).sum().item()
            raise ValueError(f"kuramoto_simulation: coupling_matrix contains NaN. NaN count: {nan_count}")
        
        # Check for +inf (not allowed, -inf from masking is OK)
        if torch.any(torch.isinf(coupling_matrix)):
            inf_mask = torch.isinf(coupling_matrix)
            inf_values = coupling_matrix[inf_mask]
            # Check if any inf values are positive (not -inf)
            pos_inf = inf_values > 0
            if torch.any(pos_inf):
                raise ValueError(f"kuramoto_simulation: coupling_matrix contains +Inf (positive infinity). -inf from masking is OK, but +inf is not allowed")
        
        batch_size, seq_len, _ = Q.shape
        
        # Reset phase history for information flow tracking at start of simulation
        if self.track_information_flow:
            self.phases_history = []
        
        # Initialize slow memory buffer for dual-timescale coupling (if enabled)
        slow_memory = None
        if self.use_dual_timescale:
            # Initialize to zero (will be populated on first step)
            slow_memory = torch.zeros(batch_size, seq_len, seq_len, device=Q.device, dtype=Q.dtype)
        
        # Initialize phases: natural frequencies from K embeddings
        # Use first dimension of K as natural frequency
        natural_frequencies = K[:, :, 0]  # [batch, seq_len]
        
        # Validate natural frequencies
        if torch.any(torch.isnan(natural_frequencies)) or torch.any(torch.isinf(natural_frequencies)):
            raise ValueError(f"kuramoto_simulation: natural_frequencies contains NaN or Inf")
        
        # Initialize phases randomly or from natural frequencies
        phases = torch.randn(batch_size, seq_len, device=Q.device) * 0.1
        phases = phases + natural_frequencies * self.dt  # Start near natural frequency
        
        # Validate initial phases
        if torch.any(torch.isnan(phases)) or torch.any(torch.isinf(phases)):
            raise ValueError(f"kuramoto_simulation: initial phases contain NaN or Inf after initialization")
        
        # Amplitudes from V embeddings (use first dimension)
        # For Stuart-Landau, start with small amplitudes that can grow
        if self.use_stuart_landau:
            amplitudes = torch.softmax(V[:, :, 0], dim=-1) * 0.1  # Start small
        else:
            amplitudes = torch.softmax(V[:, :, 0], dim=-1)  # [batch, seq_len]
        
        # Validate amplitudes
        if torch.any(torch.isnan(amplitudes)) or torch.any(torch.isinf(amplitudes)):
            raise ValueError(f"kuramoto_simulation: amplitudes contain NaN or Inf after initialization")
        if torch.any(amplitudes < 0):
            raise ValueError(f"kuramoto_simulation: amplitudes contain negative values")
        
        # Query driving frequencies (from Q embeddings)
        driving_frequencies = Q[:, :, 0]  # [batch, seq_len]
        
        # Validate driving frequencies
        if torch.any(torch.isnan(driving_frequencies)) or torch.any(torch.isinf(driving_frequencies)):
            raise ValueError(f"kuramoto_simulation: driving_frequencies contains NaN or Inf")
        
        # Gains for Stuart-Landau dynamics (learnable excitability)
        if self.use_stuart_landau:
            # Use full V embedding for gain computation
            gains = self.gain_mlp(V).squeeze(-1)  # [batch, seq_len]
            # Sanitize any accidental NaN/Inf to keep dynamics stable
            try:
                gains = torch.nan_to_num(gains, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass
            # Validate gains
            if torch.any(torch.isnan(gains)) or torch.any(torch.isinf(gains)):
                raise ValueError(f"kuramoto_simulation: gains contain NaN or Inf")
        else:
            gains = None
        
        # Apply critical coupling tuning if enabled
        if self.use_critical_tuning:
            # Initial order parameter estimate
            initial_R = self.compute_order_parameter(phases, amplitudes)
            if torch.any(torch.isnan(initial_R)) or torch.any(torch.isinf(initial_R)):
                raise ValueError(f"kuramoto_simulation: initial_R contains NaN or Inf")
            coupling_matrix = self.critical_tuner(
                coupling_matrix, 
                natural_frequencies,
                order_parameter=initial_R
            )
            # Validate coupling matrix after tuning - allow -inf from masking
            if torch.any(torch.isnan(coupling_matrix)):
                nan_count = torch.isnan(coupling_matrix).sum().item()
                raise ValueError(f"kuramoto_simulation: coupling_matrix contains NaN after critical_tuner. NaN count: {nan_count}")
            
            # Check for +inf (not allowed, -inf from masking is OK)
            if torch.any(torch.isinf(coupling_matrix)):
                inf_mask = torch.isinf(coupling_matrix)
                inf_values = coupling_matrix[inf_mask]
                pos_inf = inf_values > 0
                if torch.any(pos_inf):
                    raise ValueError(f"kuramoto_simulation: coupling_matrix contains +Inf after critical_tuner. -inf from masking is OK")
        
        # Delay line setup (per-head, uniform τ steps)
        # Support explicit delay_dt by converting to steps relative to current dt
        eff_tau_steps = int(getattr(self, "tau_steps", 0))
        try:
            dt_local = float(self.dt)
            delay_dt = float(getattr(self, "delay_dt", 0.0))
            if (delay_dt is not None) and (delay_dt > 0.0) and (dt_local > 0.0):
                eff_tau_steps = max(eff_tau_steps, int(round(delay_dt / dt_local)))
        except Exception:
            pass
        delay_line = DelayLine(
            (batch_size, seq_len),
            DelayConfig(capacity=max(2, eff_tau_steps + 4)),
            like=phases,
        ) if (((self.use_delays or getattr(self, "use_learnable_delays", False)) and eff_tau_steps > 0)) else None
        if delay_line is not None:
            delay_line.write(phases.detach())

        # History tracking for metrics
        phases_history = []
        order_params_history = []
        last_msf: Optional[Dict] = None
        safety_stats = {
            "min_R": float("inf"),
            "max_R": 0.0,
            "clamp_count": 0,
            "last_clamp_scale": 1.0,
            "last_R": None,
        }
        structural_bias_state: Optional[torch.Tensor] = None
        bias_stats = {
            "steps": 0,
            "mean_norm": 0.0,
            "max_norm": 0.0,
        }
        viz_phases_history: Optional[List[torch.Tensor]] = [] if self.store_visualization_traces else None
        viz_order_history: Optional[List[torch.Tensor]] = [] if self.store_visualization_traces else None
        
        # Time-reversible synchronization (if enabled)
        tr_metrics: Optional[Dict[str, Any]] = None
        device_label = str(phases.device)
        if self.use_time_reversible:
            tr_meta = {
                "n_steps": self.n_sim_steps,
                "dt": float(self.dt),
                "batch": batch_size,
                "seq_len": seq_len,
                "adaptive": bool(self.tr_adaptive_steps),
            }
            with self._trace_operation("kuramoto.time_reversible", device=device_label, metadata=tr_meta):
                # Prepare gains for time-reversible integration
                if gains is None:
                    # Create dummy gains if not using Stuart-Landau
                    gains_tr = torch.ones(batch_size, seq_len, device=phases.device, dtype=phases.dtype)
                else:
                    gains_tr = gains  # [batch, seq_len]
                
                # Prepare phase lag
                phase_lag = self.phase_lag if self.use_sakaguchi else None
                
                # Prepare harmonics if enabled
                if getattr(self, "use_harmonics", False) and (getattr(self, "harmonics_buf", None) is not None):
                    harm_t = self.harmonics_buf.to(device=phases.device, dtype=phases.dtype)
                    kappa_t = self.kappa_harm.to(device=phases.device, dtype=phases.dtype)  # type: ignore
                    alpha_t = self.alpha_harm.to(device=phases.device, dtype=phases.dtype)  # type: ignore
                else:
                    harm_t = None
                    kappa_t = None
                    alpha_t = None
                
                # Determine window length
                if self.tr_adaptive_steps:
                    # Adaptive time-reversible integration
                    phases_final, amplitudes_final, tr_metrics = self._time_reversible_adaptive(
                        phases, amplitudes, coupling_matrix, natural_frequencies,
                        self.dt, self.n_sim_steps, self.tr_convergence_threshold,
                        gains_tr, phase_lag=phase_lag,
                        harmonics=harm_t, kappa_harm=kappa_t, alpha_harm=alpha_t
                    )
                    phases = phases_final
                    if amplitudes_final is not None:
                        amplitudes = amplitudes_final
                else:
                    # Fixed window time-reversible integration
                    if self.tr_window_length is None:
                        # Auto-compute window length
                        initial_R = self.compute_order_parameter(phases, amplitudes).mean().item()
                        coupling_strength_avg = coupling_matrix.abs().mean().item()
                        window = compute_optimal_window_length(
                            coupling_strength_avg, initial_R, self.dt
                        )
                    else:
                        window = self.tr_window_length
                    
                    # Run time-reversible integration
                    phases_final, amplitudes_final, tr_metrics = self._time_reversible_integrate(
                        phases, amplitudes, coupling_matrix, natural_frequencies,
                        self.dt, window, gains_tr,
                        phase_lag=phase_lag,
                        harmonics=harm_t, kappa_harm=kappa_t, alpha_harm=alpha_t
                    )
                    phases = phases_final
                    if amplitudes_final is not None:
                        amplitudes = amplitudes_final
                
                # Track metrics for time-reversible
                if self.track_metrics and tr_metrics is not None:
                    # Store reversibility metrics
                    self.metrics['time_reversible'] = tr_metrics
        
        # Run simulation loop (skip if time-reversible was used)
        if not self.use_time_reversible:
            sim_meta = {
                "n_steps": self.n_sim_steps,
                "dt": float(self.dt),
                "batch": batch_size,
                "seq_len": seq_len,
            }
            with self._trace_operation("kuramoto.simulation", device=device_label, metadata=sim_meta):
                current_R_scalar: Optional[float] = None
                for step in range(self.n_sim_steps):
                    # Validate phases and amplitudes before each step
                    if torch.any(torch.isnan(phases)) or torch.any(torch.isinf(phases)):
                        raise ValueError(f"kuramoto_simulation: phases contain NaN or Inf at step {step} (before update)")
                    if torch.any(torch.isnan(amplitudes)) or torch.any(torch.isinf(amplitudes)):
                        raise ValueError(f"kuramoto_simulation: amplitudes contain NaN or Inf at step {step} (before update)")
                    
                    # Track history for metrics/visualization
                    need_snapshot = self.track_metrics or self.store_visualization_traces
                    phase_snapshot = phases.clone().detach() if need_snapshot else None
                    if self.track_metrics:
                        if phase_snapshot is not None:
                            phases_history.append(phase_snapshot)
                        R = self.compute_order_parameter(phases, amplitudes)
                        if torch.any(torch.isnan(R)) or torch.any(torch.isinf(R)):
                            raise ValueError(f"kuramoto_simulation: order parameter R contains NaN or Inf at step {step}")
                        order_params_history.append(R.clone().detach())
                        R_current = R
                    else:
                        R_current = self.compute_order_parameter(phases, amplitudes)

                    if self.store_visualization_traces:
                        if viz_phases_history is not None and phase_snapshot is not None:
                            viz_phases_history.append(phase_snapshot)
                            if len(viz_phases_history) > self.visualization_history_limit:
                                viz_phases_history.pop(0)
                        if viz_order_history is not None and isinstance(R_current, torch.Tensor):
                            viz_order_history.append(R_current.clone().detach())
                            if len(viz_order_history) > self.visualization_history_limit:
                                viz_order_history.pop(0)

                    try:
                        current_R_scalar = float(R_current.mean().detach().cpu().item())
                    except Exception:
                        current_R_scalar = float(R_current.mean().detach().item())
                    safety_stats["min_R"] = min(safety_stats["min_R"], current_R_scalar)
                    safety_stats["max_R"] = max(safety_stats["max_R"], current_R_scalar)

                    # Optional telemetry emit (JSONL)
                    if self.telemetry_writer is not None:
                        try:
                            C_val = float(R_current.mean().detach().item())
                        except Exception:
                            C_val = 0.0
                        try:
                            D_val = float(xy_energy_torch(phases, coupling_matrix).mean().detach().item())
                        except Exception:
                            D_val = 0.0
                        self.telemetry_writer.emit(
                            {
                                "dt": float(self.dt),
                                "step": int(step),
                                "order_param": C_val,
                                "xy_energy": D_val,
                                "batch": batch_size,
                                "seq_len": seq_len,
                                "device": device_label,
                            },
                            event_type="simulation.step",
                        )
                
                # Use Stuart-Landau + Heun if enabled, otherwise fall back to original
                # MSF evaluation and optional autotune
                if MSF_AVAILABLE and (self.use_msf_autotune or self.use_msf_regularizer) and (self.msf_eval_every > 0) and (step % self.msf_eval_every == 0):
                    with self._trace_operation("msf.evaluate", device=device_label, metadata={"autotune": bool(self.use_msf_autotune)}):
                        try:
                            last_msf = self._compute_msf_signals(coupling_matrix, phases)
                            if self.use_msf_autotune and isinstance(last_msf, dict):
                                sigmas = []
                                for d in last_msf.get("sigma_crit", []):
                                    if isinstance(d, dict):
                                        for v in d.values():
                                            try:
                                                vf = float(v)
                                                if math.isfinite(vf) and vf > 0.0:
                                                    sigmas.append(vf)
                                            except Exception:
                                                pass
                                if len(sigmas) > 0:
                                    sigma_min = min(sigmas)
                                    prev_K = float(getattr(self, "_K_runtime", 1.0))
                                    new_K = float(prev_K * self.msf_autotune_safety * sigma_min)
                                    # Avoid degenerate scales
                                    new_K = float(max(1e-6, min(new_K, 1e6)))
                                    scale_adj = float(new_K / max(1e-6, prev_K))
                                    self._K_runtime = float(new_K)
                                    coupling_matrix = coupling_matrix * scale_adj
                        except Exception:
                            pass

                # PID Autopilot: adjust coupling (K) and excitability offset (alpha) using R and var(R)
                if getattr(self, "use_pid_autopilot", False) and self.pid is not None:
                    with self._trace_operation("pid.autopilot", device=device_label, metadata={"target_R": float(self.target_R)}):
                        try:
                            R_mean = float(self.compute_order_parameter(phases, amplitudes).mean().detach().item())
                        except Exception:
                            R_mean = 0.0
                        if len(order_params_history) > 1:
                            varR_t = torch.var(torch.stack(order_params_history, dim=1), dim=1).mean()
                            varR_f = float(varR_t.detach().item())
                        else:
                            varR_f = 0.0
                        prev_K = self._K_runtime
                        dK, d_alpha = self.pid.update(R=R_mean, varR=varR_f)
                        self._K_runtime = float(max(0.0, prev_K + dK))
                        scale_adj = (self._K_runtime / max(1e-6, prev_K)) if prev_K != 0.0 else 1.0
                        coupling_matrix = coupling_matrix * scale_adj
                        # Alpha control: offset (default) or direct phase_lag nudging
                        alpha_mode = str(getattr(self, "alpha_control_mode", "offset"))
                        if bool(getattr(self, "use_alpha_controller", False)) and hasattr(self, "phase_lag") and (alpha_mode == "phase_lag"):
                            try:
                                # Small bounded step to keep stability
                                step = torch.tensor(d_alpha, device=self.phase_lag.device, dtype=self.phase_lag.dtype).clamp(-0.05, 0.05)
                                with torch.no_grad():
                                    self.phase_lag.add_(step)
                            except Exception:
                                pass
                        else:
                            self.alpha_offset = float(self.alpha_offset + d_alpha)
                
                # Adaptive coupling: adapt coupling during simulation
                if self.use_adaptive_coupling and step > 0:
                    adapt_meta = {"performance": bool(self.use_performance_adaptation)}
                    with self._trace_operation("adaptive.coupling", device=device_label, metadata=adapt_meta):
                        try:
                            # Performance-driven adaptation (if enabled)
                            if self.use_performance_adaptation and self.performance_adaptive is not None:
                                # Get current performance metric (if available from metrics dict)
                                # For now, use order parameter as proxy if performance not available
                                current_R = self.compute_order_parameter(phases, amplitudes)
                                current_performance = current_R.mean().item()
                                coupling_matrix = self.performance_adaptive.adapt_based_on_performance(
                                    coupling_matrix,
                                    current_performance,
                                )
                            
                            # Gradient-based adaptation (if enabled)
                            elif self.use_gradient_adaptation and adapt_coupling_gradient is not None:
                                if self.adaptation_signal == "order_parameter":
                                    # Define metric function for gradient-based adaptation
                                    def metric_fn(K, ph):
                                        # Simple order parameter computation
                                        complex_phases = torch.exp(1j * ph)
                                        R = torch.abs(complex_phases.mean(dim=-1))
                                        return R.mean()
                                    
                                    coupling_matrix = adapt_coupling_gradient(
                                        coupling_matrix,
                                        phases,
                                        metric_fn,
                                        target_value=self.adaptation_target,
                                        learning_rate=self.adaptation_rate,
                                        n_iterations=3,  # Few iterations for efficiency
                                    )
                            
                            # Local adaptation (if enabled)
                            elif self.use_local_adaptation and self.adaptive_coupling is not None:
                                # Compute local order parameters for each oscillator
                                # Local R_i = |Σ_j K_ij exp(iθ_j)| / Σ_j |K_ij|
                                batch_size, seq_len = phases.shape
                                complex_phases = torch.exp(1j * phases)  # [batch, seq_len]
                                local_order_params = torch.zeros(batch_size, seq_len, device=phases.device, dtype=phases.dtype)
                                
                                # Replace -inf with 0 for coupling matrix (masked positions)
                                coupling_safe = coupling_matrix.clone()
                                coupling_safe[torch.isinf(coupling_safe) & (coupling_safe < 0)] = 0.0
                                
                                for i in range(seq_len):
                                    # Neighbors (coupled oscillators) - check if any batch has neighbors
                                    neighbors = coupling_safe[:, i] > 1e-6  # [batch, seq_len]
                                    # Sum of complex phases weighted by coupling
                                    weighted_sum = (coupling_safe[:, i].unsqueeze(-1) * complex_phases).sum(dim=-1)  # [batch]
                                    coupling_sum = coupling_safe[:, i].sum(dim=-1)  # [batch]
                                    local_R = torch.abs(weighted_sum) / (coupling_sum + 1e-8)  # [batch]
                                    local_order_params[:, i] = local_R
                                
                                coupling_matrix = self.adaptive_coupling.adapt_coupling_local(
                                    coupling_matrix,
                                    phases,
                                    local_order_params,
                                    target_local_R=self.adaptation_target,
                                )
                            
                            # Feedback-based adaptation (default)
                            elif self.adaptive_coupling is not None:
                                # Compute current metric for adaptation
                                if self.adaptation_signal == "order_parameter":
                                    current_R = self.compute_order_parameter(phases, amplitudes)
                                    current_metric = current_R.mean().item()
                                else:
                                    current_metric = None
                                
                                if current_metric is not None:
                                    coupling_matrix = self.adaptive_coupling.adapt_coupling_feedback(
                                        coupling_matrix,
                                        current_metric,
                                        self.adaptation_target,
                                    )
                        except Exception:
                            pass  # Fail silently if adaptation fails
            
                # Phase safety enforcement (keep system within critical window)
                if self.use_order_param_safety and current_R_scalar is not None:
                    clamp_applied = False
                    clamp_scale = 1.0
                    if current_R_scalar > self.R_max:
                        overshoot = current_R_scalar - self.R_max
                        clamp_scale = 1.0 - self.R_clamp_gain * overshoot
                        clamp_scale = max(clamp_scale, 0.1)
                        clamp_applied = clamp_scale < 0.999
                    elif current_R_scalar < self.R_min:
                        undershoot = self.R_min - current_R_scalar
                        clamp_scale = 1.0 + self.R_clamp_gain * undershoot
                        clamp_scale = min(clamp_scale, 5.0)
                        clamp_applied = clamp_scale > 1.001
                    if clamp_applied:
                        coupling_matrix = coupling_matrix * clamp_scale
                        if hasattr(self, "_K_runtime"):
                            try:
                                self._K_runtime = float(max(0.0, self._K_runtime * clamp_scale))
                            except Exception:
                                pass
                        safety_stats["clamp_count"] += 1
                        safety_stats["last_clamp_scale"] = clamp_scale
                        safety_stats["last_R"] = current_R_scalar

                # Structural bias injection using instantaneous coherence
                if self.use_structural_bias and self.structural_bias_gain != 0.0:
                    phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)
                    coherence = torch.cos(phase_diff)
                    if self.structural_bias_threshold > -1.0:
                        bias_mask = (coherence > self.structural_bias_threshold).float()
                    else:
                        bias_mask = torch.ones_like(coherence)
                    bias_increment = coherence * bias_mask
                    bias_increment = bias_increment - torch.diag_embed(torch.diagonal(bias_increment, dim1=-2, dim2=-1))
                    if structural_bias_state is None or structural_bias_state.shape != bias_increment.shape:
                        structural_bias_state = bias_increment
                    else:
                        structural_bias_state = (
                            self.structural_bias_decay * structural_bias_state.detach()
                            + (1.0 - self.structural_bias_decay) * bias_increment
                        )
                    bias_to_apply = 0.5 * (structural_bias_state + structural_bias_state.transpose(-1, -2))
                    bias_to_apply = self.structural_bias_gain * bias_to_apply
                    finite_mask = torch.isfinite(coupling_matrix).float()
                    bias_to_apply = bias_to_apply * finite_mask
                    coupling_matrix = coupling_matrix + bias_to_apply
                    try:
                        bias_norm = float(bias_to_apply.abs().mean().detach().cpu().item())
                        bias_stats["steps"] += 1
                        bias_stats["mean_norm"] += bias_norm
                        bias_stats["max_norm"] = max(bias_stats["max_norm"], bias_norm)
                    except Exception:
                        pass

            # Store phase history for information flow tracking
            if self.track_information_flow:
                if not hasattr(self, 'phases_history') or self.phases_history is None:
                    self.phases_history = []
                self.phases_history.append(phases.clone().detach())
                # Limit history size
                if len(self.phases_history) > self.info_flow_history_length:
                    self.phases_history.pop(0)
            
            gains_eff = gains + (self.alpha_offset if getattr(self, "use_pid_autopilot", False) else 0.0)
            if self.use_stuart_landau and self.use_heun:
                # Full Stuart-Landau dynamics with Heun integrator
                phase_lag = self.phase_lag if self.use_sakaguchi else None
                # Prepare harmonic params if enabled
                if getattr(self, "use_harmonics", False) and (getattr(self, "harmonics_buf", None) is not None):
                    harm_t = self.harmonics_buf.to(device=phases.device, dtype=phases.dtype)
                    kappa_t = self.kappa_harm.to(device=phases.device, dtype=phases.dtype)  # type: ignore
                    alpha_t = self.alpha_harm.to(device=phases.device, dtype=phases.dtype)  # type: ignore
                else:
                    harm_t = None
                    kappa_t = None
                    alpha_t = None
                phases, amplitudes = heun_step_torch(
                    phases, amplitudes, gains_eff, coupling_matrix,
                    natural_frequencies, self.dt,
                    phase_lag=phase_lag,
                    clamp_max=1.0,
                    harmonics=harm_t, kappa_harm=kappa_t, alpha_harm=alpha_t
                )
                
                # Validate after heun_step
                if torch.any(torch.isnan(phases)) or torch.any(torch.isinf(phases)):
                    raise ValueError(f"kuramoto_simulation: phases contain NaN or Inf at step {step} after heun_step")
                if torch.any(torch.isnan(amplitudes)) or torch.any(torch.isinf(amplitudes)):
                    raise ValueError(f"kuramoto_simulation: amplitudes contain NaN or Inf at step {step} after heun_step")
                
                # Dual-timescale coupling contribution (AMPA fast + NMDA slow)
                if self.use_dual_timescale:
                    phase_lag = self.phase_lag if self.use_sakaguchi else None
                    dual_force, slow_memory = self.compute_dual_timescale_coupling(
                        phases, coupling_matrix, self.dt, slow_memory, phase_lag, amplitudes
                    )
                    # Integrate additional dual-timescale force
                    phases = (phases + self.dt * dual_force) % (2 * np.pi)

                # Add query driving forces (not in standard Stuart-Landau)
                query_phase_diff = driving_frequencies.unsqueeze(-1) - phases.unsqueeze(-2)  # [batch, seq_len, seq_len]
                query_forces = torch.sum(
                    torch.sin(query_phase_diff) * amplitudes.unsqueeze(-2),
                    dim=-1
                ) * 0.1  # Small coupling to query
                phases = (phases + self.dt * query_forces) % (2 * np.pi)
                
                # Validate after query forces
                if torch.any(torch.isnan(phases)) or torch.any(torch.isinf(phases)):
                    raise ValueError(f"kuramoto_simulation: phases contain NaN or Inf at step {step} after query_forces")

                # Learnable fractional delays (additive) if enabled
                if delay_line is not None and getattr(self, "use_learnable_delays", False):
                    try:
                        tau_raw = self.tau_proj(K).squeeze(-1)  # [batch, seq_len]
                        # convert to steps with softplus; clip to ring buffer range
                        ds = torch.clamp(torch.nn.functional.softplus(tau_raw) / max(1e-6, self.dt),
                                         0.0, float(delay_line.capacity - 1))
                        theta_tau = delay_line.read_fractional(ds)  # [batch, seq_len]
                        diff = theta_tau.unsqueeze(-2) - phases.unsqueeze(-1)  # [batch, seq_len, seq_len]
                        s = torch.sin(diff)
                        # Replace -inf with 0 to avoid NaN from -inf * 0 operations
                        coupling_matrix_safe = coupling_matrix.clone()
                        coupling_matrix_safe[torch.isinf(coupling_matrix_safe) & (coupling_matrix_safe < 0)] = 0.0
                        frac_force = torch.sum(coupling_matrix_safe * s, dim=-1)  # [batch, seq_len]
                        phases = (phases + self.dt * self.delay_gain * frac_force) % (2 * np.pi)
                    except Exception:
                        pass
                # Uniform integer delay (additive) if enabled
                if delay_line is not None and self.use_delays and (eff_tau_steps > 0):
                    d_force = uniform_delayed_coupling_force(phases, delay_line, coupling_matrix, eff_tau_steps)
                    phases = (phases + self.dt * self.delay_gain * d_force) % (2 * np.pi)
                
                # Apply stochastic noise if enabled (after Heun step)
                if self.use_stochastic_dynamics and self.stochastic_dynamics is not None:
                    phases, amplitudes = self.stochastic_dynamics.apply_noise(phases, amplitudes, self.dt)
                
            elif self.use_stuart_landau:
                # Stuart-Landau with Euler
                phase_lag = self.phase_lag if self.use_sakaguchi else None
                # Prepare harmonic params if enabled
                if getattr(self, "use_harmonics", False) and (getattr(self, "harmonics_buf", None) is not None):
                    harm_t = self.harmonics_buf.to(device=phases.device, dtype=phases.dtype)
                    kappa_t = self.kappa_harm.to(device=phases.device, dtype=phases.dtype)  # type: ignore
                    alpha_t = self.alpha_harm.to(device=phases.device, dtype=phases.dtype)  # type: ignore
                else:
                    harm_t = None
                    kappa_t = None
                    alpha_t = None
                phase_force, amplitude_force = stuart_landau_rhs_torch(
                    phases, amplitudes, gains_eff, coupling_matrix, phase_lag,
                    harmonics=harm_t, kappa_harm=kappa_t, alpha_harm=alpha_t
                )
                
                # Validate forces
                if torch.any(torch.isnan(phase_force)) or torch.any(torch.isinf(phase_force)):
                    raise ValueError(f"kuramoto_simulation: phase_force contains NaN or Inf at step {step}")
                if torch.any(torch.isnan(amplitude_force)) or torch.any(torch.isinf(amplitude_force)):
                    raise ValueError(f"kuramoto_simulation: amplitude_force contains NaN or Inf at step {step}")
                
                phases = (phases + self.dt * (natural_frequencies + phase_force)) % (2 * np.pi)
                amplitudes = amplitudes + self.dt * amplitude_force
                amplitudes = soft_clamp_torch(amplitudes, clamp_max=1.0)
                amplitudes = torch.clamp(amplitudes, min=0.0)
                
                # Validate after Euler step
                if torch.any(torch.isnan(phases)) or torch.any(torch.isinf(phases)):
                    raise ValueError(f"kuramoto_simulation: phases contain NaN or Inf at step {step} after Euler step")
                if torch.any(torch.isnan(amplitudes)) or torch.any(torch.isinf(amplitudes)):
                    raise ValueError(f"kuramoto_simulation: amplitudes contain NaN or Inf at step {step} after Euler step")
                
                # Dual-timescale coupling contribution (AMPA fast + NMDA slow)
                if self.use_dual_timescale:
                    phase_lag = self.phase_lag if self.use_sakaguchi else None
                    dual_force, slow_memory = self.compute_dual_timescale_coupling(
                        phases, coupling_matrix, self.dt, slow_memory, phase_lag, amplitudes
                    )
                    # Integrate additional dual-timescale force
                    phases = (phases + self.dt * dual_force) % (2 * np.pi)

                # Add query driving forces
                query_phase_diff = driving_frequencies.unsqueeze(-1) - phases.unsqueeze(-2)
                query_forces = torch.sum(
                    torch.sin(query_phase_diff) * amplitudes.unsqueeze(-2),
                    dim=-1
                ) * 0.1
                phases = (phases + self.dt * query_forces) % (2 * np.pi)
                
                # Validate after query forces
                if torch.any(torch.isnan(phases)) or torch.any(torch.isinf(phases)):
                    raise ValueError(f"kuramoto_simulation: phases contain NaN or Inf at step {step} after query_forces (Euler)")
                
                # Apply stochastic noise if enabled (after Euler step)
                if self.use_stochastic_dynamics and self.stochastic_dynamics is not None:
                    phases, amplitudes = self.stochastic_dynamics.apply_noise(phases, amplitudes, self.dt)
                
            else:
                # Original Kuramoto-Sakaguchi dynamics
                phase_forces = torch.zeros_like(phases)
                
                # Replace -inf with 0 to avoid NaN from -inf * 0 operations
                coupling_matrix_safe = coupling_matrix.clone()
                coupling_matrix_safe[torch.isinf(coupling_matrix_safe) & (coupling_matrix_safe < 0)] = 0.0
                
                phase_lag = self.phase_lag if self.use_sakaguchi else None
                
                # Use dual-timescale coupling if enabled
                if self.use_dual_timescale:
                    # Dual-timescale coupling (fast + slow)
                    # Note: amplitudes are passed for voltage gating, but coupling force is not pre-weighted
                    coupling_force, slow_memory = self.compute_dual_timescale_coupling(
                        phases, coupling_matrix_safe, self.dt, slow_memory, phase_lag, amplitudes
                    )
                    # Weight by amplitudes (same as standard coupling)
                    coupling_sum = coupling_force * amplitudes
                else:
                    # Standard coupling force computation
                    # Pass amplitudes for memristive coupling (Phase 7)
                    coupling_force_matrix = self._compute_coupling_force(
                        phases, coupling_matrix_safe, phase_lag, amplitudes
                    )
                    
                    # Sum coupling forces over all oscillators (weighted by amplitudes)
                    coupling_sum = torch.sum(
                        coupling_force_matrix * amplitudes.unsqueeze(1),
                        dim=-1
                    )  # [batch, seq_len]
                
                # Query driving forces: sin(ω_Q_i - θ_j) for each query-key pair
                query_phase_diff = driving_frequencies.unsqueeze(2) - phases.unsqueeze(1)  # [batch, seq_len, seq_len]
                query_forces = torch.sum(
                    torch.sin(query_phase_diff) * amplitudes.unsqueeze(1),
                    dim=-1
                )  # [batch, seq_len]
                
                # Total force
                phase_forces = coupling_sum + query_forces
                
                # Integrate phases: dθ/dt = ω + forces
                # Validate phase_forces before integration
                if torch.any(torch.isnan(phase_forces)) or torch.any(torch.isinf(phase_forces)):
                    raise ValueError(f"kuramoto_simulation: phase_forces contain NaN or Inf at step {step}")
                
                phases = phases + self.dt * (natural_frequencies + phase_forces)
                
                # Wrap phases to [0, 2π)
                phases = phases % (2 * np.pi)
                
                # Validate after integration
                if torch.any(torch.isnan(phases)) or torch.any(torch.isinf(phases)):
                    raise ValueError(f"kuramoto_simulation: phases contain NaN or Inf at step {step} after integration (Kuramoto)")

                # Learnable fractional delays (additive) if enabled
                if delay_line is not None and getattr(self, "use_learnable_delays", False):
                    try:
                        tau_raw = self.tau_proj(K).squeeze(-1)  # [batch, seq_len]
                        ds = torch.clamp(torch.nn.functional.softplus(tau_raw) / max(1e-6, self.dt),
                                         0.0, float(delay_line.capacity - 1))
                        theta_tau = delay_line.read_fractional(ds)
                        diff = theta_tau.unsqueeze(-2) - phases.unsqueeze(-1)
                        s = torch.sin(diff)
                        # Replace -inf with 0 to avoid NaN from -inf * 0 operations
                        coupling_matrix_safe = coupling_matrix.clone()
                        coupling_matrix_safe[torch.isinf(coupling_matrix_safe) & (coupling_matrix_safe < 0)] = 0.0
                        frac_force = torch.sum(coupling_matrix_safe * s, dim=-1)
                        phases = (phases + self.dt * self.delay_gain * frac_force) % (2 * np.pi)
                    except Exception:
                        pass
                # Uniform integer delay (additive) if enabled
                if delay_line is not None and self.use_delays and (eff_tau_steps > 0):
                    d_force = uniform_delayed_coupling_force(phases, delay_line, coupling_matrix, eff_tau_steps)
                    phases = (phases + self.dt * self.delay_gain * d_force) % (2 * np.pi)
            
            # Push current state into delay line
            if delay_line is not None:
                delay_line.write(phases.detach())
            
            # Final validation after all updates for this step
            if torch.any(torch.isnan(phases)) or torch.any(torch.isinf(phases)):
                raise ValueError(f"kuramoto_simulation: phases contain NaN or Inf at step {step} (end of step)")
            if torch.any(torch.isnan(amplitudes)) or torch.any(torch.isinf(amplitudes)):
                raise ValueError(f"kuramoto_simulation: amplitudes contain NaN or Inf at step {step} (end of step)")

            # Adaptive critical tuning during simulation (optional)
            if self.use_critical_tuning and step > 0 and step % 5 == 0:
                R = self.compute_order_parameter(phases, amplitudes)
                if torch.any(torch.isnan(R)) or torch.any(torch.isinf(R)):
                    raise ValueError(f"kuramoto_simulation: R contains NaN or Inf during adaptive tuning at step {step}")
                coupling_matrix = self.critical_tuner(
                    coupling_matrix,
                    natural_frequencies,
                    order_parameter=R
                )
                # Validate coupling matrix after adaptive tuning - allow -inf from masking
                if torch.any(torch.isnan(coupling_matrix)):
                    nan_count = torch.isnan(coupling_matrix).sum().item()
                    raise ValueError(f"kuramoto_simulation: coupling_matrix contains NaN after adaptive tuning at step {step}. NaN count: {nan_count}")
                
                # Check for +inf (not allowed, -inf from masking is OK)
                if torch.any(torch.isinf(coupling_matrix)):
                    inf_mask = torch.isinf(coupling_matrix)
                    inf_values = coupling_matrix[inf_mask]
                    pos_inf = inf_values > 0
                    if torch.any(pos_inf):
                        raise ValueError(f"kuramoto_simulation: coupling_matrix contains +Inf after adaptive tuning at step {step}. -inf from masking is OK")
        
        # Validate phases and amplitudes before computing attention scores
        if torch.any(torch.isnan(phases)) or torch.any(torch.isinf(phases)):
            nan_count = torch.isnan(phases).sum().item() if torch.any(torch.isnan(phases)) else 0
            raise ValueError(f"kuramoto_simulation: phases contain NaN or Inf before attention_scores computation. NaN count: {nan_count}")
        if torch.any(torch.isnan(amplitudes)) or torch.any(torch.isinf(amplitudes)):
            nan_count = torch.isnan(amplitudes).sum().item() if torch.any(torch.isnan(amplitudes)) else 0
            raise ValueError(f"kuramoto_simulation: amplitudes contain NaN or Inf before attention_scores computation. NaN count: {nan_count}")
        if torch.any(torch.isnan(driving_frequencies)) or torch.any(torch.isinf(driving_frequencies)):
            nan_count = torch.isnan(driving_frequencies).sum().item() if torch.any(torch.isnan(driving_frequencies)) else 0
            raise ValueError(f"kuramoto_simulation: driving_frequencies contain NaN or Inf before attention_scores computation. NaN count: {nan_count}")
        
        # Compute attention scores from phase coherence
        # A_ij = cos(θ_Q_i - θ_K_j) weighted by amplitudes
        query_phases = driving_frequencies * (self.n_sim_steps * self.dt)  # Query phases
        
        # Validate query_phases
        if torch.any(torch.isnan(query_phases)) or torch.any(torch.isinf(query_phases)):
            nan_count = torch.isnan(query_phases).sum().item() if torch.any(torch.isnan(query_phases)) else 0
            raise ValueError(f"kuramoto_simulation: query_phases contain NaN or Inf. NaN count: {nan_count}")
        
        attention_scores = torch.cos(
            query_phases.unsqueeze(2) - phases.unsqueeze(1)
        )  # [batch, seq_len, seq_len]
        
        # Validate attention_scores after cos
        if torch.any(torch.isnan(attention_scores)) or torch.any(torch.isinf(attention_scores)):
            nan_count = torch.isnan(attention_scores).sum().item() if torch.any(torch.isnan(attention_scores)) else 0
            raise ValueError(f"kuramoto_simulation: attention_scores contain NaN or Inf after cos. NaN count: {nan_count}")
        
        # Weight by amplitudes
        attention_scores = attention_scores * amplitudes.unsqueeze(1)
        
        # Validate attention_scores after weighting
        if torch.any(torch.isnan(attention_scores)) or torch.any(torch.isinf(attention_scores)):
            nan_count = torch.isnan(attention_scores).sum().item() if torch.any(torch.isnan(attention_scores)) else 0
            raise ValueError(f"kuramoto_simulation: attention_scores contain NaN or Inf after weighting. NaN count: {nan_count}")
        
        # Compute final metrics
        metrics = {}
        if self.use_order_param_safety:
            min_R_val = safety_stats["min_R"] if safety_stats["min_R"] != float("inf") else float("nan")
            max_R_val = safety_stats["max_R"] if safety_stats["max_R"] != float("-inf") else float("nan")
            metrics['order_param_safety'] = {
                'min_R': torch.tensor(min_R_val, device=phases.device, dtype=phases.dtype),
                'max_R': torch.tensor(max_R_val, device=phases.device, dtype=phases.dtype),
                'clamp_count': torch.tensor(float(safety_stats["clamp_count"]), device=phases.device, dtype=phases.dtype),
                'last_clamp_scale': torch.tensor(float(safety_stats["last_clamp_scale"]), device=phases.device, dtype=phases.dtype),
                'last_R': torch.tensor(
                    float(safety_stats["last_R"]) if safety_stats["last_R"] is not None else float('nan'),
                    device=phases.device,
                    dtype=phases.dtype,
                ),
            }
        if self.use_structural_bias and bias_stats["steps"] > 0:
            mean_norm = bias_stats["mean_norm"] / max(1, bias_stats["steps"])
            metrics['structural_bias'] = {
                'mean_norm': torch.tensor(mean_norm, device=phases.device, dtype=phases.dtype),
                'max_norm': torch.tensor(bias_stats["max_norm"], device=phases.device, dtype=phases.dtype),
                'steps': torch.tensor(float(bias_stats["steps"]), device=phases.device, dtype=phases.dtype),
            }
        if self.use_time_reversible and tr_metrics is not None:
            metrics['time_reversible'] = tr_metrics
        if self.track_metrics:
            final_R = self.compute_order_parameter(phases, amplitudes)
            # Expose compact time series for physics losses (Z(t), dt)
            try:
                if len(order_params_history) > 0:
                    # Complex Z_t using last tracked phases and amplitudes history
                    ph_hist = torch.stack(phases_history, dim=1)  # [batch, T, seq]
                    
                    # Validate phases_history before computing Z_time
                    if torch.any(torch.isnan(ph_hist)) or torch.any(torch.isinf(ph_hist)):
                        raise ValueError(f"kuramoto_simulation: phases_history contains NaN or Inf when computing Z_time")
                    
                    # Validate final amplitudes
                    if torch.any(torch.isnan(amplitudes)) or torch.any(torch.isinf(amplitudes)):
                        raise ValueError(f"kuramoto_simulation: amplitudes contain NaN or Inf when computing Z_time")
                    
                    amp_hist = amplitudes.unsqueeze(1).expand(-1, ph_hist.shape[1], -1)  # [batch, T, seq]
                    
                    # Validate amp_hist after expansion
                    if torch.any(torch.isnan(amp_hist)) or torch.any(torch.isinf(amp_hist)):
                        raise ValueError(f"kuramoto_simulation: amp_hist contains NaN or Inf when computing Z_time")
                    
                    # Compute complex exponential
                    exp_ph = torch.exp(1j * ph_hist)
                    if torch.any(torch.isnan(exp_ph)) or torch.any(torch.isinf(exp_ph)):
                        raise ValueError(f"kuramoto_simulation: exp(1j * ph_hist) contains NaN or Inf")
                    
                    Z_t = torch.mean(amp_hist * exp_ph, dim=-1)  # [batch, T]
                    
                    # Validate Z_t after computation
                    if torch.any(torch.isnan(Z_t)) or torch.any(torch.isinf(Z_t)):
                        raise ValueError(f"kuramoto_simulation: Z_t contains NaN or Inf after computation")
                    
                    metrics['Z_time'] = Z_t
                    # Expose raw phase trajectory for downstream analysis (e.g., covariance comparisons)
                    metrics['phases_time'] = ph_hist  # [batch, T, seq]
                    metrics['dt'] = torch.tensor(float(self.dt), device=Z_t.device, dtype=Z_t.dtype)
            except Exception as e:
                # Re-raise ValueError to preserve the diagnostic information
                if isinstance(e, ValueError):
                    raise
                # For other exceptions, wrap with context
                raise ValueError(f"kuramoto_simulation: Error computing Z_time: {type(e).__name__}: {e}") from e
            # Also store final order parameter phase angle φ for inter-head coherence
            try:
                complex_ph = amplitudes * torch.exp(1j * phases)  # [batch, seq_len]
                R_complex = torch.sum(complex_ph, dim=-1)         # [batch]
                final_phi = torch.angle(R_complex)                 # [batch]
                metrics['final_order_phase'] = final_phi
            except Exception:
                pass
            metrics['final_order_parameter'] = final_R
            # Optional harmonic energy diagnostics (higher-order coupling stack)
            try:
                if getattr(self, "use_harmonics", False) and (getattr(self, "harmonics_buf", None) is not None):
                    harm_orders = self.harmonics_buf.to(device=phases.device, dtype=phases.dtype)
                    # Safe nonnegative weights from coupling (mask -inf to 0 via relu)
                    Wpos = torch.relu(coupling_matrix)
                    theta_i = phases.unsqueeze(-1)
                    theta_j = phases.unsqueeze(-2)
                    phase_diff_ji = theta_j - theta_i  # [B, N, N]
                    energies_per_k = []
                    energy_total = torch.zeros(phases.shape[0], device=phases.device, dtype=phases.dtype)
                    for idx in range(int(harm_orders.shape[0])):
                        m_k = harm_orders[idx]
                        a_k = self.alpha_harm[idx].to(device=phases.device, dtype=phases.dtype) if getattr(self, "alpha_harm", None) is not None else torch.tensor(0.0, device=phases.device, dtype=phases.dtype)
                        cos_term = torch.cos(m_k * phase_diff_ji - a_k)
                        E_k = 0.5 * torch.sum(Wpos * (1.0 - cos_term), dim=(-2, -1))  # [B]
                        energies_per_k.append(E_k)
                        energy_total = energy_total + E_k
                    # Package metrics
                    try:
                        metrics['harmonics'] = {
                            'orders': harm_orders.detach(),
                            'energies_per_k': energies_per_k,
                            'energy_total': energy_total,
                        }
                    except Exception:
                        pass
            except Exception:
                pass
            
            if len(phases_history) > 1:
                phases_tensor = torch.stack(phases_history, dim=1)  # [batch, n_steps, seq_len]
                order_params_tensor = torch.stack(order_params_history, dim=1)  # [batch, n_steps]
                metastability_metrics = self.compute_metastability_metrics(
                    phases_tensor, order_params_tensor
                )
                metrics.update(metastability_metrics)
                
                # Analytical variance computation (if enabled)
                if self.use_analytical_variance and self.analytics is not None:
                    try:
                        # Estimate thermodynamic parameters
                        # Get average coupling strength from coupling matrix
                        K_avg = float(coupling_matrix.abs().mean().detach().item())
                        lambda_param = float(self.phase_lag.detach().item()) if self.use_sakaguchi else 0.0
                        r_inf_est = float(final_R.mean().detach().item())
                        
                        # Estimate omega_inf from natural frequencies (from K embeddings)
                        # natural_frequencies was computed earlier in the function
                        omega_inf_est = 0.0  # Default, will be estimated from phases if needed
                        
                        # Compute analytical variance
                        if r_inf_est > 0.1:  # Partially synchronized
                            analytical_var, R_ps_0 = self.analytics.compute_variance_ps(
                                K=K_avg,
                                r_inf=r_inf_est,
                                omega_inf=omega_inf_est,
                                lambda_param=lambda_param
                            )
                            seq_len = phases.shape[-1]
                            metrics['analytical_variance'] = torch.tensor(
                                analytical_var / seq_len,  # Scale by 1/N
                                device=final_R.device
                            ).expand_as(final_R)
                            metrics['analytical_R_ps_0'] = torch.tensor(
                                R_ps_0 / seq_len,
                                device=final_R.device
                            ).expand_as(final_R)
                            metrics['analytical_state'] = 'partially_synchronized'
                        else:  # Incoherent
                            analytical_var, mean_R = self.analytics.compute_variance_incoherent()
                            seq_len = phases.shape[-1]
                            metrics['analytical_variance'] = torch.tensor(
                                analytical_var / seq_len,  # Scale by 1/N
                                device=final_R.device
                            ).expand_as(final_R)
                            metrics['analytical_mean_R'] = torch.tensor(
                                mean_R / np.sqrt(seq_len),  # Mean scales as 1/√N
                                device=final_R.device
                            ).expand_as(final_R)
                            metrics['analytical_state'] = 'incoherent'
                        
                        # Comparison metrics
                        if 'order_param_variance' in metrics:
                            num_var = metrics['order_param_variance']
                            ana_var = metrics['analytical_variance']
                            metrics['variance_ratio'] = num_var / (ana_var + 1e-10)
                            metrics['variance_error'] = torch.abs(num_var - ana_var)
                            metrics['variance_relative_error'] = metrics['variance_error'] / (ana_var + 1e-10)
                    except Exception as e:
                        # Silently fail if analytical computation fails
                        pass
        
        # Compute CDNS metrics if enabled
        if self.track_cdns:
            # Consonance (order parameter) - already computed
            C = final_R if self.track_metrics else self.compute_order_parameter(phases, amplitudes)
            
            # Dissonance (XY energy)
            D = xy_energy_torch(phases, coupling_matrix)  # [batch]
            
            # Gradients for potential use in training
            C_grad = grad_order_parameter_torch(phases, amplitudes)  # [batch, seq_len]
            D_grad = xy_energy_grad_torch(phases, coupling_matrix)  # [batch, seq_len]
            
            metrics['cdns'] = {
                'consonance': C,
                'dissonance': D,
                'consonance_grad': C_grad,
                'dissonance_grad': D_grad,
            }

            # Optional full CDNS (Noise/Signal) using spectral modes
            if getattr(self, "use_cdns_full", False):
                try:
                    # Build Laplacian eigen-basis from first batch
                    K0 = coupling_matrix[0].detach()
                    d0 = torch.sum(K0, dim=1)
                    L0 = torch.diag(d0) - K0
                    try:
                        _lams0, V_eig0 = torch.linalg.eigh(L0)
                    except Exception:
                        L0_cpu = L0.to("cpu")
                        _lams0, V_eig0 = torch.linalg.eigh(L0_cpu)
                        V_eig0 = V_eig0.to(K0.device, dtype=K0.dtype)
                except Exception:
                    V_eig0 = None

                try:
                    cdns_full = compute_cdns_torch(
                        Z=V, phases=phases, weights=coupling_matrix,
                        phis=V_eig0, top_m=int(getattr(self, "cdns_top_m", 0))
                    )
                    metrics['cdns_full'] = {
                        'noise': cdns_full.N,
                        'signal': cdns_full.S,
                    }
                except Exception:
                    pass
            
            # Extended CDNS metrics (dimensionality, integration, complexity, etc.)
            if self.use_extended_cdns:
                try:
                    extended = compute_extended_cdns(
                        phases=phases,
                        amplitudes=amplitudes,
                        coupling_matrix=coupling_matrix,
                        consonance=C,
                        dissonance=D,
                        noise=metrics.get('cdns_full', {}).get('noise'),
                        signal=metrics.get('cdns_full', {}).get('signal'),
                        phases_prev=getattr(self, '_phases_prev', None)
                    )
                    metrics['extended_cdns'] = extended.to_dict()
                    # Store current phases for next flow computation
                    self._phases_prev = phases.detach()
                except Exception as e:
                    # Log error for debugging
                    import warnings
                    warnings.warn(f"Extended CDNS metrics failed: {e}")
                    # Still create empty dict so extraction doesn't fail
                    metrics['extended_cdns'] = {}
        
        # Attach MSF metrics if available or requested
        if MSF_AVAILABLE and (self.use_msf_regularizer or self.use_msf_autotune):
            try:
                if last_msf is None:
                    last_msf = self._compute_msf_signals(coupling_matrix, phases)
                if isinstance(last_msf, dict):
                    metrics['msf'] = last_msf
            except Exception:
                pass

        # Attach Hodge decomposition metrics if enabled
        if self.use_hodge_decomposition:
            try:
                hodge = hodge_decompose_coupling(coupling_matrix, return_harmonic_dim=True)
                metrics['hodge'] = {
                    'harmonic_dim': hodge['harmonic_dim'].float().mean().item(),  # Average across batch
                    'harmonic_strength': hodge['harmonic'].abs().mean().item(),
                    'standard_strength': hodge['exact'].abs().mean().item(),
                    'coexact_strength': hodge['coexact'].abs().mean().item(),
                }
            except Exception as e:
                # Log error but don't fail
                import warnings
                warnings.warn(f"Hodge metrics computation failed: {e}")
                metrics['hodge'] = {
                    'harmonic_dim': 0.0,
                    'harmonic_strength': 0.0,
                    'standard_strength': 0.0,
                    'coexact_strength': 0.0,
                }

        # Attach ReLU coupling metrics if using ReLU or hybrid coupling
        if self.coupling_type in ["relu", "hybrid"]:
            try:
                # natural_frequencies is defined at the start of kuramoto_simulation
                # Compute frequency statistics for ReLU coupling analysis
                natural_freq_mean = natural_frequencies.mean(dim=-1)  # [batch]
                natural_freq_max = natural_frequencies.max(dim=-1)[0]  # [batch]
                
                # For ReLU coupling, synchronized frequency should be closer to max
                # For sine coupling, synchronized frequency should be closer to mean
                # We track both for comparison
                metrics['relu_coupling'] = {
                    'coupling_type': self.coupling_type,
                    'relu_weight': self.relu_weight if self.coupling_type == "hybrid" else (1.0 if self.coupling_type == "relu" else 0.0),
                    'natural_freq_mean': natural_freq_mean.mean().item(),
                    'natural_freq_max': natural_freq_max.mean().item(),
                    'frequency_bias': (natural_freq_max - natural_freq_mean).mean().item(),  # Expected bias for ReLU
                }
            except Exception as e:
                # Log error but don't fail
                import warnings
                warnings.warn(f"ReLU coupling metrics computation failed: {e}")
                metrics['relu_coupling'] = {
                    'coupling_type': self.coupling_type,
                    'relu_weight': self.relu_weight if self.coupling_type == "hybrid" else (1.0 if self.coupling_type == "relu" else 0.0),
                }

        # Attach dual-timescale coupling metrics if enabled
        if self.use_dual_timescale:
            try:
                # Compute mean phase difference (MPD) metric
                mpd = self.compute_mean_phase_difference(phases, coupling_matrix)
                
                # Compute phase-locking value (PLV) for comparison
                # PLV = |<exp(i*phase_diff)>| for coupled pairs
                phase_diff = phases.unsqueeze(-1) - phases.unsqueeze(-2)  # [batch, seq_len, seq_len]
                coupled_mask = coupling_matrix > 1e-6
                exp_phase_diff = torch.exp(1j * phase_diff) * coupled_mask.float()
                plv = torch.abs(exp_phase_diff.mean(dim=(-2, -1)))  # [batch]
                
                metrics['dual_timescale'] = {
                    'fast_tau': self.fast_tau,
                    'slow_tau': self.slow_tau,
                    'fast_coupling_strength': self.fast_coupling_strength,
                    'slow_coupling_strength': self.slow_coupling_strength,
                    'use_voltage_gating': self.use_voltage_gating,
                    'mean_phase_difference': mpd.mean().item(),  # Average MPD across batch
                    'phase_locking_value': plv.mean().item(),  # Average PLV across batch
                    'mpd_plv_ratio': (mpd / (plv + 1e-6)).mean().item(),  # MPD/PLV ratio
                    'timescale_separation_index': float(self.slow_tau / (self.fast_tau + 1e-6)),
                }
            except Exception as e:
                # Log error but don't fail
                import warnings
                warnings.warn(f"Dual-timescale metrics computation failed: {e}")
                metrics['dual_timescale'] = {
                    'fast_tau': self.fast_tau,
                    'slow_tau': self.slow_tau,
                    'fast_coupling_strength': self.fast_coupling_strength,
                    'slow_coupling_strength': self.slow_coupling_strength,
                    'use_voltage_gating': self.use_voltage_gating,
                    'mean_phase_difference': 0.0,
                    'phase_locking_value': 0.0,
                    'mpd_plv_ratio': 0.0,
                }

        # Optional regularizers (per-batch) using CDNS and criticality
        # Always compute criticality loss; compute dissonance if needed
        final_R_local = metrics.get('final_order_parameter', self.compute_order_parameter(phases, amplitudes))
        
        # Validate final_R_local before computing criticality loss
        if torch.any(torch.isnan(final_R_local)) or torch.any(torch.isinf(final_R_local)):
            # Fallback to zero regularizer if R is invalid
            criticality_loss = torch.zeros(phases.shape[0], device=phases.device, dtype=phases.dtype)
        else:
            criticality_loss = (final_R_local - self.target_R) ** 2  # [batch]
        
        if self.lambda_dissonance != 0.0:
            # If CDNS already computed, reuse D; else compute quickly
            if 'cdns' in metrics and 'dissonance' in metrics['cdns']:
                dissonance_loss = metrics['cdns']['dissonance']
                # Validate dissonance_loss
                if torch.any(torch.isnan(dissonance_loss)) or torch.any(torch.isinf(dissonance_loss)):
                    dissonance_loss = torch.zeros_like(criticality_loss)
            else:
                dissonance_loss = xy_energy_torch(phases, coupling_matrix)
                # Validate computed dissonance_loss
                if torch.any(torch.isnan(dissonance_loss)) or torch.any(torch.isinf(dissonance_loss)):
                    dissonance_loss = torch.zeros_like(criticality_loss)
        else:
            dissonance_loss = torch.zeros_like(criticality_loss)
        
        # Base regularizers (criticality and dissonance)
        total_reg = self.lambda_criticality * criticality_loss + self.lambda_dissonance * dissonance_loss

        # Alpha target regularizer (learnable Sakaguchi lag control)
        alpha_reg = torch.zeros_like(criticality_loss)
        try:
            lam_alpha = float(getattr(self, "lambda_alpha", 0.0))
            alpha_tgt = float(getattr(self, "alpha_target", 0.0))
            if lam_alpha != 0.0 and getattr(self, "use_sakaguchi", False) and hasattr(self, "phase_lag"):
                alpha_err = (self.phase_lag - alpha_tgt) ** 2
                alpha_reg = lam_alpha * alpha_err.expand_as(criticality_loss)
                total_reg = total_reg + alpha_reg
        except Exception:
            pass

        # Harmonic energy regularizer (optional)
        harmonics_reg = torch.zeros_like(criticality_loss)
        try:
            lam_h = float(getattr(self, "lambda_harmonics", 0.0))
            if lam_h != 0.0 and isinstance(metrics, dict) and ("harmonics" in metrics):
                E_tot = metrics['harmonics'].get('energy_total', None)
                if E_tot is not None:
                    harmonics_reg = lam_h * E_tot
                    total_reg = total_reg + harmonics_reg
        except Exception:
            pass
        
        # MSF regularizer (optional)
        msf_loss = torch.zeros_like(criticality_loss)
        if MSF_AVAILABLE and self.use_msf_regularizer:
            try:
                if last_msf is None:
                    last_msf = self._compute_msf_signals(coupling_matrix, phases)
                if isinstance(last_msf, dict):
                    msf_loss = self._msf_regularizer_loss(last_msf, batch=phases.shape[0], device=phases.device, dtype=phases.dtype)
                    total_reg = total_reg + float(self.lambda_msf) * msf_loss
            except Exception:
                msf_loss = torch.zeros_like(criticality_loss)

        # Validate total_reg before storing
        if torch.any(torch.isnan(total_reg)) or torch.any(torch.isinf(total_reg)):
            # Fallback to zero if total_reg is invalid
            total_reg = torch.zeros_like(criticality_loss)

        # Optional valence regularizers (per-batch) using CDNS and criticality
        valence_reg = torch.zeros_like(total_reg)
        if getattr(self, "use_cdns_full", False) and ('cdns_full' in metrics):
            N_full = metrics['cdns_full']['noise']
            S_full = metrics['cdns_full']['signal']
            valence_reg = self.lambda_noise * N_full + self.lambda_signal * (1.0 - S_full)
            total_reg = total_reg + valence_reg

        metrics['regularizers'] = {
            'criticality_loss': criticality_loss,
            'dissonance_loss': dissonance_loss,
            'alpha_reg': alpha_reg,
            'harmonics_loss': harmonics_reg,
            'valence_loss': valence_reg,
            'msf_loss': msf_loss,
            'total': total_reg,
        }

        # Optional torsor/geometry regularizer (frustration alignment)
        if getattr(self, "lambda_frustration", 0.0) != 0.0:
            try:
                # features: f = [cos θ, sin θ]
                f = torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1)  # [batch, seq_len, 2]
                # rotation matrix from global phase lag
                if getattr(self, "use_sakaguchi", False) and hasattr(self, "phase_lag"):
                    psi = self.phase_lag
                    cpsi = torch.cos(psi)
                    spsi = torch.sin(psi)
                else:
                    cpsi = torch.tensor(1.0, device=phases.device, dtype=phases.dtype)
                    spsi = torch.tensor(0.0, device=phases.device, dtype=phases.dtype)
                Rm_mat = torch.stack([
                    torch.stack([cpsi, -spsi]),
                    torch.stack([spsi,  cpsi]),
                ])  # [2,2]
                # rotate f_j
                f_rot = torch.matmul(f, Rm_mat.T)  # [batch, seq_len, 2]
                # pairwise differences
                fi = f.unsqueeze(2)      # [batch, seq_len, 1, 2]
                frj = f_rot.unsqueeze(1) # [batch, 1, seq_len, 2]
                diff2 = (fi - frj).pow(2).sum(dim=-1)  # [batch, seq_len, seq_len]
                # weights from coupling (row-normalized, nonnegative)
                A = torch.relu(coupling_matrix)
                W = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
                E_frust = (W * diff2).sum(dim=(-2, -1))  # [batch]
                torsor_loss = self.lambda_frustration * E_frust
                metrics['regularizers']['torsor_loss'] = torsor_loss
                metrics['regularizers']['total'] = metrics['regularizers']['total'] + torsor_loss
            except Exception:
                pass
        
        # Cache dynamics for visualization / telemetry consumers
        try:
            self._last_phases = phases.detach().clone()
        except Exception:
            self._last_phases = None
        if amplitudes is not None:
            try:
                self._last_amplitudes = amplitudes.detach().clone()
            except Exception:
                self._last_amplitudes = None
        else:
            self._last_amplitudes = None
        if self.store_visualization_traces:
            if viz_phases_history:
                try:
                    phase_hist_tensor = torch.stack(viz_phases_history, dim=1)
                    self._last_phase_history = phase_hist_tensor.detach().clone()
                except Exception:
                    self._last_phase_history = None
            else:
                self._last_phase_history = None
            if viz_order_history:
                try:
                    order_hist_tensor = torch.stack(viz_order_history, dim=1)
                    self._last_order_parameter_history = order_hist_tensor.detach().clone()
                except Exception:
                    self._last_order_parameter_history = None
            else:
                self._last_order_parameter_history = None
            try:
                self._last_attention_scores = attention_scores.detach().clone()
            except Exception:
                self._last_attention_scores = None
            try:
                self._last_coupling_matrix = coupling_matrix.detach().clone()
            except Exception:
                self._last_coupling_matrix = None
        else:
            self._last_phase_history = None
            self._last_order_parameter_history = None
            self._last_attention_scores = None
            self._last_coupling_matrix = None

        # Store final amplitudes in metrics for chimera death detection
        if amplitudes is not None:
            metrics['final_amplitudes'] = amplitudes
        
        return attention_scores, phases, metrics
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_metrics: bool = False) -> torch.Tensor:
        """
        Forward pass of Enhanced Resonance Attention.
        
        Args:
            x: Input embeddings [batch, seq_len, d_model]
            mask: Optional attention mask [batch, seq_len, seq_len]
            return_metrics: Whether to return metastability metrics
            
        Returns:
            Output embeddings [batch, seq_len, d_model] or tuple with metrics
        """
        # Validate input x first
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            nan_count = torch.isnan(x).sum().item() if torch.any(torch.isnan(x)) else 0
            inf_count = torch.isinf(x).sum().item() if torch.any(torch.isinf(x)) else 0
            raise ValueError(f"ResonanceAttentionHead.forward: input x contains NaN or Inf. NaN count: {nan_count}, Inf count: {inf_count}")
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Generate Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_k]
        K = self.W_k(x)  # [batch, seq_len, d_k]
        V = self.W_v(x)  # [batch, seq_len, d_v]

        # Sanitize Q, K, V to prevent NaN/Inf propagation
        try:
            Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
            K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
            V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        
        # Validate Q, K, V embeddings after sanitation
        if torch.any(torch.isnan(Q)) or torch.any(torch.isinf(Q)):
            nan_count = torch.isnan(Q).sum().item() if torch.any(torch.isnan(Q)) else 0
            inf_count = torch.isinf(Q).sum().item() if torch.any(torch.isinf(Q)) else 0
            raise ValueError(f"ResonanceAttentionHead.forward: Q embeddings contain NaN or Inf. NaN count: {nan_count}, Inf count: {inf_count}")
        if torch.any(torch.isnan(K)) or torch.any(torch.isinf(K)):
            nan_count = torch.isnan(K).sum().item() if torch.any(torch.isnan(K)) else 0
            inf_count = torch.isinf(K).sum().item() if torch.any(torch.isinf(K)) else 0
            raise ValueError(f"ResonanceAttentionHead.forward: K embeddings contain NaN or Inf. NaN count: {nan_count}, Inf count: {inf_count}")
        if torch.any(torch.isnan(V)) or torch.any(torch.isinf(V)):
            nan_count = torch.isnan(V).sum().item() if torch.any(torch.isnan(V)) else 0
            inf_count = torch.isinf(V).sum().item() if torch.any(torch.isinf(V)) else 0
            raise ValueError(f"ResonanceAttentionHead.forward: V embeddings contain NaN or Inf. NaN count: {nan_count}, Inf count: {inf_count}")
        
        # Compute coupling matrix
        coupling_matrix = self.compute_coupling_matrix(Q, K)
        
        # Validate coupling matrix after computation
        if torch.any(torch.isnan(coupling_matrix)) or torch.any(torch.isinf(coupling_matrix)):
            raise ValueError(f"ResonanceAttentionHead.forward: coupling_matrix contains NaN or Inf after compute_coupling_matrix")
        
        # Apply multi-scale structures (hierarchical coupling and/or scale-free topology)
        if self.multiscale_structures is not None:
            try:
                seq_len = coupling_matrix.shape[-1]
                coupling_matrix = self.multiscale_structures.apply_to_coupling(
                    coupling_matrix,
                    seq_len,
                    device=coupling_matrix.device,
                )
            except Exception as e:
                # Don't fail on multi-scale structure errors
                if self.track_metrics:
                    metrics.setdefault('multiscale_errors', []).append(str(e))
        
        # Apply frequency-dependent coupling if enabled (requires phase history)
        if self.use_frequency_dependent_coupling and self.frequency_analyzer is not None:
            try:
                # Use shared phase history if available, otherwise use frequency-specific history
                phase_history = getattr(self, 'phases_history', None)
                if phase_history is None:
                    phase_history = getattr(self, 'phases_history_freq', [])
                
                if len(phase_history) >= 10:
                    coupling_matrix = compute_frequency_dependent_coupling(
                        coupling_matrix,
                        phase_history,
                        self.frequency_bands,
                        self.coupling_per_band,
                        self.dt,
                    )
            except Exception as e:
                # Don't fail on frequency-dependent coupling errors
                if self.track_metrics:
                    metrics.setdefault('frequency_coupling_errors', []).append(str(e))
        
        # Apply mask if provided
        if mask is not None:
            coupling_matrix = coupling_matrix.masked_fill(mask == 0, float('-inf'))
            # After masking, -inf is expected, but check for NaN
            if torch.any(torch.isnan(coupling_matrix)):
                raise ValueError(f"ResonanceAttentionHead.forward: coupling_matrix contains NaN after masking")
        
        # Final validation right before simulation (catch any issues)
        # Always check for NaN - this should never happen
        if torch.any(torch.isnan(coupling_matrix)):
            nan_count = torch.isnan(coupling_matrix).sum().item()
            raise ValueError(f"ResonanceAttentionHead.forward: coupling_matrix contains NaN before simulation. NaN count: {nan_count}")
        
        # Check for inf - handle masked vs unmasked cases
        if torch.any(torch.isinf(coupling_matrix)):
            if mask is None:
                # No mask - any inf is bad
                inf_count = torch.isinf(coupling_matrix).sum().item()
                raise ValueError(f"ResonanceAttentionHead.forward: coupling_matrix contains Inf before simulation (no mask). Inf count: {inf_count}")
            else:
                # With mask - check non-masked positions only
                non_masked = mask != 0
                if torch.any(non_masked):
                    non_masked_coupling = coupling_matrix[non_masked]
                    # Check for inf that's not -inf (which is expected from masking)
                    inf_in_non_masked = torch.isinf(non_masked_coupling)
                    if torch.any(inf_in_non_masked):
                        # Check if any inf values are not -inf
                        inf_values = non_masked_coupling[inf_in_non_masked]
                        not_neg_inf = inf_values != float('-inf')
                        if torch.any(not_neg_inf):
                            raise ValueError(f"ResonanceAttentionHead.forward: coupling_matrix contains non--inf Inf in non-masked positions before simulation")
        
        # Run enhanced Kuramoto-Sakaguchi simulation (with optional temporal multiplexing)
        final_phases = None  # Initialize for bifurcation analysis
        if getattr(self, "use_temporal_multiplex", False) and (len(getattr(self, "tm_dts", [])) > 1):
            scores_list = []
            metrics_list = []
            # Preserve runtime state
            _save_dt = self.dt
            _save_alpha = getattr(self, "alpha_offset", 0.0)
            _save_nsteps = int(self.n_sim_steps)
            try:
                M = len(self.tm_dts)
                tm_nsteps = getattr(self, "tm_nsteps", [])
                for i in range(M):
                    # Override dt, n_steps and optional alpha offset per stream
                    self.dt = float(self.tm_dts[i])
                    if isinstance(tm_nsteps, (list, tuple)) and (i < len(tm_nsteps)) and (tm_nsteps[i] is not None):
                        try:
                            self.n_sim_steps = int(tm_nsteps[i])
                        except Exception:
                            self.n_sim_steps = _save_nsteps
                    alpha_extra = float(self.tm_alpha_offsets[i]) if i < len(self.tm_alpha_offsets) else 0.0
                    self.alpha_offset = float(_save_alpha + alpha_extra)
                    # Validate coupling_matrix before each simulation call
                    if torch.any(torch.isnan(coupling_matrix)) or (torch.any(torch.isinf(coupling_matrix)) and mask is None):
                        raise ValueError(f"ResonanceAttentionHead.forward: coupling_matrix contains NaN or Inf before temporal multiplex simulation {i}")
                    s_i, _ph, m_i = self.kuramoto_simulation(Q, K, V, coupling_matrix)
                    scores_list.append(s_i)
                    metrics_list.append(m_i)
                    # Store phases from first stream for bifurcation analysis
                    if final_phases is None:
                        final_phases = _ph
            finally:
                # Restore
                self.dt = _save_dt
                self.alpha_offset = _save_alpha
                self.n_sim_steps = _save_nsteps
            # Fuse attention scores
            if self.tm_logits is not None:
                w = torch.softmax(self.tm_logits, dim=0)  # [M]
            else:
                # Optional fixed weights; else uniform
                tm_w = getattr(self, "tm_fixed_weights", None)
                if isinstance(tm_w, (list, tuple)) and len(tm_w) == len(scores_list):
                    w_raw = torch.tensor([float(x) for x in tm_w], device=Q.device, dtype=scores_list[0].dtype)
                    s = torch.clamp(w_raw.sum(), min=1e-8)
                    w = w_raw / s
                else:
                    w = torch.ones(len(scores_list), device=Q.device, dtype=scores_list[0].dtype) / max(1, len(scores_list))
            attention_scores = 0.0
            for i, s_i in enumerate(scores_list):
                attention_scores = attention_scores + w[i] * s_i
            # Pick metrics from first stream and blend key scalars if available
            metrics = metrics_list[0] if len(metrics_list) > 0 else {}
            if isinstance(metrics, dict):
                # Blend final_order_parameter
                Rs = []
                for i, m_i in enumerate(metrics_list):
                    R_i = m_i.get("final_order_parameter", None) if isinstance(m_i, dict) else None
                    if R_i is not None:
                        Rs.append((i, R_i))
                if len(Rs) == len(scores_list):
                    Rw = 0.0
                    for i, R_i in Rs:
                        Rw = Rw + w[i] * R_i  # w[i] scalar broadcasts over batch
                    metrics["final_order_parameter"] = Rw
                # Blend harmonics energy if available
                if all(isinstance(m_i, dict) and ("harmonics" in m_i) for m_i in metrics_list):
                    try:
                        Ew = 0.0
                        for i, m_i in enumerate(metrics_list):
                            Et = m_i["harmonics"].get("energy_total", None)
                            if Et is not None:
                                Ew = Ew + w[i] * Et
                        if isinstance(Ew, torch.Tensor):
                            # Use base dict and override energy_total
                            metrics.setdefault("harmonics", {})
                            metrics["harmonics"]["energy_total"] = Ew
                    except Exception:
                        pass
        else:
            # Final validation right before simulation call with detailed diagnostics
            has_nan = torch.any(torch.isnan(coupling_matrix))
            has_inf = torch.any(torch.isinf(coupling_matrix))
            if has_nan:
                nan_count = torch.isnan(coupling_matrix).sum().item()
                nan_locations = torch.nonzero(torch.isnan(coupling_matrix), as_tuple=False)
                raise ValueError(f"ResonanceAttentionHead.forward: coupling_matrix contains NaN right before kuramoto_simulation call. NaN count: {nan_count}, locations: {nan_locations[:5] if len(nan_locations) > 0 else 'none'}")
            # Check for inf only if not masked (masked positions can have -inf)
            if mask is None and has_inf:
                inf_count = torch.isinf(coupling_matrix).sum().item()
                raise ValueError(f"ResonanceAttentionHead.forward: coupling_matrix contains Inf right before kuramoto_simulation call. Inf count: {inf_count}")
            attention_scores, final_phases, metrics = self.kuramoto_simulation(Q, K, V, coupling_matrix)
        
        # Update phase history for frequency analysis
        if (self.analyze_frequency_domain or self.use_frequency_dependent_coupling) and final_phases is not None:
            try:
                if self.frequency_analyzer is not None:
                    self.frequency_analyzer.update_history(final_phases)
                # Also update shared history if it exists
                if hasattr(self, 'phases_history_freq'):
                    self.phases_history_freq.append(final_phases.clone())
                    if len(self.phases_history_freq) > self.frequency_analysis_history_length:
                        self.phases_history_freq.pop(0)
            except Exception as e:
                # Don't fail on phase history update errors
                if self.track_metrics:
                    metrics.setdefault('frequency_history_errors', []).append(str(e))
        
        # Frequency domain analysis metrics
        if return_metrics and self.analyze_frequency_domain and self.frequency_analyzer is not None:
            try:
                freq_analysis = self.frequency_analyzer.analyze()
                if freq_analysis:
                    metrics['frequency_domain'] = freq_analysis
            except Exception as e:
                # Don't fail on frequency analysis errors
                if self.track_metrics:
                    metrics.setdefault('frequency_analysis_errors', []).append(str(e))
        
        # Nested synchronization metrics (multi-scale)
        if return_metrics and self.multiscale_structures is not None and final_phases is not None:
            try:
                seq_len = final_phases.shape[-1]
                nested_sync = self.multiscale_structures.compute_nested_sync(
                    final_phases,
                    seq_len,
                    device=final_phases.device,
                )
                if nested_sync is not None:
                    metrics['nested_synchronization'] = nested_sync
            except Exception as e:
                # Don't fail on nested synchronization errors
                if self.track_metrics:
                    metrics.setdefault('nested_sync_errors', []).append(str(e))
        
        # Store metrics for external access
        if self.track_metrics:
            self.metrics = metrics
        
        # Bifurcation analysis and phase transition tracking
        if (self.detect_bifurcations or self.track_phase_transitions) and self.bifurcation_analyzer is not None:
            try:
                # Get current order parameter and coupling strength
                current_R = metrics.get('final_order_parameter', None)
                if current_R is not None:
                    # Handle batch dimension - use mean if tensor
                    if isinstance(current_R, torch.Tensor):
                        current_R_val = float(current_R.mean().detach().cpu().item())
                    else:
                        current_R_val = float(current_R)
                    
                    # Get effective coupling strength
                    if hasattr(self, '_K_runtime'):
                        current_K = float(self._K_runtime)
                    elif hasattr(self, 'coupling_strength'):
                        current_K = float(self.coupling_strength)
                    else:
                        current_K = 1.0
                    
                    # Update history
                    self.order_parameter_history.append(current_R_val)
                    self.coupling_strength_history.append(current_K)
                    
                    # Limit history size
                    if len(self.order_parameter_history) > self.history_max_length:
                        self.order_parameter_history.pop(0)
                        self.coupling_strength_history.pop(0)
                    
                    # Detect bifurcations
                    if self.detect_bifurcations and len(self.order_parameter_history) >= 10:
                        # Get final phases for bifurcation detection
                        # Phases are mainly used for shape compatibility; detection relies on order parameter history
                        if final_phases is not None:
                            phases_for_analysis = final_phases
                        else:
                            # Use dummy phases - detection mainly uses order parameter history
                            phases_for_analysis = torch.zeros(1, attention_scores.shape[-1], device=attention_scores.device)
                        
                        bifurcations = self.bifurcation_analyzer.detect_bifurcations(
                            phases_for_analysis,
                            current_K,
                            current_R_val,
                            self.order_parameter_history,
                        )
                        metrics['bifurcations'] = bifurcations
                    
                    # Track phase transitions
                    if self.track_phase_transitions and len(self.order_parameter_history) >= 2:
                        transitions = self.bifurcation_analyzer.track_phase_transition(
                            self.order_parameter_history,
                            self.coupling_strength_history,
                            self.phase_transition_threshold,
                        )
                        metrics['phase_transitions'] = transitions
            except Exception as e:
                # Don't fail on bifurcation analysis errors
                if self.track_metrics:
                    metrics.setdefault('bifurcation_errors', []).append(str(e))
        
        # Information flow & propagation metrics
        if self.track_information_flow and self.info_flow_analyzer is not None:
            try:
                if hasattr(self, 'phases_history') and len(self.phases_history) >= self.info_flow_analyzer.transfer_entropy_k + self.info_flow_analyzer.transfer_entropy_l + 1:
                    # Compute information flow metrics
                    info_flow_results = self.info_flow_analyzer.compute_information_flow(
                        self.phases_history,
                        compute_transfer_entropy=True,
                        compute_granger=True,
                    )
                    
                    # Add to metrics
                    if 'transfer_entropy' in info_flow_results:
                        transfer_matrix = info_flow_results['transfer_entropy']
                        metrics['information_flow'] = {
                            'transfer_entropy': transfer_matrix,
                            'mean_transfer': transfer_matrix.mean(dim=(-2, -1)),
                            'max_transfer': transfer_matrix.max(dim=-1)[0].max(dim=-1)[0],
                        }
                        
                        # Identify bottlenecks
                        bottlenecks = self.info_flow_analyzer.identify_bottlenecks(transfer_matrix)
                        metrics['information_flow']['bottlenecks'] = bottlenecks
                    
                    if 'granger_causality' in info_flow_results:
                        granger_matrix = info_flow_results['granger_causality']
                        if 'information_flow' not in metrics:
                            metrics['information_flow'] = {}
                        metrics['information_flow']['granger_causality'] = granger_matrix
                        metrics['information_flow']['mean_granger'] = granger_matrix.mean(dim=(-2, -1))
            except Exception as e:
                # Don't fail on information flow analysis errors
                if self.track_metrics:
                    metrics.setdefault('information_flow_errors', []).append(str(e))
        
        # Stochastic dynamics metrics
        if self.use_stochastic_dynamics and self.stochastic_dynamics is not None:
            try:
                metrics['noise'] = {
                    'noise_type': self.stochastic_dynamics.noise_type,
                    'noise_strength': self.stochastic_dynamics.noise_strength,
                    'applied': True,
                }
                
                if self.stochastic_dynamics.stochastic_resonance:
                    # Track for resonance detection
                    current_R = metrics.get('final_order_parameter', None)
                    if current_R is not None:
                        # Handle batch dimension
                        if isinstance(current_R, torch.Tensor):
                            current_R_val = float(current_R.mean().detach().cpu().item())
                        else:
                            current_R_val = float(current_R)
                        
                        self.stochastic_dynamics.sync_vs_noise_history.append({
                            'noise': self.stochastic_dynamics.noise_strength,
                            'R': current_R_val,
                        })
                        
                        # Detect resonance
                        if len(self.stochastic_dynamics.sync_vs_noise_history) >= 10:
                            order_history = [h['R'] for h in self.stochastic_dynamics.sync_vs_noise_history]
                            noise_history = [h['noise'] for h in self.stochastic_dynamics.sync_vs_noise_history]
                            resonance = self.stochastic_dynamics.detect_stochastic_resonance(
                                order_history, noise_history
                            )
                            metrics['stochastic_resonance'] = resonance
            except Exception as e:
                # Don't fail on stochastic dynamics errors
                if self.track_metrics:
                    metrics.setdefault('stochastic_dynamics_errors', []).append(str(e))
        
        # Chimera death detection metrics
        if self.detect_chimera_death and self.use_stuart_landau:
            try:
                # Need final phases and amplitudes for chimera death detection
                if final_phases is not None:
                    # Get final amplitudes from simulation
                    # Check if amplitudes are available in metrics or need to be retrieved
                    final_amplitudes = None
                    if 'final_amplitudes' in metrics:
                        final_amplitudes = metrics['final_amplitudes']
                    elif hasattr(self, '_last_amplitudes'):
                        final_amplitudes = self._last_amplitudes
                    
                    if final_amplitudes is not None:
                        # Detect chimera death
                        cd_metrics = self.detect_chimera_death(
                            final_phases,
                            final_amplitudes,
                            coupling_matrix,
                        )
                        metrics['chimera_death'] = cd_metrics
                        
                        # Higher-order coupling analysis
                        ho_metrics = self.analyze_higher_order_coupling(
                            coupling_matrix,
                            final_phases,
                            final_amplitudes,
                        )
                        metrics['higher_order_coupling'] = ho_metrics
            except Exception as e:
                # Don't fail on chimera death detection errors
                if self.track_metrics:
                    metrics.setdefault('chimera_death_errors', []).append(str(e))
        
        # Attach lag synchronization metrics if enabled
        if self.detect_lag_synchronization:
            try:
                if final_phases is not None:
                    lag_sync_metrics = self.detect_lag_synchronization(
                        final_phases,
                        coupling_matrix,
                        self.lag_similarity_threshold,
                        self.max_lag
                    )
                    
                    # Detect complete synchronization for comparison (Phase 7)
                    complete_sync_metrics = self.detect_complete_synchronization(
                        final_phases,
                        coupling_matrix,
                        sync_threshold=0.01
                    )
                    
                    metrics['lag_synchronization'] = {
                        'lag_sync_quality': lag_sync_metrics['lag_sync_quality'].mean().item(),
                        'lag_sync_quality_per_batch': lag_sync_metrics['lag_sync_quality'],
                        'optimal_lag_mean': lag_sync_metrics['optimal_lag'].float().mean().item(),
                        'min_similarity_mean': lag_sync_metrics['min_similarity'].mean().item(),
                        'lag_sync_pairs': lag_sync_metrics['lag_sync_pairs'],
                        'n_lag_sync_pairs': [len(pairs) for pairs in lag_sync_metrics['lag_sync_pairs']],
                    }
                    
                    # Phase 7: Comparison between lag sync and complete sync
                    metrics['synchronization_comparison'] = {
                        'lag_sync_quality': lag_sync_metrics['lag_sync_quality'].mean().item(),
                        'complete_sync_quality': complete_sync_metrics['sync_quality'].mean().item(),
                        'coupling_type': self.coupling_type,
                        'lag_sync_pairs_count': sum(len(pairs) for pairs in lag_sync_metrics['lag_sync_pairs']),
                        'complete_sync_pairs_count': complete_sync_metrics['is_complete_sync'].sum().item() // 2,  # Divide by 2 to avoid double counting
                        'sync_type': 'lag' if lag_sync_metrics['lag_sync_quality'].mean() < complete_sync_metrics['sync_quality'].mean() else 'complete',
                    }
            except Exception as e:
                # Don't fail on lag synchronization detection errors
                if self.track_metrics:
                    metrics.setdefault('lag_sync_errors', []).append(str(e))
        
        # Validate attention scores
        if torch.any(torch.isnan(attention_scores)) or torch.any(torch.isinf(attention_scores)):
            nan_count = torch.isnan(attention_scores).sum().item() if torch.any(torch.isnan(attention_scores)) else 0
            inf_count = torch.isinf(attention_scores).sum().item() if torch.any(torch.isinf(attention_scores)) else 0
            raise ValueError(f"ResonanceAttentionHead.forward: attention_scores contain NaN or Inf. NaN count: {nan_count}, Inf count: {inf_count}")
        
        # Resonant attention distribution with stabilized softmax
        logits = attention_scores * self.scale
        logits = torch.clamp(logits, -50.0, 50.0)
        attn_res = torch.softmax(logits, dim=-1)
        try:
            attn_res = torch.nan_to_num(attn_res, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        
        # Validate attn_res
        if torch.any(torch.isnan(attn_res)) or torch.any(torch.isinf(attn_res)):
            nan_count = torch.isnan(attn_res).sum().item() if torch.any(torch.isnan(attn_res)) else 0
            inf_count = torch.isinf(attn_res).sum().item() if torch.any(torch.isinf(attn_res)) else 0
            raise ValueError(f"ResonanceAttentionHead.forward: attn_res contains NaN or Inf. NaN count: {nan_count}, Inf count: {inf_count}")

        # Optional hybrid readout with QK^T
        if self.hybrid_readout and self.hybrid_logit is not None:
            qk_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [batch, seq_len, seq_len]
            qk_scores = torch.clamp(qk_scores, -50.0, 50.0)
            attn_qk = torch.softmax(qk_scores, dim=-1)
            try:
                attn_qk = torch.nan_to_num(attn_qk, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass
            gamma = torch.sigmoid(self.hybrid_logit)  # scalar in (0,1)
            attention_dist = (1.0 - gamma) * attn_res + gamma * attn_qk
        else:
            attention_dist = attn_res

        # Final safety: sanitize attention distribution
        try:
            attention_dist = torch.nan_to_num(attention_dist, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        
        # Validate attention_dist
        if torch.any(torch.isnan(attention_dist)) or torch.any(torch.isinf(attention_dist)):
            nan_count = torch.isnan(attention_dist).sum().item() if torch.any(torch.isnan(attention_dist)) else 0
            inf_count = torch.isinf(attention_dist).sum().item() if torch.any(torch.isinf(attention_dist)) else 0
            raise ValueError(f"ResonanceAttentionHead.forward: attention_dist contains NaN or Inf. NaN count: {nan_count}, Inf count: {inf_count}")

        if self.store_visualization_traces:
            try:
                self._last_attention_distribution = attention_dist.detach().clone()
            except Exception:
                self._last_attention_distribution = None
        else:
            self._last_attention_distribution = None

        # Apply attention to values
        output = torch.bmm(attention_dist, V)  # [batch, seq_len, d_v]
        # Pre-projection stabilization: clamp magnitude and sanitize
        try:
            output = torch.clamp(output, min=-1e4, max=1e4)
            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
        except Exception:
            pass
        
        # Validate output before projection
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            nan_count = torch.isnan(output).sum().item() if torch.any(torch.isnan(output)) else 0
            inf_count = torch.isinf(output).sum().item() if torch.any(torch.isinf(output)) else 0
            raise ValueError(f"ResonanceAttentionHead.forward: output before W_o contains NaN or Inf. NaN count: {nan_count}, Inf count: {inf_count}")
        
        # Output projection with parameter sanitization
        try:
            with torch.no_grad():
                if torch.any(torch.isnan(self.W_o.weight)) or torch.any(torch.isinf(self.W_o.weight)):
                    self.W_o.weight.data = torch.nan_to_num(self.W_o.weight.data, nan=0.0, posinf=0.0, neginf=0.0)
                if getattr(self.W_o, "bias", None) is not None and (torch.any(torch.isnan(self.W_o.bias)) or torch.any(torch.isinf(self.W_o.bias))):
                    self.W_o.bias.data = torch.nan_to_num(self.W_o.bias.data, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        output = self.W_o(output)  # [batch, seq_len, d_model]
        # Post-projection stabilization
        try:
            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
            output = torch.clamp(output, min=-1e4, max=1e4)
        except Exception:
            pass
        
        # Validate final output
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            nan_count = torch.isnan(output).sum().item() if torch.any(torch.isnan(output)) else 0
            inf_count = torch.isinf(output).sum().item() if torch.any(torch.isinf(output)) else 0
            raise ValueError(f"ResonanceAttentionHead.forward: final output contains NaN or Inf. NaN count: {nan_count}, Inf count: {inf_count}")

        # Optional auxiliary interference readout mixing
        if getattr(self, "use_aux_interference", False):
            try:
                B, T, _ = V.shape
                A = torch.relu(coupling_matrix)
                d = torch.sum(A, dim=-1)
                I = torch.eye(T, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, T, T)
                L = torch.diag_embed(d) - A
                Lp = L + float(self.aux_eps) * I
                y = torch.zeros_like(V)
                iters = max(1, int(getattr(self, "aux_iters", 3)))
                step = float(getattr(self, "aux_step", 0.1))
                for _ in range(iters):
                    y = y - step * (torch.bmm(Lp, y) - V)
                y_proj = self.W_auxo(y) if getattr(self, "W_auxo", None) is not None else y
                gamma_aux = torch.sigmoid(self.aux_logit) if getattr(self, "aux_logit", None) is not None else torch.tensor(0.0, device=output.device, dtype=output.dtype)
                output = (1.0 - gamma_aux) * output + gamma_aux * y_proj
                if self.track_metrics:
                    try:
                        self.metrics["aux_gamma"] = float(gamma_aux.detach().item())
                    except Exception:
                        pass
            except Exception:
                pass
        
        if return_metrics:
            return output, metrics
        return output

    def enable_visualization_cache(self, enabled: bool, history_limit: Optional[int] = None) -> None:
        """
        Toggle caching of per-step dynamics for downstream visualization tooling.
        """
        self.store_visualization_traces = bool(enabled)
        if history_limit is not None:
            self.visualization_history_limit = int(max(1, history_limit))

    def _prepare_export_tensor(self, tensor: Optional[torch.Tensor], detach: bool = True, cpu: bool = False) -> Optional[torch.Tensor]:
        """
        Clone/detach helper to keep cached tensors safe for downstream consumers.
        """
        if tensor is None:
            return None
        out = tensor
        if detach:
            out = out.detach()
        if cpu:
            out = out.cpu()
        if detach:
            out = out.clone()
        return out

    def get_last_phases(self, *, detach: bool = True, cpu: bool = False) -> Optional[torch.Tensor]:
        """
        Return the most recent final-phase snapshot from the head.
        """
        return self._prepare_export_tensor(self._last_phases, detach=detach, cpu=cpu)

    def get_visualization_snapshot(self, *, detach: bool = True, cpu: bool = False) -> Optional[Dict[str, Any]]:
        """
        Build a dictionary with the most recent dynamics suitable for visualization.
        """
        if self._last_phases is None:
            return None
        payload: Dict[str, Any] = {
            "phases": self._prepare_export_tensor(self._last_phases, detach=detach, cpu=cpu),
            "phase_history": self._prepare_export_tensor(self._last_phase_history, detach=detach, cpu=cpu),
            "order_parameter_history": self._prepare_export_tensor(self._last_order_parameter_history, detach=detach, cpu=cpu),
            "attention_scores": self._prepare_export_tensor(self._last_attention_scores, detach=detach, cpu=cpu),
            "attention_distribution": self._prepare_export_tensor(self._last_attention_distribution, detach=detach, cpu=cpu),
            "coupling_matrix": self._prepare_export_tensor(self._last_coupling_matrix, detach=detach, cpu=cpu),
        }
        metrics = getattr(self, "metrics", None)
        if isinstance(metrics, dict):
            final_R = metrics.get("final_order_parameter")
            if torch.is_tensor(final_R):
                payload["final_order_parameter"] = self._prepare_export_tensor(final_R, detach=detach, cpu=cpu)
            elif isinstance(final_R, (float, int)):
                payload["final_order_parameter"] = float(final_R)
            cdns = metrics.get("cdns_full") or metrics.get("cdns")
            if cdns is not None:
                payload["cdns"] = cdns
        return payload


class ResonanceTransformerBlock(nn.Module):
    """
    Complete Transformer block with Resonance Attention.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = None,
                 n_sim_steps: int = 15, dropout: float = 0.1, telemetry: bool = False,
                 # Constraint path (ANE) knobs
                 use_beam_splitter: bool = True, bs_layers: int = 2,
                 use_spectral_head: bool = True, spectral_bands: int = 4,
                 # Registry / Temporal pass-through
                 use_registry_kernel: bool = False,
                 registry_kernel_type: str = "blend",
                 registry_rank: int = 32,
                 registry_temperature: float = 1.0,
                 use_temporal_multiplex: bool = False,
                 tm_dts=None,
                 tm_alpha_offsets=None,
                 tm_learned_mix: bool = True,
                 # Remote synchronization
                 use_remote_synchronization: bool = False,
                 rs_threshold: float = 0.1,
                 rs_top_k_bridges: int = 10):
        """
        Initialize Resonance Transformer Block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension (default: 4 * d_model)
            n_sim_steps: Kuramoto simulation steps per head
            dropout: Dropout rate
            telemetry: If True, emit JSONL telemetry during simulation
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        
        # Multi-head Resonance Attention
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads

        # Constraint path (ANE) modules
        self.beam_splitter = None
        self.spectral_head = None
        if use_beam_splitter:
            assert d_model % 2 == 0, "BeamSplitterUnitaryStack requires even d_model"
            self.beam_splitter = BeamSplitterUnitaryStack(d_model=d_model, n_layers=int(bs_layers))
        if use_spectral_head:
            self.spectral_head = SpectralThresholdHead(d_model=d_model, bands=int(spectral_bands), fixed_basis="dct8", learn_thresholds=True)
        
        self.attention_heads = nn.ModuleList([
            ResonanceAttentionHead(
                d_model=self.head_dim,
                d_k=self.head_dim,
                d_v=self.head_dim,
                n_sim_steps=n_sim_steps,
                use_sakaguchi=True,
                use_critical_tuning=True,
                use_coupling_kernel=True,
                track_metrics=True,
                use_stuart_landau=True,  # Enable Stuart-Landau dynamics
                use_heun=True,  # Use Heun integrator
                track_cdns=True,  # Track CDNS metrics
                use_extended_cdns=True,
                # Pass-through: registry kernels
                use_registry_kernel=use_registry_kernel,
                registry_kernel_type=registry_kernel_type,
                registry_rank=registry_rank,
                registry_temperature=registry_temperature,
                # Pass-through: temporal multiplexing
                use_temporal_multiplex=use_temporal_multiplex,
                tm_dts=tm_dts,
                tm_alpha_offsets=tm_alpha_offsets,
                tm_learned_mix=tm_learned_mix,
                telemetry=telemetry
            )
            for _ in range(n_heads)
        ])
        
        self.attention_output = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

        # Remote synchronization settings
        self.use_remote_synchronization = bool(use_remote_synchronization)
        self.rs_threshold = float(rs_threshold)
        self.rs_top_k_bridges = int(rs_top_k_bridges)

        # Block-level metrics (e.g., inter-head coherence)
        self.metrics: Dict[str, torch.Tensor] = {}
    
    def compute_indirect_coupling(
        self,
        attention_patterns: torch.Tensor,  # [batch, n_heads, seq_len, seq_len]
        head_phases: torch.Tensor,         # [batch, n_heads]
    ) -> torch.Tensor:
        """
        Compute indirect coupling between heads via shared tokens.
        
        Indirect coupling: Heads i and j synchronize if they attend to
        similar token patterns, even without direct connection.
        
        Args:
            attention_patterns: Attention scores from each head [batch, n_heads, seq_len, seq_len]
            head_phases: Final phase angles from each head [batch, n_heads]
            
        Returns:
            indirect_K: [batch, n_heads, n_heads] indirect coupling matrix
        """
        batch_size, n_heads, seq_len, _ = attention_patterns.shape
        device = attention_patterns.device
        dtype = attention_patterns.dtype
        
        # Compute attention similarity between heads
        indirect_K = torch.zeros(batch_size, n_heads, n_heads, device=device, dtype=dtype)
        
        for i in range(n_heads):
            for j in range(i+1, n_heads):
                # Attention patterns for heads i and j
                attn_i = attention_patterns[:, i]  # [batch, seq_len, seq_len]
                attn_j = attention_patterns[:, j]
                
                # Flatten attention patterns
                attn_i_flat = attn_i.reshape(batch_size, -1)
                attn_j_flat = attn_j.reshape(batch_size, -1)
                
                # Cosine similarity
                # Normalize for cosine similarity
                attn_i_norm = torch.nn.functional.normalize(attn_i_flat, p=2, dim=-1)
                attn_j_norm = torch.nn.functional.normalize(attn_j_flat, p=2, dim=-1)
                cos_sim = (attn_i_norm * attn_j_norm).sum(dim=-1)  # [batch]
                
                # Token overlap (which tokens both heads attend to)
                # Get most attended token per source position
                attn_i_max = attn_i.argmax(dim=-1)  # [batch, seq_len] - most attended token per position
                attn_j_max = attn_j.argmax(dim=-1)
                overlap = (attn_i_max == attn_j_max).float().mean(dim=-1)  # [batch]
                
                # Combined indirect coupling
                indirect_K[:, i, j] = indirect_K[:, j, i] = (cos_sim + overlap) / 2
        
        return indirect_K
    
    def detect_remote_synchronization(
        self,
        head_phases: torch.Tensor,           # [batch, n_heads]
        indirect_coupling: torch.Tensor,      # [batch, n_heads, n_heads]
        direct_coupling: Optional[torch.Tensor] = None,  # [batch, n_heads, n_heads]
    ) -> Dict[str, Any]:
        """
        Detect remote synchronization patterns.
        
        RS occurs when:
        - Indirect coupling is strong
        - Direct coupling is weak (or absent)
        - Phase coherence is high despite weak direct coupling
        
        Args:
            head_phases: Final phase angles from each head [batch, n_heads]
            indirect_coupling: Indirect coupling matrix [batch, n_heads, n_heads]
            direct_coupling: Optional direct coupling matrix [batch, n_heads, n_heads]
            
        Returns:
            Dictionary with RS analysis results
        """
        batch_size, n_heads = head_phases.shape
        device = head_phases.device
        dtype = head_phases.dtype
        
        # Compute phase coherence between heads
        phase_diff = head_phases.unsqueeze(-1) - head_phases.unsqueeze(-2)  # [batch, n_heads, n_heads]
        coherence = torch.abs(torch.exp(1j * phase_diff))  # [batch, n_heads, n_heads]
        
        # Direct synchronization (if direct coupling exists)
        if direct_coupling is not None:
            direct_mask = direct_coupling > self.rs_threshold
            direct_mask_sum = direct_mask.sum(dim=(-2, -1)).clamp(min=1)
            direct_sync = (coherence * direct_mask).sum(dim=(-2, -1)) / direct_mask_sum
        else:
            direct_sync = torch.zeros(batch_size, device=device, dtype=dtype)
        
        # Remote synchronization: high coherence + high indirect coupling + low direct coupling
        indirect_mask = indirect_coupling > self.rs_threshold
        if direct_coupling is not None:
            weak_direct_mask = direct_coupling < self.rs_threshold
            rs_mask = indirect_mask & weak_direct_mask
        else:
            rs_mask = indirect_mask
        
        rs_mask_sum = rs_mask.sum(dim=(-2, -1)).clamp(min=1)
        remote_sync = (coherence * rs_mask).sum(dim=(-2, -1)) / rs_mask_sum
        
        # Identify RS pairs
        rs_pairs = []
        rs_strength = torch.zeros(batch_size, n_heads, n_heads, device=device, dtype=dtype)
        
        for i in range(n_heads):
            for j in range(i+1, n_heads):
                if rs_mask[:, i, j].any():
                    rs_pairs.append((i, j))
                    rs_strength[:, i, j] = rs_strength[:, j, i] = coherence[:, i, j] * indirect_coupling[:, i, j]
        
        return {
            'rs_pairs': rs_pairs,
            'rs_strength': rs_strength,
            'direct_sync': direct_sync,
            'remote_sync': remote_sync,
            'rs_ratio': remote_sync / (direct_sync + 1e-6),  # Ratio of RS to direct sync
        }
    
    def identify_bridge_tokens(
        self,
        attention_patterns: torch.Tensor,  # [batch, n_heads, seq_len, seq_len]
        head_phases: torch.Tensor,         # [batch, n_heads]
    ) -> Dict[str, torch.Tensor]:
        """
        Identify tokens that act as bridges between heads.
        
        Bridge tokens:
        - Attended to by multiple heads
        - High variance in attention across heads
        - Strong coupling to multiple head phases
        
        Args:
            attention_patterns: Attention scores from each head [batch, n_heads, seq_len, seq_len]
            head_phases: Final phase angles from each head [batch, n_heads]
            
        Returns:
            Dictionary with bridge token indices and strengths
        """
        batch_size, n_heads, seq_len, _ = attention_patterns.shape
        device = attention_patterns.device
        dtype = attention_patterns.dtype
        
        # Attention to each token (sum over source positions)
        token_attention = attention_patterns.sum(dim=2)  # [batch, n_heads, seq_len]
        
        # Number of heads attending to each token (above threshold)
        heads_attending = (token_attention > 0.1).sum(dim=1).float()  # [batch, seq_len]
        
        # Variance of attention across heads
        attn_variance = token_attention.var(dim=1)  # [batch, seq_len]
        
        # Coupling to head phases (correlation)
        # For each token, measure how attention correlates with head phases
        token_phase_coupling = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        for t in range(seq_len):
            attn_to_token = token_attention[:, :, t]  # [batch, n_heads]
            # Compute correlation between attention and phases
            # Use cosine similarity as proxy for correlation
            attn_norm = torch.nn.functional.normalize(attn_to_token, p=2, dim=-1)
            phase_norm = torch.nn.functional.normalize(head_phases, p=2, dim=-1)
            phase_corr = (attn_norm * phase_norm).sum(dim=-1).abs()  # [batch]
            token_phase_coupling[:, t] = phase_corr
        
        # Combined bridge score
        bridge_score = (
            heads_attending / n_heads +  # Normalized by number of heads
            attn_variance +              # High variance = bridge
            token_phase_coupling         # Strong phase coupling
        ) / 3
        
        # Top-k bridge tokens
        top_k = min(self.rs_top_k_bridges, seq_len)
        _, top_indices = torch.topk(bridge_score, k=top_k, dim=-1)
        
        return {
            'bridge_tokens': top_indices,
            'bridge_strength': torch.gather(bridge_score, -1, top_indices),
        }
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output embeddings [batch, seq_len, d_model]
        """
        # Constraint path preprocessing (ANE-friendly)
        h = x
        if self.beam_splitter is not None:
            h = self.beam_splitter(h)
        if self.spectral_head is not None:
            h = self.spectral_head(h)

        # Split into heads
        batch_size, seq_len, _ = h.shape
        x_heads = h.view(batch_size, seq_len, self.n_heads, self.head_dim)
        x_heads = x_heads.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        
        # Apply each attention head
        head_outputs = []
        head_phases_list = []
        for i, head in enumerate(self.attention_heads):
            head_x = x_heads[:, i, :, :]  # Extract head i
            head_out = head(head_x, mask)
            head_outputs.append(head_out)
            # Collect per-head final order phase if available for inter-head metric
            try:
                m = getattr(head, "metrics", {}) or {}
                if isinstance(m, dict) and ("final_order_phase" in m):
                    head_phases_list.append(m["final_order_phase"])  # [batch]
            except Exception:
                pass

        # Inter-head coherence metric: R_heads = |(1/H) Σ_h e^{i φ_h}|
        if len(head_phases_list) >= 2:
            try:
                # Stack to [H, batch] then transpose -> [batch, H]
                Hphi = torch.stack(head_phases_list, dim=0).transpose(0, 1)  # [batch, H]
                z = torch.exp(1j * Hphi)                                     # [batch, H], complex
                R_heads_complex = torch.mean(z, dim=-1)                       # [batch]
                R_heads = torch.abs(R_heads_complex).real                     # [batch]
                self.metrics["inter_head_coherence"] = R_heads
            except Exception:
                self.metrics["inter_head_coherence"] = torch.zeros(x.shape[0], device=x.device)
        
        # Remote synchronization analysis
        if self.use_remote_synchronization and len(head_phases_list) >= 2:
            try:
                # Stack head phases: [batch, n_heads]
                head_phases = torch.stack(head_phases_list, dim=0).transpose(0, 1)  # [batch, n_heads]
                
                # Collect attention patterns from each head
                # We need to recompute or access attention_scores from each head
                # Since they're not stored, we'll compute them from the head's Q, K, V and phases
                # For now, we'll use a simplified approach: compute attention patterns from head outputs
                # Actually, we can access the attention patterns by re-running the head's forward
                # but that's inefficient. Instead, let's store attention_scores in head metrics temporarily.
                
                # Alternative: compute attention patterns from coupling matrices if available
                # Or we can modify heads to store attention_scores when RS is enabled
                # For now, let's use the head outputs to approximate attention patterns
                # We'll use the head outputs as a proxy for attention patterns
                attention_patterns = []
                for i, head in enumerate(self.attention_heads):
                    head_x = x_heads[:, i, :, :]
                    # Get Q, K from head
                    Q = head.W_q(head_x)  # [batch, seq_len, head_dim]
                    K = head.W_k(head_x)  # [batch, seq_len, head_dim]
                    # Compute attention pattern from Q, K similarity (approximation)
                    # This approximates the attention pattern before Kuramoto simulation
                    attn_pattern = torch.bmm(Q, K.transpose(1, 2))  # [batch, seq_len, seq_len]
                    # Apply softmax to get attention distribution
                    d_k = Q.shape[-1]
                    attn_pattern = torch.softmax(attn_pattern / (d_k ** 0.5), dim=-1)
                    attention_patterns.append(attn_pattern)
                
                # Stack attention patterns: [batch, n_heads, seq_len, seq_len]
                attention_patterns = torch.stack(attention_patterns, dim=1)
                
                # Compute indirect coupling
                indirect_K = self.compute_indirect_coupling(attention_patterns, head_phases)
                
                # Detect remote synchronization
                rs_analysis = self.detect_remote_synchronization(
                    head_phases,
                    indirect_coupling=indirect_K
                )
                
                # Identify bridge tokens
                bridge_tokens = self.identify_bridge_tokens(attention_patterns, head_phases)
                
                # Store RS metrics
                self.metrics['remote_synchronization'] = {
                    **rs_analysis,
                    'bridge_tokens': bridge_tokens,
                    'indirect_coupling': indirect_K,
                }
            except Exception as e:
                # Log error but don't fail
                import warnings
                warnings.warn(f"Remote synchronization analysis failed: {e}")
                self.metrics['remote_synchronization'] = {
                    'rs_pairs': [],
                    'rs_strength': torch.zeros(batch_size, self.n_heads, self.n_heads, device=x.device),
                    'direct_sync': torch.zeros(batch_size, device=x.device),
                    'remote_sync': torch.zeros(batch_size, device=x.device),
                    'rs_ratio': torch.zeros(batch_size, device=x.device),
                }

        # Concatenate heads
        multi_head = torch.cat(head_outputs, dim=-1)  # [batch, seq_len, d_model]
        multi_head = self.attention_output(multi_head)
        
        # Residual connection and normalization
        out = self.norm1(h + self.dropout(multi_head))
        
        # Feed-forward
        ff_out = self.ff(out)
        out = self.norm2(out + ff_out)
        
        return out


class ResonanceTransformer(nn.Module):
    """
    Full Resonance Transformer model.
    
    Replaces standard Transformer attention with Kuramoto-based resonance attention.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = None, max_seq_len: int = 512,
                 n_sim_steps: int = 15, dropout: float = 0.1):
        """
        Initialize Resonance Transformer.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            n_sim_steps: Kuramoto simulation steps per attention head
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            ResonanceTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_sim_steps=n_sim_steps,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional mask [batch, seq_len]
            
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Create attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            mask = mask.expand(batch_size, 1, seq_len, seq_len)
        else:
            mask = None
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
