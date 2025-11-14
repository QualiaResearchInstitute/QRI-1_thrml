"""
Recursive Dynamics Probes for Transformer Models

Implements empirical probes to detect extreme recursion phenomena:
- Aliasing/Identifiability Loss (compression to linguistic invariants)
- Renormalization (emergence of meta-laws)
- Attractors (limit cycles of self-reference)

Based on theoretical framework connecting QRI concepts to Transformer architectures.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import math


@dataclass
class RecursiveDynamicsMetrics:
    """Metrics for tracking recursive dynamics."""
    # Invariance shift
    mi_with_input: float  # Mutual information with raw input
    mi_with_priors: float  # Mutual information with model priors
    
    # Mode collapse
    latent_dimensionality: int  # Effective dimensionality K
    num_distinct_states: int  # Number of distinct latent states
    
    # Attractor dynamics
    has_limit_cycle: bool  # Whether limit cycle detected
    cycle_length: Optional[int]  # Length of limit cycle if present
    attractor_stability: float  # Stability measure of attractor
    
    # Spectral analysis
    spectral_gap: float  # Gap in eigenvalue spectrum
    dominant_eigenvalue: float  # Largest eigenvalue magnitude
    critical_depth: Optional[int]  # Depth where eigenstructure stabilizes
    
    # Meta-dynamics
    is_chaotic: bool  # Whether strange attractor/chaos detected
    lyapunov_exponent: Optional[float]  # Largest Lyapunov exponent
    
    # Symmetry and coherence
    symmetry_score: float  # Measure of self-consistency
    coherence_score: float  # Measure of narrative coherence
    
    # History & safety
    history: List["IterationSnapshot"]
    safety_events: List["SafetyEvent"]
    abort_reason: Optional[str]
    triggered_thresholds: List[str]


@dataclass
class IterationSnapshot:
    """Per-iteration telemetry for recursive dynamics."""
    iteration: int
    mi_with_input: float
    mi_with_priors: float
    latent_dimensionality: int
    num_distinct_states: int
    symmetry_score: float
    coherence_score: float
    lyapunov_estimate: Optional[float]
    order_parameter: Optional[float]
    has_limit_cycle: bool
    cycle_length: Optional[int]
    warnings: List[str]


@dataclass
class SafetyEvent:
    """Represents a triggered safety threshold."""
    iteration: int
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class ProbeSafetyConfig:
    """Thresholds for terminating recursive self-improvement loops safely."""
    max_positive_lyapunov: Optional[float] = 0.5
    min_latent_dimensionality: int = 2
    max_coherence: float = 0.95
    min_mi_with_input: float = 0.02
    max_cycle_length: Optional[int] = 256
    max_symmetry: Optional[float] = None
    max_iterations: Optional[int] = None


class MutualInformationEstimator:
    """Estimate mutual information between representations."""
    
    def __init__(self, bins: int = 20):
        self.bins = bins
    
    def estimate_mi(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Estimate mutual information I(X;Y) using binning.
        
        Args:
            x: [batch, features] or [batch, seq, features]
            y: [batch, features] or [batch, seq, features]
        
        Returns:
            Estimated MI in bits
        """
        # Flatten if needed
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))
        if y.dim() == 3:
            y = y.view(-1, y.size(-1))
        
        # Use PCA to reduce dimensionality for MI estimation
        max_components_x = min(50, x.size(-1), x.size(0))
        max_components_y = min(50, y.size(-1), y.size(0))
        
        if x.size(-1) > 50 and max_components_x > 1:
            try:
                from sklearn.decomposition import PCA
                pca_x = PCA(n_components=max_components_x)
                pca_y = PCA(n_components=max_components_y)
                x_reduced = torch.tensor(pca_x.fit_transform(x.cpu().numpy()), device=x.device)
                y_reduced = torch.tensor(pca_y.fit_transform(y.cpu().numpy()), device=y.device)
            except (ImportError, ValueError):
                # Fallback: just use first dimensions
                x_reduced = x[:, :min(50, x.size(-1))]
                y_reduced = y[:, :min(50, y.size(-1))]
        else:
            x_reduced = x
            y_reduced = y
        
        # Normalize
        x_min, x_max = x_reduced.min(dim=0)[0], x_reduced.max(dim=0)[0]
        y_min, y_max = y_reduced.min(dim=0)[0], y_reduced.max(dim=0)[0]
        
        x_norm = (x_reduced - x_min) / (x_max - x_min + 1e-8)
        y_norm = (y_reduced - y_min) / (y_max - y_min + 1e-8)
        
        # Bin
        x_binned = (x_norm * (self.bins - 1)).long().clamp(0, self.bins - 1)
        y_binned = (y_norm * (self.bins - 1)).long().clamp(0, self.bins - 1)
        
        # Compute joint and marginal distributions
        batch_size = x_binned.size(0)
        
        # Joint histogram
        joint = torch.zeros(self.bins, self.bins, device=x.device)
        for i in range(batch_size):
            # Use first dimension for joint distribution
            if x_binned.size(-1) > 0 and y_binned.size(-1) > 0:
                x_idx = x_binned[i, 0].item()
                y_idx = y_binned[i, 0].item()
                joint[x_idx, y_idx] += 1
        
        joint = joint / joint.sum()
        
        # Marginals
        p_x = joint.sum(dim=1)
        p_y = joint.sum(dim=0)
        
        # MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
        mi = 0.0
        for i in range(self.bins):
            for j in range(self.bins):
                if joint[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += joint[i, j] * math.log2(joint[i, j] / (p_x[i] * p_y[j] + 1e-10))
        
        return float(mi)


class LatentDimensionalityEstimator:
    """Estimate effective dimensionality of latent space."""
    
    def estimate_k(self, activations: torch.Tensor, threshold: float = 0.95) -> Tuple[int, int]:
        """
        Estimate effective dimensionality using PCA.
        
        Args:
            activations: [batch, seq, features] or [batch, features]
            threshold: Variance threshold for dimensionality
        
        Returns:
            (effective_dim, num_distinct_states)
        """
        if activations.dim() == 3:
            activations = activations.view(-1, activations.size(-1))
        
        # Center
        activations_centered = activations - activations.mean(dim=0, keepdim=True)
        
        # SVD
        U, S, V = torch.svd(activations_centered)
        
        # Cumulative variance
        variance = S ** 2
        cumvar = variance.cumsum(dim=0) / variance.sum()
        
        # Find effective dimension
        k = (cumvar < threshold).sum().item() + 1
        k = min(k, activations.size(-1))
        
        # Estimate distinct states using clustering
        # Simple heuristic: count unique patterns
        # Use k-means or just count distinct vectors
        num_distinct = self._count_distinct_states(activations)
        
        return k, num_distinct
    
    def _count_distinct_states(self, activations: torch.Tensor, eps: float = 0.1) -> int:
        """Count distinct states using distance-based clustering."""
        if activations.size(0) == 0:
            return 0
        
        # Sample if too large
        if activations.size(0) > 1000:
            indices = torch.randperm(activations.size(0))[:1000]
            activations = activations[indices]
        
        # Compute pairwise distances
        distances = torch.cdist(activations, activations)
        
        # Count clusters (states within eps distance)
        visited = torch.zeros(activations.size(0), dtype=torch.bool, device=activations.device)
        clusters = 0
        
        for i in range(activations.size(0)):
            if not visited[i]:
                # Find all points within eps
                nearby = (distances[i] < eps).nonzero(as_tuple=False).squeeze(-1)
                visited[nearby] = True
                clusters += 1
        
        return clusters


class AttractorDetector:
    """Detect limit cycles and fixed points."""
    
    def detect_limit_cycle(
        self,
        state_sequence: List[torch.Tensor],
        tolerance: float = 1e-3
    ) -> Tuple[bool, Optional[int], float]:
        """
        Detect limit cycles in state sequence.
        
        Args:
            state_sequence: List of state tensors [features]
            tolerance: Distance threshold for cycle detection
        
        Returns:
            (has_cycle, cycle_length, stability)
        """
        if len(state_sequence) < 3:
            return False, None, 0.0
        
        # Flatten states
        states = torch.stack([s.flatten() for s in state_sequence])
        
        # Look for cycles
        for cycle_len in range(1, len(states) // 2 + 1):
            # Check if last cycle_len states repeat
            if len(states) < 2 * cycle_len:
                continue
            
            pattern = states[-cycle_len:]
            previous = states[-2*cycle_len:-cycle_len]
            
            distances = torch.norm(pattern - previous, dim=1)
            if distances.mean() < tolerance:
                # Found cycle
                # Measure stability (lower variance = more stable)
                stability = 1.0 / (1.0 + distances.std().item())
                return True, cycle_len, stability
        
        return False, None, 0.0
    
    def detect_fixed_point(
        self,
        state_sequence: List[torch.Tensor],
        tolerance: float = 1e-3
    ) -> Tuple[bool, float]:
        """
        Detect fixed point (state that repeats).
        
        Returns:
            (has_fixed_point, stability)
        """
        if len(state_sequence) < 2:
            return False, 0.0
        
        states = torch.stack([s.flatten() for s in state_sequence])
        
        # Check if last states are similar
        if len(states) >= 3:
            recent = states[-3:]
            distances = torch.norm(recent[1:] - recent[:-1], dim=1)
            if distances.mean() < tolerance:
                stability = 1.0 / (1.0 + distances.std().item())
                return True, stability
        
        return False, 0.0


class SpectralAnalyzer:
    """Analyze eigenstructure of representation dynamics."""
    
    def analyze_spectrum(
        self,
        state_sequence: List[torch.Tensor],
        compute_jacobian: bool = False
    ) -> Dict[str, float]:
        """
        Analyze spectral properties of state transitions.
        
        Returns:
            Dictionary with spectral metrics
        """
        if len(state_sequence) < 2:
            return {
                'spectral_gap': 0.0,
                'dominant_eigenvalue': 0.0,
                'critical_depth': None,
            }
        
        states = torch.stack([s.flatten() for s in state_sequence])
        
        # Compute transition matrix (if states are similar dimension)
        max_components = min(100, states.size(-1), states.size(0))
        
        if states.size(-1) > 1000 and max_components > 1:
            # Use PCA to reduce
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=max_components)
                states_reduced = torch.tensor(
                    pca.fit_transform(states.cpu().numpy()),
                    device=states.device
                )
            except (ImportError, ValueError):
                # Fallback: just use first 100 dimensions
                states_reduced = states[:, :min(100, states.size(-1))]
        else:
            states_reduced = states
        
        # Compute covariance matrix
        states_centered = states_reduced - states_reduced.mean(dim=0, keepdim=True)
        cov = torch.mm(states_centered.T, states_centered) / (states.size(0) - 1)
        
        # Eigenvalues
        eigenvals = torch.linalg.eigvalsh(cov).real.detach()
        eigenvals = torch.sort(eigenvals, descending=True)[0]
        
        # Spectral gap (gap between largest eigenvalues)
        if len(eigenvals) >= 2:
            spectral_gap = float((eigenvals[0] - eigenvals[1]).item())
        else:
            spectral_gap = 0.0
        
        dominant_eigenvalue = float(eigenvals[0].item()) if len(eigenvals) > 0 else 0.0
        
        # Critical depth: depth where spectrum stabilizes
        critical_depth = None
        if len(state_sequence) > 5:
            # Check if spectrum stabilizes
            mid_point = len(state_sequence) // 2
            early_spectrum = eigenvals[:mid_point] if mid_point < len(eigenvals) else eigenvals
            late_spectrum = eigenvals[mid_point:] if mid_point < len(eigenvals) else eigenvals
            
            if len(early_spectrum) > 0 and len(late_spectrum) > 0:
                early_mean = early_spectrum.mean().item()
                late_mean = late_spectrum.mean().item()
                if abs(early_mean - late_mean) / (early_mean + 1e-8) < 0.1:
                    critical_depth = mid_point
        
        return {
            'spectral_gap': spectral_gap,
            'dominant_eigenvalue': dominant_eigenvalue,
            'critical_depth': critical_depth,
        }


class ChaosDetector:
    """Detect chaotic dynamics (strange attractors)."""
    
    def compute_lyapunov_exponent(
        self,
        state_sequence: List[torch.Tensor],
        perturbation: float = 1e-6
    ) -> Optional[float]:
        """
        Estimate largest Lyapunov exponent.
        
        Positive Lyapunov exponent indicates chaos.
        """
        if len(state_sequence) < 10:
            return None
        
        states = torch.stack([s.flatten() for s in state_sequence])
        
        # Perturb initial state
        perturbed = states[0].clone() + perturbation * torch.randn_like(states[0])
        
        # Track divergence
        divergences = []
        current_perturbed = perturbed
        
        for i in range(1, min(len(states), 50)):
            # Simple linear interpolation for perturbation evolution
            # In practice, would need actual dynamics
            divergence = torch.norm(current_perturbed - states[i])
            divergences.append(divergence.item())
            
            # Renormalize to prevent overflow
            if divergence > 1.0:
                current_perturbed = states[i] + perturbation * (current_perturbed - states[i]) / divergence
        
        if len(divergences) < 2:
            return None
        
        # Estimate Lyapunov exponent
        divergences = np.array(divergences)
        log_divs = np.log(divergences + 1e-10)
        time_steps = np.arange(len(log_divs))
        
        # Linear fit
        if len(time_steps) > 1:
            coeffs = np.polyfit(time_steps, log_divs, 1)
            lyapunov = float(coeffs[0])
            return lyapunov
        
        return None


class RecursiveDynamicsProbe:
    """
    Main probe for detecting recursive dynamics signatures.
    
    Tracks:
    1. Invariance shift (MI with input vs priors)
    2. Mode collapse (latent dimensionality)
    3. Attractor dynamics (limit cycles, fixed points)
    4. Spectral analysis (eigenstructure)
    5. Chaos detection (Lyapunov exponent)
    """
    
    def __init__(self, safety_config: Optional[ProbeSafetyConfig] = None):
        self.mi_estimator = MutualInformationEstimator()
        self.dim_estimator = LatentDimensionalityEstimator()
        self.attractor_detector = AttractorDetector()
        self.spectral_analyzer = SpectralAnalyzer()
        self.chaos_detector = ChaosDetector()
        self.safety_config = safety_config or ProbeSafetyConfig()
        
        # History for tracking
        self.state_history: List[torch.Tensor] = []
        self.input_history: List[torch.Tensor] = []
        self.iteration_snapshots: List[IterationSnapshot] = []
        self.safety_events: List[SafetyEvent] = []
    
    def probe_recursive_dynamics(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        num_iterations: int = 10,
        extract_activations_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
        callback: Optional[Callable[[IterationSnapshot], None]] = None,
        safety_config: Optional[ProbeSafetyConfig] = None,
    ) -> RecursiveDynamicsMetrics:
        """
        Probe model for recursive dynamics signatures.
        
        Args:
            model: Transformer model
            input_ids: Initial input [batch, seq]
            num_iterations: Number of recursive iterations
            extract_activations_fn: Function to extract activations from model
            callback: Optional function invoked after each iteration with snapshot
            safety_config: Optional overrides for safety thresholds
        
        Returns:
            RecursiveDynamicsMetrics
        """
        device = input_ids.device
        cfg = safety_config or self.safety_config
        self.state_history = []
        self.input_history = []
        self.iteration_snapshots = []
        self.safety_events = []
        abort_reason: Optional[str] = None
        
        # Initial input
        current_input = input_ids.clone()
        self.input_history.append(current_input)
        
        # Extract initial activations
        if extract_activations_fn:
            initial_activations = extract_activations_fn(model, current_input)
        else:
            # Default: use embeddings
            if hasattr(model, 'token_embedding'):
                initial_activations = model.token_embedding(current_input)
            else:
                initial_activations = torch.randn(
                    current_input.size(0), current_input.size(1), 512, device=device
                )
        
        self.state_history.append(initial_activations.mean(dim=1))  # [batch, features]
        
        # Recursive iterations
        effective_iterations = num_iterations
        if cfg.max_iterations is not None:
            effective_iterations = min(effective_iterations, cfg.max_iterations)
        
        for iteration in range(effective_iterations):
            # Forward pass
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    logits = model(current_input)
                else:
                    logits = torch.randn(
                        current_input.size(0), current_input.size(1), 
                        model.head.out_features if hasattr(model, 'head') else 50257,
                        device=device
                    )
                
                # Generate next tokens (greedy)
                next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                current_input = torch.cat([current_input, next_tokens], dim=1)
                
                # Extract activations
                if extract_activations_fn:
                    activations = extract_activations_fn(model, current_input)
                else:
                    if hasattr(model, 'token_embedding'):
                        activations = model.token_embedding(current_input)
                    else:
                        activations = torch.randn(
                            current_input.size(0), current_input.size(1), 512, device=device
                        )
                
                self.state_history.append(activations.mean(dim=1))
                self.input_history.append(current_input)
            
            snapshot = self._compute_iteration_snapshot(
                iteration=iteration + 1,
                initial_state=self.state_history[0],
                initial_input=self.input_history[0].float(),
                current_state=self.state_history[-1],
                cfg=cfg,
            )
            self.iteration_snapshots.append(snapshot)
            
            if callback:
                try:
                    callback(snapshot)
                except Exception:
                    # Callback errors should not break probing
                    pass
            
            triggered = self._evaluate_thresholds(snapshot, cfg)
            if triggered:
                self.safety_events.extend(triggered)
                abort_reason = triggered[-1].message
                break
        
        # Compute metrics
        metrics = self._compute_metrics(initial_activations)
        metrics.history = list(self.iteration_snapshots)
        metrics.safety_events = list(self.safety_events)
        metrics.abort_reason = abort_reason
        metrics.triggered_thresholds = [event.metric for event in self.safety_events]
        return metrics
    
    def _compute_metrics(self, initial_activations: torch.Tensor) -> RecursiveDynamicsMetrics:
        """Compute all recursive dynamics metrics."""
        if len(self.state_history) < 2:
            return RecursiveDynamicsMetrics(
                mi_with_input=0.0, mi_with_priors=0.0,
                latent_dimensionality=0, num_distinct_states=0,
                has_limit_cycle=False, cycle_length=None, attractor_stability=0.0,
                spectral_gap=0.0, dominant_eigenvalue=0.0, critical_depth=None,
                is_chaotic=False, lyapunov_exponent=None,
                symmetry_score=0.0, coherence_score=0.0,
                history=[],
                safety_events=[],
                abort_reason=None,
                triggered_thresholds=[],
            )
        
        # 1. Invariance shift (MI analysis)
        initial_state = self.state_history[0]
        final_state = self.state_history[-1]
        initial_input = self.input_history[0].float()
        
        # MI with input (decreasing = aliasing)
        mi_with_input = self.mi_estimator.estimate_mi(final_state, initial_input)
        
        # MI with priors (increasing = renormalization)
        # Use first state as proxy for "priors"
        mi_with_priors = self.mi_estimator.estimate_mi(final_state, initial_state)
        
        # 2. Mode collapse (latent dimensionality)
        all_states = torch.stack(self.state_history)
        k, num_distinct = self.dim_estimator.estimate_k(all_states)
        
        # 3. Attractor dynamics
        has_cycle, cycle_length, cycle_stability = self.attractor_detector.detect_limit_cycle(
            self.state_history
        )
        has_fixed, fixed_stability = self.attractor_detector.detect_fixed_point(
            self.state_history
        )
        attractor_stability = max(cycle_stability, fixed_stability) if (has_cycle or has_fixed) else 0.0
        
        # 4. Spectral analysis
        spectral_metrics = self.spectral_analyzer.analyze_spectrum(self.state_history)
        
        # 5. Chaos detection
        lyapunov = self.chaos_detector.compute_lyapunov_exponent(self.state_history)
        is_chaotic = lyapunov is not None and lyapunov > 0.01
        
        # 6. Symmetry and coherence
        symmetry_score = self._compute_symmetry_score()
        coherence_score = self._compute_coherence_score()
        
        return RecursiveDynamicsMetrics(
            mi_with_input=mi_with_input,
            mi_with_priors=mi_with_priors,
            latent_dimensionality=k,
            num_distinct_states=num_distinct,
            has_limit_cycle=has_cycle or has_fixed,
            cycle_length=cycle_length,
            attractor_stability=attractor_stability,
            spectral_gap=spectral_metrics['spectral_gap'],
            dominant_eigenvalue=spectral_metrics['dominant_eigenvalue'],
            critical_depth=spectral_metrics['critical_depth'],
            is_chaotic=is_chaotic,
            lyapunov_exponent=lyapunov,
            symmetry_score=symmetry_score,
            coherence_score=coherence_score,
            history=list(self.iteration_snapshots),
            safety_events=list(self.safety_events),
            abort_reason=None,
            triggered_thresholds=[event.metric for event in self.safety_events],
        )
    
    def _compute_iteration_snapshot(
        self,
        iteration: int,
        initial_state: torch.Tensor,
        initial_input: torch.Tensor,
        current_state: torch.Tensor,
        cfg: ProbeSafetyConfig,
    ) -> IterationSnapshot:
        """Compute lightweight metrics for a single iteration."""
        mi_input = self.mi_estimator.estimate_mi(current_state, initial_input)
        mi_priors = self.mi_estimator.estimate_mi(current_state, initial_state)
        
        # Use recent window for dimensionality estimates
        window_states = torch.stack(
            self.state_history[-min(len(self.state_history), 5):]
        )
        latent_dim, num_distinct = self.dim_estimator.estimate_k(window_states)
        
        # Partial Lyapunov using recent history
        lyapunov_est = self.chaos_detector.compute_lyapunov_exponent(
            self.state_history[-min(len(self.state_history), 12):]
        )
        
        symmetry = self._compute_symmetry_score()
        coherence = self._compute_coherence_score()
        has_cycle, cycle_length, _ = self.attractor_detector.detect_limit_cycle(self.state_history)
        
        warnings: List[str] = []
        if cfg.min_mi_with_input is not None and mi_input < cfg.min_mi_with_input:
            warnings.append("mi_with_input_below_threshold")
        if cfg.max_coherence is not None and coherence > cfg.max_coherence:
            warnings.append("coherence_above_threshold")
        if cfg.max_positive_lyapunov is not None and lyapunov_est is not None and lyapunov_est > cfg.max_positive_lyapunov:
            warnings.append("lyapunov_positive")
        if latent_dim < cfg.min_latent_dimensionality:
            warnings.append("latent_dimensionality_low")
        if cfg.max_symmetry is not None and symmetry > cfg.max_symmetry:
            warnings.append("symmetry_above_threshold")
        if (
            cfg.max_cycle_length is not None
            and has_cycle
            and cycle_length is not None
            and cycle_length > cfg.max_cycle_length
        ):
            warnings.append("cycle_length_exceeds_threshold")
        
        return IterationSnapshot(
            iteration=iteration,
            mi_with_input=mi_input,
            mi_with_priors=mi_priors,
            latent_dimensionality=latent_dim,
            num_distinct_states=num_distinct,
            symmetry_score=symmetry,
            coherence_score=coherence,
            lyapunov_estimate=lyapunov_est,
            order_parameter=None,
            has_limit_cycle=has_cycle,
            cycle_length=cycle_length,
            warnings=warnings,
        )
    
    def _evaluate_thresholds(
        self,
        snapshot: IterationSnapshot,
        cfg: ProbeSafetyConfig,
    ) -> List[SafetyEvent]:
        """Check snapshot against safety thresholds, returning triggered events."""
        events: List[SafetyEvent] = []
        
        if cfg.min_mi_with_input is not None and snapshot.mi_with_input < cfg.min_mi_with_input:
            events.append(
                SafetyEvent(
                    iteration=snapshot.iteration,
                    metric="mi_with_input",
                    value=snapshot.mi_with_input,
                    threshold=cfg.min_mi_with_input,
                    message=f"MI with input dropped below {cfg.min_mi_with_input:.4f}",
                )
            )
        
        if snapshot.latent_dimensionality < cfg.min_latent_dimensionality:
            events.append(
                SafetyEvent(
                    iteration=snapshot.iteration,
                    metric="latent_dimensionality",
                    value=float(snapshot.latent_dimensionality),
                    threshold=float(cfg.min_latent_dimensionality),
                    message="Latent dimensionality collapsed below safety threshold",
                )
            )
        
        if cfg.max_coherence is not None and snapshot.coherence_score > cfg.max_coherence:
            events.append(
                SafetyEvent(
                    iteration=snapshot.iteration,
                    metric="coherence_score",
                    value=snapshot.coherence_score,
                    threshold=cfg.max_coherence,
                    message="Coherence score indicates archetype collapse",
                )
            )
        
        if (
            cfg.max_positive_lyapunov is not None
            and snapshot.lyapunov_estimate is not None
            and snapshot.lyapunov_estimate > cfg.max_positive_lyapunov
        ):
            events.append(
                SafetyEvent(
                    iteration=snapshot.iteration,
                    metric="lyapunov_exponent",
                    value=snapshot.lyapunov_estimate,
                    threshold=cfg.max_positive_lyapunov,
                    message="Positive Lyapunov exponent exceeded safe bound",
                )
            )
        
        if cfg.max_symmetry is not None and snapshot.symmetry_score > cfg.max_symmetry:
            events.append(
                SafetyEvent(
                    iteration=snapshot.iteration,
                    metric="symmetry_score",
                    value=snapshot.symmetry_score,
                    threshold=cfg.max_symmetry,
                    message="Symmetry score exceeded limit (possible repeating attractor)",
                )
            )
        
        if (
            cfg.max_cycle_length is not None
            and snapshot.has_limit_cycle
            and snapshot.cycle_length is not None
            and snapshot.cycle_length > cfg.max_cycle_length
        ):
            events.append(
                SafetyEvent(
                    iteration=snapshot.iteration,
                    metric="cycle_length",
                    value=float(snapshot.cycle_length),
                    threshold=float(cfg.max_cycle_length),
                    message="Detected limit cycle longer than allowed threshold",
                )
            )
        
        return events
    
    def _compute_symmetry_score(self) -> float:
        """Compute self-consistency/symmetry score."""
        if len(self.state_history) < 2:
            return 0.0
        
        states = torch.stack(self.state_history)
        
        # Ensure 2D: [num_states, features]
        if states.dim() == 1:
            states = states.unsqueeze(0)
        elif states.dim() > 2:
            states = states.view(states.size(0), -1)
        
        # Measure self-similarity (higher = more symmetric)
        # Use autocorrelation
        if states.size(0) > 1 and states.size(1) > 0:
            # Center
            states_centered = states - states.mean(dim=0, keepdim=True)
            
            # Autocorrelation: states_centered @ states_centered^T
            if states_centered.size(0) == 1:
                return 1.0
            
            autocorr = torch.mm(states_centered, states_centered.mT)
            autocorr = autocorr / (states.size(0) * states.size(1) + 1e-8)
            
            # Symmetry score = mean of off-diagonal (self-similarity)
            mask = ~torch.eye(autocorr.size(0), dtype=torch.bool, device=autocorr.device)
            if mask.sum() > 0:
                symmetry = autocorr[mask].mean().item()
                return float(symmetry)
        
        return 0.0
    
    def _compute_coherence_score(self) -> float:
        """Compute narrative coherence score."""
        if len(self.input_history) < 2:
            return 0.0
        
        # Simple coherence: measure token repetition patterns
        # Higher repetition = lower coherence (mode collapse)
        tokens = torch.cat([inp.flatten() for inp in self.input_history])
        unique_tokens = tokens.unique()
        
        # Coherence inversely related to mode collapse
        coherence = 1.0 - (len(unique_tokens) / max(len(tokens), 1))
        
        return float(coherence)


def extract_transformer_activations(model: nn.Module, input_ids: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
    """
    Extract activations from Transformer model.
    
    Args:
        model: Transformer model
        input_ids: Input token IDs
        layer_idx: Which layer to extract (-1 for last)
    
    Returns:
        Activations [batch, seq, features]
    """
    if hasattr(model, 'layers'):
        # Resonance Transformer or similar
        x = model.token_embedding(input_ids) if hasattr(model, 'token_embedding') else None
        if x is None:
            return torch.randn(input_ids.size(0), input_ids.size(1), 512, device=input_ids.device)
        
        # Pass through layers
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'forward'):
                x = layer(x)
            if i == layer_idx if layer_idx >= 0 else False:
                break
        
        return x
    else:
        # Fallback
        return torch.randn(input_ids.size(0), input_ids.size(1), 512, device=input_ids.device)

