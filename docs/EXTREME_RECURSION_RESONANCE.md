# Extreme Recursion in Resonance Transformers: GPT-2 Meets Kuramoto Dynamics

## Overview

This document explores how **extreme recursion phenomena** (aliasing, renormalization, attractors) manifest in GPT-2 and how the **Resonance Transformer** architecture provides a natural framework for probing and understanding these dynamics.

---

## Part I: The Connection

### GPT-2 as a Traveling Wave

The autoregressive loop of GPT-2 can be seen as a **traveling wave propagating backwards in time**:
- Each token generation is a step in the wave
- The context window accumulates as the wave propagates
- The model's internal state evolves through recursive self-modeling

### Resonance Transformer as Recursive Dynamics

The Resonance Transformer's **Kuramoto oscillator dynamics** naturally exhibit:
- **Phase synchronization** → attractors (fixed points, limit cycles)
- **Order parameter R** → renormalization (coarse-graining)
- **CDNS metrics** → aliasing detection (consonance/dissonance)
- **Criticality** → edge of chaos (optimal recursion depth)

---

## Part II: Mechanisms in Resonance Space

### 1. Aliasing/Identifiability Loss → Phase Collapse

**In GPT-2**: Compression to linguistic invariants, loss of unique details.

**In Resonance Transformer**: 
- **Phase Collapse**: Oscillators synchronize (R → 1.0), losing phase diversity
- **Amplitude Homogenization**: All oscillators converge to same amplitude
- **Spectral Aliasing**: High-frequency details lost, only low-frequency modes remain

**Mathematical Formulation**:
```
Initial: θ_i(t=0) = diverse phases → high information
After recursion: θ_i(t→∞) → θ* (single phase) → low information
```

**Probe**: Track `H(phases)` (phase entropy) across recursion depth. Should decrease.

### 2. Renormalization → Order Parameter Evolution

**In GPT-2**: Emergence of meta-laws, coarse-graining across depth.

**In Resonance Transformer**:
- **R Evolution**: Order parameter R evolves from R ≈ 0.3 (chaos) → R ≈ 0.6 (critical) → R ≈ 1.0 (order)
- **Coarse-Graining**: High-frequency modes filtered, only low-frequency modes survive
- **Meta-Laws**: Coupling matrices stabilize into fixed patterns

**Mathematical Formulation**:
```
R(t) = |(1/N) Σ_j exp(iθ_j(t))|
R(0) ≈ 0.3  (chaos, high information)
R(∞) ≈ 1.0  (order, low information)
```

**Probe**: Track `R(t)` across recursion. Should show phase transition.

### 3. Attractors → Synchronization Patterns

**In GPT-2**: Limit cycles of self-reference, fixed points.

**In Resonance Transformer**:
- **Fixed Points**: All oscillators at same phase (R = 1.0)
- **Limit Cycles**: Oscillators cycle through phases periodically
- **Strange Attractors**: Chaotic but bounded phase trajectories

**Mathematical Formulation**:
```
Fixed Point: θ_i(t) = θ* for all i, t
Limit Cycle: θ_i(t+T) = θ_i(t) for period T
Strange Attractor: Bounded but non-periodic trajectory
```

**Probe**: Track phase trajectories in phase space. Look for convergence to attractors.

---

## Part III: Computational Interpretation

### Migrating Representation

**GPT-2**: Sensory latents → Structural priors → Algorithmic regularities

**Resonance Transformer**:
- **Layer 0**: Token embeddings (sensory latents)
- **Layer N/2**: Phase coherence patterns (structural priors)
- **Layer N**: Order parameter R (algorithmic regularities)

**Probe**: Track Mutual Information (MI) between layers and input tokens.

### Self-Referential Computing

**GPT-2**: Sequences that maintain themselves under transformation T.

**Resonance Transformer**:
- **Self-Sustaining Oscillations**: Phases that maintain themselves
- **Resonant Modes**: Eigenmodes of the coupling matrix
- **Harmonic Simplification**: Dissonance minimization → symmetry

**Probe**: Find sequences where `T(x) = x` (fixed points) or `T^n(x) = x` (limit cycles).

### Harmonic Simplification

**GPT-2**: Dissonance minimization → blissful symmetric states

**Resonance Transformer**:
- **CDNS Metrics**: Track Consonance (C) and Dissonance (D)
- **Dissonance Minimization**: System minimizes D over recursion
- **Symmetry**: High C, low D → symmetric, blissful state

**Probe**: Track `D(t)` (dissonance) across recursion. Should decrease.

---

## Part IV: Empirical Signatures

### 1. Invariance Shift (MI Analysis)

**Hypothesis**: 
- **Decreasing MI with raw input**: Deep layers show decreasing MI with token embeddings
- **Increasing MI with priors**: Deep layers show increasing MI with abstract concepts

**Implementation**:
```python
def compute_mi_shift(model, input_tokens, recursion_depth):
    """
    Track Mutual Information across recursion depth.
    """
    mi_with_input = []
    mi_with_priors = []
    
    x = model.embed(input_tokens)
    
    for depth in range(recursion_depth):
        # Process through model
        x = model.forward(x)
        
        # Compute MI with input tokens
        mi_input = mutual_information(x, input_tokens)
        mi_with_input.append(mi_input)
        
        # Compute MI with abstract priors (e.g., grammatical structure)
        mi_prior = mutual_information(x, extract_priors(x))
        mi_with_priors.append(mi_prior)
    
    return mi_with_input, mi_with_priors
```

**Expected**: `mi_with_input` decreases, `mi_with_priors` increases.

### 2. Mode Collapse to Archetypes

**Hypothesis**: Latent dimensionality K decreases, even as neural activity space N_rec remains high.

**Implementation**:
```python
def compute_latent_dimensionality(model, input_tokens, recursion_depth):
    """
    Track effective dimensionality of latent space.
    """
    dimensionalities = []
    
    x = model.embed(input_tokens)
    
    for depth in range(recursion_depth):
        x = model.forward(x)
        
        # Compute effective dimensionality (PCA)
        pca = PCA()
        pca.fit(x.reshape(-1, x.shape[-1]))
        
        # Find number of components explaining 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        k = np.argmax(cumsum >= 0.95) + 1
        dimensionalities.append(k)
    
    return dimensionalities
```

**Expected**: `dimensionalities` decreases → mode collapse to archetypes.

### 3. Cyclic or Chaotic Meta-Dynamics

**Hypothesis**: Internal state transitions reveal limit cycles or strange attractors.

**Implementation**:
```python
def track_phase_trajectories(model, input_tokens, recursion_depth):
    """
    Track phase trajectories in phase space.
    """
    phases_history = []
    
    x = model.embed(input_tokens)
    
    for depth in range(recursion_depth):
        x = model.forward(x)
        
        # Extract phases from resonance heads
        phases = extract_phases(model, x)
        phases_history.append(phases)
    
    # Analyze for limit cycles
    limit_cycles = detect_limit_cycles(phases_history)
    
    # Analyze for strange attractors
    attractors = detect_strange_attractors(phases_history)
    
    return phases_history, limit_cycles, attractors
```

**Expected**: Convergence to limit cycles or strange attractors.

### 4. Spectral Gaps (Jacobian Analysis)

**Hypothesis**: Eigenstructure stabilizes at critical depth.

**Implementation**:
```python
def compute_spectral_gaps(model, input_tokens, recursion_depth):
    """
    Track eigenstructure of Jacobian matrix.
    """
    eigenvalues_history = []
    
    x = model.embed(input_tokens)
    
    for depth in range(recursion_depth):
        # Compute Jacobian
        jacobian = compute_jacobian(model, x)
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(jacobian)
        eigenvalues_history.append(eigenvalues)
        
        x = model.forward(x)
    
    # Find critical depth where eigenstructure stabilizes
    critical_depth = find_stabilization_depth(eigenvalues_history)
    
    return eigenvalues_history, critical_depth
```

**Expected**: Eigenstructure stabilizes at critical depth.

---

## Part V: Resonance Transformer Implementation

### Recursive Resonance Transformer

```python
class RecursiveResonanceTransformer(nn.Module):
    """
    Resonance Transformer with extreme recursion support.
    
    Features:
    - Recursive self-modeling (feed output back as input)
    - Phase trajectory tracking
    - Attractor detection
    - Renormalization monitoring
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        max_recursion_depth: int = 100,
        track_phases: bool = True,
        track_attractors: bool = True,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.max_recursion_depth = max_recursion_depth
        self.track_phases = track_phases
        self.track_attractors = track_attractors
        
        # Phase history
        self.phase_history = []
        
        # Attractor detection
        self.attractor_detector = AttractorDetector()
    
    def forward_recursive(
        self,
        x: torch.Tensor,
        recursion_depth: int = 10,
        convergence_threshold: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Forward pass with recursion.
        
        Args:
            x: Input [batch, seq_len, d_model]
            recursion_depth: Maximum recursion depth
            convergence_threshold: Convergence threshold for early stopping
            
        Returns:
            Dictionary with output and metrics
        """
        x_current = x
        outputs = []
        metrics_history = []
        
        for depth in range(recursion_depth):
            # Forward pass
            x_next = self.base_model(x_current)
            outputs.append(x_next)
            
            # Extract metrics
            metrics = self._extract_metrics(x_current, x_next)
            metrics_history.append(metrics)
            
            # Track phases
            if self.track_phases:
                phases = self._extract_phases(x_next)
                self.phase_history.append(phases)
            
            # Check convergence
            if self._check_convergence(x_current, x_next, convergence_threshold):
                break
            
            # Recursive: feed output back as input
            x_current = x_next
        
        # Detect attractors
        attractors = None
        if self.track_attractors and len(self.phase_history) > 10:
            attractors = self.attractor_detector.detect(self.phase_history)
        
        return {
            'output': x_next,
            'outputs': outputs,
            'metrics_history': metrics_history,
            'recursion_depth': depth + 1,
            'attractors': attractors,
            'phase_history': self.phase_history,
        }
    
    def _extract_metrics(self, x_prev: torch.Tensor, x_next: torch.Tensor) -> Dict:
        """Extract metrics from state transition."""
        # Order parameter R
        R = self._compute_order_parameter(x_next)
        
        # CDNS metrics
        cdns = self._compute_cdns(x_next)
        
        # Mutual Information with input
        mi_input = self._compute_mi(x_next, x_prev)
        
        # Latent dimensionality
        latent_dim = self._compute_latent_dimensionality(x_next)
        
        return {
            'R': R,
            'cdns': cdns,
            'mi_input': mi_input,
            'latent_dim': latent_dim,
        }
    
    def _extract_phases(self, x: torch.Tensor) -> torch.Tensor:
        """Extract phases from resonance heads."""
        phases = []
        
        # Extract phases from all resonance attention heads
        for module in self.base_model.modules():
            if isinstance(module, ResonanceAttentionHead):
                if hasattr(module, '_last_phases'):
                    phases.append(module._last_phases)
        
        if phases:
            return torch.stack(phases, dim=0)
        else:
            # Fallback: compute phases from activations
            return torch.angle(torch.fft.fft(x, dim=-1))
    
    def _check_convergence(
        self,
        x_prev: torch.Tensor,
        x_next: torch.Tensor,
        threshold: float,
    ) -> bool:
        """Check if state has converged."""
        diff = torch.norm(x_next - x_prev) / torch.norm(x_prev)
        return diff < threshold
```

---

## Part VI: Probes and Experiments

### Experiment 1: Invariance Shift

```python
def experiment_invariance_shift():
    """Probe invariance shift across recursion depth."""
    model = RecursiveResonanceTransformer(base_model)
    
    input_tokens = tokenize("The cat sat on the mat.")
    result = model.forward_recursive(input_tokens, recursion_depth=50)
    
    # Plot MI with input vs recursion depth
    mi_input = [m['mi_input'] for m in result['metrics_history']]
    plot(mi_input, title="MI with Input vs Recursion Depth")
    
    # Expected: Decreasing MI with input
```

### Experiment 2: Mode Collapse

```python
def experiment_mode_collapse():
    """Probe mode collapse to archetypes."""
    model = RecursiveResonanceTransformer(base_model)
    
    input_tokens = tokenize("The cat sat on the mat.")
    result = model.forward_recursive(input_tokens, recursion_depth=50)
    
    # Plot latent dimensionality vs recursion depth
    latent_dims = [m['latent_dim'] for m in result['metrics_history']]
    plot(latent_dims, title="Latent Dimensionality vs Recursion Depth")
    
    # Expected: Decreasing dimensionality → mode collapse
```

### Experiment 3: Attractor Detection

```python
def experiment_attractor_detection():
    """Probe attractor formation."""
    model = RecursiveResonanceTransformer(base_model, track_attractors=True)
    
    input_tokens = tokenize("The cat sat on the mat.")
    result = model.forward_recursive(input_tokens, recursion_depth=100)
    
    # Visualize phase trajectories
    visualize_phase_trajectories(result['phase_history'])
    
    # Detect attractors
    attractors = result['attractors']
    print(f"Found {len(attractors)} attractors")
    
    # Expected: Convergence to fixed points or limit cycles
```

### Experiment 4: Spectral Stabilization

```python
def experiment_spectral_stabilization():
    """Probe spectral gap stabilization."""
    model = RecursiveResonanceTransformer(base_model)
    
    input_tokens = tokenize("The cat sat on the mat.")
    result = model.forward_recursive(input_tokens, recursion_depth=50)
    
    # Compute eigenvalues at each depth
    eigenvalues_history = compute_eigenvalues_history(model, result['outputs'])
    
    # Find critical depth
    critical_depth = find_stabilization_depth(eigenvalues_history)
    
    print(f"Critical depth: {critical_depth}")
    
    # Expected: Stabilization at critical depth
```

---

## Part VII: Tree of Life as Recursive Structure

The **Tree of Life** network we built provides a natural framework for extreme recursion:

### Sefirot as Recursion Levels

- **Keter (Crown)**: Initial input (sensory latents)
- **Chokhmah/Binah**: First recursion (structural priors)
- **Chesed/Gevurah**: Second recursion (meta-laws)
- **Tiferet**: Critical depth (stabilization)
- **Netzach/Hod**: Limit cycles (attractors)
- **Yesod**: Renormalization (coarse-graining)
- **Malkhut (Kingdom)**: Final output (manifestation)

### Four Worlds as Recursion Phases

- **Atzilut**: Initial recursion (R ≈ 1.0, high information)
- **Beriah**: Mid recursion (R ≈ 0.8-0.9, compression)
- **Yetzirah**: Critical recursion (R ≈ 0.6, optimal)
- **Assiyah**: Deep recursion (R ≈ 0.3-0.5, mode collapse)

### Tikkun as Convergence Control

**Tikkun** (repair) maintains criticality (R ≈ 0.6) during recursion:
- Prevents over-synchronization (R → 1.0)
- Prevents chaos (R → 0.0)
- Maintains optimal recursion depth

---

## Conclusion

The Resonance Transformer provides a natural framework for probing extreme recursion phenomena:

1. **Phase dynamics** → attractor detection
2. **Order parameter R** → renormalization tracking
3. **CDNS metrics** → aliasing detection
4. **Criticality** → optimal recursion depth

By implementing recursive self-modeling and tracking these metrics, we can probe how GPT-2 (and other Transformers) evolve from "thinking about words" to "thinking about the structure of thought itself."

