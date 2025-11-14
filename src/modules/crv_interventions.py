"""
CRV Intervention Engine: Performs targeted interventions on oscillator features.

Suppresses problematic features or amplifies underactive features to correct reasoning errors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Callable, Any
import torch
import torch.nn as nn


class OscillatorIntervention:
    """
    Performs targeted interventions on oscillator features to correct reasoning.
    
    Can suppress problematic features or amplify underactive features via forward hooks.
    """
    
    def __init__(self):
        """Initialize intervention engine."""
        self.active_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.intervention_history: List[Dict[str, Any]] = []
    
    def suppress_feature(
        self,
        model: nn.Module,
        feature_id: int,
        layer_idx: int,
        head_idx: int = 0,
        value: float = 0.0,
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Suppress a specific oscillator feature by clamping its activation.
        
        Args:
            model: Model to intervene on
            feature_id: Feature ID to suppress (e.g., oscillator index)
            layer_idx: Layer index (0-based)
            head_idx: Head index (0-based, default: 0)
            value: Value to clamp to (default: 0.0)
        
        Returns:
            Hook handle (call .remove() to undo intervention)
        """
        head = self._get_resonance_head(model, layer_idx, head_idx)
        if head is None:
            raise ValueError(f"Could not find resonance head at layer {layer_idx}, head {head_idx}")
        
        def hook_fn(module, input, output):
            """
            Hook function to suppress feature activation.
            
            Note: This is a simplified intervention. In practice, you'd want to
            intervene on specific oscillator features (phases, amplitudes, coupling).
            This hook intervenes on the head's output, which is a proxy for
            suppressing the feature's influence.
            """
            # For now, we'll need to modify the head's internal state
            # This is a placeholder - actual implementation depends on head structure
            return output
        
        # Register hook on the head's forward method
        handle = head.register_forward_hook(hook_fn)
        self.active_hooks.append(handle)
        
        # Store intervention info
        self.intervention_history.append({
            'type': 'suppress',
            'feature_id': feature_id,
            'layer_idx': layer_idx,
            'head_idx': head_idx,
            'value': value,
        })
        
        return handle
    
    def amplify_feature(
        self,
        model: nn.Module,
        feature_id: int,
        layer_idx: int,
        head_idx: int = 0,
        amplification_factor: float = 2.0,
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Amplify a specific oscillator feature by multiplying its activation.
        
        Args:
            model: Model to intervene on
            feature_id: Feature ID to amplify
            layer_idx: Layer index (0-based)
            head_idx: Head index (0-based, default: 0)
            amplification_factor: Factor to multiply activation by (default: 2.0)
        
        Returns:
            Hook handle (call .remove() to undo intervention)
        """
        head = self._get_resonance_head(model, layer_idx, head_idx)
        if head is None:
            raise ValueError(f"Could not find resonance head at layer {layer_idx}, head {head_idx}")
        
        def hook_fn(module, input, output):
            """
            Hook function to amplify feature activation.
            """
            # Placeholder - actual implementation depends on head structure
            return output
        
        handle = head.register_forward_hook(hook_fn)
        self.active_hooks.append(handle)
        
        self.intervention_history.append({
            'type': 'amplify',
            'feature_id': feature_id,
            'layer_idx': layer_idx,
            'head_idx': head_idx,
            'amplification_factor': amplification_factor,
        })
        
        return handle
    
    def suppress_phase_cluster(
        self,
        model: nn.Module,
        cluster_id: int,
        layer_idx: int,
        head_idx: int = 0,
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Suppress a phase coherence cluster (synchronized oscillator group).
        
        Args:
            model: Model to intervene on
            cluster_id: Phase cluster ID to suppress
            layer_idx: Layer index
            head_idx: Head index
        
        Returns:
            Hook handle
        """
        # This would require access to phase clustering information
        # For now, use generic suppress_feature
        return self.suppress_feature(model, cluster_id, layer_idx, head_idx)
    
    def amplify_phase_cluster(
        self,
        model: nn.Module,
        cluster_id: int,
        layer_idx: int,
        head_idx: int = 0,
        amplification_factor: float = 2.0,
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Amplify a phase coherence cluster.
        
        Args:
            model: Model to intervene on
            cluster_id: Phase cluster ID to amplify
            layer_idx: Layer index
            head_idx: Head index
            amplification_factor: Amplification factor
        
        Returns:
            Hook handle
        """
        return self.amplify_feature(model, cluster_id, layer_idx, head_idx, amplification_factor)
    
    def suppress_coupling_hub(
        self,
        model: nn.Module,
        hub_position: int,
        layer_idx: int,
        head_idx: int = 0,
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Suppress a coupling hub (highly connected oscillator).
        
        Args:
            model: Model to intervene on
            hub_position: Position of coupling hub
            layer_idx: Layer index
            head_idx: Head index
        
        Returns:
            Hook handle
        """
        return self.suppress_feature(model, hub_position, layer_idx, head_idx)
    
    def intervene_on_phases(
        self,
        model: nn.Module,
        layer_idx: int,
        head_idx: int = 0,
        phase_mask: Optional[torch.Tensor] = None,
        phase_offsets: Optional[torch.Tensor] = None,
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Intervene directly on oscillator phases.
        
        Args:
            model: Model to intervene on
            layer_idx: Layer index
            head_idx: Head index
            phase_mask: Boolean mask for which phases to modify [seq_len]
            phase_offsets: Phase offsets to apply [seq_len]
        
        Returns:
            Hook handle
        """
        head = self._get_resonance_head(model, layer_idx, head_idx)
        if head is None:
            raise ValueError(f"Could not find resonance head at layer {layer_idx}, head {head_idx}")
        
        def hook_fn(module, input, output):
            """
            Hook to modify phases during simulation.
            
            Note: This requires modifying the head's internal simulation state.
            This is a placeholder - actual implementation would need to
            access and modify phases_history or phases during kuramoto_simulation.
            """
            # Would need to modify phases before/during simulation
            return output
        
        handle = head.register_forward_hook(hook_fn)
        self.active_hooks.append(handle)
        
        self.intervention_history.append({
            'type': 'phase_intervention',
            'layer_idx': layer_idx,
            'head_idx': head_idx,
            'phase_mask': phase_mask,
            'phase_offsets': phase_offsets,
        })
        
        return handle
    
    def intervene_on_coupling(
        self,
        model: nn.Module,
        layer_idx: int,
        head_idx: int = 0,
        coupling_mask: Optional[torch.Tensor] = None,
        coupling_multiplier: Optional[torch.Tensor] = None,
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Intervene on coupling matrix.
        
        Args:
            model: Model to intervene on
            layer_idx: Layer index
            head_idx: Head index
            coupling_mask: Boolean mask for which couplings to modify [seq_len, seq_len]
            coupling_multiplier: Multiplier to apply to couplings [seq_len, seq_len]
        
        Returns:
            Hook handle
        """
        head = self._get_resonance_head(model, layer_idx, head_idx)
        if head is None:
            raise ValueError(f"Could not find resonance head at layer {layer_idx}, head {head_idx}")
        
        def hook_fn(module, input, output):
            """
            Hook to modify coupling matrix.
            
            Would need to modify coupling_matrix before simulation.
            """
            return output
        
        handle = head.register_forward_hook(hook_fn)
        self.active_hooks.append(handle)
        
        self.intervention_history.append({
            'type': 'coupling_intervention',
            'layer_idx': layer_idx,
            'head_idx': head_idx,
            'coupling_mask': coupling_mask,
            'coupling_multiplier': coupling_multiplier,
        })
        
        return handle
    
    def _get_resonance_head(
        self,
        model: nn.Module,
        layer_idx: int,
        head_idx: int = 0,
    ) -> Optional[nn.Module]:
        """
        Get resonance head at specified layer and head index.
        
        Args:
            model: Model
            layer_idx: Layer index (0-based)
            head_idx: Head index (0-based)
        
        Returns:
            ResonanceAttentionHead or None if not found
        """
        try:
            # Access via model structure (GPT-2 style)
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layers = model.transformer.h
                if 0 <= layer_idx < len(layers):
                    layer = layers[layer_idx]
                    if hasattr(layer, 'attn'):
                        attn = layer.attn
                        # Check if it's a ResonanceGPT2AttentionAdapter
                        if hasattr(attn, 'res_heads'):
                            heads = attn.res_heads
                            if 0 <= head_idx < len(heads):
                                return heads[head_idx]
            
            # Fallback: iterate through all heads
            from modules.resonance_gpt2_adapter import iter_resonance_heads
            
            heads_list = list(iter_resonance_heads(model))
            # Simple mapping: assume heads are ordered by layer then head index
            # This is approximate - actual mapping depends on model structure
            total_heads = len(heads_list)
            if total_heads > 0:
                # Estimate: assume equal heads per layer
                n_layers = len(model.transformer.h) if hasattr(model, 'transformer') else 1
                heads_per_layer = total_heads // n_layers if n_layers > 0 else total_heads
                idx = layer_idx * heads_per_layer + head_idx
                if 0 <= idx < len(heads_list):
                    return heads_list[idx]
            
            return None
        except Exception as e:
            print(f"Warning: Failed to get resonance head: {e}")
            return None
    
    def remove_all_hooks(self) -> None:
        """Remove all active intervention hooks."""
        for hook in self.active_hooks:
            hook.remove()
        self.active_hooks.clear()
    
    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """Get history of all interventions."""
        return self.intervention_history.copy()


class InterventionResult:
    """Result of an intervention attempt."""
    
    def __init__(
        self,
        success: bool,
        original_output: str,
        corrected_output: str,
        intervention_type: str,
        feature_id: Optional[int] = None,
        confidence_before: Optional[float] = None,
        confidence_after: Optional[float] = None,
    ):
        self.success = success
        self.original_output = original_output
        self.corrected_output = corrected_output
        self.intervention_type = intervention_type
        self.feature_id = feature_id
        self.confidence_before = confidence_before
        self.confidence_after = confidence_after
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'original_output': self.original_output,
            'corrected_output': self.corrected_output,
            'intervention_type': self.intervention_type,
            'feature_id': self.feature_id,
            'confidence_before': self.confidence_before,
            'confidence_after': self.confidence_after,
        }


def correct_reasoning_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    incorrect_step: str,
    crv_classifier: Any,  # CRVDiagnosticClassifier
    graph_builder: Any,  # OscillatorAttributionGraph
    feature_extractor: Any,  # GraphFeatureExtractor
    intervention: OscillatorIntervention,
    max_interventions: int = 5,
) -> Tuple[str, InterventionResult]:
    """
    Attempt to correct reasoning via targeted interventions.
    
    Process:
    1. Run model, detect incorrect step via CRV
    2. Extract attribution graph, identify problematic features
    3. Try interventions (suppress/amplify features)
    4. Re-run, check if corrected
    
    Args:
        model: Model to correct
        input_ids: Input token IDs [batch, seq_len]
        incorrect_step: Text of incorrect step
        crv_classifier: Trained CRV classifier
        graph_builder: OscillatorAttributionGraph instance
        feature_extractor: GraphFeatureExtractor instance
        intervention: OscillatorIntervention instance
        max_interventions: Maximum number of interventions to try
    
    Returns:
        (corrected_output, intervention_result)
    """
    # Step 1: Run model and detect error
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
    
    # Extract oscillator features
    oscillator_features = extract_oscillator_features(model)
    
    # Build graph and extract features
    graph = graph_builder.build_graph(oscillator_features, logits)
    features = feature_extractor.extract_features(graph)
    
    # Predict correctness
    is_correct, confidence = crv_classifier.predict(features)
    
    if is_correct:
        # Already correct
        return incorrect_step, InterventionResult(
            success=True,
            original_output=incorrect_step,
            corrected_output=incorrect_step,
            intervention_type="none",
            confidence_before=confidence,
            confidence_after=confidence,
        )
    
    # Step 2: Identify problematic features
    feature_importance = crv_classifier.get_feature_importance()
    
    # Get feature nodes from graph
    problematic_features = identify_problematic_features(graph, features, feature_importance)
    
    # Step 3: Try interventions
    original_output = incorrect_step
    
    for i, (feature_id, feature_type, layer_idx, head_idx) in enumerate(problematic_features[:max_interventions]):
        try:
            # Try suppressing problematic feature
            handle = intervention.suppress_feature(
                model,
                feature_id=feature_id,
                layer_idx=layer_idx,
                head_idx=head_idx,
            )
            
            # Re-run model
            with torch.no_grad():
                new_outputs = model(input_ids)
                new_logits = new_outputs.logits if hasattr(new_outputs, "logits") else new_outputs
            
            # Check if corrected
            new_oscillator_features = extract_oscillator_features(model)
            new_graph = graph_builder.build_graph(new_oscillator_features, new_logits)
            new_features = feature_extractor.extract_features(new_graph)
            new_correct, new_confidence = crv_classifier.predict(new_features)
            
            # Remove hook
            handle.remove()
            
            if new_correct:
                # Success!
                return original_output, InterventionResult(
                    success=True,
                    original_output=original_output,
                    corrected_output=original_output,  # Would need to decode from logits
                    intervention_type="suppress",
                    feature_id=feature_id,
                    confidence_before=confidence,
                    confidence_after=new_confidence,
                )
        except Exception as e:
            print(f"Warning: Intervention {i} failed: {e}")
            continue
    
    # Failed to correct
    return original_output, InterventionResult(
        success=False,
        original_output=original_output,
        corrected_output=original_output,
        intervention_type="suppress",
        confidence_before=confidence,
        confidence_after=confidence,
    )


def extract_oscillator_features(model: nn.Module) -> Dict[str, Any]:
    """
    Extract oscillator features from model.
    
    Placeholder - would need to access resonance head metrics.
    
    Args:
        model: Model
    
    Returns:
        Dictionary with oscillator features
    """
    # This would extract phases, amplitudes, coupling from resonance heads
    # For now, return empty dict
    return {}


def identify_problematic_features(
    graph: Any,  # nx.DiGraph
    features: Dict[str, float],
    feature_importance: Dict[str, float],
) -> List[Tuple[int, str, int, int]]:
    """
    Identify problematic features from graph and feature importance.
    
    Args:
        graph: Attribution graph
        features: Extracted features
        feature_importance: Feature importance from classifier
    
    Returns:
        List of (feature_id, feature_type, layer_idx, head_idx) tuples
    """
    problematic = []
    
    # Identify features with high influence but low correctness signal
    # This is a simplified heuristic - would need more sophisticated analysis
    
    # For now, return empty list
    return problematic

