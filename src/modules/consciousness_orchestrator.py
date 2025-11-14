"""
Consciousness-Aware Multi-Model Orchestrator

Combines multi-model coordination with consciousness circuit patterns:
- Information flows through synchronized oscillator networks
- High synchronization = High throughput = "BRRRRRRRRR"
- Cross-model information routing via circuit paths
- Parallel processing through oscillator clusters
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn

from modules.multi_model_orchestrator import (
    MultiModelResonanceOrchestrator,
    ModelRegistration,
    CrossModelBridge,
)
from modules.consciousness_circuit import (
    compute_information_flow,
    compute_circuit_throughput,
    compute_parallel_clusters,
    compute_circuit_paths,
    ConsciousnessCircuitLayer,
)


class ConsciousnessAwareOrchestrator(MultiModelResonanceOrchestrator):
    """
    Multi-model orchestrator enhanced with consciousness circuit patterns.
    
    Features:
    - Extract resonance state (phases, amplitudes) from all models
    - Compute information flow through the multi-model network
    - Route information along high-coherence circuit paths
    - Optimize for maximum throughput ("BRRRRRRRRR")
    - Visualize circuit topology and clusters
    """
    
    def __init__(
        self,
        enable_cross_model_routing: bool = True,
        enable_shared_state: bool = True,
        enable_metrics_aggregation: bool = True,
        enable_circuit_processing: bool = True,
        coherence_threshold: float = 0.7,
        sync_frequency: int = 1,
    ):
        super().__init__(
            enable_cross_model_routing=enable_cross_model_routing,
            enable_shared_state=enable_shared_state,
            enable_metrics_aggregation=enable_metrics_aggregation,
            sync_frequency=sync_frequency,
        )
        
        self.enable_circuit_processing = enable_circuit_processing
        self.coherence_threshold = coherence_threshold
        
        # Consciousness circuit layer for processing
        if enable_circuit_processing:
            self.circuit_layer = ConsciousnessCircuitLayer(
                coherence_threshold=coherence_threshold,
                track_throughput=True,
            )
        
        # Circuit state tracking
        self.circuit_state: Dict = {
            "phases": None,
            "amplitudes": None,
            "coupling_matrix": None,
            "throughput": None,
            "clusters": None,
        }
    
    def extract_resonance_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract resonance state (phases, amplitudes, coupling) from all registered models.
        
        Returns:
            phases: [batch, n_oscillators] aggregated phases from all models
            amplitudes: [batch, n_oscillators] aggregated amplitudes
            coupling_matrix: [batch, n_oscillators, n_oscillators] cross-model coupling
        """
        all_phases = []
        all_amplitudes = []
        model_indices = {}  # Track which oscillators belong to which model
        
        current_idx = 0
        
        for model_id, reg in self.models.items():
            model_phases = []
            model_amplitudes = []
            
            for adapter in reg.adapters:
                state = adapter.get_resonance_state()
                
                # Extract phases and amplitudes from each head
                for h_idx, head in enumerate(adapter.res_heads):
                    if hasattr(head, 'phase') and head.phase is not None:
                        phase = head.phase
                        if isinstance(phase, torch.Tensor):
                            # Average across sequence length if needed
                            if phase.dim() > 1:
                                phase = phase.mean(dim=-1) if phase.dim() > 1 else phase
                            model_phases.append(phase)
                    
                    if hasattr(head, 'amplitude') and head.amplitude is not None:
                        amp = head.amplitude
                        if isinstance(amp, torch.Tensor):
                            if amp.dim() > 1:
                                amp = amp.mean(dim=-1) if amp.dim() > 1 else amp
                            model_amplitudes.append(amp)
            
            if model_phases:
                # Stack phases from this model
                model_phases_tensor = torch.stack(model_phases, dim=-1)  # [batch, n_heads]
                model_indices[model_id] = (current_idx, current_idx + model_phases_tensor.shape[-1])
                current_idx += model_phases_tensor.shape[-1]
                all_phases.append(model_phases_tensor)
            
            if model_amplitudes:
                model_amplitudes_tensor = torch.stack(model_amplitudes, dim=-1)
                all_amplitudes.append(model_amplitudes_tensor)
        
        if not all_phases:
            # Fallback: create dummy state
            batch_size = 1
            n_oscillators = len(self.models) * 8  # Estimate
            phases = torch.randn(batch_size, n_oscillators) * 0.1
            amplitudes = torch.ones(batch_size, n_oscillators) * 0.5
        else:
            # Concatenate across models
            phases = torch.cat(all_phases, dim=-1)  # [batch, total_oscillators]
            
            if all_amplitudes:
                amplitudes = torch.cat(all_amplitudes, dim=-1)
            else:
                amplitudes = torch.ones_like(phases) * 0.5
        
        # Build coupling matrix based on bridges
        batch_size, n_oscillators = phases.shape
        coupling_matrix = torch.zeros(batch_size, n_oscillators, n_oscillators, device=phases.device)
        
        # Initialize with small random coupling
        coupling_matrix = torch.randn_like(coupling_matrix) * 0.1
        
        # Add coupling based on bridges
        for bridge in self.bridges:
            if not bridge.enabled:
                continue
            
            source_range = model_indices.get(bridge.source_model_id)
            target_range = model_indices.get(bridge.target_model_id)
            
            if source_range and target_range:
                s_start, s_end = source_range
                t_start, t_end = target_range
                
                # Add coupling between source and target oscillators
                coupling_matrix[:, s_start:s_end, t_start:t_end] += bridge.coupling_strength
                coupling_matrix[:, t_start:t_end, s_start:s_end] += bridge.coupling_strength
        
        # Make symmetric
        coupling_matrix = (coupling_matrix + coupling_matrix.transpose(-2, -1)) / 2
        
        return phases, amplitudes, coupling_matrix
    
    def compute_circuit_metrics(self) -> Dict[str, Any]:
        """
        Compute consciousness circuit metrics from current resonance state.
        
        Returns:
            Dictionary with throughput, flow, clusters, etc.
        """
        if not self.enable_circuit_processing:
            return {}
        
        try:
            phases, amplitudes, coupling_matrix = self.extract_resonance_state()
            
            # Compute throughput metrics
            throughput_metrics = compute_circuit_throughput(
                phases,
                coupling_matrix,
                amplitudes,
            )
            
            # Compute information flow
            flow = compute_information_flow(phases, coupling_matrix, amplitudes)
            
            # Compute clusters
            _, clusters = compute_parallel_clusters(
                phases,
                coupling_matrix,
                self.coherence_threshold,
            )
            
            # Store circuit state
            self.circuit_state = {
                "phases": phases.detach(),
                "amplitudes": amplitudes.detach(),
                "coupling_matrix": coupling_matrix.detach(),
                "throughput": throughput_metrics.get("throughput", torch.tensor(0.0)),
                "clusters": clusters,
                "flow": flow.detach(),
            }
            
            metrics = {
                **throughput_metrics,
                "n_clusters": len(clusters),
                "cluster_sizes": [len(c) for c in clusters],
                "flow_mean": flow.mean().item(),
                "flow_max": flow.abs().max().item(),
            }
            
            return metrics
            
        except Exception as e:
            # Fallback on error
            return {"error": str(e)}
    
    def optimize_for_throughput(
        self,
        n_iterations: int = 5,
        learning_rate: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Optimize model coordination for maximum throughput.
        
        Uses consciousness circuit processing to maximize information flow.
        """
        if not self.enable_circuit_processing:
            return {}
        
        phases, amplitudes, coupling_matrix = self.extract_resonance_state()
        
        # Process through circuit layer
        optimized_phases = phases.clone()
        all_metrics = []
        
        for iteration in range(n_iterations):
            if hasattr(self, 'circuit_layer'):
                optimized_phases, metrics = self.circuit_layer(
                    optimized_phases,
                    coupling_matrix,
                    amplitudes,
                )
                all_metrics.append(metrics)
            
            # Compute throughput
            throughput_metrics = compute_circuit_throughput(
                optimized_phases,
                coupling_matrix,
                amplitudes,
            )
            
            # Update coupling based on throughput (simple heuristic)
            # Increase coupling where flow is high
            flow = compute_information_flow(optimized_phases, coupling_matrix, amplitudes)
            flow_abs = flow.abs()
            
            # Adaptive coupling update
            coupling_update = learning_rate * flow_abs.mean(dim=0, keepdim=True)
            coupling_matrix = coupling_matrix + coupling_update
            coupling_matrix = (coupling_matrix + coupling_matrix.transpose(-2, -1)) / 2  # Keep symmetric
        
        # Update circuit state
        self.circuit_state["phases"] = optimized_phases.detach()
        self.circuit_state["coupling_matrix"] = coupling_matrix.detach()
        
        # Aggregate metrics
        final_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    if isinstance(values[0], torch.Tensor):
                        final_metrics[key] = torch.stack(values).mean()
                    else:
                        final_metrics[key] = sum(values) / len(values)
        
        final_metrics.update(compute_circuit_throughput(optimized_phases, coupling_matrix, amplitudes))
        
        return final_metrics
    
    def find_circuit_paths(
        self,
        source_model_id: str,
        target_model_id: str,
        max_path_length: int = 5,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """
        Find circuit paths from source to target model through the oscillator network.
        
        Returns:
            paths: List of paths (oscillator indices)
            path_strengths: Path strengths
        """
        phases, amplitudes, coupling_matrix = self.extract_resonance_state()
        
        # Find model oscillator ranges
        model_indices = {}
        current_idx = 0
        
        for model_id, reg in self.models.items():
            n_oscillators = sum(len(adapter.res_heads) for adapter in reg.adapters)
            model_indices[model_id] = (current_idx, current_idx + n_oscillators)
            current_idx += n_oscillators
        
        source_range = model_indices.get(source_model_id)
        target_range = model_indices.get(target_model_id)
        
        if not source_range or not target_range:
            return [], torch.tensor([])
        
        # Use first oscillator from each model as representative
        source_idx = source_range[0]
        target_idx = target_range[0]
        
        paths, path_strengths = compute_circuit_paths(
            phases,
            coupling_matrix,
            source_idx,
            target_idx,
            max_path_length,
        )
        
        return paths, path_strengths
    
    def get_circuit_topology(self) -> Dict[str, Any]:
        """
        Get circuit topology information (clusters, flow, etc.).
        """
        metrics = self.compute_circuit_metrics()
        
        return {
            "circuit_state": self.circuit_state,
            "metrics": metrics,
            "models": list(self.models.keys()),
            "bridges": [
                {
                    "source": b.source_model_id,
                    "target": b.target_model_id,
                    "coupling": b.coupling_strength,
                    "type": b.bridge_type,
                }
                for b in self.bridges
            ],
        }
    
    def _sync_shared_state(self):
        """Override to include circuit metrics."""
        super()._sync_shared_state()
        
        if self.enable_circuit_processing:
            circuit_metrics = self.compute_circuit_metrics()
            self.shared_state["circuit_throughput"] = circuit_metrics.get("throughput", torch.tensor(0.0))
            self.shared_state["circuit_clusters"] = circuit_metrics.get("n_clusters", 0)
    
    def coordinate_ensemble(
        self,
        model_outputs: Dict[str, torch.Tensor],
        aggregation_method: str = "resonance_weighted",
        weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Override to use circuit throughput for weighting.
        """
        if aggregation_method == "circuit_weighted" and self.enable_circuit_processing:
            # Weight by circuit throughput contribution
            circuit_metrics = self.compute_circuit_metrics()
            
            if weights is None:
                weights = {}
                for model_id in model_outputs.keys():
                    # Use throughput contribution as weight
                    # Models with higher synchronization contribute more
                    throughput = circuit_metrics.get("throughput", torch.tensor(0.5))
                    weights[model_id] = float(throughput.mean().item()) if isinstance(throughput, torch.Tensor) else 0.5
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            weighted = sum(
                weights[mid] * output
                for mid, output in model_outputs.items()
            )
            return weighted
        
        # Fall back to parent implementation
        return super().coordinate_ensemble(model_outputs, aggregation_method, weights)

