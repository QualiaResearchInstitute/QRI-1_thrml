"""
Multi-Model Resonance Orchestrator

Coordinates multiple models (vision, language, etc.) with shared resonance dynamics,
cross-model attention routing, and unified metrics collection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
import threading

import torch
import torch.nn as nn

try:
    from transformers import PreTrainedModel  # type: ignore
except Exception:
    PreTrainedModel = object  # type: ignore


@dataclass
class ModelRegistration:
    """Registration info for a model in the orchestrator."""
    model_id: str
    model: nn.Module
    model_type: str  # "vision", "language", "multimodal", etc.
    adapters: List[Any] = field(default_factory=list)  # List of ResonanceAdapterBase instances
    metrics_history: List[Dict] = field(default_factory=list)
    resonance_state: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class CrossModelBridge:
    """Bridge configuration for cross-model resonance."""
    source_model_id: str
    target_model_id: str
    source_layer_idx: Optional[int] = None
    target_layer_idx: Optional[int] = None
    coupling_strength: float = 0.1
    bridge_type: str = "attention"  # "attention", "metrics", "state"
    enabled: bool = True


class MultiModelResonanceOrchestrator:
    """
    Orchestrates multiple models with shared resonance dynamics.
    
    Features:
    - Unified metrics collection across all models
    - Cross-model attention routing (vision -> language, etc.)
    - Shared resonance state synchronization
    - Cross-modal resonance bridges
    - Coordinated parameter updates
    - Ensemble prediction coordination
    """
    
    def __init__(
        self,
        enable_cross_model_routing: bool = True,
        enable_shared_state: bool = True,
        enable_metrics_aggregation: bool = True,
        sync_frequency: int = 1,  # Sync every N forward passes
    ):
        self.enable_cross_model_routing = enable_cross_model_routing
        self.enable_shared_state = enable_shared_state
        self.enable_metrics_aggregation = enable_metrics_aggregation
        self.sync_frequency = sync_frequency
        
        # Registered models
        self.models: Dict[str, ModelRegistration] = {}
        
        # Cross-model bridges
        self.bridges: List[CrossModelBridge] = []
        
        # Shared resonance state (aggregated across models)
        self.shared_state: Dict = {
            "global_consonance": 0.0,
            "global_dissonance": 0.0,
            "global_order_parameter": 0.0,
            "cross_model_coupling": {},
        }
        
        # Metrics aggregation
        self.global_metrics: Dict = {}
        self.metrics_history: List[Dict] = []
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Forward pass counter
        self._forward_count = 0
    
    def register_model(
        self,
        model_id: str,
        model: nn.Module,
        model_type: str = "generic",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Register a model with the orchestrator.
        
        Args:
            model_id: Unique identifier for the model
            model: The model instance (should have resonance adapters)
            model_type: Type of model ("vision", "language", "multimodal", etc.)
            metadata: Optional metadata about the model
        
        Returns:
            The model_id (for chaining)
        """
        with self._lock:
            if model_id in self.models:
                raise ValueError(f"Model {model_id} already registered")
            
            # Find all resonance adapters in the model
            adapters = []
            for module in model.modules():
                # Check if it's a resonance adapter (has res_heads attribute)
                if hasattr(module, "res_heads") and hasattr(module, "adapter_id"):
                    adapters.append(module)
                    # Register this orchestrator with the adapter
                    module.orchestrator = self
            
            registration = ModelRegistration(
                model_id=model_id,
                model=model,
                model_type=model_type,
                adapters=adapters,
                metadata=metadata or {},
            )
            
            self.models[model_id] = registration
            
            # Initialize shared state entry
            self.shared_state["cross_model_coupling"][model_id] = {}
            
            self._emit_event("model_registered", {
                "model_id": model_id,
                "model_type": model_type,
                "num_adapters": len(adapters),
            })
            
            return model_id
    
    def add_bridge(
        self,
        source_model_id: str,
        target_model_id: str,
        coupling_strength: float = 0.1,
        bridge_type: str = "attention",
        source_layer_idx: Optional[int] = None,
        target_layer_idx: Optional[int] = None,
    ) -> CrossModelBridge:
        """
        Add a cross-model resonance bridge.
        
        Args:
            source_model_id: Source model ID
            target_model_id: Target model ID
            coupling_strength: Strength of coupling (0.0 to 1.0)
            bridge_type: Type of bridge ("attention", "metrics", "state")
            source_layer_idx: Optional source layer index
            target_layer_idx: Optional target layer index
        
        Returns:
            The created bridge
        """
        if source_model_id not in self.models:
            raise ValueError(f"Source model {source_model_id} not registered")
        if target_model_id not in self.models:
            raise ValueError(f"Target model {target_model_id} not registered")
        
        bridge = CrossModelBridge(
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            source_layer_idx=source_layer_idx,
            target_layer_idx=target_layer_idx,
            coupling_strength=coupling_strength,
            bridge_type=bridge_type,
        )
        
        self.bridges.append(bridge)
        self._emit_event("bridge_added", {
            "source": source_model_id,
            "target": target_model_id,
            "bridge_type": bridge_type,
        })
        
        return bridge
    
    def on_adapter_event(
        self,
        adapter_id: str,
        event: str,
        data: Dict,
    ):
        """Called by adapters to notify orchestrator of events."""
        with self._lock:
            self._forward_count += 1
            
            # Update metrics if this is a forward completion
            if event == "forward_complete":
                self._update_metrics(adapter_id, data)
            
            # Sync shared state periodically
            if self._forward_count % self.sync_frequency == 0:
                self._sync_shared_state()
            
            # Apply cross-model bridges
            if self.enable_cross_model_routing:
                self._apply_bridges()
    
    def _update_metrics(self, adapter_id: str, data: Dict):
        """Update metrics from an adapter."""
        # Find which model this adapter belongs to
        model_id = None
        for mid, reg in self.models.items():
            if any(a.adapter_id == adapter_id for a in reg.adapters):
                model_id = mid
                break
        
        if model_id is None:
            return
        
        # Store metrics
        if "metrics" in data:
            self.models[model_id].metrics_history.append(data["metrics"])
            # Keep only recent history
            if len(self.models[model_id].metrics_history) > 100:
                self.models[model_id].metrics_history.pop(0)
        
        # Update global metrics
        if self.enable_metrics_aggregation:
            self._aggregate_global_metrics()
    
    def _aggregate_global_metrics(self):
        """Aggregate metrics across all models."""
        aggregated = {}
        
        for model_id, reg in self.models.items():
            if not reg.metrics_history:
                continue
            
            # Get latest metrics
            latest = reg.metrics_history[-1] if reg.metrics_history else {}
            
            # Aggregate by key
            for key, value in latest.items():
                if isinstance(value, (int, float, torch.Tensor)):
                    tensor_val = value if isinstance(value, torch.Tensor) else torch.tensor(float(value))
                    if key not in aggregated:
                        aggregated[key] = []
                    aggregated[key].append(tensor_val)
        
        # Average across models
        for key, values in aggregated.items():
            if values:
                try:
                    stacked = torch.stack([v.float() if isinstance(v, torch.Tensor) else torch.tensor(float(v)) for v in values])
                    aggregated[key] = stacked.mean().item()
                except Exception:
                    aggregated[key] = sum(float(v) for v in values) / len(values)
        
        self.global_metrics = aggregated
        self.metrics_history.append(aggregated.copy())
        
        # Keep only recent history
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
    
    def _sync_shared_state(self):
        """Synchronize shared resonance state across models."""
        if not self.enable_shared_state:
            return
        
        # Aggregate resonance states from all adapters
        all_states = []
        for reg in self.models.values():
            for adapter in reg.adapters:
                state = adapter.get_resonance_state()
                if state:
                    all_states.append(state)
        
        if not all_states:
            return
        
        # Compute global metrics from states
        # This is a simplified version - can be extended
        try:
            # Aggregate order parameters
            order_params = []
            for state in all_states:
                for key, value in state.items():
                    if "order" in key.lower() or "R" in key:
                        if isinstance(value, torch.Tensor):
                            order_params.append(value.mean().item())
                        else:
                            order_params.append(float(value))
            
            if order_params:
                self.shared_state["global_order_parameter"] = sum(order_params) / len(order_params)
        except Exception:
            pass
    
    def _apply_bridges(self):
        """Apply cross-model bridges."""
        for bridge in self.bridges:
            if not bridge.enabled:
                continue
            
            source_reg = self.models.get(bridge.source_model_id)
            target_reg = self.models.get(bridge.target_model_id)
            
            if source_reg is None or target_reg is None:
                continue
            
            if bridge.bridge_type == "attention":
                self._apply_attention_bridge(bridge, source_reg, target_reg)
            elif bridge.bridge_type == "metrics":
                self._apply_metrics_bridge(bridge, source_reg, target_reg)
            elif bridge.bridge_type == "state":
                self._apply_state_bridge(bridge, source_reg, target_reg)
    
    def _apply_attention_bridge(
        self,
        bridge: CrossModelBridge,
        source_reg: ModelRegistration,
        target_reg: ModelRegistration,
    ):
        """Apply attention-based cross-model bridge."""
        # Get source metrics/state
        source_metrics = {}
        if source_reg.metrics_history:
            source_metrics = source_reg.metrics_history[-1]
        
        # Influence target adapters
        target_adapters = target_reg.adapters
        if bridge.target_layer_idx is not None:
            target_adapters = [target_adapters[bridge.target_layer_idx]] if bridge.target_layer_idx < len(target_adapters) else []
        
        for adapter in target_adapters:
            # Extract influence from source metrics
            influence = self._compute_influence(source_metrics, bridge.coupling_strength)
            adapter.set_cross_model_influence(influence, bridge.source_model_id)
    
    def _apply_metrics_bridge(
        self,
        bridge: CrossModelBridge,
        source_reg: ModelRegistration,
        target_reg: ModelRegistration,
    ):
        """Apply metrics-based cross-model bridge."""
        # Copy relevant metrics from source to target
        if source_reg.metrics_history:
            source_metrics = source_reg.metrics_history[-1]
            # This can be used to modulate target model behavior
            # Implementation depends on specific use case
            pass
    
    def _apply_state_bridge(
        self,
        bridge: CrossModelBridge,
        source_reg: ModelRegistration,
        target_reg: ModelRegistration,
    ):
        """Apply state-based cross-model bridge."""
        # Synchronize resonance state between models
        source_state = source_reg.resonance_state
        target_state = target_reg.resonance_state
        
        # Blend states based on coupling strength
        for key, value in source_state.items():
            if key in target_state:
                blended = (
                    bridge.coupling_strength * value +
                    (1 - bridge.coupling_strength) * target_state[key]
                )
                target_state[key] = blended
    
    def _compute_influence(self, metrics: Dict, coupling_strength: float) -> float:
        """Compute influence strength from metrics."""
        # Simple heuristic: use order parameter or consonance
        if "order_parameter" in metrics or "R" in str(metrics):
            for key in ["order_parameter", "R", "cdns.R"]:
                if key in metrics:
                    val = float(metrics[key])
                    return coupling_strength * val
        
        # Fallback: use coupling strength directly
        return coupling_strength
    
    def get_global_metrics(self) -> Dict:
        """Get aggregated global metrics."""
        return self.global_metrics.copy()
    
    def get_model_metrics(self, model_id: str) -> List[Dict]:
        """Get metrics history for a specific model."""
        if model_id not in self.models:
            return []
        return self.models[model_id].metrics_history.copy()
    
    def get_shared_state(self) -> Dict:
        """Get shared resonance state."""
        return self.shared_state.copy()
    
    def synchronize_parameters(
        self,
        model_ids: Optional[List[str]] = None,
        sync_alpha: float = 0.1,
    ):
        """
        Synchronize resonance parameters across models.
        
        Args:
            model_ids: List of model IDs to sync (None = all models)
            sync_alpha: Synchronization strength (0.0 to 1.0)
        """
        if model_ids is None:
            model_ids = list(self.models.keys())
        
        # Get all adapters from selected models
        all_adapters = []
        for model_id in model_ids:
            if model_id in self.models:
                all_adapters.extend(self.models[model_id].adapters)
        
        if len(all_adapters) < 2:
            return
        
        # Average parameters across adapters
        # This is a simplified version - can be extended
        for adapter in all_adapters:
            for head in adapter.res_heads:
                # Example: synchronize coupling strength
                if hasattr(head, 'coupling_strength'):
                    # Get average from all other adapters
                    other_strengths = [
                        h.coupling_strength
                        for a in all_adapters
                        for h in a.res_heads
                        if hasattr(h, 'coupling_strength') and h is not head
                    ]
                    if other_strengths:
                        avg_strength = sum(other_strengths) / len(other_strengths)
                        head.coupling_strength = (
                            sync_alpha * avg_strength +
                            (1 - sync_alpha) * head.coupling_strength
                        )
    
    def coordinate_ensemble(
        self,
        model_outputs: Dict[str, torch.Tensor],
        aggregation_method: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Coordinate ensemble predictions from multiple models.
        
        Args:
            model_outputs: Dict mapping model_id -> output tensor
            aggregation_method: "weighted_average", "consensus", "resonance_weighted"
            weights: Optional weights for each model
        
        Returns:
            Aggregated output tensor
        """
        if not model_outputs:
            raise ValueError("No model outputs provided")
        
        outputs = list(model_outputs.values())
        
        if aggregation_method == "weighted_average":
            if weights is None:
                weights = {mid: 1.0 / len(model_outputs) for mid in model_outputs.keys()}
            weighted = sum(
                weights[mid] * output
                for mid, output in model_outputs.items()
            )
            return weighted
        
        elif aggregation_method == "resonance_weighted":
            # Weight by resonance metrics (order parameter, consonance, etc.)
            if weights is None:
                weights = {}
                for model_id in model_outputs.keys():
                    if model_id in self.models:
                        metrics = self.get_model_metrics(model_id)
                        if metrics:
                            latest = metrics[-1]
                            # Use order parameter or consonance as weight
                            weight = latest.get("order_parameter", latest.get("cdns.C", 0.5))
                            weights[model_id] = float(weight)
                        else:
                            weights[model_id] = 1.0 / len(model_outputs)
                    else:
                        weights[model_id] = 1.0 / len(model_outputs)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            weighted = sum(
                weights[mid] * output
                for mid, output in model_outputs.items()
            )
            return weighted
        
        elif aggregation_method == "consensus":
            # Simple average
            return sum(outputs) / len(outputs)
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def _emit_event(self, event_type: str, data: Dict):
        """Emit event to registered handlers."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(event_type, data)
            except Exception:
                pass
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        self.event_handlers[event_type].append(handler)
    
    def unregister_model(self, model_id: str):
        """Unregister a model."""
        with self._lock:
            if model_id in self.models:
                # Remove bridges involving this model
                self.bridges = [
                    b for b in self.bridges
                    if b.source_model_id != model_id and b.target_model_id != model_id
                ]
                del self.models[model_id]
                self._emit_event("model_unregistered", {"model_id": model_id})

