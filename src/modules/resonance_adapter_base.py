"""
Base class for resonance attention adapters.

Provides common functionality for wrapping transformer attention modules
with resonance dynamics, enabling multi-model coordination.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

from resonance_transformer import ResonanceAttentionHead


class ResonanceAdapterBase(nn.Module, ABC):
    """
    Base class for resonance attention adapters.
    
    Provides common functionality for:
    - Resonance head management
    - Metrics aggregation
    - Multi-model coordination hooks
    - Unified interface for different transformer architectures
    """
    
    def __init__(
        self,
        base_attn: nn.Module,
        n_heads: int,
        d_model: int,
        *,
        attention_type: str = "resonance",
        hybrid_alpha: float = 0.8,
        resonance_kwargs: Optional[Dict] = None,
        adapter_id: Optional[str] = None,
        orchestrator: Optional[Any] = None,  # MultiModelResonanceOrchestrator
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.base_attn = base_attn
        self.n_heads = int(n_heads)
        self.d_model = int(d_model)
        self.head_dim = self.d_model // self.n_heads
        self.attention_type = attention_type
        self.hybrid_alpha = float(hybrid_alpha)
        self.adapter_id = adapter_id or f"adapter_{id(self)}"
        self.orchestrator = orchestrator
        
        resonance_kwargs = resonance_kwargs or {}
        
        # Build resonance heads
        heads: List[ResonanceAttentionHead] = []
        for h_idx in range(self.n_heads):
            head = self._create_resonance_head(h_idx, resonance_kwargs)
            heads.append(head)
        self.res_heads = nn.ModuleList(heads)
        
        # Storage for metrics
        self.last_metrics: Dict = {}
        self.last_output: Optional[torch.Tensor] = None
        
        # CoT mode
        self.cot_mode: bool = False
        self._original_n_sim_steps: List[int] = [head.n_sim_steps for head in heads]
    
    def _create_resonance_head(
        self,
        head_idx: int,
        resonance_kwargs: Dict,
    ) -> ResonanceAttentionHead:
        """Create a resonance attention head with standard configuration."""
        return ResonanceAttentionHead(
            d_model=self.head_dim,
            d_k=self.head_dim,
            d_v=self.head_dim,
            n_sim_steps=int(resonance_kwargs.get("n_sim_steps", 10)),
            dt=float(resonance_kwargs.get("dt", 0.01)),
            use_sakaguchi=bool(resonance_kwargs.get("use_sakaguchi", True)),
            use_stuart_landau=bool(resonance_kwargs.get("use_stuart_landau", True)),
            use_heun=bool(resonance_kwargs.get("use_heun", True)),
            track_cdns=bool(resonance_kwargs.get("track_cdns", True)),
            use_extended_cdns=bool(resonance_kwargs.get("use_extended_cdns", False)),
            use_coupling_kernel=bool(resonance_kwargs.get("use_coupling_kernel", True)),
            lambda_criticality=float(resonance_kwargs.get("lambda_criticality", 0.0)),
            lambda_dissonance=float(resonance_kwargs.get("lambda_dissonance", 0.0)),
            target_R=float(resonance_kwargs.get("target_R", 0.6)),
            store_visualization_traces=bool(resonance_kwargs.get("store_visualization_traces", False)),
            visualization_history_limit=int(resonance_kwargs.get("visualization_history_limit", max(4, int(resonance_kwargs.get("n_sim_steps", 10))))),
            hybrid_readout=True,
            hybrid_mix_init=float(resonance_kwargs.get("hybrid_mix_init", self.hybrid_alpha)),
            telemetry=bool(resonance_kwargs.get("telemetry", False)),
            use_spectral_gating=bool(resonance_kwargs.get("use_spectral_gating", True)),
            spectral_num_bands=int(resonance_kwargs.get("spectral_num_bands", 4)),
            use_delays=bool(resonance_kwargs.get("use_delays", False)),
            tau_steps=int(resonance_kwargs.get("tau_steps", 0)),
            delay_gain=float(resonance_kwargs.get("delay_gain", 1.0)),
            use_learnable_delays=bool(resonance_kwargs.get("use_learnable_delays", False)),
            use_temporal_multiplex=bool(resonance_kwargs.get("use_temporal_multiplex", False)),
            tm_dts=[float(x) for x in resonance_kwargs.get("tm_dts", [])],
            tm_alpha_offsets=[float(x) for x in resonance_kwargs.get("tm_alpha_offsets", [])],
            tm_learned_mix=bool(resonance_kwargs.get("tm_learned_mix", True)),
            use_pid_autopilot=bool(resonance_kwargs.get("use_pid_autopilot", False) or resonance_kwargs.get("use_alpha_controller", False)),
            use_harmonics=bool(resonance_kwargs.get("use_harmonics", False)),
            harmonics=resonance_kwargs.get("harmonics", None),
            kappa_harm=resonance_kwargs.get("kappa_harm", None),
            alpha_harm=resonance_kwargs.get("alpha_harm", None),
            use_msf_regularizer=bool(resonance_kwargs.get("use_msf_regularizer", False)),
            lambda_msf=float(resonance_kwargs.get("lambda_msf", 0.0)),
            msf_target=str(resonance_kwargs.get("msf_target", "critical")),
            msf_n_clusters=int(resonance_kwargs.get("msf_n_clusters", 8)),
            use_msf_autotune=bool(resonance_kwargs.get("use_msf_autotune", False)),
            msf_autotune_safety=float(resonance_kwargs.get("msf_autotune_safety", 0.9)),
            msf_eval_every=int(resonance_kwargs.get("msf_eval_every", 5)),
        )
    
    def _aggregate_head_metrics(self, metric_list: List[Dict]) -> Dict:
        """Aggregate metrics across heads."""
        agg: Dict = {}
        for m in metric_list:
            for k, v in (m or {}).items():
                try:
                    if isinstance(v, dict):
                        for sk, sv in v.items():
                            key = f"{k}.{sk}"
                            if isinstance(sv, torch.Tensor):
                                agg.setdefault(key, []).append(sv.detach())
                    elif isinstance(v, torch.Tensor):
                        agg.setdefault(k, []).append(v.detach())
                except Exception:
                    pass
        red: Dict = {}
        for k, items in agg.items():
            try:
                stacked = torch.stack([t.float().mean().reshape(1) for t in items], dim=0)
                red[k] = stacked.mean()
            except Exception:
                red[k] = items[0] if items else None
        return red
    
    def _notify_orchestrator(self, event: str, data: Optional[Dict] = None):
        """Notify orchestrator of events (if registered)."""
        if self.orchestrator is not None:
            try:
                self.orchestrator.on_adapter_event(self.adapter_id, event, data or {})
            except Exception:
                pass
    
    @abstractmethod
    def _forward_resonance(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through resonance heads. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _forward_vanilla(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass through vanilla attention. Must be implemented by subclasses."""
        pass
    
    def toggle_cot_mode(self, enabled: bool):
        """Toggle Chain-of-Thought mode."""
        self.cot_mode = enabled
        for i, head in enumerate(self.res_heads):
            if enabled:
                head.n_sim_steps = self._original_n_sim_steps[i] * 2
            else:
                head.n_sim_steps = self._original_n_sim_steps[i]
    
    def get_metrics(self) -> Dict:
        """Get aggregated metrics from last forward pass."""
        return self.last_metrics.copy()
    
    def get_resonance_state(self) -> Dict:
        """Get current resonance state (phases, amplitudes, etc.)."""
        state = {}
        for h_idx, head in enumerate(self.res_heads):
            if hasattr(head, 'phase'):
                state[f"head_{h_idx}_phase"] = head.phase.detach() if hasattr(head.phase, 'detach') else head.phase
            if hasattr(head, 'amplitude'):
                state[f"head_{h_idx}_amplitude"] = head.amplitude.detach() if hasattr(head.amplitude, 'detach') else head.amplitude
        return state
    
    def set_cross_model_influence(self, influence_strength: float, source_adapter_id: Optional[str] = None):
        """Set cross-model influence strength (called by orchestrator)."""
        # This can be used to modulate coupling based on other models' states
        for head in self.res_heads:
            if hasattr(head, 'coupling_kernel') and hasattr(head.coupling_kernel, 'set_external_influence'):
                head.coupling_kernel.set_external_influence(influence_strength)
