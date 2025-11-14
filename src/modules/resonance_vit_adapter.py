from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    # Transformers is optional in this repo; import types for hints only
    from transformers import PreTrainedModel  # type: ignore
except Exception:  # pragma: no cover
    PreTrainedModel = object  # type: ignore

# Reuse our resonance head implementation
from resonance_transformer import ResonanceAttentionHead


class ResonanceViTAttentionAdapter(nn.Module):
    """
    Drop-in replacement for ViT/CLIP style attention that routes through ResonanceAttentionHead(s)
    under a config flag. Provides a safe fallback to the original ViT attention.

    Notes:
    - This adapter works with ViTSelfAttention and CLIPSelfAttention modules
    - ViT uses bidirectional attention (no causal mask), unlike GPT-2
    - We split hidden_states into per-head slices and let each resonance head process its slice
    - Hybrid readout is enabled so outputs blend resonance and vanilla QK^T
    - Supports both ViT (google/vit-base-patch16-224) and CLIP (openai/clip-vit-large-patch14) architectures
    """

    def __init__(
        self,
        base_attn: nn.Module,
        n_heads: int,
        d_model: int,
        *,
        attention_type: str = "resonance",  # "resonance" | "vanilla"
        hybrid_alpha: float = 0.9,          # weight on vanilla QK path in [0,1]; higher is closer to vanilla
        resonance_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.base_attn = base_attn
        self.n_heads = int(n_heads)
        self.d_model = int(d_model)
        self.head_dim = self.d_model // self.n_heads
        self.attention_type = attention_type
        self.hybrid_alpha = float(hybrid_alpha)
        resonance_kwargs = resonance_kwargs or {}

        # Build resonance heads (one per ViT head slice)
        heads: List[ResonanceAttentionHead] = []
        for _ in range(self.n_heads):
            head = ResonanceAttentionHead(
                d_model=self.head_dim,
                d_k=self.head_dim,
                d_v=self.head_dim,
                # Small-N defaults; can be overridden via resonance_kwargs
                n_sim_steps=int(resonance_kwargs.get("n_sim_steps", 10)),
                dt=float(resonance_kwargs.get("dt", 0.01)),
                use_sakaguchi=bool(resonance_kwargs.get("use_sakaguchi", True)),
                use_stuart_landau=bool(resonance_kwargs.get("use_stuart_landau", True)),
                use_heun=bool(resonance_kwargs.get("use_heun", True)),
                track_cdns=bool(resonance_kwargs.get("track_cdns", True)),
                use_extended_cdns=bool(resonance_kwargs.get("use_extended_cdns", False)),
                use_coupling_kernel=bool(resonance_kwargs.get("use_coupling_kernel", True)),
                # Regularizers (kept small by default)
                lambda_criticality=float(resonance_kwargs.get("lambda_criticality", 0.0)),
                lambda_dissonance=float(resonance_kwargs.get("lambda_dissonance", 0.0)),
                target_R=float(resonance_kwargs.get("target_R", 0.6)),
                store_visualization_traces=bool(resonance_kwargs.get("store_visualization_traces", False)),
                visualization_history_limit=int(resonance_kwargs.get("visualization_history_limit", max(4, int(resonance_kwargs.get("n_sim_steps", 10))))),
                # Hybrid readout: enable and initialize with requested alpha
                hybrid_readout=True,
                hybrid_mix_init=float(resonance_kwargs.get("hybrid_mix_init", self.hybrid_alpha)),
                # Stabilizers
                telemetry=bool(resonance_kwargs.get("telemetry", False)),
                use_spectral_gating=bool(resonance_kwargs.get("use_spectral_gating", True)),
                spectral_num_bands=int(resonance_kwargs.get("spectral_num_bands", 4)),
                # Delays
                use_delays=bool(resonance_kwargs.get("use_delays", False)),
                tau_steps=int(resonance_kwargs.get("tau_steps", 0)),
                delay_gain=float(resonance_kwargs.get("delay_gain", 1.0)),
                use_learnable_delays=bool(resonance_kwargs.get("use_learnable_delays", False)),
                # Multiplex (core toggles)
                use_temporal_multiplex=bool(resonance_kwargs.get("use_temporal_multiplex", False)),
                tm_dts=[float(x) for x in resonance_kwargs.get("tm_dts", [])],
                tm_alpha_offsets=[float(x) for x in resonance_kwargs.get("tm_alpha_offsets", [])],
                tm_learned_mix=bool(resonance_kwargs.get("tm_learned_mix", True)),
                # PID autopilot for K/alpha control
                use_pid_autopilot=bool(resonance_kwargs.get("use_pid_autopilot", False) or resonance_kwargs.get("use_alpha_controller", False)),
                # Harmonics scaffold
                use_harmonics=bool(resonance_kwargs.get("use_harmonics", False)),
                harmonics=resonance_kwargs.get("harmonics", None),
                kappa_harm=resonance_kwargs.get("kappa_harm", None),
                alpha_harm=resonance_kwargs.get("alpha_harm", None),
                # MSF integration knobs
                use_msf_regularizer=bool(resonance_kwargs.get("use_msf_regularizer", False)),
                lambda_msf=float(resonance_kwargs.get("lambda_msf", 0.0)),
                msf_target=str(resonance_kwargs.get("msf_target", "critical")),
                msf_n_clusters=int(resonance_kwargs.get("msf_n_clusters", 8)),
                use_msf_autotune=bool(resonance_kwargs.get("use_msf_autotune", False)),
                msf_autotune_safety=float(resonance_kwargs.get("msf_autotune_safety", 0.9)),
                msf_eval_every=int(resonance_kwargs.get("msf_eval_every", 5)),
            )
            # Optional extras set as attributes (supported by head via getattr defaults)
            # Delay dt (converted inside head)
            if "delay_dt" in resonance_kwargs:
                try:
                    head.delay_dt = float(resonance_kwargs["delay_dt"])
                except Exception:
                    head.delay_dt = 0.0
            # Harmonics regularizer
            if "lambda_harmonics" in resonance_kwargs:
                try:
                    head.lambda_harmonics = float(resonance_kwargs["lambda_harmonics"])
                except Exception:
                    head.lambda_harmonics = 0.0
            # Temporal multiplex lane controls
            if "tm_nsteps" in resonance_kwargs:
                try:
                    head.tm_nsteps = [int(x) if x is not None else None for x in resonance_kwargs["tm_nsteps"]]
                except Exception:
                    head.tm_nsteps = []
            if "tm_fixed_weights" in resonance_kwargs:
                try:
                    head.tm_fixed_weights = [float(x) for x in resonance_kwargs["tm_fixed_weights"]]
                except Exception:
                    head.tm_fixed_weights = None
            # Alpha control and regularization
            learn_alpha = bool(resonance_kwargs.get("learn_alpha", False))
            use_alpha_controller = bool(resonance_kwargs.get("use_alpha_controller", False))
            head.use_alpha_controller = use_alpha_controller
            head.alpha_control_mode = str(resonance_kwargs.get("alpha_control_mode", "phase_lag"))
            head.alpha_target = float(resonance_kwargs.get("alpha_target", 0.0))
            head.lambda_alpha = float(resonance_kwargs.get("lambda_alpha", 0.0))
            # If controller is used or learning disabled, freeze direct grads on phase_lag
            if hasattr(head, "phase_lag"):
                try:
                    head.phase_lag.requires_grad = bool(learn_alpha and not use_alpha_controller)
                except Exception:
                    pass
            heads.append(head)
        self.res_heads = nn.ModuleList(heads)

        # Reuse original output projection (ViT uses 'dense', CLIP may use 'dense' or 'out_proj')
        self.dense: Optional[nn.Linear] = getattr(base_attn, "dense", None)
        if self.dense is None:
            self.dense = getattr(base_attn, "out_proj", None)

        # Storage for last forward's metrics (aggregated)
        self.last_metrics: Dict = {}
        
        # CoT mode toggle
        self.cot_mode: bool = False
        self._original_n_sim_steps: List[int] = [head.n_sim_steps for head in heads]

    def _forward_vanilla(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # Directly call the original ViT/CLIP attention module to preserve behavior
        # ViTSelfAttention returns a tuple: (attn_output, attn_probs) or just attn_output
        result = self.base_attn(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        # Ensure consistent return format
        if isinstance(result, tuple):
            return result
        return result, None

    def _aggregate_head_metrics(self, metric_list: List[Dict]) -> Dict:
        agg: Dict = {}
        # Simple aggregation: average tensors with matching keys when shapes are compatible
        for m in metric_list:
            for k, v in (m or {}).items():
                try:
                    if isinstance(v, dict):
                        # Flatten one level for 'cdns' and similar dict metrics
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
                # Fallback: store first item
                red[k] = items[0]
        return red

    def _forward_resonance(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Resonance path: split hidden_states into head slices, run each ResonanceAttentionHead,
        concatenate, and project through original dense layer.
        
        Note: ViT uses bidirectional attention (no causal mask), so we only apply padding masks.
        """
        bsz, seqlen, d_model = hidden_states.shape
        assert d_model == self.d_model, f"Expected hidden size {self.d_model}, got {d_model}"
        # Split into [B, H, T, Dh]
        x_heads = hidden_states.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        # Defensive sanitation: replace NaN/Inf coming from upstream with zeros
        try:
            x_heads = torch.nan_to_num(x_heads, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        
        # Build bidirectional attention mask (ViT doesn't use causal masking)
        # Start with all True (all positions can attend to all positions)
        attn_mask = torch.ones(seqlen, seqlen, device=hidden_states.device, dtype=torch.bool).unsqueeze(0).expand(bsz, -1, -1)
        
        # Apply padding mask if provided
        if attention_mask is not None:
            try:
                if attention_mask.dim() == 2:
                    # [B, T] -> [B, T, T] where True means "can attend"
                    pad = attention_mask != 0  # ViT uses 1 for valid tokens, 0 for padding
                    pad = pad[:, None, :] & pad[:, :, None]
                    attn_mask = attn_mask & pad
                elif attention_mask.dim() == 3:
                    # [B, T, T] mask directly
                    # ViT additive mask: 0 means "can attend", -inf means "cannot attend"
                    # Convert to boolean: True where mask == 0
                    pad = (attention_mask == 0).to(dtype=torch.bool)
                    if pad.shape[-2:] == (seqlen, seqlen):
                        attn_mask = attn_mask & pad
            except Exception:
                pass
        
        outputs: List[torch.Tensor] = []
        head_metrics: List[Dict] = []

        for h, head in enumerate(self.res_heads):
            x_h = x_heads[:, h, :, :]  # [B, T, Dh]
            # Sanitize per-head slice to avoid hard failures inside head
            if torch.isnan(x_h).any() or torch.isinf(x_h).any():
                try:
                    x_h = torch.nan_to_num(x_h, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    # As last resort, zero the slice
                    x_h = torch.zeros_like(x_h)
            if output_attentions:
                y_h, m_h = head(x_h, mask=attn_mask, return_metrics=True)  # head handles its own dynamics
                head_metrics.append(m_h or {})
            else:
                y_h = head(x_h, mask=attn_mask, return_metrics=False)
            outputs.append(y_h)

        # Concatenate heads back to [B, T, D]
        x_attn = torch.stack(outputs, dim=1)  # [B, H, T, Dh]
        x_attn = x_attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)

        # Project through ViT's dense layer if available
        if self.dense is not None:
            x_attn = self.dense(x_attn)

        # Save an aggregated snapshot of metrics
        self.last_metrics = self._aggregate_head_metrics(head_metrics) if head_metrics else {}
        
        # Store output for orchestrator
        self.last_output = x_attn
        
        # Notify orchestrator of forward completion
        if hasattr(self, '_notify_orchestrator'):
            self._notify_orchestrator("forward_complete", {
                "metrics": self.last_metrics,
                "output_shape": list(x_attn.shape),
            })

        # We do not return attentions for now (can be extended if needed)
        attn_probs = None
        return x_attn, attn_probs

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,  # Accept any other kwargs for future compatibility
    ):
        # Safe fallback to vanilla attention
        if self.attention_type == "vanilla":
            result = self._forward_vanilla(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
            # ViTSelfAttention returns tuple when output_attentions=True, otherwise just tensor
            if output_attentions:
                return result if isinstance(result, tuple) else (result, None)
            return result[0] if isinstance(result, tuple) else result

        # Resonance attention path
        x_attn, attn_probs = self._forward_resonance(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )

        # Match ViTSelfAttention return signature: tuple when output_attentions=True, otherwise just tensor
        if output_attentions:
            return x_attn, attn_probs
        return x_attn
    
    def toggle_cot_mode(self, enabled: bool):
        """
        Toggle Chain-of-Thought mode on/off.
        
        When enabled:
        - Increases simulation steps for richer dynamics during reasoning
        - Enables enhanced metric collection for CoT diagnosis
        
        When disabled:
        - Resets to normal simulation steps
        - Faster inference for direct answers
        """
        self.cot_mode = enabled
        
        # Adjust simulation steps based on CoT mode
        for i, head in enumerate(self.res_heads):
            if enabled:
                # Increase steps for richer reasoning dynamics
                head.n_sim_steps = self._original_n_sim_steps[i] * 2
            else:
                # Reset to original steps
                head.n_sim_steps = self._original_n_sim_steps[i]


def patch_vit_attention(
    model: PreTrainedModel,
    *,
    attention_type: str = "resonance",
    hybrid_alpha: float = 0.9,
    resonance_kwargs: Optional[Dict] = None,
) -> PreTrainedModel:
    """
    Replace ViT/CLIP self-attention modules with ResonanceViTAttentionAdapter.

    Args:
        model: A ViT or CLIP model (e.g., ViTForImageClassification, CLIPModel)
        attention_type: "resonance" or "vanilla"
        hybrid_alpha: initial hybrid mix weight on vanilla QK path (0..1); only used for resonance path
        resonance_kwargs: extra kwargs forwarded to ResonanceAttentionHead init

    Returns:
        The model with its attention modules replaced (in-place) and returned for convenience.
    """
    resonance_kwargs = resonance_kwargs or {}

    # Resolve config and module tree
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Unsupported model type for ViT patching; expected a model with .config")

    d_model = int(getattr(cfg, "hidden_size", getattr(cfg, "hidden_dim", 0)))
    n_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", 0)))
    if d_model <= 0 or n_heads <= 0:
        raise ValueError(f"Invalid ViT config: hidden={d_model}, heads={n_heads}")

    # Walk encoder layers - ViT uses 'encoder' -> 'layer', CLIP uses 'vision_model' -> 'encoder' -> 'layer'
    encoder = None
    if hasattr(model, "vit") or hasattr(model, "vision_model"):
        # CLIP model structure
        vision_model = getattr(model, "vision_model", getattr(model, "vit", None))
        if vision_model is not None:
            encoder = getattr(vision_model, "encoder", None)
    elif hasattr(model, "encoder"):
        # Direct ViT model structure
        encoder = model.encoder
    
    if encoder is None:
        raise ValueError("Could not find encoder in model structure. Expected 'encoder' or 'vision_model.encoder'")
    
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise ValueError("ViT encoder layers not found at encoder.layer")

    # Replace each layer's self-attention
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "attention", None)
        if attn is None:
            continue
        
        # Get the actual self-attention module (may be nested)
        self_attn = getattr(attn, "self", None)
        if self_attn is None:
            # Some models have self_attn directly on the layer
            self_attn = getattr(layer, "self_attn", None)
        
        if self_attn is None:
            continue
        
        adapter = ResonanceViTAttentionAdapter(
            self_attn,
            n_heads=n_heads,
            d_model=d_model,
            attention_type=attention_type,
            hybrid_alpha=hybrid_alpha,
            resonance_kwargs=resonance_kwargs,
        )
        # Move the adapter to the same device as the layer
        try:
            adapter.to(next(layer.parameters()).device)
        except Exception:
            pass
        
        # Replace the self-attention module
        if hasattr(attn, "self"):
            setattr(attn, "self", adapter)
        elif hasattr(layer, "self_attn"):
            setattr(layer, "self_attn", adapter)

    return model


def iter_resonance_heads(model: nn.Module):
    """
    Yield all ResonanceAttentionHead modules inside a patched ViT/CLIP model.
    Useful for calibration utilities / metric collection.
    """
    for m in model.modules():
        if isinstance(m, ResonanceViTAttentionAdapter):
            for h in m.res_heads:
                yield h


def set_cot_mode(model: nn.Module, enabled: bool):
    """
    Toggle Chain-of-Thought mode for all ResonanceViTAttentionAdapter instances in model.
    
    Args:
        model: Model with patched resonance attention
        enabled: If True, enable CoT mode (more simulation steps, richer dynamics)
                 If False, disable CoT mode (faster inference)
    """
    for adapter in model.modules():
        if isinstance(adapter, ResonanceViTAttentionAdapter):
            adapter.toggle_cot_mode(enabled)
