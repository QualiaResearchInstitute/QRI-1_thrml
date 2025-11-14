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


class ResonanceGPT2AttentionAdapter(nn.Module):
    """
    Drop-in replacement for GPT-2 style attention that routes through ResonanceAttentionHead(s)
    under a config flag. Provides a safe fallback to the original GPT-2 attention.

    Notes:
    - This adapter does NOT implement KV caching or cross-attention; it targets training/finetune.
    - To reduce risk, hybrid_readout is enabled so outputs blend resonance and vanilla QK^T.
    - We split hidden_states into per-head slices and let each resonance head process its slice.
      This is not a perfect reuse of the GPT-2 QKV fused projection (c_attn), but preserves the
      attention ABI and allows a recovery path via hybrid blending toward vanilla behavior.
    """

    def __init__(
        self,
        base_attn: nn.Module,
        n_heads: int,
        d_model: int,
        *,
        attention_type: str = "resonance",  # "resonance" | "vanilla"
        hybrid_alpha: float = 0.8,          # weight on vanilla QK path in [0,1]; higher is closer to vanilla
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

        # Build resonance heads (one per GPT-2 head slice)
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

        # Reuse original output projection to stay close to GPT-2 ABI
        self.c_proj: Optional[nn.Linear] = getattr(base_attn, "c_proj", None)

        # Storage for last forward's metrics (aggregated)
        self.last_metrics: Dict = {}
        
        # CoT mode toggle
        self.cot_mode: bool = False
        self._original_n_sim_steps: List[int] = [head.n_sim_steps for head in heads]

    def _forward_vanilla(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        layer_past=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # Directly call the original GPT-2 attention module to preserve behavior
        return self.base_attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

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
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, None, Optional[torch.Tensor]]:
        """
        Resonance path: split hidden_states into head slices, run each ResonanceAttentionHead,
        concatenate, and project through original c_proj.
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
        # Build causal + padding mask for resonance head (boolean [B, T, T])
        attn_mask = torch.tril(torch.ones(seqlen, seqlen, device=hidden_states.device, dtype=torch.bool)).unsqueeze(0).expand(bsz, -1, -1)
        if attention_mask is not None:
            try:
                if attention_mask.dim() == 2:
                    # Combine with padding mask
                    pad = attention_mask != 0
                    pad = pad[:, None, :] & pad[:, :, None]
                    attn_mask = attn_mask & pad
                elif attention_mask.dim() == 4:
                    # HF additive mask: allowed where mask == 0
                    pad4 = (attention_mask == 0)
                    pad3 = pad4.squeeze(1).to(dtype=torch.bool)
                    if pad3.shape[-2:] == (seqlen, seqlen):
                        attn_mask = attn_mask & pad3
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

        # Project through GPT-2's c_proj if available
        if self.c_proj is not None:
            x_attn = self.c_proj(x_attn)

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

        # We do not return attentions or cache
        attn_probs = None
        present = None
        return x_attn, present, attn_probs

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # Newer transformers API
        **kwargs,  # Accept any other kwargs for future compatibility
    ):
        # Handle past_key_values (newer transformers) -> layer_past (older API)
        if past_key_values is not None and layer_past is None:
            layer_past = past_key_values
        
        # Safe fallback to vanilla attention
        if self.attention_type == "vanilla":
            return self._forward_vanilla(
                hidden_states,
                attention_mask,
                layer_past=layer_past,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        # Resonance attention path (no KV cache; mask unused)
        x_attn, present, attn_probs = self._forward_resonance(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )

        # Match GPT-2 attention return signature: (attn_output, present) (+attn_probs if requested)
        if output_attentions:
            return x_attn, present, attn_probs
        return x_attn, present
    
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


def patch_gpt2_attention(
    model: PreTrainedModel,
    *,
    attention_type: str = "resonance",
    hybrid_alpha: float = 0.8,
    resonance_kwargs: Optional[Dict] = None,
) -> PreTrainedModel:
    """
    Replace GPT-2 self-attention modules with ResonanceGPT2AttentionAdapter.

    Args:
        model: A GPT-2 family model (e.g., GPT2LMHeadModel)
        attention_type: "resonance" or "vanilla"
        hybrid_alpha: initial hybrid mix weight on vanilla QK path (0..1); only used for resonance path
        resonance_kwargs: extra kwargs forwarded to ResonanceAttentionHead init

    Returns:
        The model with its attention modules replaced (in-place) and returned for convenience.
    """
    resonance_kwargs = resonance_kwargs or {}

    # Resolve config and module tree
    cfg = getattr(model, "config", None)
    if cfg is None or not hasattr(model, "transformer"):
        raise ValueError("Unsupported model type for GPT-2 patching; expected a GPT-2-like model with .transformer")

    d_model = int(getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 0)))
    n_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)))
    if d_model <= 0 or n_heads <= 0:
        raise ValueError(f"Invalid GPT-2 config: hidden={d_model}, heads={n_heads}")

    # Walk each decoder block's self-attention
    blocks = getattr(model.transformer, "h", None)
    if blocks is None:
        raise ValueError("GPT-2 transformer blocks not found at model.transformer.h")
    for idx, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        adapter = ResonanceGPT2AttentionAdapter(
            attn,
            n_heads=n_heads,
            d_model=d_model,
            attention_type=attention_type,
            hybrid_alpha=hybrid_alpha,
            resonance_kwargs=resonance_kwargs,
        )
        # Move the adapter to the same device as the block
        try:
            adapter.to(next(block.parameters()).device)
        except Exception:
            pass
        setattr(block, "attn", adapter)

    return model


def iter_resonance_heads(model: nn.Module):
    """
    Yield all ResonanceAttentionHead modules inside a patched GPT-2 model.
    Useful for calibration utilities / metric collection.
    """
    for m in model.modules():
        if isinstance(m, ResonanceGPT2AttentionAdapter):
            for h in m.res_heads:
                yield h


def set_cot_mode(model: nn.Module, enabled: bool):
    """
    Toggle Chain-of-Thought mode for all ResonanceGPT2AttentionAdapter instances in model.
    
    Args:
        model: Model with patched resonance attention
        enabled: If True, enable CoT mode (more simulation steps, richer dynamics)
                 If False, disable CoT mode (faster inference)
    """
    for adapter in model.modules():
        if isinstance(adapter, ResonanceGPT2AttentionAdapter):
            adapter.toggle_cot_mode(enabled)
