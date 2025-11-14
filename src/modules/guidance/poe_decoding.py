from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from modules.multimodal.critics import Critic, tokens_to_strings
from modules.guidance.projectors import Projector
from modules.guidance.consensus import ConsensusGovernor
try:
    from modules.live_metrics_emitter import RTLiveEmitter, extract_head_phases_1d
except Exception:
    RTLiveEmitter = None  # type: ignore
    def extract_head_phases_1d(model):  # type: ignore
        return None


def _default_should_pulse(step: int, schedule_cfg: Dict[str, Any], model_metrics: Optional[Dict[str, Any]] = None) -> bool:
    pulse_every = int(schedule_cfg.get("pulse_every", 2))
    if pulse_every <= 1:
        return True
    return (step % pulse_every) == 0


@torch.no_grad()
def guided_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    critics: List[Critic],
    lambdas: Dict[str, float],
    projectors: Optional[List[Projector]] = None,
    schedule_cfg: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    attention_mask: Optional[torch.Tensor] = None,
    critic_context: Optional[Dict[str, Any]] = None,
    governor: Optional[ConsensusGovernor] = None,
    telemetry_emitter: Optional["RTLiveEmitter"] = None,
) -> torch.Tensor:
    """
    Product-of-experts guided decoding over next-token logits.
    - Adds sum_c ( -lambda_c * calibrated(ΔE_c) ) to top-K token logits when pulsing.
    - Applies projectors to top-K candidate logits (safety, schemas).
    - Samples with temperature and top-p.
    """
    device = input_ids.device
    schedule_cfg = schedule_cfg or {}
    K = int(schedule_cfg.get("top_k", 32))
    should_pulse = schedule_cfg.get("should_pulse_fn", _default_should_pulse)
    projectors = projectors or []
    critic_context = critic_context or {}
    # Initialize live telemetry emitter if requested
    emitter = telemetry_emitter
    if emitter is None:
        try:
            if bool((schedule_cfg or {}).get("rt_metrics", False)) and RTLiveEmitter is not None:
                emitter = RTLiveEmitter()
        except Exception:
            emitter = None

    # Prime critics with external context (e.g., image/audio)
    for c in critics:
        try:
            c.prime({**critic_context, "prefix_text": ""})
        except Exception:
            # If prime fails, disable critic
            c.enabled = False

    generated = input_ids
    for step in range(max_new_tokens):
        outputs = model(generated, attention_mask=attention_mask)  # type: ignore
        logits = outputs.logits if hasattr(outputs, "logits") else outputs  # [B, T, V] or [T, V]
        if logits.dim() == 2:
            next_logits = logits[-1, :].unsqueeze(0)  # [1, V]
        else:
            next_logits = logits[:, -1, :]  # [B, V]

        # Temperature first
        next_logits = next_logits / max(temperature, 1e-6)

        # Compute top-K
        topk_vals, topk_idx = torch.topk(next_logits, k=min(K, next_logits.shape[-1]), dim=-1)

        # Build candidate strings for B=1 case (this MVP focuses on batch size 1)
        # For general B>1, extend by looping per batch.
        prefix_text = ""
        try:
            prefix_text = tokenizer.decode(generated[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)  # type: ignore
        except Exception:
            pass
        token_ids = topk_idx[0].tolist()
        cand_strs = tokens_to_strings(tokenizer, token_ids)

        # Pulse energies?
        model_metrics = getattr(model, "last_metrics", None)
        pulse = should_pulse(step, schedule_cfg, model_metrics)
        adjusted_topk = topk_vals.clone()
        if pulse:
            # For each critic: compute ΔE for candidates, calibrate, apply -lambda * ΔE to logits.
            critic_zscores_accum: Dict[str, float] = {}
            for c in critics:
                if not c.enabled:
                    continue
                try:
                    dE = c.delta_for_candidates(prefix_text, cand_strs)
                    if len(dE) != adjusted_topk.shape[-1]:
                        continue
                    z = c.calibrate(dE)
                    lam = float(lambdas.get(c.name, 0.0))
                    # Subtract because higher energy is worse
                    delta = torch.tensor(z, device=adjusted_topk.device, dtype=adjusted_topk.dtype)
                    adjusted_topk = adjusted_topk - lam * delta.unsqueeze(0)
                    # Track a representative z-score (e.g., mean magnitude) for governor
                    critic_zscores_accum[c.name] = float(abs(torch.tensor(z).float().mean().item()))
                except Exception:
                    continue

            # Apply projectors on top-k candidates
            for p in projectors:
                try:
                    adjusted_topk = p.project_topk(topk_idx, adjusted_topk, cand_strs, {"prefix_text": prefix_text})
                except Exception:
                    continue
        else:
            critic_zscores_accum = {}

        # Scatter adjusted top-k back into full logits
        next_logits_scattered = next_logits.clone()
        next_logits_scattered.scatter_(1, topk_idx, adjusted_topk)

        allow_commit = False
        # Consensus governor (optional): if allow_commit is True, bias toward EOS slightly
        if governor is not None:
            decision = governor.decide(model_metrics=model_metrics, critic_zscores=critic_zscores_accum)
            allow_commit = bool(decision.get("allow_commit", False))
            if decision.get("allow_commit", False):
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if eos_id is not None:
                    # Small bias to encourage ending when consensus is high
                    next_logits_scattered[:, eos_id] = next_logits_scattered[:, eos_id] + 0.5

        # Top-p filtering
        probs = F.softmax(next_logits_scattered, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            probs = probs.scatter(1, sorted_idx, torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs))
            # Renormalize
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

        # Telemetry emit (decimated inside emitter)
        if emitter is not None:
            try:
                phases_1d = extract_head_phases_1d(model) if callable(extract_head_phases_1d) else None
                # Build top-K entries with probabilities and logit deltas
                base = topk_vals[0].detach().float()
                adj = adjusted_topk[0].detach().float()
                idx = topk_idx[0]
                p_vec = probs[0].detach().float()
                topk_list = []
                J = int(idx.shape[0])
                for j in range(J):
                    tok_id = int(idx[j].item())
                    tok_str = cand_strs[j] if j < len(cand_strs) else str(tok_id)
                    pj = float(p_vec[tok_id].item())
                    base_logit = float(base[j].item())
                    delta_logit = float((adj[j] - base[j]).item())
                    topk_list.append({
                        "token": tok_str,
                        "p": pj,
                        "base_logit": base_logit,
                        "delta_logit": delta_logit,
                    })
                consensus_state = "commit" if allow_commit else "negotiate"
                emitter.maybe_emit(
                    step=step,
                    phases_1d=phases_1d,
                    energies_z=critic_zscores_accum if pulse else {},
                    topk=topk_list,
                    consensus=consensus_state,
                )
            except Exception:
                pass

        # Early stop on EOS if tokenizer has eos_token_id
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and int(next_token[0].item()) == int(eos_id):
            break

    return generated
