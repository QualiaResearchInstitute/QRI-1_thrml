"""
RNA Transcriber: Ephemeral adapter proposals for GPT-2 repair.

Provides:
- RNAAdapter for proposing and applying ephemeral deltas
- Conflict resolution strategies
- Guardrail enforcement
- Rollback support via ApplyToken
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from modules.genome_registry import Address, DeltaSpec, Guardrails


@dataclass
class RNAProposal:
    """An ephemeral RNA proposal."""
    proposal_id: str
    specs: List[DeltaSpec]
    created_at: float
    guardrails: Guardrails


@dataclass
class ApplyToken:
    """Token for tracking applied RNA proposals (enables rollback)."""
    token_id: str
    proposal_id: str
    hooks: List[Any] = field(default_factory=list)  # Forward hooks to remove
    original_state: Dict[str, Any] = field(default_factory=dict)  # For rollback


class RNAAdapter:
    """
    Ephemeral adapter manager for RNA proposals.
    
    Proposes, applies, and manages rollback of temporary deltas before
    they are accepted into DNA.
    """
    
    def __init__(self, guardrails: Guardrails):
        """
        Initialize RNA adapter.
        
        Args:
            guardrails: Guardrails to enforce on proposals
        """
        self.guardrails = guardrails
        self._active_tokens: Dict[str, ApplyToken] = {}
    
    def propose(self, specs: List[DeltaSpec]) -> List[RNAProposal]:
        """
        Propose RNA deltas with guardrail checks.
        
        Args:
            specs: List of delta specs to propose
        
        Returns:
            List of RNAProposal objects (one per spec, or merged if strategy allows)
        
        Raises:
            ValueError: If guardrails are violated
        """
        # Validate guardrails
        for spec in specs:
            self._check_guardrails(spec)
        
        # Create proposals (one per spec for simplicity)
        proposals = []
        for spec in specs:
            proposal = RNAProposal(
                proposal_id=str(uuid.uuid4()),
                specs=[spec],
                created_at=time.time(),
                guardrails=self.guardrails,
            )
            proposals.append(proposal)
        
        return proposals
    
    def _check_guardrails(self, spec: DeltaSpec):
        """Check guardrails for a single delta spec."""
        if spec.type == 'lora':
            payload = spec.payload
            U = payload.get('U')
            V = payload.get('V')
            if U is None or V is None:
                raise ValueError("LoRA payload missing U or V")
            # Check norm (will be checked against target weight at apply time)
            # For now, just check payload structure
        elif spec.type == 'alpha':
            alpha_delta = spec.payload.get('alpha_delta')
            if alpha_delta is None:
                raise ValueError("Alpha payload missing alpha_delta")
            if abs(alpha_delta) > self.guardrails.max_norms['alpha_abs']:
                raise ValueError(f"Alpha delta {alpha_delta} exceeds max {self.guardrails.max_norms['alpha_abs']}")
        elif spec.type == 'kernel_weight':
            weights = spec.payload.get('weights')
            if weights is None:
                raise ValueError("Kernel weight payload missing weights")
            # L2 norm check will happen at apply time
    
    def apply(
        self,
        proposal_id: str,
        model: nn.Module,
        proposals: List[RNAProposal],
    ) -> ApplyToken:
        """
        Apply an RNA proposal to the model.
        
        Args:
            proposal_id: ID of proposal to apply
            model: GPT-2 model to patch
            proposals: List of proposals (to find the one matching proposal_id)
        
        Returns:
            ApplyToken for rollback
        
        Raises:
            ValueError: If proposal not found or application fails
        """
        # Find proposal
        proposal = None
        for p in proposals:
            if p.proposal_id == proposal_id:
                proposal = p
                break
        
        if proposal is None:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        # Apply each spec in proposal
        token = ApplyToken(
            token_id=str(uuid.uuid4()),
            proposal_id=proposal_id,
        )
        
        hooks = []
        original_state = {}
        
        for spec in proposal.specs:
            hook, orig = self._apply_delta(model, spec)
            hooks.append(hook)
            original_state[spec.address.to_canonical()] = orig
        
        token.hooks = hooks
        token.original_state = original_state
        
        self._active_tokens[token.token_id] = token
        return token
    
    def _apply_delta(self, model: nn.Module, spec: DeltaSpec) -> Tuple[Any, Dict[str, Any]]:
        """
        Apply a single delta to the model.
        
        Returns:
            (hook_handle, original_state_dict) for rollback
        """
        # Resolve parameter pointer
        param_ptr = self._resolve_param_pointer(model, spec.address)
        
        if spec.type == 'lora':
            return self._apply_lora(param_ptr, spec, model)
        elif spec.type == 'alpha':
            return self._apply_alpha(param_ptr, spec, model)
        elif spec.type == 'kernel_weight':
            return self._apply_kernel_weight(param_ptr, spec, model)
        else:
            raise ValueError(f"Unknown delta type: {spec.type}")
    
    def _resolve_param_pointer(self, model: nn.Module, address: Address) -> Tuple[nn.Module, str]:
        """
        Resolve GPT-2 parameter pointer from address.
        
        Returns:
            (module, param_name) tuple
        """
        layer_idx = address.layer
        
        # Navigate to transformer block
        if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'h'):
            raise ValueError("Model missing transformer.h structure")
        
        block = model.transformer.h[layer_idx]
        
        if address.param == 'alpha':
            # Alpha is stored on resonance head
            if not hasattr(block, 'attn') or not hasattr(block.attn, 'res_heads'):
                raise ValueError(f"Layer {layer_idx} missing resonance heads")
            head_idx = address.head
            if head_idx >= len(block.attn.res_heads):
                raise ValueError(f"Head {head_idx} out of range")
            head = block.attn.res_heads[head_idx]
            return (head, 'phase_lag')  # Alpha maps to phase_lag parameter
        
        elif address.param == 'lora':
            # LoRA attaches to attention projections
            target = address.target
            if target in ['q', 'k', 'v']:
                # QKV are fused in c_attn
                return (block.attn, 'c_attn')
            elif target == 'o':
                return (block.attn, 'c_proj')
            elif target == 'mlp_in':
                return (block.mlp, 'c_fc')
            elif target == 'mlp_out':
                return (block.mlp, 'c_proj')
            else:
                raise ValueError(f"Unknown LoRA target: {target}")
        
        elif address.param == 'kernel_weight':
            # Kernel weights are on resonance head
            if not hasattr(block, 'attn') or not hasattr(block.attn, 'res_heads'):
                raise ValueError(f"Layer {layer_idx} missing resonance heads")
            # Band-level weights (assuming stored as attribute or parameter)
            head = block.attn.res_heads[0]  # Default to first head if not per-head
            return (head, f'kernel_weights_band_{address.band}')
        
        raise ValueError(f"Cannot resolve address: {address.to_canonical()}")
    
    def _apply_lora(
        self,
        param_ptr: Tuple[nn.Module, str],
        spec: DeltaSpec,
        model: nn.Module,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Apply LoRA delta via forward hook."""
        module, param_name = param_ptr
        payload = spec.payload
        U = torch.tensor(payload['U'], dtype=torch.float32)
        V = torch.tensor(payload['V'], dtype=torch.float32)
        scale = payload.get('scale', 1.0)
        
        # Get original weight
        original_weight = getattr(module, param_name).weight.data.clone()
        
        # Compute LoRA delta: U @ V * scale
        lora_delta = (U @ V) * scale
        
        # Check norm guardrail
        fro_norm_delta = torch.norm(lora_delta, p='fro').item()
        fro_norm_orig = torch.norm(original_weight, p='fro').item()
        rel_norm = fro_norm_delta / (fro_norm_orig + 1e-10)
        
        if rel_norm > self.guardrails.max_norms['lora_fro_rel']:
            raise ValueError(f"LoRA norm {rel_norm} exceeds {self.guardrails.max_norms['lora_fro_rel']}")
        
        # Store original state
        original_state = {
            'weight': original_weight.clone(),
        }
        
        # Apply via forward hook (non-destructive)
        def lora_hook(module, input, output):
            # For linear layers, output = input @ weight^T + bias
            # We want: output = input @ (weight + lora_delta)^T + bias
            # So: output = input @ weight^T + input @ lora_delta^T + bias
            if isinstance(module, nn.Linear):
                lora_addition = input[0] @ lora_delta.T
                return output + lora_addition
            return output
        
        handle = module.register_forward_hook(lora_hook)
        return handle, original_state
    
    def _apply_alpha(
        self,
        param_ptr: Tuple[nn.Module, str],
        spec: DeltaSpec,
        model: nn.Module,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Apply alpha delta (phase lag adjustment)."""
        module, param_name = param_ptr
        payload = spec.payload
        alpha_delta = payload['alpha_delta']
        
        # Get original phase_lag
        if not hasattr(module, param_name):
            raise ValueError(f"Module missing {param_name}")
        original_param = getattr(module, param_name)
        if isinstance(original_param, nn.Parameter):
            original_value = original_param.data.clone()
        else:
            original_value = torch.tensor(original_param, dtype=torch.float32)
        
        # Store original state
        original_state = {
            'value': original_value.clone(),
        }
        
        # Apply delta (in-place for alpha)
        if isinstance(original_param, nn.Parameter):
            with torch.no_grad():
                original_param.data += alpha_delta
        else:
            setattr(module, param_name, original_value + alpha_delta)
        
        # Return None hook (no hook needed for direct modification)
        return None, original_state
    
    def _apply_kernel_weight(
        self,
        param_ptr: Tuple[nn.Module, str],
        spec: DeltaSpec,
        model: nn.Module,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Apply kernel weight delta."""
        module, param_name = param_ptr
        payload = spec.payload
        weights = torch.tensor(payload['weights'], dtype=torch.float32)
        
        # Get original weights (if exists)
        if hasattr(module, param_name):
            original_weights = getattr(module, param_name)
            if isinstance(original_weights, nn.Parameter):
                original_value = original_weights.data.clone()
            else:
                original_value = torch.tensor(original_weights, dtype=torch.float32)
        else:
            # Create if doesn't exist
            original_value = torch.zeros_like(weights)
            setattr(module, param_name, nn.Parameter(original_value))
        
        # Check L2 norm guardrail
        l2_norm_delta = torch.norm(weights, p=2).item()
        l2_norm_orig = torch.norm(original_value, p=2).item()
        rel_norm = l2_norm_delta / (l2_norm_orig + 1e-10) if l2_norm_orig > 0 else float('inf')
        
        if rel_norm > self.guardrails.max_norms['kernel_weight_l2_rel']:
            raise ValueError(f"Kernel weight norm {rel_norm} exceeds {self.guardrails.max_norms['kernel_weight_l2_rel']}")
        
        # Store original state
        original_state = {
            'value': original_value.clone(),
        }
        
        # Apply delta
        if isinstance(getattr(module, param_name), nn.Parameter):
            with torch.no_grad():
                getattr(module, param_name).data += weights
        else:
            setattr(module, param_name, original_value + weights)
        
        return None, original_state
    
    def rollback(self, token: ApplyToken, model: nn.Module) -> None:
        """
        Rollback applied RNA proposal.
        
        Args:
            token: ApplyToken from apply()
            model: Model to rollback
        """
        # Remove hooks
        for hook in token.hooks:
            if hook is not None:
                hook.remove()
        
        # Restore original state
        for canonical_addr, orig_state in token.original_state.items():
            # Parse address and restore
            address = Address.from_canonical(canonical_addr)
            param_ptr = self._resolve_param_pointer(model, address)
            module, param_name = param_ptr
            
            if 'weight' in orig_state:
                # LoRA: restore weight (though we used hooks, so weight unchanged)
                pass  # Hook already removed
            elif 'value' in orig_state:
                # Alpha or kernel_weight: restore value
                if isinstance(getattr(module, param_name), nn.Parameter):
                    getattr(module, param_name).data.copy_(orig_state['value'])
                else:
                    setattr(module, param_name, orig_state['value'].clone())
        
        # Remove token
        self._active_tokens.pop(token.token_id, None)
    
    def merge_conflicts(
        self,
        proposals: List[RNAProposal],
        strategy: Literal['sum', 'last', 'bounded-sum'] = 'last',
    ) -> List[RNAProposal]:
        """
        Merge conflicting proposals (same address).
        
        Args:
            proposals: List of proposals to merge
            strategy: Merge strategy
        
        Returns:
            Merged proposals
        """
        # Group by address
        by_address: Dict[str, List[DeltaSpec]] = {}
        for proposal in proposals:
            for spec in proposal.specs:
                addr_key = spec.address.to_canonical()
                by_address.setdefault(addr_key, []).append(spec)
        
        # Merge conflicts
        merged_specs = []
        for addr_key, specs in by_address.items():
            if len(specs) == 1:
                merged_specs.append(specs[0])
            else:
                # Merge multiple specs
                if strategy == 'last':
                    merged_specs.append(specs[-1])
                elif strategy == 'sum':
                    merged = self._merge_specs_sum(specs)
                    merged_specs.append(merged)
                elif strategy == 'bounded-sum':
                    merged = self._merge_specs_bounded_sum(specs)
                    merged_specs.append(merged)
        
        # Create new proposal
        if merged_specs:
            return [RNAProposal(
                proposal_id=str(uuid.uuid4()),
                specs=merged_specs,
                created_at=time.time(),
                guardrails=self.guardrails,
            )]
        return []
    
    def _merge_specs_sum(self, specs: List[DeltaSpec]) -> DeltaSpec:
        """Merge specs by summing (for compatible types)."""
        if len(specs) == 0:
            raise ValueError("Cannot merge empty specs")
        
        base = specs[0]
        if base.type == 'alpha':
            total_delta = sum(s.payload.get('alpha_delta', 0) for s in specs)
            return DeltaSpec(
                address=base.address,
                type=base.type,
                payload={'alpha_delta': total_delta},
                provenance={'merged_from': [s.delta_id for s in specs]},
            )
        else:
            # For LoRA/kernel_weight, use last (summing is complex)
            return specs[-1]
    
    def _merge_specs_bounded_sum(self, specs: List[DeltaSpec]) -> DeltaSpec:
        """Merge specs by summing then clamping to guardrails."""
        merged = self._merge_specs_sum(specs)
        
        # Clamp to guardrails
        if merged.type == 'alpha':
            max_abs = self.guardrails.max_norms['alpha_abs']
            merged.payload['alpha_delta'] = max(-max_abs, min(max_abs, merged.payload['alpha_delta']))
        
        return merged


# Utility functions for sampling proposals
def sample_alpha_delta(
    layer: int,
    head: int,
    alpha_range: Tuple[float, float] = (-0.1, 0.1),
) -> DeltaSpec:
    """Sample a random alpha delta proposal."""
    import random
    alpha_delta = random.uniform(*alpha_range)
    return DeltaSpec(
        address=Address(layer=layer, param='alpha', head=head),
        type='alpha',
        payload={'alpha_delta': alpha_delta},
    )


def sample_lora_rank2(
    layer: int,
    target: Literal['q', 'k', 'v', 'o', 'mlp_in', 'mlp_out'],
    d_out: int,
    d_in: int,
    r: int = 2,
    scale: float = 1.0,
) -> DeltaSpec:
    """Sample a random LoRA rank-2 proposal."""
    import random
    U = [[random.gauss(0, 0.01) for _ in range(r)] for _ in range(d_out)]
    V = [[random.gauss(0, 0.01) for _ in range(d_in)] for _ in range(r)]
    return DeltaSpec(
        address=Address(layer=layer, param='lora', target=target),
        type='lora',
        payload={'U': U, 'V': V, 'scale': scale},
    )


def sample_kernel_band_reweight(
    layer: int,
    band: int,
    n_bands: int,
    weight_range: Tuple[float, float] = (-0.05, 0.05),
) -> DeltaSpec:
    """Sample kernel weight reweighting for a band."""
    import random
    weights = [random.uniform(*weight_range) for _ in range(n_bands)]
    return DeltaSpec(
        address=Address(layer=layer, param='kernel_weight', band=band),
        type='kernel_weight',
        payload={'weights': weights},
    )


__all__ = [
    'RNAProposal',
    'ApplyToken',
    'RNAAdapter',
    'sample_alpha_delta',
    'sample_lora_rank2',
    'sample_kernel_band_reweight',
]


