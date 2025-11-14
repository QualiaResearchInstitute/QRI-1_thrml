"""
CRV Data Collector: Collects step-level reasoning traces from resonance transformer.

Extracts oscillator features (phases, amplitudes, coupling) per reasoning step
for use in Circuit-based Reasoning Verification (CRV).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import numpy as np

try:
    from modules.resonance_gpt2_adapter import iter_resonance_heads
except ImportError:
    iter_resonance_heads = None


@dataclass
class StepTrace:
    """Single reasoning step trace with oscillator features."""
    step_idx: int
    step_text: str
    correct: Optional[bool] = None
    oscillator_features: Optional[Dict[str, Any]] = None
    attribution: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling torch tensors."""
        d = asdict(self)
        # Convert torch tensors to numpy arrays
        if d.get("oscillator_features"):
            for k, v in d["oscillator_features"].items():
                if isinstance(v, torch.Tensor):
                    d["oscillator_features"][k] = v.detach().cpu().numpy().tolist()
        if d.get("attribution"):
            for k, v in d["attribution"].items():
                if isinstance(v, torch.Tensor):
                    d["attribution"][k] = v.detach().cpu().numpy().tolist()
        return d


class CRVDataCollector:
    """
    Collects step-level reasoning traces from resonance transformer.
    
    Extracts oscillator features (phases, amplitudes, coupling) per step
    and stores them in structured format for CRV analysis.
    """
    
    def __init__(
        self,
        output_dir: str,
        enabled: bool = True,
        collect_interval: int = 1,
    ):
        """
        Initialize CRV data collector.
        
        Args:
            output_dir: Directory to save traces (JSONL format)
            enabled: Whether collection is enabled
            collect_interval: Collect every N steps (1 = every step)
        """
        self.output_dir = output_dir
        self.enabled = enabled
        self.collect_interval = collect_interval
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for current batch traces
        self.current_traces: List[StepTrace] = []
        self.global_step = 0
        
        # Hooks for extracting features
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._captured_features: Dict[str, Any] = {}
    
    def register_hooks(self, model: nn.Module) -> None:
        """
        Register forward hooks to capture oscillator features.
        
        Args:
            model: Model to hook into (should have resonance heads)
        """
        if not self.enabled:
            return
        
        # Clear existing hooks
        self.remove_hooks()
        
        # Hook into resonance attention heads
        if iter_resonance_heads is None:
            print("[CRV] Warning: iter_resonance_heads not available, skipping hooks")
            return
        
        try:
            for layer_idx, head in enumerate(iter_resonance_heads(model)):
                # Create closure to capture layer_idx
                def make_hook(layer: int, head_ref: nn.Module):
                    def hook_fn(module, input, output):
                        # Capture metrics if available
                        if hasattr(head_ref, 'last_metrics'):
                            metrics = head_ref.last_metrics
                            key = f"layer_{layer}"
                            if key not in self._captured_features:
                                self._captured_features[key] = []
                            self._captured_features[key].append({
                                "metrics": metrics.copy() if isinstance(metrics, dict) else {},
                            })
                    return hook_fn
                
                # Register hook on the head's forward method
                handle = head.register_forward_hook(make_hook(layer_idx, head))
                self.hooks.append(handle)
        except Exception as e:
            print(f"[CRV] Warning: Failed to register hooks: {e}")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def collect_step_trace(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        step_text: str,
        step_idx: int,
        ground_truth: Optional[bool] = None,
    ) -> StepTrace:
        """
        Collect oscillator features for a single reasoning step.
        
        Args:
            model: Model to extract features from
            input_ids: Input token IDs [batch, seq_len]
            step_text: Text of the reasoning step
            step_idx: Index of the step in the reasoning chain
            ground_truth: Ground truth correctness label (if available)
        
        Returns:
            StepTrace with oscillator features
        """
        if not self.enabled:
            return StepTrace(step_idx=step_idx, step_text=step_text, correct=ground_truth)
        
        # Clear captured features
        self._captured_features.clear()
        
        # Run forward pass (hooks will capture features)
        with torch.no_grad():
            try:
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
            except Exception as e:
                print(f"[CRV] Warning: Forward pass failed: {e}")
                return StepTrace(step_idx=step_idx, step_text=step_text, correct=ground_truth)
        
        # Extract oscillator features from captured data
        oscillator_features = self._extract_oscillator_features(model)
        
        # Extract attribution (simplified - can be enhanced later)
        attribution = self._extract_attribution(model, logits)
        
        return StepTrace(
            step_idx=step_idx,
            step_text=step_text,
            correct=ground_truth,
            oscillator_features=oscillator_features,
            attribution=attribution,
        )
    
    def _extract_oscillator_features(self, model: nn.Module) -> Dict[str, Any]:
        """
        Extract oscillator features from model's resonance heads.
        
        Returns:
            Dictionary with phases, amplitudes, coupling, CDNS metrics, etc.
        """
        features = {}
        
        if iter_resonance_heads is None:
            return features
        
        try:
            phases_list = []
            amplitudes_list = []
            coupling_list = []
            order_params_list = []
            cdns_list = []
            
            for layer_idx, head in enumerate(iter_resonance_heads(model)):
                # Try to get metrics from head
                metrics = {}
                if hasattr(head, 'last_metrics'):
                    metrics = head.last_metrics
                elif hasattr(head, 'metrics'):
                    metrics = head.metrics
                
                # Extract phases and amplitudes if available
                # Note: These might not be stored by default, may need to modify head
                if 'phases_history' in metrics:
                    phases_list.append(metrics['phases_history'])
                if 'amplitudes_history' in metrics:
                    amplitudes_list.append(metrics['amplitudes_history'])
                if 'coupling_matrix' in metrics:
                    coupling_list.append(metrics['coupling_matrix'])
                if 'order_param_history' in metrics:
                    order_params_list.append(metrics['order_param_history'])
                if 'final_order_parameter' in metrics:
                    order_params_list.append(metrics['final_order_parameter'])
                
                # Extract CDNS metrics
                cdns = {}
                if 'cdns' in metrics:
                    cdns = metrics['cdns']
                elif 'cdns.consonance' in metrics:
                    cdns = {
                        'consonance': metrics.get('cdns.consonance'),
                        'dissonance': metrics.get('cdns.dissonance'),
                        'noise': metrics.get('cdns.noise'),
                        'signal': metrics.get('cdns.signal'),
                    }
                
                if cdns:
                    cdns_list.append(cdns)
            
            if phases_list:
                features['phases'] = torch.stack(phases_list) if len(phases_list) > 1 else phases_list[0]
            if amplitudes_list:
                features['amplitudes'] = torch.stack(amplitudes_list) if len(amplitudes_list) > 1 else amplitudes_list[0]
            if coupling_list:
                features['coupling_matrix'] = torch.stack(coupling_list) if len(coupling_list) > 1 else coupling_list[0]
            if order_params_list:
                features['order_parameter'] = torch.stack(order_params_list) if len(order_params_list) > 1 else order_params_list[0]
            if cdns_list:
                features['cdns'] = cdns_list
            
        except Exception as e:
            print(f"[CRV] Warning: Failed to extract oscillator features: {e}")
        
        return features
    
    def _extract_attribution(
        self,
        model: nn.Module,
        logits: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Extract attribution scores (simplified version).
        
        Can be enhanced later with gradient-based attribution or attention weights.
        
        Args:
            model: Model
            logits: Output logits [batch, seq_len, vocab_size]
        
        Returns:
            Dictionary with attribution scores
        """
        attribution = {}
        
        # For now, use logit probabilities as simple attribution
        try:
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
            attribution['top_logit_probs'] = top_probs.detach().cpu()
            attribution['top_logit_indices'] = top_indices.detach().cpu()
        except Exception as e:
            print(f"[CRV] Warning: Failed to extract attribution: {e}")
        
        return attribution
    
    def collect_batch_traces(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        step_texts: List[str],
        ground_truths: Optional[List[bool]] = None,
    ) -> List[StepTrace]:
        """
        Collect traces for a batch of reasoning steps.
        
        Args:
            model: Model
            input_ids: Input token IDs [batch, seq_len]
            step_texts: List of step texts (one per batch item)
            ground_truths: Optional list of correctness labels
        
        Returns:
            List of StepTrace objects
        """
        traces = []
        batch_size = input_ids.shape[0]
        
        for i in range(batch_size):
            step_text = step_texts[i] if i < len(step_texts) else f"step_{i}"
            ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            
            trace = self.collect_step_trace(
                model=model,
                input_ids=input_ids[i:i+1],  # Single item batch
                step_text=step_text,
                step_idx=i,
                ground_truth=ground_truth,
            )
            traces.append(trace)
        
        return traces
    
    def save_trace(self, trace: StepTrace, filename: Optional[str] = None) -> None:
        """
        Save a single trace to JSONL file.
        
        Args:
            trace: StepTrace to save
            filename: Optional filename (defaults to trace_<step_idx>.jsonl)
        """
        if not self.enabled:
            return
        
        if filename is None:
            filename = f"trace_step_{trace.step_idx:06d}.jsonl"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            trace_dict = trace.to_dict()
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trace_dict) + '\n')
        except Exception as e:
            print(f"[CRV] Warning: Failed to save trace: {e}")
    
    def save_traces(self, traces: List[StepTrace], step: Optional[int] = None) -> None:
        """
        Save multiple traces to JSONL file.
        
        Args:
            traces: List of StepTrace objects
            step: Optional global step number for filename
        """
        if not self.enabled or not traces:
            return
        
        filename = f"traces_step_{step:06d}.jsonl" if step is not None else "traces.jsonl"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                for trace in traces:
                    trace_dict = trace.to_dict()
                    f.write(json.dumps(trace_dict) + '\n')
        except Exception as e:
            print(f"[CRV] Warning: Failed to save traces: {e}")
    
    def save_problem_trace(
        self,
        problem_id: str,
        problem_text: str,
        steps: List[StepTrace],
    ) -> None:
        """
        Save a complete problem trace with all steps.
        
        Args:
            problem_id: Unique identifier for the problem
            problem_text: Text of the problem
            steps: List of StepTrace objects for each step
        """
        if not self.enabled:
            return
        
        problem_trace = {
            "problem_id": problem_id,
            "problem_text": problem_text,
            "steps": [step.to_dict() for step in steps],
        }
        
        filename = f"problem_{problem_id}.jsonl"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(problem_trace) + '\n')
        except Exception as e:
            print(f"[CRV] Warning: Failed to save problem trace: {e}")

