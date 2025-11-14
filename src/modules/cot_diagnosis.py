"""
Chain-of-Thought Diagnosis Module

Diagnoses CoT reasoning quality using resonance metrics (R, CDNS, criticality).
"""

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn

from .resonance_gpt2_adapter import iter_resonance_heads


class CoTDiagnostic:
    """
    Diagnoses Chain-of-Thought reasoning quality using resonance metrics.
    
    Evaluates CoT based on:
    1. Resonance metrics (order parameter R, CDNS)
    2. Criticality (how close to optimal R ≈ 0.6)
    3. CoT structure (step count, clarity)
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize CoT diagnostic.
        
        Args:
            model: Model with ResonanceGPT2AttentionAdapter instances
        """
        self.model = model
        
    def diagnose_cot(
        self,
        problem: str,
        cot_rationale: str,
        answer: str,
        resonance_metrics: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Diagnose CoT quality based on resonance metrics and structure.
        
        Args:
            problem: The problem being solved
            cot_rationale: The Chain-of-Thought reasoning steps
            answer: The final answer
            resonance_metrics: Optional pre-collected metrics (if None, collects fresh)
            
        Returns:
            Dictionary with diagnostic scores:
            - resonance_health: Overall health of resonance dynamics (0-1)
            - criticality_score: How close to optimal criticality R≈0.6 (0-1)
            - dissonance_score: How low dissonance is (0-1, higher is better)
            - cot_complexity: Normalized step count (0-1)
            - overall_quality: Combined quality score (0-1)
            - order_parameter_R: Actual R value
            - consonance: Consonance metric
            - dissonance: Dissonance metric
        """
        if resonance_metrics is None:
            resonance_metrics = self._collect_metrics()
            
        # Extract metrics
        R = resonance_metrics.get('final_order_parameter', 0.5)
        cdns = resonance_metrics.get('cdns', {})
        if isinstance(cdns, dict):
            dissonance = cdns.get('dissonance', 0.0)
            consonance = cdns.get('consonance', 0.0)
        else:
            dissonance = 0.0
            consonance = 0.0
            
        # Compute diagnostic scores
        # Criticality: closer to 0.6 is better (optimal range 0.5-0.7)
        criticality_score = 1.0 - abs(R - 0.6) / 0.6
        criticality_score = max(0.0, min(1.0, criticality_score))
        
        # Dissonance: lower is better (phases aligned)
        dissonance_score = max(0.0, 1.0 - min(dissonance, 1.0))
        
        # Resonance health: combination of criticality and low dissonance
        resonance_health = (criticality_score + dissonance_score) / 2.0
        
        # CoT structure analysis
        cot_lines = [line.strip() for line in cot_rationale.split('\n') if line.strip()]
        cot_steps = len(cot_lines)
        cot_complexity = min(1.0, cot_steps / 10.0)  # Normalize to 0-1, 10 steps = max
        
        # Overall quality (weighted combination)
        overall_quality = (
            0.4 * resonance_health +
            0.3 * criticality_score +
            0.2 * dissonance_score +
            0.1 * cot_complexity
        )
        
        return {
            'resonance_health': float(resonance_health),
            'criticality_score': float(criticality_score),
            'dissonance_score': float(dissonance_score),
            'cot_complexity': float(cot_complexity),
            'overall_quality': float(overall_quality),
            'order_parameter_R': float(R),
            'consonance': float(consonance),
            'dissonance': float(dissonance),
            'cot_step_count': int(cot_steps),
        }
        
    def _collect_metrics(self) -> Dict:
        """
        Collect resonance metrics from all ResonanceAttentionHead modules.
        
        Returns:
            Aggregated metrics dictionary
        """
        metrics = {}
        head_count = 0
        
        for head in iter_resonance_heads(self.model):
            head_count += 1
            
            # Try to get metrics from head's last forward pass
            if hasattr(head, 'last_metrics'):
                head_metrics = head.last_metrics
            else:
                # Fallback: try to get from adapter
                head_metrics = {}
                
            # Aggregate metrics
            for k, v in head_metrics.items():
                if isinstance(v, dict):
                    # Nested dict (e.g., 'cdns')
                    if k not in metrics:
                        metrics[k] = {}
                    for sk, sv in v.items():
                        if isinstance(sv, torch.Tensor):
                            if sk not in metrics[k]:
                                metrics[k][sk] = []
                            metrics[k][sk].append(sv.detach().cpu())
                        elif isinstance(sv, (int, float)):
                            if sk not in metrics[k]:
                                metrics[k][sk] = []
                            metrics[k][sk].append(float(sv))
                elif isinstance(v, torch.Tensor):
                    if k not in metrics:
                        metrics[k] = []
                    metrics[k].append(v.detach().cpu())
                elif isinstance(v, (int, float)):
                    if k not in metrics:
                        metrics[k] = []
                    metrics[k].append(float(v))
        
        # Average metrics across heads
        avg_metrics = {}
        for k, values in metrics.items():
            if isinstance(values, dict):
                # Nested dict
                avg_metrics[k] = {}
                for sk, sv_list in values.items():
                    if sv_list:
                        if isinstance(sv_list[0], torch.Tensor):
                            stacked = torch.stack(sv_list)
                            avg_metrics[k][sk] = stacked.mean().item()
                        else:
                            avg_metrics[k][sk] = sum(sv_list) / len(sv_list)
            elif isinstance(values, list) and values:
                if isinstance(values[0], torch.Tensor):
                    stacked = torch.stack(values)
                    avg_metrics[k] = stacked.mean().item()
                else:
                    avg_metrics[k] = sum(values) / len(values)
            else:
                avg_metrics[k] = values
                
        # Ensure we have defaults
        if 'final_order_parameter' not in avg_metrics:
            avg_metrics['final_order_parameter'] = 0.5
        if 'cdns' not in avg_metrics:
            avg_metrics['cdns'] = {
                'consonance': 0.5,
                'dissonance': 0.2,
            }
            
        return avg_metrics

