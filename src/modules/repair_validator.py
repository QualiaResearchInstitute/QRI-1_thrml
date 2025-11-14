"""
Repair Validator: Shadow-run validation for GPT-2 repair proposals.

Provides:
- TelemetryProbe for recording metrics
- Scorer for computing quality scores
- RepairValidator for shadow-running and decision making
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from modules.genome_registry import Address, DeltaSpec, Guardrails
from modules.rna_transcriber import RNAProposal, RNAAdapter, ApplyToken


@dataclass
class MetricsWindow:
    """Sliding window of metrics."""
    R: List[float] = field(default_factory=list)
    var_R: List[float] = field(default_factory=list)
    cdns: List[float] = field(default_factory=list)
    cov_disc: List[float] = field(default_factory=list)
    latency_ms: List[float] = field(default_factory=list)
    
    def add(
        self,
        R: float,
        var_R: float,
        cdns: float,
        cov_disc: float,
        latency_ms: float,
    ):
        """Add a metrics point."""
        self.R.append(R)
        self.var_R.append(var_R)
        self.cdns.append(cdns)
        self.cov_disc.append(cov_disc)
        self.latency_ms.append(latency_ms)
    
    def mean(self) -> Dict[str, float]:
        """Compute mean metrics."""
        return {
            'R': statistics.mean(self.R) if self.R else 0.0,
            'var_R': statistics.mean(self.var_R) if self.var_R else 0.0,
            'cdns': statistics.mean(self.cdns) if self.cdns else 0.0,
            'cov_disc': statistics.mean(self.cov_disc) if self.cov_disc else 0.0,
            'latency_ms': statistics.mean(self.latency_ms) if self.latency_ms else 0.0,
        }
    
    def std(self) -> Dict[str, float]:
        """Compute std metrics."""
        return {
            'R': statistics.stdev(self.R) if len(self.R) > 1 else 0.0,
            'var_R': statistics.stdev(self.var_R) if len(self.var_R) > 1 else 0.0,
            'cdns': statistics.stdev(self.cdns) if len(self.cdns) > 1 else 0.0,
            'cov_disc': statistics.stdev(self.cov_disc) if len(self.cov_disc) > 1 else 0.0,
            'latency_ms': statistics.stdev(self.latency_ms) if len(self.latency_ms) > 1 else 0.0,
        }


class TelemetryProbe:
    """
    Minimal adapter for recording telemetry metrics.
    
    Accepts precomputed values from existing telemetry infrastructure.
    """
    
    def __init__(self, scorer: 'Scorer'):
        """
        Initialize probe.
        
        Args:
            scorer: Scorer instance for computing quality scores
        """
        self.scorer = scorer
        self._window = MetricsWindow()
    
    def record(
        self,
        step: int,
        R: float,
        var_R: float,
        cdns: float,
        cov_disc: float,
        latency_ms: float,
    ) -> None:
        """
        Record metrics for a step.
        
        Args:
            step: Step number
            R: Order parameter
            var_R: Order parameter variance
            cdns: CDNS metric (consonance/dissonance)
            cov_disc: Covariance discordance
            latency_ms: Latency in milliseconds
        """
        self._window.add(R, var_R, cdns, cov_disc, latency_ms)
    
    def window(self, window_size: int) -> MetricsWindow:
        """
        Get metrics window (last N steps).
        
        Args:
            window_size: Size of window
        
        Returns:
            MetricsWindow with last window_size steps
        """
        w = MetricsWindow()
        n = min(window_size, len(self._window.R))
        if n > 0:
            w.R = self._window.R[-n:]
            w.var_R = self._window.var_R[-n:]
            w.cdns = self._window.cdns[-n:]
            w.cov_disc = self._window.cov_disc[-n:]
            w.latency_ms = self._window.latency_ms[-n:]
        return w


class Scorer:
    """
    Quality scorer for repair proposals.
    
    Computes scalar quality score: q = wR*R - wVar*var_R - wCDNS*cdns - wCov*cov_disc - wLat*latency_penalty
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        latency_threshold_ms: float = 2.0,
    ):
        """
        Initialize scorer.
        
        Args:
            weights: Weight dict with keys 'R', 'var_R', 'cdns', 'cov_disc', 'latency'
            latency_threshold_ms: Threshold for latency penalty (piecewise linear)
        """
        self.weights = weights or {
            'R': 1.0,
            'var_R': 0.5,
            'cdns': 0.3,
            'cov_disc': 0.2,
            'latency': 0.1,
        }
        self.latency_threshold_ms = latency_threshold_ms
    
    def __call__(
        self,
        R: float,
        var_R: float,
        cdns: float,
        cov_disc: float,
        latency_ms: float,
    ) -> float:
        """
        Compute quality score.
        
        Args:
            R: Order parameter
            var_R: Order parameter variance
            cdns: CDNS metric
            cov_disc: Covariance discordance
            latency_ms: Latency in milliseconds
        
        Returns:
            Quality score (higher is better)
        """
        # Latency penalty (piecewise linear)
        if latency_ms <= self.latency_threshold_ms:
            latency_penalty = 0.0
        else:
            latency_penalty = (latency_ms - self.latency_threshold_ms) / self.latency_threshold_ms
        
        q = (
            self.weights['R'] * R -
            self.weights['var_R'] * var_R -
            self.weights['cdns'] * cdns -
            self.weights['cov_disc'] * cov_disc -
            self.weights['latency'] * latency_penalty
        )
        return q


@dataclass
class ValidationResult:
    """Result of validating a proposal."""
    proposal_id: str
    accepted: bool
    baseline_q: float
    candidate_q: float
    baseline_latency_ms: float
    candidate_latency_ms: float
    q_delta: float
    latency_delta_ms: float
    cohens_d: Optional[float] = None  # Effect size
    collapse_detected: bool = False
    collapse_reason: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class RepairValidator:
    """
    Validator for shadow-running repair proposals.
    
    Shadow-runs candidate proposals against held-out prompts and decides
    acceptance based on Pareto rule with effect size.
    """
    
    def __init__(
        self,
        probe: TelemetryProbe,
        scorer: Scorer,
        guardrails: Guardrails,
        tau_ms: Optional[float] = None,
        epsilon: Optional[float] = None,
    ):
        """
        Initialize validator.
        
        Args:
            probe: TelemetryProbe for recording metrics
            scorer: Scorer for computing quality scores
            guardrails: Guardrails for collapse detection
            tau_ms: Latency threshold (default: max(1% baseline, 2ms))
            epsilon: Quality threshold (default: 0.5σ of baseline q)
        """
        self.probe = probe
        self.scorer = scorer
        self.guardrails = guardrails
        self.tau_ms = tau_ms  # Will be computed from baseline if None
        self.epsilon = epsilon  # Will be computed from baseline if None
    
    def evaluate(
        self,
        model: nn.Module,
        prompts: List[Any],  # List of tokenized prompts or strings
        proposals: List[RNAProposal],
        run_baseline: Callable[[nn.Module, List[Any]], Tuple[Dict[str, float], float]],
        run_candidate: Callable[[nn.Module, List[Any]], Tuple[Dict[str, float], float]],
        rna_adapter: RNAAdapter,
    ) -> List[ValidationResult]:
        """
        Shadow-run proposals and return validation results.
        
        Args:
            model: GPT-2 model
            prompts: List of prompts (held-out slice)
            proposals: List of RNA proposals to evaluate
            run_baseline: Function(model, prompts) -> (metrics_dict, latency_ms)
            run_candidate: Function(model, prompts, apply_fn) -> (metrics_dict, latency_ms)
            rna_adapter: RNAAdapter for applying proposals
        
        Returns:
            List of ValidationResult objects
        """
        # Run baseline
        baseline_metrics, baseline_latency_ms = run_baseline(model, prompts)
        baseline_q = self.scorer(
            R=baseline_metrics.get('R', 0.0),
            var_R=baseline_metrics.get('var_R', 0.0),
            cdns=baseline_metrics.get('cdns', 0.0),
            cov_disc=baseline_metrics.get('cov_disc', 0.0),
            latency_ms=baseline_latency_ms,
        )
        
        # Compute thresholds if not provided
        tau_ms = self.tau_ms
        if tau_ms is None:
            tau_ms = max(0.01 * baseline_latency_ms, 2.0)
        
        epsilon = self.epsilon
        if epsilon is None:
            # Use 0.5σ heuristic (would need multiple baseline runs for true σ)
            epsilon = 0.5 * abs(baseline_q) * 0.1  # Rough estimate
        
        # Evaluate each proposal
        results = []
        for proposal in proposals:
            result = self._evaluate_proposal(
                model=model,
                prompts=prompts,
                proposal=proposal,
                baseline_q=baseline_q,
                baseline_latency_ms=baseline_latency_ms,
                tau_ms=tau_ms,
                epsilon=epsilon,
                run_candidate=run_candidate,
                rna_adapter=rna_adapter,
            )
            results.append(result)
        
        return results
    
    def _evaluate_proposal(
        self,
        model: nn.Module,
        prompts: List[Any],
        proposal: RNAProposal,
        baseline_q: float,
        baseline_latency_ms: float,
        tau_ms: float,
        epsilon: float,
        run_candidate: Callable,
        rna_adapter: RNAAdapter,
    ) -> ValidationResult:
        """Evaluate a single proposal."""
        # Apply proposal
        token = rna_adapter.apply(proposal.proposal_id, model, [proposal])
        
        try:
            # Check for immediate collapse
            collapse_detected, collapse_reason = self._check_collapse(model)
            if collapse_detected:
                return ValidationResult(
                    proposal_id=proposal.proposal_id,
                    accepted=False,
                    baseline_q=baseline_q,
                    candidate_q=baseline_q,  # Not computed
                    baseline_latency_ms=baseline_latency_ms,
                    candidate_latency_ms=baseline_latency_ms,
                    q_delta=0.0,
                    latency_delta_ms=0.0,
                    collapse_detected=True,
                    collapse_reason=collapse_reason,
                )
            
            # Run candidate (proposal already applied via token above)
            candidate_metrics, candidate_latency_ms = run_candidate(model, prompts)
            candidate_q = self.scorer(
                R=candidate_metrics.get('R', 0.0),
                var_R=candidate_metrics.get('var_R', 0.0),
                cdns=candidate_metrics.get('cdns', 0.0),
                cov_disc=candidate_metrics.get('cov_disc', 0.0),
                latency_ms=candidate_latency_ms,
            )
            
            # Compute effect size (Cohen's d approximation)
            q_delta = candidate_q - baseline_q
            latency_delta_ms = candidate_latency_ms - baseline_latency_ms
            
            # Simple Cohen's d estimate (would need std for proper calculation)
            cohens_d = q_delta / (abs(baseline_q) + 1e-10) if abs(baseline_q) > 0 else 0.0
            
            # Decision rule: Pareto with thresholds
            accepted = (
                candidate_latency_ms <= baseline_latency_ms + tau_ms and
                candidate_q >= baseline_q + epsilon
            )
            
            return ValidationResult(
                proposal_id=proposal.proposal_id,
                accepted=accepted,
                baseline_q=baseline_q,
                candidate_q=candidate_q,
                baseline_latency_ms=baseline_latency_ms,
                candidate_latency_ms=candidate_latency_ms,
                q_delta=q_delta,
                latency_delta_ms=latency_delta_ms,
                cohens_d=cohens_d,
                collapse_detected=False,
                metrics=candidate_metrics,
            )
        
        finally:
            # Always rollback
            rna_adapter.rollback(token, model)
    
    def _check_collapse(self, model: nn.Module) -> Tuple[bool, Optional[str]]:
        """
        Check for collapse signals.
        
        Returns:
            (collapsed, reason) tuple
        """
        # Extract metrics from resonance heads
        from modules.resonance_gpt2_adapter import iter_resonance_heads
        
        all_R = []
        all_cdns = []
        has_nan = False
        
        for head in iter_resonance_heads(model):
            metrics = getattr(head, 'metrics', {}) or {}
            
            if 'final_order_parameter' in metrics:
                R = metrics['final_order_parameter']
                if isinstance(R, torch.Tensor):
                    if torch.isnan(R).any() or torch.isinf(R).any():
                        has_nan = True
                    R_val = float(R.mean().item() if R.numel() > 1 else R.item())
                    all_R.append(R_val)
            
            if 'cdns' in metrics:
                cdns_dict = metrics['cdns']
                if isinstance(cdns_dict, dict):
                    cdns_val = cdns_dict.get('consonance', torch.tensor(0.0))
                    if isinstance(cdns_val, torch.Tensor):
                        if torch.isnan(cdns_val).any() or torch.isinf(cdns_val).any():
                            has_nan = True
                        cdns_val = float(cdns_val.mean().item() if cdns_val.numel() > 1 else cdns_val.item())
                        all_cdns.append(cdns_val)
        
        # Check guardrails
        if has_nan and self.guardrails.collapse['abort_on_nan']:
            return True, "NaN detected"
        
        if all_R:
            mean_R = sum(all_R) / len(all_R)
            if mean_R < self.guardrails.collapse['R_min']:
                return True, f"R={mean_R:.3f} < {self.guardrails.collapse['R_min']}"
        
        if all_cdns:
            mean_cdns = sum(all_cdns) / len(all_cdns)
            if mean_cdns > self.guardrails.collapse['cdns_max']:
                return True, f"cdns={mean_cdns:.3f} > {self.guardrails.collapse['cdns_max']}"
        
        return False, None
    
    def accepts(self, result: ValidationResult) -> bool:
        """Check if validation result is accepted."""
        return result.accepted and not result.collapse_detected
    
    def to_dna_deltas(self, accepted: List[ValidationResult], proposals: List[RNAProposal]) -> List[DeltaSpec]:
        """
        Convert accepted validation results to DNA deltas.
        
        Args:
            accepted: List of accepted ValidationResult objects
            proposals: Original proposals
        
        Returns:
            List of DeltaSpec objects ready for DNA
        """
        # Map proposal_id to proposal
        proposal_map = {p.proposal_id: p for p in proposals}
        
        dna_deltas = []
        for result in accepted:
            proposal = proposal_map.get(result.proposal_id)
            if proposal is None:
                continue
            
            # Add metrics_delta to each spec
            for spec in proposal.specs:
                spec.metrics_delta = {
                    'resonance': result.q_delta,
                    'latency_ms': result.latency_delta_ms,
                    'sample_n': 1,  # Would be actual sample count in real usage
                }
                spec.provenance = {
                    'source': 'validator',
                    'rna_ids': [proposal.proposal_id],
                    'notes': f"Accepted: q_delta={result.q_delta:.4f}, latency_delta={result.latency_delta_ms:.2f}ms",
                }
                dna_deltas.append(spec)
        
        return dna_deltas


__all__ = [
    'TelemetryProbe',
    'Scorer',
    'RepairValidator',
    'ValidationResult',
    'MetricsWindow',
]

