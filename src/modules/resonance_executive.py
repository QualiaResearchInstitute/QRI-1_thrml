"""
Resonance Executive Controller

Implements an OGI/TI-inspired executive that coordinates resonance modules,
criticality controllers, and safety probes. The controller observes global
oscillatory metrics, applies corrective actions (PID target adjustments,
harmonic leak tuning, curriculum throttling hooks), and logs every decision.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

try:
    from modules.recursive_dynamics_probes import RecursiveDynamicsMetrics  # type: ignore
except Exception:  # pragma: no cover - optional dependency during tests
    RecursiveDynamicsMetrics = None  # type: ignore


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


@dataclass
class ExecutiveConfig:
    """Configuration parameters for the resonance executive controller."""

    target_order_parameter: float = 0.6
    order_param_band: float = 0.05
    pid_adjust_rate: float = 0.02
    harmonic_leak_bounds: Tuple[float, float] = (5e-4, 1e-2)
    harmonic_leak_adjust: float = 5e-4
    dissonance_scale_up: float = 1.1
    dissonance_scale_down: float = 0.95
    max_probe_coherence: float = 0.92
    chaos_disable_temporal_multiplex: bool = True
    chaos_reduce_dt: bool = True
    chaos_dt_scale: float = 0.8
    clamp_target_r: Tuple[float, float] = (0.05, 0.95)
    decision_log_path: Optional[str] = None
    retain_history: int = 200

    def __post_init__(self) -> None:
        if self.order_param_band <= 0:
            raise ValueError("order_param_band must be > 0")
        if not (0 < self.pid_adjust_rate <= 0.5):
            raise ValueError("pid_adjust_rate must be in (0, 0.5]")
        lo, hi = self.harmonic_leak_bounds
        if not (0 < lo < hi):
            raise ValueError("harmonic_leak_bounds must satisfy 0 < lo < hi")
        if self.harmonic_leak_adjust <= 0:
            raise ValueError("harmonic_leak_adjust must be > 0")
        if not (0 < self.target_order_parameter < 1):
            raise ValueError("target_order_parameter must lie in (0, 1)")
        if not (self.clamp_target_r[0] < self.clamp_target_r[1]):
            raise ValueError("clamp_target_r lower bound must be < upper bound")
        if not (0 < self.max_probe_coherence < 1):
            raise ValueError("max_probe_coherence must lie in (0, 1)")
        if not (0 < self.chaos_dt_scale <= 1):
            raise ValueError("chaos_dt_scale must lie in (0, 1]")


@dataclass
class Adjustment:
    """Single adjustment applied by the executive."""

    scope: str  # e.g. "head", "global"
    key: str
    old_value: Optional[float]
    new_value: Optional[float]
    delta: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutiveDecision:
    """Decision summary per step."""

    step: int
    epoch: int
    avg_order_parameter: float
    avg_dissonance: float
    avg_order_variance: float
    avg_harmonic_dim: float
    probe_metrics: Optional[Dict[str, Any]]
    adjustments: List[Adjustment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["adjustments"] = [asdict(adj) for adj in self.adjustments]
        return payload


class ResonanceExecutive:
    """
    Coordinates resonance modules by monitoring oscillator metrics and applying
    control actions that keep the system within safe, productive regimes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[ExecutiveConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or ExecutiveConfig()
        self.heads = self._collect_heads(model)
        self.decision_history: List[ExecutiveDecision] = []
        self.log_path: Optional[Path] = None
        if self.config.decision_log_path:
            self.log_path = Path(self.config.decision_log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _collect_heads(model: torch.nn.Module) -> List[torch.nn.Module]:
        heads: List[torch.nn.Module] = []
        for module in model.modules():
            if hasattr(module, "metrics") and hasattr(module, "target_R"):
                heads.append(module)
        return heads

    def _gather_metrics(self) -> Dict[str, float]:
        orders: List[float] = []
        dissonances: List[float] = []
        variances: List[float] = []
        harmonic_dims: List[float] = []

        for head in self.heads:
            metrics = getattr(head, "metrics", None)
            if not metrics or not isinstance(metrics, dict):
                continue
            if "final_order_parameter" in metrics:
                try:
                    orders.append(float(metrics["final_order_parameter"].mean().item()))
                except Exception:
                    pass
            if "order_param_variance" in metrics:
                try:
                    variances.append(float(metrics["order_param_variance"].mean().item()))
                except Exception:
                    pass
            cdns = metrics.get("cdns", {})
            if isinstance(cdns, dict) and "dissonance" in cdns:
                try:
                    dissonances.append(float(cdns["dissonance"].mean().item()))
                except Exception:
                    pass
            hodge = metrics.get("hodge", {})
            if isinstance(hodge, dict) and "harmonic_dim" in hodge:
                try:
                    harmonic_dims.append(float(hodge["harmonic_dim"]))
                except Exception:
                    pass

        return {
            "order": _mean(orders),
            "variance": _mean(variances),
            "dissonance": _mean(dissonances),
            "harmonic_dim": _mean(harmonic_dims),
        }

    def step(
        self,
        *,
        step: int,
        epoch: int,
        probe_metrics: Optional[RecursiveDynamicsMetrics] = None,
    ) -> ExecutiveDecision:
        """
        Observe latest metrics and apply executive control actions.
        """
        agg = self._gather_metrics()
        adjustments: List[Adjustment] = []
        warnings: List[str] = []

        target = self.config.target_order_parameter
        band = self.config.order_param_band

        # Adjust target_R to nudge toward band
        avg_order = agg["order"]
        if math.isfinite(avg_order) and avg_order > 0:
            if avg_order < target - band:
                for idx, head in enumerate(self.heads):
                    if hasattr(head, "target_R"):
                        old = float(getattr(head, "target_R"))
                        new = min(self.config.clamp_target_r[1], old + self.config.pid_adjust_rate)
                        setattr(head, "target_R", new)
                        adjustments.append(
                            Adjustment(
                                scope="head",
                                key=f"target_R[{idx}]",
                                old_value=old,
                                new_value=new,
                                delta=new - old,
                            )
                        )
            elif avg_order > target + band:
                for idx, head in enumerate(self.heads):
                    if hasattr(head, "target_R"):
                        old = float(getattr(head, "target_R"))
                        new = max(self.config.clamp_target_r[0], old - self.config.pid_adjust_rate)
                        setattr(head, "target_R", new)
                        adjustments.append(
                            Adjustment(
                                scope="head",
                                key=f"target_R[{idx}]",
                                old_value=old,
                                new_value=new,
                                delta=new - old,
                            )
                        )

        # Harmonic leak tuning
        avg_harmonic_dim = agg["harmonic_dim"]
        if math.isfinite(avg_harmonic_dim) and avg_harmonic_dim > 0:
            leak_lo, leak_hi = self.config.harmonic_leak_bounds
            for idx, head in enumerate(self.heads):
                if hasattr(head, "harmonic_leak_rate"):
                    old = float(getattr(head, "harmonic_leak_rate"))
                    if avg_harmonic_dim < 1.0:
                        new = min(leak_hi, old + self.config.harmonic_leak_adjust)
                    else:
                        new = max(leak_lo, old - self.config.harmonic_leak_adjust)
                    if new != old:
                        setattr(head, "harmonic_leak_rate", new)
                        adjustments.append(
                            Adjustment(
                                scope="head",
                                key=f"harmonic_leak_rate[{idx}]",
                                old_value=old,
                                new_value=new,
                                delta=new - old,
                            )
                        )

        # Probe-driven guardrails
        probe_payload: Optional[Dict[str, Any]] = None
        if probe_metrics is not None and RecursiveDynamicsMetrics is not None:
            probe_payload = {
                "mi_with_input": probe_metrics.mi_with_input,
                "mi_with_priors": probe_metrics.mi_with_priors,
                "coherence_score": probe_metrics.coherence_score,
                "is_chaotic": probe_metrics.is_chaotic,
                "lyapunov_exponent": probe_metrics.lyapunov_exponent,
            }

            if probe_metrics.coherence_score > self.config.max_probe_coherence:
                warnings.append(
                    f"Probe coherence {probe_metrics.coherence_score:.3f} exceeds "
                    f"{self.config.max_probe_coherence:.3f}"
                )
                for idx, head in enumerate(self.heads):
                    if hasattr(head, "lambda_dissonance"):
                        old = float(getattr(head, "lambda_dissonance"))
                        new = old * self.config.dissonance_scale_up
                        setattr(head, "lambda_dissonance", new)
                        adjustments.append(
                            Adjustment(
                                scope="head",
                                key=f"lambda_dissonance[{idx}]",
                                old_value=old,
                                new_value=new,
                                delta=new - old,
                            )
                        )

            if probe_metrics.is_chaotic:
                warnings.append("Chaos detected by probe; applying stabilizers")
                for idx, head in enumerate(self.heads):
                    if self.config.chaos_disable_temporal_multiplex and hasattr(head, "use_temporal_multiplex"):
                        old = bool(getattr(head, "use_temporal_multiplex"))
                        if old:
                            setattr(head, "use_temporal_multiplex", False)
                            adjustments.append(
                                Adjustment(
                                    scope="head",
                                    key=f"use_temporal_multiplex[{idx}]",
                                    old_value=float(old),
                                    new_value=0.0,
                                    delta=-1.0,
                                )
                            )
                    if self.config.chaos_reduce_dt and hasattr(head, "dt"):
                        old_dt = float(getattr(head, "dt"))
                        new_dt = max(1e-4, old_dt * self.config.chaos_dt_scale)
                        if new_dt != old_dt:
                            setattr(head, "dt", new_dt)
                            adjustments.append(
                                Adjustment(
                                    scope="head",
                                    key=f"dt[{idx}]",
                                    old_value=old_dt,
                                    new_value=new_dt,
                                    delta=new_dt - old_dt,
                                )
                            )

        decision = ExecutiveDecision(
            step=step,
            epoch=epoch,
            avg_order_parameter=agg["order"],
            avg_dissonance=agg["dissonance"],
            avg_order_variance=agg["variance"],
            avg_harmonic_dim=agg["harmonic_dim"],
            probe_metrics=probe_payload,
            adjustments=adjustments,
            warnings=warnings,
        )

        self.decision_history.append(decision)
        if self.config.retain_history and len(self.decision_history) > self.config.retain_history:
            self.decision_history = self.decision_history[-self.config.retain_history :]

        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(decision.to_dict()) + "\n")

        return decision

    def latest_decision(self) -> Optional[ExecutiveDecision]:
        return self.decision_history[-1] if self.decision_history else None

