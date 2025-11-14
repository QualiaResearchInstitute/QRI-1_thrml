"""
Self-modification loop utilities that integrate recursive dynamics probes
with configurable safety thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .recursive_dynamics_probes import (
    RecursiveDynamicsMetrics,
    RecursiveDynamicsProbe,
    ProbeSafetyConfig,
)


@dataclass
class SelfModificationStepResult:
    """Result for a single self-modification attempt."""

    step: int
    accepted: bool
    safety_triggered: bool
    candidate_metrics: RecursiveDynamicsMetrics
    previous_metrics: RecursiveDynamicsMetrics
    metadata: Dict[str, Union[str, float, int, bool]]
    abort_reason: Optional[str]


class SelfModificationLoop:
    """
    Orchestrates recursive self-improvement attempts with safety monitoring.

    The loop evaluates candidate model updates using the RecursiveDynamicsProbe.
    If a safety threshold is triggered, the loop halts and returns the captured
    metrics so callers can roll back the change.
    """

    def __init__(
        self,
        probe: Optional[RecursiveDynamicsProbe] = None,
    ):
        self.probe = probe or RecursiveDynamicsProbe()

    def run(
        self,
        *,
        initial_model: nn.Module,
        input_ids: torch.Tensor,
        steps: int,
        mutate_fn: Callable[[nn.Module, int], Tuple[nn.Module, Dict[str, Union[str, float, int, bool]]]],
        accept_fn: Callable[[RecursiveDynamicsMetrics, RecursiveDynamicsMetrics], bool],
        probe_iterations: int = 10,
        safety_config: Optional[ProbeSafetyConfig] = None,
        log_path: Optional[Union[str, Path]] = None,
        callback: Optional[Callable[[SelfModificationStepResult], None]] = None,
    ) -> List[SelfModificationStepResult]:
        """
        Execute the self-modification loop.

        Args:
            initial_model: Starting model instance.
            input_ids: Prompt tokens used for recursive probing (shape [batch, seq]).
            steps: Maximum number of modification attempts.
            mutate_fn: Callable returning a (candidate_model, metadata) pair.
            accept_fn: Predicate deciding whether to accept the candidate metrics.
            probe_iterations: Number of recursive iterations for the probe.
            safety_config: Optional overrides for probe safety thresholds.
            log_path: Optional JSONL file path for telemetry logging.
            callback: Optional callable invoked with each step result.
        """
        if steps <= 0:
            return []

        device = input_ids.device
        cfg = safety_config or self.probe.safety_config
        probe = self.probe

        # Ensure model lives on same device as input
        initial_model.to(device)
        current_model = initial_model

        # Baseline metrics
        baseline_metrics = probe.probe_recursive_dynamics(
            model=current_model,
            input_ids=input_ids,
            num_iterations=probe_iterations,
            safety_config=cfg,
        )

        results: List[SelfModificationStepResult] = []
        current_metrics = baseline_metrics

        log_file = None
        if log_path:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a")

            baseline_payload = {
                "step": 0,
                "accepted": True,
                "safety_triggered": bool(baseline_metrics.abort_reason),
                "metrics": asdict(baseline_metrics),
                "metadata": {"event": "baseline"},
            }
            log_file.write(json.dumps(baseline_payload) + "\n")
            log_file.flush()

        try:
            for step_idx in range(1, steps + 1):
                prev_metrics = current_metrics
                candidate_model, metadata = mutate_fn(current_model, step_idx)
                candidate_model.to(device)

                candidate_metrics = probe.probe_recursive_dynamics(
                    model=candidate_model,
                    input_ids=input_ids,
                    num_iterations=probe_iterations,
                    safety_config=cfg,
                )

                safety_triggered = bool(candidate_metrics.abort_reason) or bool(
                    candidate_metrics.safety_events
                )

                if safety_triggered:
                    accepted = False
                    abort_reason = candidate_metrics.abort_reason or "safety_event_triggered"
                else:
                    accepted = accept_fn(current_metrics, candidate_metrics)
                    abort_reason = None

                if accepted:
                    current_model = candidate_model
                    current_metrics = candidate_metrics
                else:
                    # Detach candidate model to free memory if not accepted
                    if candidate_model is not current_model:
                        self._maybe_cpu(candidate_model)

                step_result = SelfModificationStepResult(
                    step=step_idx,
                    accepted=accepted,
                    safety_triggered=safety_triggered,
                    candidate_metrics=candidate_metrics,
                    previous_metrics=prev_metrics,
                    metadata=metadata,
                    abort_reason=abort_reason,
                )

                results.append(step_result)

                if log_file:
                    payload = {
                        "step": step_idx,
                        "accepted": accepted,
                        "safety_triggered": safety_triggered,
                        "abort_reason": abort_reason,
                        "candidate_metrics": asdict(candidate_metrics),
                        "previous_metrics": asdict(prev_metrics),
                        "metadata": metadata,
                    }
                    log_file.write(json.dumps(payload) + "\n")
                    log_file.flush()

                if callback:
                    try:
                        callback(step_result)
                    except Exception:
                        # Best-effort callback; errors should not break the loop
                        pass

                if safety_triggered:
                    break

        finally:
            if log_file:
                log_file.close()

        return results

    @staticmethod
    def _maybe_cpu(model: nn.Module) -> None:
        """Move model parameters to CPU to release GPU memory."""
        try:
            model.to("cpu")
        except Exception:
            pass

