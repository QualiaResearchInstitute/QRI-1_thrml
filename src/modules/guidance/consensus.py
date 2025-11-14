from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ConsensusConfig:
    r_threshold: float = 0.85
    drift_epsilon: float = 0.05
    descent_window: int = 3
    clip_z_max: float = 1.0  # require |z_clip| <= this
    prosody_z_max: float = 1.0


class ConsensusGovernor:
    """
    Computes a simple consensus decision from model metrics (if available) and critic z-scores.
    Intended to gate emission (commit) vs continued negotiation, and modulate guidance pulses.
    """
    def __init__(self, cfg: Optional[ConsensusConfig] = None) -> None:
        self.cfg = cfg or ConsensusConfig()
        self._descent_count: int = 0
        self._last_energy: Optional[float] = None

    def _update_descent(self, energy: Optional[float]) -> None:
        if energy is None:
            self._descent_count = 0
            self._last_energy = None
            return
        if self._last_energy is None:
            self._descent_count = 1
        else:
            if energy <= self._last_energy:
                self._descent_count += 1
            else:
                self._descent_count = 0
        self._last_energy = energy

    def decide(
        self,
        model_metrics: Optional[Dict[str, Any]] = None,
        critic_zscores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, bool]:
        """
        Returns:
            {
              'should_pulse': bool,   # apply guidance now
              'allow_commit': bool,   # ok to emit/commit (high consensus)
            }
        """
        r = None
        max_drift = None
        energy = None

        # Extract from resonance metrics if present
        if model_metrics:
            r = model_metrics.get("order_parameter", None)
            max_drift = model_metrics.get("max_phase_drift", None)
            energy = model_metrics.get("energy", None)

        self._update_descent(energy)

        # Critic z-scores (lower is better)
        cz = critic_zscores or {}
        z_clip = abs(float(cz.get("clip", 0.0)))
        z_prosody = abs(float(cz.get("prosody", 0.0)))

        # Text consensus conditions
        text_ok = True
        if r is not None and r < self.cfg.r_threshold:
            text_ok = False
        if max_drift is not None and max_drift > self.cfg.drift_epsilon:
            text_ok = False
        if self._descent_count < self.cfg.descent_window:
            text_ok = False

        # Cross-modal consensus conditions
        multimodal_ok = True
        if z_clip > self.cfg.clip_z_max:
            multimodal_ok = False
        if z_prosody > self.cfg.prosody_z_max:
            multimodal_ok = False

        allow_commit = bool(text_ok and multimodal_ok)
        # Pulse guidance if not in consensus
        should_pulse = not allow_commit

        return {
            "should_pulse": should_pulse,
            "allow_commit": allow_commit,
        }


