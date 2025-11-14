from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional

import torch

try:
    from modules.telemetry import TelemetryWriter, TelemetryConfig
except Exception:
    # Fallback stub so imports don't explode if telemetry module moves
    class TelemetryConfig:  # type: ignore
        def __init__(self, **kwargs): ...
    class TelemetryWriter:  # type: ignore
        def __init__(self, config: Optional[TelemetryConfig] = None): ...
        def emit(self, data: Dict[str, Any], event_type: str = "event", severity: str = "INFO") -> None: ...


def _compute_order_parameter(phases_1d: torch.Tensor) -> Dict[str, float]:
    """
    Compute global order parameter r and phase psi from a 1D tensor of phases [N] (radians).
    Returns { "r": float, "psi": float }.
    """
    if phases_1d is None or not torch.is_tensor(phases_1d):
        return {"r": 0.0, "psi": 0.0}
    try:
        phases = phases_1d.detach().float()
        N = max(1, phases.numel())
        z = torch.exp(1j * phases)  # [N]
        R = z.mean()
        r = float(torch.abs(R).item())
        psi = float(torch.angle(R).item())
        return {"r": r, "psi": psi}
    except Exception:
        return {"r": 0.0, "psi": 0.0}


def _to_float_list(x: torch.Tensor, stride: int = 1, max_len: int = 2048) -> List[float]:
    if x is None or not torch.is_tensor(x):
        return []
    try:
        x = x.detach().float().view(-1)
        if stride > 1:
            x = x[::stride]
        if x.numel() > max_len:
            stride2 = max(1, math.ceil(x.numel() / max_len))
            x = x[::stride2]
        return [float(v) for v in x.tolist()]
    except Exception:
        return []


class RTLiveEmitter:
    """
    Lightweight decimating emitter that writes compact JSONL lines using TelemetryWriter.
    Intended to be called from within guided_generate per token/step.

    Payload shape (example):
    {
      "type": "rt-metrics",
      "t": 1731370000.123,
      "step": 124,
      "r": 0.82,
      "psi": 1.57,
      "phases": [...],         # float array (possibly decimated)
      "phases_stride": 1,      # stride used when decimating
      "energies": { "clip": 1.3, "prosody": -0.4, "schema": 0.9 },  # z-scores if available
      "topk": [
        { "token": "the", "p": 0.12, "base_logit": -2.1, "delta_logit": +0.3 }
      ],
      "consensus": "commit" | "negotiate"
    }
    """

    def __init__(
        self,
        telemetry_writer: Optional[TelemetryWriter] = None,
        *,
        hz: float = 12.0,
        wal_path: Optional[str] = None,
        max_phases: int = 2048,
    ):
        self.last_emit_t = 0.0
        self.period_s = 1.0 / max(1e-3, hz)
        self.max_phases = int(max_phases)
        if telemetry_writer is not None:
            self.tw = telemetry_writer
        else:
            # Default to repo's WAL path used by dashboard/telemetry_server.js
            default_wal = wal_path or os.path.join("spare_parts", ".wal", "telemetry.log")
            os.makedirs(os.path.dirname(default_wal) or ".", exist_ok=True)
            cfg = TelemetryConfig(
                enabled=True,
                output_path=default_wal,
                structured_logging=True,
                log_to_stdout=False,  # keep quiet in console unless explicitly desired
                sample_rate=1.0,
            )
            self.tw = TelemetryWriter(cfg)

    def maybe_emit(
        self,
        *,
        step: int,
        phases_1d: Optional[torch.Tensor],
        energies_z: Optional[Dict[str, float]],
        topk: Optional[List[Dict[str, float]]],
        consensus: Optional[str],
        r_psi: Optional[Dict[str, float]] = None,
    ) -> None:
        now = time.time()
        if (now - self.last_emit_t) < self.period_s:
            return

        payload: Dict[str, Any] = {
            "type": "rt-metrics",
            "t": now,
            "step": int(step),
            "consensus": str(consensus or "negotiate"),
        }

        # Global order parameter (compute if not provided)
        if r_psi is None:
            r_psi = _compute_order_parameter(phases_1d if phases_1d is not None else torch.tensor([]))
        payload.update({"r": float(r_psi.get("r", 0.0)), "psi": float(r_psi.get("psi", 0.0))})

        # Phases (decimate if too long)
        phases_list: List[float] = []
        stride_used = 1
        if phases_1d is not None and torch.is_tensor(phases_1d) and phases_1d.numel() > 0:
            if phases_1d.numel() > self.max_phases:
                stride_used = max(1, math.ceil(phases_1d.numel() / self.max_phases))
            phases_list = _to_float_list(phases_1d, stride=stride_used, max_len=self.max_phases)
        payload["phases"] = phases_list
        payload["phases_stride"] = int(stride_used)

        # Energies (z-scores)
        if energies_z:
            # keep it small and flat
            payload["energies"] = {k: float(v) for (k, v) in energies_z.items() if isinstance(v, (int, float))}

        # Top-K (cap to ~12)
        if topk:
            payload["topk"] = topk[:12]

        self.tw.emit(payload, event_type="rt-metrics", severity="INFO")
        self.last_emit_t = now


def extract_head_phases_1d(model: Any) -> Optional[torch.Tensor]:
    """
    Best-effort helper to extract a 1D phase vector [N] from the first available resonance head.
    """
    try:
        from modules.resonance_gpt2_adapter import iter_resonance_heads  # type: ignore
    except Exception:
        return None

    try:
        for head in iter_resonance_heads(model):
            ph = None
            if hasattr(head, "get_last_phases"):
                try:
                    ph = head.get_last_phases(detach=True, cpu=False)
                except Exception:
                    ph = None
            if ph is None and hasattr(head, "_last_phases") and head._last_phases is not None:
                ph = head._last_phases
            if torch.is_tensor(ph) and ph.dim() >= 1:
                # use last time slice if 2D
                if ph.dim() == 2:
                        ph = ph[-1]
                    return ph.view(-1)
        return None
    except Exception:
        return None
