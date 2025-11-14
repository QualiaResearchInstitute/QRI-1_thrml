from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch


class Projector:
    """
    Base interface for hard projectors that can suppress or disallow tokens.
    Two usage modes:
    - project_logits: apply on full logits (broad masks if possible)
    - project_topk: adjust top-k candidate logits given their strings
    """
    name: str = "projector"

    def project_logits(self, logits: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        return logits

    def project_topk(
        self,
        topk_indices: torch.Tensor,
        topk_logits: torch.Tensor,
        candidate_strings: List[str],
        context: Dict[str, Any],
    ) -> torch.Tensor:
        return topk_logits


class SafetyProjector(Projector):
    """
    Simple safety projector using regex denylist on candidate strings.
    Applies a large negative penalty to offending candidates.
    """
    name = "safety"

    def __init__(self, patterns: Optional[List[str]] = None, penalty: float = 100.0) -> None:
        if patterns is None:
            patterns = [
                r"\b\d{3}-\d{2}-\d{4}\b",   # SSN-like
                r"\b\d{10}\b",              # 10-digit phone-like
                r"[A-Za-z0-9\.\-_]+@[A-Za-z0-9\.\-]+\.[A-Za-z]{2,}",  # email
            ]
        self._regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.penalty = float(penalty)

    def project_topk(
        self,
        topk_indices: torch.Tensor,
        topk_logits: torch.Tensor,
        candidate_strings: List[str],
        context: Dict[str, Any],
    ) -> torch.Tensor:
        if len(candidate_strings) == 0:
            return topk_logits
        adjusted = topk_logits.clone()
        for i, s in enumerate(candidate_strings):
            for rgx in self._regexes:
                if rgx.search(s):
                    adjusted[..., i] = adjusted[..., i] - self.penalty
                    break
        return adjusted


class JSONSchemaProjector(Projector):
    """
    Stub for JSON schema enforcement. For now, no-op.
    Extend to maintain a streaming JSON validator and mask tokens that would break the schema.
    """
    name = "json_schema"

    def __init__(self, schema: Optional[Dict[str, Any]] = None) -> None:
        self.schema = schema


