"""
LADDER-style curriculum generator for recursive self-improvement training.

This module synthesizes progressively harder/easier sub-problems by
transforming language modeling batches according to controllable difficulty
fractions, inspired by Simonds & Yoshiyama (2025).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch


@dataclass
class LadderCurriculumConfig:
    """Configuration for LADDER curriculum generation."""

    levels: int = 3
    min_fraction: float = 0.35
    max_fraction: float = 1.0
    random_truncate: bool = True
    noise_prob: float = 0.05
    weight_decay: float = 0.6
    include_original: bool = True
    curriculum_mix: str = "geometric"  # or "uniform"
    min_tokens: int = 16
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.levels < 1:
            raise ValueError("levels must be >= 1")
        if not (0.0 < self.min_fraction <= 1.0):
            raise ValueError("min_fraction must be in (0, 1]")
        if not (self.min_fraction <= self.max_fraction <= 1.0):
            raise ValueError("max_fraction must be in [min_fraction, 1]")
        if not (0.0 <= self.noise_prob < 1.0):
            raise ValueError("noise_prob must be in [0, 1)")
        if self.weight_decay <= 0.0:
            raise ValueError("weight_decay must be positive")
        if self.min_tokens < 1:
            raise ValueError("min_tokens must be >= 1")


@dataclass
class CurriculumLevelBatch:
    """Represents a single curriculum level for a batch."""

    level: int
    difficulty: float
    weight: float
    input_ids: torch.Tensor
    labels: torch.Tensor
    start_index: int
    length: int
    metadata: Dict[str, float]


class LadderCurriculumGenerator:
    """
    Generates sub-problem batches by progressively truncating and smoothing
    token sequences to emulate recursive problem decomposition.
    """

    def __init__(
        self,
        seq_len: int,
        config: Optional[LadderCurriculumConfig] = None,
        pad_token_id: Optional[int] = None,
    ) -> None:
        self.seq_len = seq_len
        self.config = config or LadderCurriculumConfig()
        self.pad_token_id = pad_token_id
        if self.config.seed is not None:
            random.seed(self.config.seed)

    def fractions(self) -> List[float]:
        """Return monotonically increasing difficulty fractions."""
        levels = self.config.levels
        min_frac = self.config.min_fraction
        max_frac = self.config.max_fraction
        if levels == 1:
            return [max_frac]
        step = (max_frac - min_frac) / max(levels - 1, 1)
        return [min(1.0, min_frac + i * step) for i in range(levels)]

    def _choose_start(self, length: int) -> int:
        if not self.config.random_truncate:
            return 0
        max_start = max(0, self.seq_len - length)
        if max_start == 0:
            return 0
        return random.randint(0, max_start)

    def _apply_noise(self, tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.config.noise_prob <= 0.0 and mask is None:
            return tensor
        if mask is None:
            mask = torch.rand_like(tensor, dtype=torch.float32) < self.config.noise_prob
        shifted = torch.roll(tensor, shifts=1, dims=1)
        if self.pad_token_id is not None:
            shifted[:, 0] = self.pad_token_id
        else:
            shifted[:, 0] = tensor[:, 0]
        return torch.where(mask, shifted, tensor)

    def _weight_for_level(self, level: int, total_levels: int) -> float:
        schedule = self.config.curriculum_mix
        if schedule == "uniform":
            return 1.0 / max(total_levels, 1)
        decay = self.config.weight_decay
        exponents = total_levels - level - 1
        return decay ** exponents

    def generate(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> List[CurriculumLevelBatch]:
        """
        Produce curriculum levels ordered from easiest to hardest.

        The final level corresponds to the original task when
        `include_original=True`.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids shape")

        device = input_ids.device
        batches: List[CurriculumLevelBatch] = []
        fractions = self.fractions()

        for level, fraction in enumerate(fractions):
            length = max(self.config.min_tokens, int(self.seq_len * fraction))
            length = min(length, input_ids.size(1))
            start_idx = self._choose_start(length)
            end_idx = start_idx + length
            inputs_level = input_ids[:, start_idx:end_idx].clone()
            labels_level = labels[:, start_idx:end_idx].clone()

            noise_mask = None
            if self.config.noise_prob > 0.0:
                noise_mask = torch.rand_like(inputs_level, dtype=torch.float32) < self.config.noise_prob
                inputs_level = self._apply_noise(inputs_level, mask=noise_mask)
                labels_level = self._apply_noise(labels_level, mask=noise_mask)

            weight = self._weight_for_level(level, len(fractions))
            batches.append(
                CurriculumLevelBatch(
                    level=level,
                    difficulty=float(fraction),
                    weight=float(weight),
                    input_ids=inputs_level.to(device),
                    labels=labels_level.to(device),
                    start_index=start_idx,
                    length=inputs_level.size(1),
                    metadata={
                        "fraction": float(fraction),
                        "start": float(start_idx),
                        "weight": float(weight),
                    },
                )
            )

        if self.config.include_original and (fractions[-1] < 0.999 or not batches):
            batches.append(
                CurriculumLevelBatch(
                    level=len(batches),
                    difficulty=1.0,
                    weight=1.0,
                    input_ids=input_ids,
                    labels=labels,
                    start_index=0,
                    length=input_ids.size(1),
                    metadata={"fraction": 1.0, "start": 0.0, "weight": 1.0},
                )
            )

        return batches

