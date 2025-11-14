"""
Geometry-aware CLIP encoder wrapper.

This module provides a self-contained interface for loading the tuned CLIP model
that we use across resonance-transformer experiments.  It exposes helper methods
for embedding text, images, or paired image+text prompts while preserving the
geometric structure emphasized during fine-tuning.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import CLIPModel, CLIPProcessor


Prompt = Union[str, Image.Image, Tuple[Image.Image, str]]


@dataclass
class EncoderConfig:
    """Configuration options for the geometry-aware CLIP encoder."""

    latent_dim: int = 256
    model_name: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None
    fine_tune_path: Optional[str] = None
    use_tuned_init: bool = True
    projection_scale: float = 0.5


class GeometryAwareCLIPEncoder:
    """
    Geometry-aware CLIP encoder with optional fine-tuned projection.

    Fine-tuning targets the final projection layer so that symmetry-aware
    structure is preserved in the latent space.  When no tuned weights are
    available we fall back to a conservative initialization that keeps CLIP's
    geometric relationships mostly intact.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.model = CLIPModel.from_pretrained(self.config.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

        feature_dim = self.model.config.projection_dim
        self.projection = nn.Linear(feature_dim, self.config.latent_dim).to(self.device)
        self._initialize_projection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_text_features(**inputs)
        return self._project(features).cpu().numpy().squeeze(0)

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_image_features(**inputs)
        return self._project(features).cpu().numpy().squeeze(0)

    @torch.no_grad()
    def embed_image_text(self, image: Image.Image, text: str) -> np.ndarray:
        img_inputs = self.processor(images=image, return_tensors="pt")
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        img_features = self.model.get_image_features(**img_inputs)

        txt_inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        txt_inputs = {k: v.to(self.device) for k, v in txt_inputs.items()}
        txt_features = self.model.get_text_features(**txt_inputs)

        features = (img_features + txt_features) / 2.0
        return self._project(features).cpu().numpy().squeeze(0)

    def embed_prompt(self, prompt: Prompt) -> np.ndarray:
        if isinstance(prompt, str):
            return self.embed_text(prompt)
        if isinstance(prompt, Image.Image):
            return self.embed_image(prompt)
        if isinstance(prompt, tuple) and len(prompt) == 2:
            image, text = prompt
            if not isinstance(image, Image.Image):
                raise TypeError("Expected PIL.Image for first element of prompt tuple")
            if not isinstance(text, str):
                raise TypeError("Expected str for second element of prompt tuple")
            return self.embed_image_text(image, text)
        raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def decode_base64_image(b64: str) -> Image.Image:
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")

    @staticmethod
    def cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return matrix_norm @ vec_norm

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        features_normalized = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        latent = torch.tanh(self.projection(features_normalized) * self.config.projection_scale)
        return latent

    def _initialize_projection(self) -> None:
        if self.config.fine_tune_path and os.path.exists(self.config.fine_tune_path):
            state_dict = torch.load(self.config.fine_tune_path, map_location=self.device)
            self.projection.load_state_dict(state_dict)
            print(f"[GeometryAwareCLIPEncoder] Loaded fine-tuned projection from {self.config.fine_tune_path}")
            return

        # Fall back to tuned initialization
        gain = 0.05 if self.config.use_tuned_init else 1.0
        nn.init.xavier_uniform_(self.projection.weight, gain=gain)
        nn.init.zeros_(self.projection.bias)
        if self.config.fine_tune_path:
            print(
                f"[GeometryAwareCLIPEncoder] Warning: fine_tune_path '{self.config.fine_tune_path}' not found. "
                "Using tuned Xavier initialization instead."
            )


def batch_embed(encoder: GeometryAwareCLIPEncoder, prompts: Sequence[Prompt]) -> List[np.ndarray]:
    """Convenience helper to embed a list of prompts."""
    return [encoder.embed_prompt(prompt) for prompt in prompts]


