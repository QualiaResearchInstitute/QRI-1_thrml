from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    # Prefer HuggingFace CLIP if available
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    _HF_CLIP_AVAILABLE = True
except Exception:
    _HF_CLIP_AVAILABLE = False
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import librosa  # type: ignore
except Exception:
    librosa = None  # type: ignore


@dataclass
class CalibrationConfig:
    decay: float = 0.9
    z_clip: float = 3.0
    min_std: float = 1e-6
    alpha: float = 1.0   # base scale for mapping to logit space
    tau: float = 1.0     # smoothness in tanh mapping


class EWMAStandardizer:
    """
    Online EWMA mean/std for per-critic calibration.
    """
    def __init__(self, cfg: CalibrationConfig) -> None:
        self.cfg = cfg
        self.mean: Optional[float] = None
        self.var: Optional[float] = None
        self.initialized = False

    def update_and_standardize(self, values: List[float]) -> List[float]:
        if len(values) == 0:
            return values
        x = torch.tensor(values, dtype=torch.float32)
        m = float(x.mean().item())
        v = float(x.var(unbiased=False).item())
        if not self.initialized:
            self.mean = m
            self.var = max(v, self.cfg.min_std ** 2)
            self.initialized = True
        else:
            assert self.mean is not None and self.var is not None
            self.mean = self.cfg.decay * self.mean + (1.0 - self.cfg.decay) * m
            # For variance, blend on std^2 in the same manner
            self.var = self.cfg.decay * self.var + (1.0 - self.cfg.decay) * max(v, self.cfg.min_std ** 2)
        assert self.mean is not None and self.var is not None
        std = math.sqrt(max(self.var, self.cfg.min_std ** 2))
        z = (x - self.mean) / std
        z = torch.clamp(z, -self.cfg.z_clip, self.cfg.z_clip)
        # Map to logit-scale shift unit via tanh
        mapped = self.cfg.alpha * torch.tanh(z / max(self.cfg.tau, 1e-6))
        return mapped.tolist()


class Critic:
    """
    Base interface for multimodal critics.
    - prime(context): preload embeddings/targets given current external context
    - score_prefix(prefix): optional baseline energy for current prefix
    - delta_for_candidates(prefix, token_strs): per-candidate energy delta (positive = worse)
    - calibrate(scores): map scores to standardized comparable units (for logit shifts)
    """
    name: str = "base"

    def __init__(self, calibration: Optional[CalibrationConfig] = None) -> None:
        self.calibration = calibration or CalibrationConfig()
        self._standardizer = EWMAStandardizer(self.calibration)
        self.enabled: bool = True
        self._prime_ok: bool = False
        self._prefix_baseline: Optional[float] = None

    def prime(self, context: Dict[str, Any]) -> None:
        self._prime_ok = True
        self._prefix_baseline = None

    def score_prefix(self, text_prefix: str) -> Optional[float]:
        return None

    def delta_for_candidates(self, text_prefix: str, token_strs: List[str]) -> List[float]:
        raise NotImplementedError

    def calibrate(self, scores: List[float]) -> List[float]:
        if not self.enabled:
            return [0.0 for _ in scores]
        return self._standardizer.update_and_standardize(scores)


class ClipImageGroundingCritic(Critic):
    """
    Text-image grounding energy using CLIP embeddings.
    Energy defined as 1 - cosine_similarity(text, image).
    ΔE is difference relative to the current prefix energy.
    """
    name = "clip"

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "openai/clip-vit-base-patch32",
        calibration: Optional[CalibrationConfig] = None,
    ) -> None:
        super().__init__(calibration=calibration)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self._clip_model = None
        self._clip_processor = None
        self._img_embed: Optional[torch.Tensor] = None
        self._text_prefix_energy: Optional[float] = None
        self._load_clip()

    def _load_clip(self) -> None:
        if not _HF_CLIP_AVAILABLE:
            self.enabled = False
            return
        try:
            model = CLIPModel.from_pretrained(self.model_name)  # type: ignore
            processor = CLIPProcessor.from_pretrained(self.model_name)  # type: ignore
            self._clip_model = model.to(self.device)
            self._clip_model.eval()
            self._clip_processor = processor
            self.enabled = True
        except Exception:
            self.enabled = False

    @torch.no_grad()
    def _embed_image(self, image: Any) -> Optional[torch.Tensor]:
        if not self.enabled or self._clip_model is None or self._clip_processor is None:
            return None
        try:
            inputs = self._clip_processor(images=image, return_tensors="pt").to(self.device)  # type: ignore
            outputs = self._clip_model.get_image_features(**inputs)  # type: ignore
            emb = F.normalize(outputs, dim=-1)
            return emb
        except Exception:
            return None

    @torch.no_grad()
    def _embed_texts(self, texts: List[str]) -> Optional[torch.Tensor]:
        if not self.enabled or self._clip_model is None or self._clip_processor is None:
            return None
        try:
            inputs = self._clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)  # type: ignore
            outputs = self._clip_model.get_text_features(**inputs)  # type: ignore
            emb = F.normalize(outputs, dim=-1)
            return emb
        except Exception:
            return None

    def prime(self, context: Dict[str, Any]) -> None:
        super().prime(context)
        self._img_embed = None
        self._text_prefix_energy = None
        image = context.get("image", None)
        if image is not None:
            self._img_embed = self._embed_image(image)
            if self._img_embed is None:
                self.enabled = False

        # Optional: prepare prefix energy if provided
        prefix_text = context.get("prefix_text", None)
        if prefix_text and self._img_embed is not None:
            txt_emb = self._embed_texts([prefix_text])
            if txt_emb is not None:
                sim = torch.sum(txt_emb * self._img_embed, dim=-1)  # cosine
                self._text_prefix_energy = float((1.0 - sim).item())

    def score_prefix(self, text_prefix: str) -> Optional[float]:
        if self._img_embed is None:
            return None
        if self._text_prefix_energy is not None:
            return self._text_prefix_energy
        txt_emb = self._embed_texts([text_prefix])
        if txt_emb is None:
            return None
        sim = torch.sum(txt_emb * self._img_embed, dim=-1)
        e = float((1.0 - sim).item())
        self._text_prefix_energy = e
        return e

    def delta_for_candidates(self, text_prefix: str, token_strs: List[str]) -> List[float]:
        if not self.enabled or self._img_embed is None:
            return [0.0 for _ in token_strs]
        if len(token_strs) == 0:
            return []
        # Build candidate strings
        cands = [text_prefix + s for s in token_strs]
        txt_emb = self._embed_texts(cands)
        if txt_emb is None:
            return [0.0 for _ in token_strs]
        # Compute energies
        sim = torch.matmul(txt_emb, self._img_embed.t()).squeeze(-1)  # [K]
        e_cand = (1.0 - sim).tolist()
        # Baseline
        base = self.score_prefix(text_prefix)
        if base is None:
            base = 0.0
        deltas = [float(e - base) for e in e_cand]
        return deltas


class ProsodyValenceCritic(Critic):
    """
    Simple prosody/affect energy: align text valence to audio-derived target.
    - prime(context): expects {'audio': np.ndarray or torch.Tensor, 'sr': int}
    - Estimate a target valence in [-1, 1] using spectral centroid and RMS heuristic.
    - Text valence: small lexicon-based score on prefix+token.
    Energy ΔE = (val_text - val_target)^2 - (val_prefix - val_target)^2
    """
    name = "prosody"

    def __init__(self, calibration: Optional[CalibrationConfig] = None) -> None:
        super().__init__(calibration=calibration)
        self._target_valence: Optional[float] = None
        # Very small sentiment lexicon
        self._pos_words = set([
            "good", "great", "nice", "love", "happy", "joy", "calm", "peace",
            "bright", "clear", "beautiful", "gentle", "soft", "smile",
        ])
        self._neg_words = set([
            "bad", "sad", "angry", "hate", "fear", "pain", "harsh", "dark",
            "rough", "loud", "hard", "cry",
        ])

    def _estimate_valence_from_audio(self, audio: Any, sr: int) -> Optional[float]:
        try:
            if isinstance(audio, torch.Tensor):
                a = audio.detach().cpu().float().numpy()
            else:
                a = np.asarray(audio, dtype=np.float32) if np is not None else None  # type: ignore
            if a is None:
                return None
            if a.ndim > 1:
                a = a.mean(axis=0)
            if librosa is None:
                # Fallback: simple RMS mapped to arousal, valence neutral
                rms = float(np.sqrt(np.mean(a * a)) + 1e-8) if np is not None else 0.0  # type: ignore
                val = 0.0 + 0.0 * rms
                return float(np.clip(val, -1.0, 1.0)) if np is not None else float(val)
            # With librosa: centroid ~ brightness; map to [-1,1]
            centroid = librosa.feature.spectral_centroid(y=a, sr=sr)
            c = float(np.mean(centroid)) if np is not None else float(centroid.mean())  # type: ignore
            # Normalize centroid by Nyquist
            ny = 0.5 * sr
            norm_c = max(0.0, min(1.0, c / max(ny, 1e-6)))
            # Map bright -> positive valence lightly
            valence = 2.0 * (norm_c - 0.5)
            return float(max(-1.0, min(1.0, valence)))
        except Exception:
            return None

    def _text_valence(self, text: str) -> float:
        # Very small heuristic: average token polarity in [-1,1]
        toks = re.findall(r"[A-Za-z']+", text.lower())
        if not toks:
            return 0.0
        score = 0.0
        count = 0
        for t in toks:
            if t in self._pos_words:
                score += 1.0
                count += 1
            elif t in self._neg_words:
                score -= 1.0
                count += 1
        if count == 0:
            return 0.0
        return max(-1.0, min(1.0, score / count))

    def prime(self, context: Dict[str, Any]) -> None:
        super().prime(context)
        self._target_valence = None
        audio = context.get("audio", None)
        sr = context.get("sr", None)
        if audio is not None and sr is not None:
            self._target_valence = self._estimate_valence_from_audio(audio, int(sr))

    def delta_for_candidates(self, text_prefix: str, token_strs: List[str]) -> List[float]:
        if len(token_strs) == 0:
            return []
        target = 0.0 if self._target_valence is None else float(self._target_valence)
        base = self._text_valence(text_prefix)
        base_e = (base - target) * (base - target)
        deltas: List[float] = []
        for s in token_strs:
            v = self._text_valence(text_prefix + s)
            e = (v - target) * (v - target)
            deltas.append(float(e - base_e))
        return deltas


# Utility
def tokens_to_strings(tokenizer, token_ids: List[int]) -> List[str]:
    """
    Convert token ids to token strings without special token removal.
    """
    out: List[str] = []
    for tid in token_ids:
        try:
            s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            s = str(tid)
        out.append(s)
    return out


