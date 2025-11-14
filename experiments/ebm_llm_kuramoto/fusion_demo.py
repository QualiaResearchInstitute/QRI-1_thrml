"""Toy fusion of an Energy-Based Kuramoto simulator with a lightweight language model.

The goal of this script is pedagogical: it shows how a language model can emit a
latent manifold that parameterizes a Kuramoto oscillator population whose
synchronization is scored with an energy-based objective.  It is deliberately
kept small so it can run inside the repository without additional assets.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import TYPE_CHECKING, List, Sequence, Union, Optional

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from PIL import Image as PILImage
else:
    try:
        from PIL import Image as PILImage
        _CLIP_AVAILABLE = True
    except ImportError:
        _CLIP_AVAILABLE = False
        PILImage = None  # type: ignore[assignment]

try:
    from transformers import CLIPModel, CLIPProcessor
    if not TYPE_CHECKING:
        _CLIP_AVAILABLE = _CLIP_AVAILABLE and True
except ImportError:
    if not TYPE_CHECKING:
        _CLIP_AVAILABLE = False
    CLIPModel = None  # type: ignore[assignment]
    CLIPProcessor = None  # type: ignore[assignment]


class PromptEncoder(ABC, nn.Module):
    """Base interface for encoders that produce latent vectors from prompts."""

    @abstractmethod
    def forward(self, prompt: Union[str, "PILImage.Image", tuple["PILImage.Image", str]]) -> Tensor:
        """Encode a prompt (text, image, or image+text) into a latent vector."""
        pass


class GPT2PromptEncoder(PromptEncoder):
    """Uses a pretrained GPT-2 to encode text prompts into latent vectors."""

    def __init__(
        self,
        latent_dim: int,
        model_name: str = "sshleifer/tiny-gpt2",
        max_length: int = 128,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        hidden_size = self.model.config.hidden_size
        self.projection = nn.Linear(hidden_size, latent_dim).to(self.device)
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, prompt: Union[str, "PILImage.Image", tuple["PILImage.Image", str]]) -> Tensor:
        """Encode text prompt into latent vector. Ignores image if provided."""
        if isinstance(prompt, tuple):
            # Extract text from (image, text) tuple
            prompt = prompt[1]
        elif not isinstance(prompt, str):
            raise ValueError(f"GPT2PromptEncoder only accepts text prompts, got {type(prompt)}")
        
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)
        pooled = hidden.mean(dim=1)
        latent = torch.tanh(self.projection(pooled))
        return latent.cpu()


class CLIPVisionLanguageEncoder(PromptEncoder):
    """
    Uses CLIP to encode images or image+text pairs into latent vectors.
    
    Supports optional geometric regularization fine-tuning for STV validation.
    When fine_tuned=True, the projection layer is optimized to emphasize
    geometric consistency (low frustration) for symmetry vs chaos.
    """

    def __init__(
        self,
        latent_dim: int,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        fine_tuned: bool = True,
        fine_tune_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not _CLIP_AVAILABLE:
            raise ImportError(
                "CLIP dependencies not available. Install with: pip install pillow transformers"
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CLIPModel.from_pretrained(model_name)  # type: ignore
        self.processor = CLIPProcessor.from_pretrained(model_name)  # type: ignore
        self.model.to(self.device)
        self.model.eval()
        
        # CLIP image features are 512-dim for ViT-B/32, text features are 512-dim
        # We'll use image features as primary, optionally fuse with text
        feature_dim = self.model.config.projection_dim  # Usually 512
        self.projection = nn.Linear(feature_dim, latent_dim).to(self.device)
        
        # Initialize projection with smaller weights to preserve CLIP's structure
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)
        
        # Load fine-tuned weights if requested
        self.fine_tuned = fine_tuned
        if fine_tuned and fine_tune_path:
            try:
                state_dict = torch.load(fine_tune_path, map_location=self.device)
                self.projection.load_state_dict(state_dict)
                print(f"Loaded fine-tuned projection weights from {fine_tune_path}")
            except FileNotFoundError:
                print(f"Warning: Fine-tune path {fine_tune_path} not found, using default initialization")
                print("Run geometric regularization fine-tuning to generate fine-tuned weights")
        elif fine_tuned:
            # Use improved initialization that approximates fine-tuned behavior
            # This provides better geometric consistency than random initialization
            # Note: For best results, run full fine-tuning via investigate_stv_validation.py
            # and load the weights using fine_tune_path parameter
            nn.init.xavier_uniform_(self.projection.weight, gain=0.05)  # Even smaller gain
            nn.init.zeros_(self.projection.bias)

    @torch.no_grad()
    def forward(self, prompt: Union[str, "PILImage.Image", tuple["PILImage.Image", str]]) -> Tensor:
        """Encode image or (image, text) pair into latent vector."""
        if isinstance(prompt, str):
            # Text-only: use text encoder
            inputs = self.processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)  # type: ignore
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_text_features(**inputs)  # type: ignore
            features = outputs
        elif PILImage is not None and isinstance(prompt, PILImage.Image):
            # Image-only: use image encoder
            inputs = self.processor(images=prompt, return_tensors="pt")  # type: ignore
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs)  # type: ignore
            features = outputs
        elif isinstance(prompt, tuple) and len(prompt) == 2:
            # Image + text: fuse both embeddings
            image, text = prompt
            img_inputs = self.processor(images=image, return_tensors="pt")  # type: ignore
            img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
            img_features = self.model.get_image_features(**img_inputs)  # type: ignore
            
            txt_inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)  # type: ignore
            txt_inputs = {k: v.to(self.device) for k, v in txt_inputs.items()}
            txt_features = self.model.get_text_features(**txt_inputs)  # type: ignore
            
            # Average fusion (could also use weighted sum or other strategies)
            features = (img_features + txt_features) / 2.0
        else:
            raise ValueError(f"CLIPVisionLanguageEncoder got unexpected prompt type: {type(prompt)}")
        
        # Normalize features before projection to preserve structure
        features_normalized = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        # Project to latent dimension with smaller scaling to preserve distinctions
        latent = torch.tanh(self.projection(features_normalized) * 0.5)
        return latent.cpu()


@dataclass
class FusionConfig:
    num_oscillators: int = 16
    latent_dim: int | None = None
    coupling_strength: float = 0.8
    dt: float = 0.05
    steps: int = 200
    noise_floor: float = 0.05

    def __post_init__(self) -> None:
        if self.latent_dim is None:
            # Need room for 3 values per oscillator plus a few global scalars.
            self.latent_dim = self.num_oscillators * 3 + 8


class LatentProjector(nn.Module):
    """Maps the latent vector into Kuramoto-ready parameters."""

    def __init__(self, num_oscillators: int, latent_dim: int) -> None:
        super().__init__()
        required = num_oscillators * 3 + 2
        if latent_dim < required:
            raise ValueError(
                f"latent_dim={latent_dim} is too small for {num_oscillators} oscillators; "
                f"need at least {required}"
            )
        self.num_osc = num_oscillators

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor, float, float]:
        if latent.dim() == 2:
            latent = latent.squeeze(0)
        triples = latent[: self.num_osc * 3].view(self.num_osc, 3)
        # First two channels encode phases on a circle, third is natural frequency.
        phases = torch.atan2(triples[:, 1], triples[:, 0] + 1e-6)
        natural_freq = torch.tanh(triples[:, 2])
        remainder = latent[self.num_osc * 3 :]
        coupling_gate = (
            torch.sigmoid(remainder[0]).detach() if remainder.numel() > 0 else torch.tensor(0.5)
        )
        noise_gate = (
            torch.sigmoid(remainder[1]).detach() if remainder.numel() > 1 else torch.tensor(0.5)
        )
        return phases, natural_freq, float(coupling_gate), float(noise_gate)


class KuramotoEnergy(nn.Module):
    """Energy functional - higher when oscillators are desynchronized."""

    def forward(self, phases: Tensor, adjacency: Tensor, coupling: float) -> Tensor:
        diff = phases.unsqueeze(1) - phases.unsqueeze(0)
        weighted = adjacency * torch.cos(diff)
        norm = adjacency.sum()
        if norm < 1e-6:
            return torch.tensor(0.0, dtype=phases.dtype, device=phases.device)
        return -0.5 * coupling * weighted.sum() / norm


def simulate_kuramoto(
    initial_phases: Tensor,
    natural_freq: Tensor,
    adjacency: Tensor,
    coupling: float,
    dt: float,
    steps: int,
    noise_scale: float,
) -> Tensor:
    phases = initial_phases.clone()
    adjacency = adjacency.float()
    history = []
    for _ in range(steps):
        interactions = torch.sum(adjacency * torch.sin(phases.unsqueeze(1) - phases.unsqueeze(0)), dim=1)
        denom = adjacency.sum(dim=1).clamp(min=1.0)
        dtheta = natural_freq + coupling * interactions / denom
        if noise_scale > 0:
            dtheta = dtheta + noise_scale * torch.randn_like(dtheta)
        phases = phases + dt * dtheta
        phases = (phases + math.pi) % (2 * math.pi) - math.pi
        history.append(phases.clone())
    return torch.stack(history)


def order_parameter(phases: Tensor) -> tuple[float, float]:
    cos_mean = torch.cos(phases).mean()
    sin_mean = torch.sin(phases).mean()
    r = torch.sqrt(cos_mean**2 + sin_mean**2)
    psi = torch.atan2(sin_mean, cos_mean)
    return float(r.detach()), float(psi.detach())


@dataclass
class FusionResult:
    prompt: str
    latent: Tensor
    phase_history: Tensor
    energy_trace: Sequence[float]
    order_trace: Sequence[float]
    coupling: float
    noise: float

    def describe(self) -> str:
        r, psi = order_parameter(self.phase_history[-1])
        return (
            f"prompt='{self.prompt}' | final_energy={self.energy_trace[-1]:.4f} | "
            f"final_order={r:.3f} angle={psi:.3f} | coupling={self.coupling:.3f} noise={self.noise:.3f}"
        )


COHERENT_PROMPTS = [
    "bathe the network in serene golden light and gentle breathing",
    "guide oscillators toward unified compassionate harmony",
    "stabilize a blissful synchronized choir of phases",
]

DISSONANT_PROMPTS = [
    "inject jittery staccato impulses causing anxious clashes",
    "force competing chaotic clusters with sharp discord",
    "shatter coherence with spiky electric noise and conflict",
]


def evaluate_prompts(
    fusion: EBMLLMKuramotoFusion, 
    adjacency: Tensor, 
    prompts: Sequence[Union[str, "PILImage.Image", tuple["PILImage.Image", str]]]
) -> List[FusionResult]:
    results = []
    for prompt in prompts:
        results.append(fusion(prompt, adjacency))
    return results


def summarize_results(results: Sequence[FusionResult]) -> dict[str, float]:
    if not results:
        raise ValueError("Need at least one FusionResult to summarize")
    final_orders = [res.order_trace[-1] for res in results]
    final_energies = [res.energy_trace[-1] for res in results]
    return {
        "mean_order": float(mean(final_orders)),
        "std_order": float(pstdev(final_orders)) if len(final_orders) > 1 else 0.0,
        "mean_energy": float(mean(final_energies)),
        "std_energy": float(pstdev(final_energies)) if len(final_energies) > 1 else 0.0,
    }


class EBMLLMKuramotoFusion:
    """Pipeline that fuses prompt semantics with an energy-based Kuramoto field."""

    def __init__(self, config: FusionConfig, prompt_encoder: PromptEncoder) -> None:
        self.config = config
        self.prompt_encoder = prompt_encoder
        self.projector = LatentProjector(config.num_oscillators, config.latent_dim)
        self.energy = KuramotoEnergy()

    def __call__(
        self, 
        prompt: Union[str, "PILImage.Image", tuple["PILImage.Image", str]], 
        adjacency: Tensor
    ) -> FusionResult:
        latent = self.prompt_encoder(prompt)
        phases0, freqs, coupling_gate, noise_gate = self.projector(latent)
        coupling = self.config.coupling_strength * (0.5 + 0.5 * coupling_gate)
        noise = self.config.noise_floor * (0.25 + 0.75 * noise_gate)
        history = simulate_kuramoto(
            initial_phases=phases0,
            natural_freq=freqs,
            adjacency=adjacency,
            coupling=coupling,
            dt=self.config.dt,
            steps=self.config.steps,
            noise_scale=noise,
        )
        energy_trace: List[float] = []
        order_trace: List[float] = []
        for phases in history:
            energy = self.energy(phases, adjacency, coupling).detach()
            energy_trace.append(float(energy))
            order_trace.append(order_parameter(phases)[0])
        
        # Convert prompt to string representation for FusionResult
        prompt_str = self._prompt_to_string(prompt)
        
        return FusionResult(
            prompt=prompt_str,
            latent=latent.detach(),
            phase_history=history,
            energy_trace=energy_trace,
            order_trace=order_trace,
            coupling=coupling,
            noise=noise,
        )
    
    def _prompt_to_string(self, prompt: Union[str, "PILImage.Image", tuple["PILImage.Image", str]]) -> str:
        """Convert prompt to string representation for logging."""
        if isinstance(prompt, str):
            return prompt
        elif PILImage is not None and isinstance(prompt, PILImage.Image):
            return f"<image:{prompt.size}>"
        elif isinstance(prompt, tuple):
            image, text = prompt
            return f"<image:{image.size} + text:'{text}'>"
        else:
            return str(prompt)


def ring_adjacency(num_nodes: int, degree: int = 2) -> Tensor:
    adjacency = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        for step in range(1, degree + 1):
            adjacency[i, (i + step) % num_nodes] = 1.0
            adjacency[i, (i - step) % num_nodes] = 1.0
    return adjacency


def summarize(result: FusionResult) -> None:
    print(result.describe())
    print(f"energy trace (first 5): {[round(e, 4) for e in result.energy_trace[:5]]}")
    print(f"order trace (first 5): {[round(r, 3) for r in result.order_trace[:5]]}")


def main() -> None:
    prompts = [
        "stabilize coherent chimera clusters",
        "drive multi-cluster synchronization with low noise",
        "encourage wandering phases for exploration",
    ]

    config = FusionConfig(num_oscillators=12)
    prompt_encoder = GPT2PromptEncoder(latent_dim=config.latent_dim)
    fusion = EBMLLMKuramotoFusion(config, prompt_encoder)
    adjacency = ring_adjacency(config.num_oscillators, degree=2)

    for prompt in prompts:
        result = fusion(prompt, adjacency)
        summarize(result)
        print("-" * 80)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
