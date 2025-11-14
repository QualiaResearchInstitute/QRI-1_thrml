"""Investigate STV validation failures using threshold analysis and geometric regularization.

This script implements the theoretical framework for validating STV predictions:
1. Threshold investigation (symmetry detection threshold, precision weighting)
2. Geometric regularization analysis (frustration loss, feature synchronization)
"""

import math
import random
from pathlib import Path
from typing import List, Tuple, Union, Optional

import torch
import numpy as np
from PIL import Image, ImageDraw
from torch import nn, optim
from torch.nn import functional as F

try:
    from transformers import CLIPModel, CLIPProcessor
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    print("CLIP not available")
    exit(1)

from experiments.ebm_llm_kuramoto.diagnose_clip_embeddings import (
    create_symmetry_image,
    create_chaos_image,
)


def frustration_loss(embeddings: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute frustration loss measuring geometric inconsistency.
    
    The frustration functional (η_F) measures how much features violate geometric
    constraints. For symmetry, we want low frustration (high geometric consistency).
    
    Args:
        embeddings: [batch, dim] feature embeddings
        adjacency: Optional [batch, batch] adjacency matrix for graph-based frustration
    
    Returns:
        Scalar frustration loss
    """
    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, dim=-1)
    
    if adjacency is not None:
        # Graph-based frustration: penalize inconsistency between connected nodes
        # For symmetry, connected nodes should have similar embeddings
        diff = embeddings_norm.unsqueeze(1) - embeddings_norm.unsqueeze(0)  # [batch, batch, dim]
        distances = torch.norm(diff, dim=-1)  # [batch, batch]
        frustration = (adjacency * distances).sum() / (adjacency.sum() + 1e-8)
    else:
        # Global frustration: penalize high variance (chaos) vs low variance (symmetry)
        # Symmetry should have low variance (coherent structure)
        # Chaos should have high variance (fragmented structure)
        variance = torch.var(embeddings_norm, dim=0).mean()
        frustration = variance
    
    return frustration


def geometric_consistency_loss(
    symmetry_embeddings: torch.Tensor,
    chaos_embeddings: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Geometric consistency loss that encourages symmetry-chaos separation.
    
    This implements the "near miss parameter" (δ) concept: we want symmetry
    embeddings to be more coherent (lower frustration) than chaos embeddings.
    
    Args:
        symmetry_embeddings: [batch_sym, dim] embeddings for symmetry images
        chaos_embeddings: [batch_chaos, dim] embeddings for chaos images
        margin: Minimum margin between symmetry and chaos frustration
    
    Returns:
        Loss encouraging symmetry < chaos (with margin)
    """
    sym_frustration = frustration_loss(symmetry_embeddings)
    chaos_frustration = frustration_loss(chaos_embeddings)
    
    # We want: sym_frustration < chaos_frustration - margin
    # Loss = max(0, sym_frustration - chaos_frustration + margin)
    loss = F.relu(sym_frustration - chaos_frustration + margin)
    
    return loss


class GeometricRegularizedCLIPEncoder(nn.Module):
    """
    CLIP encoder with geometric regularization for symmetry detection.
    
    This fine-tunes the projection layer to emphasize geometric structure
    using frustration loss and feature synchronization.
    
    Note: This class implements the same interface as PromptEncoder but doesn't
    inherit from it to avoid circular imports. It can be used as a drop-in replacement.
    """
    
    def __init__(
        self,
        latent_dim: int,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        geometric_weight: float = 0.1,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Freeze CLIP base model
        
        feature_dim = self.model.config.projection_dim
        self.projection = nn.Linear(feature_dim, latent_dim).to(self.device)
        # Initialize with smaller weights to preserve CLIP structure
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)
        
        self.geometric_weight = geometric_weight
    
    def forward(self, prompt) -> torch.Tensor:
        """Encode prompt to latent vector."""
        if isinstance(prompt, str):
            inputs = self.processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_text_features(**inputs)
        elif isinstance(prompt, Image.Image):
            inputs = self.processor(images=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_image_features(**inputs)
        elif isinstance(prompt, tuple) and len(prompt) == 2:
            image, text = prompt
            img_inputs = self.processor(images=image, return_tensors="pt")
            img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
            img_features = self.model.get_image_features(**img_inputs)
            
            txt_inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            txt_inputs = {k: v.to(self.device) for k, v in txt_inputs.items()}
            txt_features = self.model.get_text_features(**txt_inputs)
            
            features = (img_features + txt_features) / 2.0
        else:
            raise ValueError(f"Unexpected prompt type: {type(prompt)}")
        
        # Normalize and project
        features_normalized = F.normalize(features, dim=-1)
        latent = torch.tanh(self.projection(features_normalized) * 0.5)
        return latent
    
    def fine_tune_on_symmetry_chaos_pairs(
        self,
        symmetry_images: List[Image.Image],
        chaos_images: List[Image.Image],
        num_epochs: int = 10,
        lr: float = 1e-3,
    ):
        """
        Fine-tune projection layer using geometric regularization.
        
        This implements feature synchronization to reduce intra-class variance
        and increase inter-class separation for symmetry vs chaos.
        """
        optimizer = optim.Adam(self.projection.parameters(), lr=lr)
        
        print("=" * 80)
        print("Fine-tuning CLIP projection with geometric regularization")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Get embeddings for symmetry and chaos images
            sym_embeddings = []
            chaos_embeddings = []
            
            with torch.no_grad():
                for img in symmetry_images:
                    inputs = self.processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    features = self.model.get_image_features(**inputs)
                    sym_embeddings.append(features)
                
                for img in chaos_images:
                    inputs = self.processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    features = self.model.get_image_features(**inputs)
                    chaos_embeddings.append(features)
            
            sym_embeddings = torch.cat(sym_embeddings, dim=0)
            chaos_embeddings = torch.cat(chaos_embeddings, dim=0)
            
            # Project to latent space
            sym_latents = self._project_features(sym_embeddings)
            chaos_latents = self._project_features(chaos_embeddings)
            
            # Compute geometric consistency loss
            geo_loss = geometric_consistency_loss(sym_latents, chaos_latents, margin=0.1)
            
            # Feature synchronization: reduce intra-class variance
            sym_intra_var = torch.var(sym_latents, dim=0).mean()
            chaos_intra_var = torch.var(chaos_latents, dim=0).mean()
            sync_loss = sym_intra_var + chaos_intra_var
            
            # Total loss
            total_loss = geo_loss + self.geometric_weight * sync_loss
            
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Geometric loss: {geo_loss.item():.6f}")
                print(f"  Sync loss: {sync_loss.item():.6f}")
                print(f"  Total loss: {total_loss.item():.6f}")
                print(f"  Sym frustration: {frustration_loss(sym_latents).item():.6f}")
                print(f"  Chaos frustration: {frustration_loss(chaos_latents).item():.6f}")
        
        print("Fine-tuning complete!")
        print("=" * 80)
    
    def _project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project CLIP features to latent space."""
        features_normalized = F.normalize(features, dim=-1)
        latent = torch.tanh(self.projection(features_normalized) * 0.5)
        return latent


def analyze_threshold_sensitivity(
    symmetry_images: List[Image.Image],
    chaos_images: List[Image.Image],
    thresholds: List[float] = [0.01, 0.05, 0.10, 0.15, 0.20],
) -> dict:
    """
    Analyze how different thresholds affect symmetry-chaos separation.
    
    This investigates the "symmetry detection threshold" and "near miss parameter (δ)"
    concepts from predictive processing theory.
    """
    from experiments.ebm_llm_kuramoto.fusion_demo import (
        CLIPVisionLanguageEncoder,
        EBMLLMKuramotoFusion,
        FusionConfig,
        ring_adjacency,
        evaluate_prompts,
        summarize_results,
    )
    
    device = "cpu"
    config = FusionConfig(num_oscillators=6, steps=80, dt=0.02, coupling_strength=0.4)
    adjacency = ring_adjacency(config.num_oscillators, degree=1)
    
    results = {}
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        
        encoder = CLIPVisionLanguageEncoder(latent_dim=config.latent_dim, device=device)
        fusion = EBMLLMKuramotoFusion(config, encoder)
        
        sym_results = evaluate_prompts(fusion, adjacency, symmetry_images)
        chaos_results = evaluate_prompts(fusion, adjacency, chaos_images)
        
        sym_stats = summarize_results(sym_results)
        chaos_stats = summarize_results(chaos_results)
        
        order_diff = sym_stats["mean_order"] - chaos_stats["mean_order"]
        energy_diff = sym_stats["mean_energy"] - chaos_stats["mean_energy"]
        
        # Check if threshold is met
        order_passes = order_diff > threshold
        energy_passes = energy_diff < -threshold  # Negative because lower energy is better
        
        results[threshold] = {
            "order_diff": order_diff,
            "energy_diff": energy_diff,
            "order_passes": order_passes,
            "energy_passes": energy_passes,
            "sym_order": sym_stats["mean_order"],
            "chaos_order": chaos_stats["mean_order"],
            "sym_energy": sym_stats["mean_energy"],
            "chaos_energy": chaos_stats["mean_energy"],
        }
        
        print(f"  Order diff: {order_diff:.4f} (pass: {order_passes})")
        print(f"  Energy diff: {energy_diff:.4f} (pass: {energy_passes})")
    
    return results


def main():
    """Run comprehensive STV validation investigation."""
    print("=" * 80)
    print("STV Validation Investigation")
    print("=" * 80)
    
    # Create diverse test images
    symmetry_images = [create_symmetry_image() for _ in range(5)]
    chaos_images = [create_chaos_image(seed=42 + i) for i in range(5)]
    
    # 1. Threshold sensitivity analysis
    print("\n" + "=" * 80)
    print("1. Threshold Sensitivity Analysis")
    print("=" * 80)
    threshold_results = analyze_threshold_sensitivity(symmetry_images, chaos_images)
    
    # 2. Geometric regularization fine-tuning
    print("\n" + "=" * 80)
    print("2. Geometric Regularization Fine-Tuning")
    print("=" * 80)
    
    encoder = GeometricRegularizedCLIPEncoder(
        latent_dim=26,  # For 6 oscillators
        device="cpu",
        geometric_weight=0.1,
    )
    
    encoder.fine_tune_on_symmetry_chaos_pairs(
        symmetry_images[:3],  # Use subset for training
        chaos_images[:3],
        num_epochs=10,
        lr=1e-3,
    )

    # Persist the fine-tuned projection for downstream services (e.g. oscilleditor)
    projection_path = Path(__file__).resolve().parent / "geometry_clip_projection.pt"
    torch.save(encoder.projection.state_dict(), projection_path)
    print(f"Saved fine-tuned projection weights to {projection_path}")
    
    # 3. Test fine-tuned encoder
    print("\n" + "=" * 80)
    print("3. Testing Fine-Tuned Encoder")
    print("=" * 80)
    
    from experiments.ebm_llm_kuramoto.fusion_demo import (
        EBMLLMKuramotoFusion,
        FusionConfig,
        ring_adjacency,
        evaluate_prompts,
        summarize_results,
    )
    
    # Use fine-tuned encoder directly (it already implements the interface)
    wrapper = encoder
    config = FusionConfig(num_oscillators=6, steps=80, dt=0.02, coupling_strength=0.4)
    fusion = EBMLLMKuramotoFusion(config, wrapper)
    adjacency = ring_adjacency(config.num_oscillators, degree=1)
    
    sym_results = evaluate_prompts(fusion, adjacency, symmetry_images[3:])  # Test on held-out
    chaos_results = evaluate_prompts(fusion, adjacency, chaos_images[3:])
    
    sym_stats = summarize_results(sym_results)
    chaos_stats = summarize_results(chaos_results)
    
    print(f"\nFine-tuned results:")
    print(f"  Symmetry - Order: {sym_stats['mean_order']:.4f}, Energy: {sym_stats['mean_energy']:.4f}")
    print(f"  Chaos - Order: {chaos_stats['mean_order']:.4f}, Energy: {chaos_stats['mean_energy']:.4f}")
    print(f"  Order diff: {sym_stats['mean_order'] - chaos_stats['mean_order']:.4f}")
    print(f"  Energy diff: {sym_stats['mean_energy'] - chaos_stats['mean_energy']:.4f}")
    
    if sym_stats['mean_order'] > chaos_stats['mean_order']:
        print("  ✓ Order: Symmetry > Chaos (PASS)")
    else:
        print("  ✗ Order: Symmetry < Chaos (FAIL)")
    
    if sym_stats['mean_energy'] < chaos_stats['mean_energy']:
        print("  ✓ Energy: Symmetry < Chaos (PASS)")
    else:
        print("  ✗ Energy: Symmetry > Chaos (FAIL)")
    
    print("\n" + "=" * 80)
    print("Investigation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

