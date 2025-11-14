"""Diagnose the projection layer and its effect on latent vectors."""

import math
import random

import torch
import numpy as np
from PIL import Image, ImageDraw

from experiments.ebm_llm_kuramoto.fusion_demo import (
    CLIPVisionLanguageEncoder,
    FusionConfig,
    EBMLLMKuramotoFusion,
    ring_adjacency,
    order_parameter,
    KuramotoEnergy,
)

from experiments.ebm_llm_kuramoto.diagnose_clip_embeddings import (
    create_symmetry_image,
    create_chaos_image,
)


def analyze_projection_effects():
    """Analyze how the projection layer affects the latent space."""
    device = "cpu"
    latent_dim = 26  # For 6 oscillators
    
    encoder = CLIPVisionLanguageEncoder(latent_dim=latent_dim, device=device)
    
    # Create diverse test images
    symmetry_images = []
    chaos_images = []
    
    # Create symmetry images with slight variations
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        img = create_symmetry_image()
        symmetry_images.append(img)
    
    # Create chaos images with different seeds
    for seed in [42, 43, 44]:
        random.seed(seed)
        img = create_chaos_image()
        chaos_images.append(img)
    
    # Get CLIP embeddings (before projection)
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    
    sym_clip_embeddings = []
    chaos_clip_embeddings = []
    
    with torch.no_grad():
        for img in symmetry_images:
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            sym_clip_embeddings.append(features)
        
        for img in chaos_images:
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            chaos_clip_embeddings.append(features)
    
    # Get projected latent vectors
    sym_latents = []
    chaos_latents = []
    
    with torch.no_grad():
        for img in symmetry_images:
            latent = encoder(img)
            sym_latents.append(latent)
        
        for img in chaos_images:
            latent = encoder(img)
            chaos_latents.append(latent)
    
    print("=" * 80)
    print("Projection Layer Analysis")
    print("=" * 80)
    
    # Analyze CLIP embeddings
    print("\nCLIP Embeddings (before projection):")
    sym_clip_mean = torch.stack(sym_clip_embeddings).mean(dim=0)
    chaos_clip_mean = torch.stack(chaos_clip_embeddings).mean(dim=0)
    
    clip_similarity = torch.mm(sym_clip_mean, chaos_clip_mean.t()).item()
    print(f"  Mean symmetry-chaos cosine similarity: {clip_similarity:.4f}")
    
    # Intra-class similarities
    sym_sym_sims = []
    for i in range(3):
        for j in range(i+1, 3):
            sim = torch.mm(sym_clip_embeddings[i], sym_clip_embeddings[j].t()).item()
            sym_sym_sims.append(sim)
    
    chaos_chaos_sims = []
    for i in range(3):
        for j in range(i+1, 3):
            sim = torch.mm(chaos_clip_embeddings[i], chaos_clip_embeddings[j].t()).item()
            chaos_chaos_sims.append(sim)
    
    print(f"  Symmetry intra-class similarity: {np.mean(sym_sym_sims):.4f} ± {np.std(sym_sym_sims):.4f}")
    print(f"  Chaos intra-class similarity: {np.mean(chaos_chaos_sims):.4f} ± {np.std(chaos_chaos_sims):.4f}")
    
    # Analyze projected latents
    print("\nProjected Latents (after projection):")
    sym_latent_mean = torch.stack(sym_latents).mean(dim=0)
    chaos_latent_mean = torch.stack(chaos_latents).mean(dim=0)
    
    latent_similarity = torch.mm(sym_latent_mean, chaos_latent_mean.t()).item()
    print(f"  Mean symmetry-chaos cosine similarity: {latent_similarity:.4f}")
    
    # Check if projection preserves or destroys structure
    print(f"\n  Symmetry latent stats: mean={sym_latent_mean.mean().item():.4f}, std={sym_latent_mean.std().item():.4f}")
    print(f"  Chaos latent stats: mean={chaos_latent_mean.mean().item():.4f}, std={chaos_latent_mean.std().item():.4f}")
    
    # Now test actual Kuramoto dynamics
    print("\n" + "=" * 80)
    print("Kuramoto Dynamics Analysis")
    print("=" * 80)
    
    config = FusionConfig(num_oscillators=6, steps=80, dt=0.02, coupling_strength=0.4)
    fusion = EBMLLMKuramotoFusion(config, encoder)
    adjacency = ring_adjacency(config.num_oscillators, degree=1)
    
    symmetry_results = []
    chaos_results = []
    
    for img in symmetry_images:
        result = fusion(img, adjacency)
        symmetry_results.append(result)
    
    for img in chaos_images:
        result = fusion(img, adjacency)
        chaos_results.append(result)
    
    # Compute statistics
    sym_orders = [r.order_trace[-1] for r in symmetry_results]
    chaos_orders = [r.order_trace[-1] for r in chaos_results]
    
    sym_energies = [r.energy_trace[-1] for r in symmetry_results]
    chaos_energies = [r.energy_trace[-1] for r in chaos_results]
    
    print(f"\nSymmetry images:")
    print(f"  Mean order: {np.mean(sym_orders):.4f} ± {np.std(sym_orders):.4f}")
    print(f"  Mean energy: {np.mean(sym_energies):.4f} ± {np.std(sym_energies):.4f}")
    print(f"  Order range: [{min(sym_orders):.4f}, {max(sym_orders):.4f}]")
    
    print(f"\nChaos images:")
    print(f"  Mean order: {np.mean(chaos_orders):.4f} ± {np.std(chaos_orders):.4f}")
    print(f"  Mean energy: {np.mean(chaos_energies):.4f} ± {np.std(chaos_energies):.4f}")
    print(f"  Order range: [{min(chaos_orders):.4f}, {max(chaos_orders):.4f}]")
    
    print(f"\nDifference (symmetry - chaos):")
    print(f"  Order difference: {np.mean(sym_orders) - np.mean(chaos_orders):.4f}")
    print(f"  Energy difference: {np.mean(sym_energies) - np.mean(chaos_energies):.4f}")
    
    # Check if the issue is in the projection or the dynamics
    print("\n" + "=" * 80)
    print("Diagnosis")
    print("=" * 80)
    
    if np.mean(sym_orders) > np.mean(chaos_orders):
        print("✓ Order: Symmetry > Chaos (as expected)")
    else:
        print("✗ Order: Symmetry < Chaos (unexpected!)")
    
    if np.mean(sym_energies) < np.mean(chaos_energies):
        print("✓ Energy: Symmetry < Chaos (as expected)")
    else:
        print("✗ Energy: Symmetry > Chaos (unexpected!)")
    
    print("\nPossible issues:")
    print("1. CLIP embeddings are too similar (0.73 similarity)")
    print("2. Random projection may be scrambling geometric information")
    print("3. Synthetic images may not capture true symmetry/chaos distinction")
    print("4. Projection layer needs better initialization or training")


if __name__ == "__main__":
    analyze_projection_effects()

