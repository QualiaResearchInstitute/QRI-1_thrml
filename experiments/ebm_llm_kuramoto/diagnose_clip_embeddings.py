"""Diagnostic script to investigate CLIP embeddings for symmetry vs chaos images."""

import math
import random
from typing import Optional

import torch
import numpy as np
from PIL import Image, ImageDraw

try:
    from transformers import CLIPModel, CLIPProcessor
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    print("CLIP not available")
    exit(1)


def create_symmetry_image(size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Create a synthetic image with high symmetry (coherent visual)."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Draw symmetric mandala-like pattern
    for i in range(8):
        angle = i * math.pi / 4
        radius = min(size) // 3
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        # Draw symmetric circles
        for r in [20, 40, 60]:
            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                outline="blue",
                width=2,
            )
    
    # Draw central symmetric pattern
    draw.ellipse(
        [center_x - 30, center_y - 30, center_x + 30, center_y + 30],
        fill="gold",
        outline="blue",
        width=2,
    )
    
    return img


def create_chaos_image(size: tuple[int, int] = (224, 224), seed: Optional[int] = None) -> Image.Image:
    """Create a synthetic image with chaotic structure (dissonant visual)."""
    img = Image.new("RGB", size, color="black")
    draw = ImageDraw.Draw(img)
    
    # Draw random, chaotic lines and shapes
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(42)  # For reproducibility
    for _ in range(50):
        x1 = random.randint(0, size[0])
        y1 = random.randint(0, size[1])
        x2 = random.randint(0, size[0])
        y2 = random.randint(0, size[1])
        color = random.choice(["red", "orange", "yellow"])
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
    
    # Add random rectangles
    for _ in range(20):
        x = random.randint(0, size[0] - 30)
        y = random.randint(0, size[1] - 30)
        w = random.randint(10, 30)
        h = random.randint(10, 30)
        color = random.choice(["red", "purple"])
        draw.rectangle([x, y, x + w, y + h], fill=color)
    
    return img


def analyze_embeddings():
    """Analyze CLIP embeddings for symmetry and chaos images."""
    device = "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    
    # Create test images
    symmetry_img = create_symmetry_image()
    chaos_img = create_chaos_image()
    
    # Get image embeddings
    with torch.no_grad():
        sym_inputs = processor(images=symmetry_img, return_tensors="pt")
        sym_inputs = {k: v.to(device) for k, v in sym_inputs.items()}
        sym_features = model.get_image_features(**sym_inputs)
        sym_features = sym_features / sym_features.norm(dim=-1, keepdim=True)  # Normalize
        
        chaos_inputs = processor(images=chaos_img, return_tensors="pt")
        chaos_inputs = {k: v.to(device) for k, v in chaos_inputs.items()}
        chaos_features = model.get_image_features(**chaos_inputs)
        chaos_features = chaos_features / chaos_features.norm(dim=-1, keepdim=True)  # Normalize
    
    # Get text embeddings for comparison
    text_prompts = [
        "symmetric harmonious pattern",
        "chaotic fragmented pattern",
        "geometric mandala",
        "random noise",
    ]
    
    text_features_list = []
    with torch.no_grad():
        for text in text_prompts:
            txt_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
            txt_features = model.get_text_features(**txt_inputs)
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
            text_features_list.append(txt_features)
    
    # Compute similarities
    print("=" * 80)
    print("CLIP Embedding Analysis")
    print("=" * 80)
    
    print(f"\nSymmetry image embedding norm: {sym_features.norm().item():.4f}")
    print(f"Chaos image embedding norm: {chaos_features.norm().item():.4f}")
    
    # Cosine similarity between symmetry and chaos images
    img_similarity = torch.mm(sym_features, chaos_features.t()).item()
    print(f"\nImage-to-image cosine similarity (symmetry vs chaos): {img_similarity:.4f}")
    
    # Image-to-text similarities
    print("\nImage-to-text cosine similarities:")
    for i, text in enumerate(text_prompts):
        sym_text_sim = torch.mm(sym_features, text_features_list[i].t()).item()
        chaos_text_sim = torch.mm(chaos_features, text_features_list[i].t()).item()
        print(f"  '{text}':")
        print(f"    Symmetry image: {sym_text_sim:.4f}")
        print(f"    Chaos image:    {chaos_text_sim:.4f}")
        print(f"    Difference:     {sym_text_sim - chaos_text_sim:.4f}")
    
    # Analyze embedding statistics
    print("\nEmbedding statistics:")
    print(f"Symmetry image - mean: {sym_features.mean().item():.6f}, std: {sym_features.std().item():.6f}")
    print(f"  min: {sym_features.min().item():.6f}, max: {sym_features.max().item():.6f}")
    print(f"Chaos image - mean: {chaos_features.mean().item():.6f}, std: {chaos_features.std().item():.6f}")
    print(f"  min: {chaos_features.min().item():.6f}, max: {chaos_features.max().item():.6f}")
    
    # Check if embeddings are actually different
    diff = (sym_features - chaos_features).abs()
    print(f"\nAbsolute difference between embeddings:")
    print(f"  mean: {diff.mean().item():.6f}")
    print(f"  max: {diff.max().item():.6f}")
    print(f"  L2 distance: {(sym_features - chaos_features).norm().item():.4f}")
    
    # Test multiple symmetry/chaos images
    print("\n" + "=" * 80)
    print("Testing multiple images")
    print("=" * 80)
    
    symmetry_images = [create_symmetry_image() for _ in range(3)]
    chaos_images = [create_chaos_image() for _ in range(3)]
    
    sym_embeddings = []
    chaos_embeddings = []
    
    with torch.no_grad():
        for img in symmetry_images:
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            sym_embeddings.append(features)
        
        for img in chaos_images:
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            chaos_embeddings.append(features)
    
    # Compute intra-class and inter-class similarities
    sym_sym_sims = []
    chaos_chaos_sims = []
    sym_chaos_sims = []
    
    for i in range(3):
        for j in range(i+1, 3):
            sim = torch.mm(sym_embeddings[i], sym_embeddings[j].t()).item()
            sym_sym_sims.append(sim)
            
            sim = torch.mm(chaos_embeddings[i], chaos_embeddings[j].t()).item()
            chaos_chaos_sims.append(sim)
        
        for j in range(3):
            sim = torch.mm(sym_embeddings[i], chaos_embeddings[j].t()).item()
            sym_chaos_sims.append(sim)
    
    print(f"\nIntra-class similarities (symmetry-symmetry):")
    print(f"  mean: {np.mean(sym_sym_sims):.4f}, std: {np.std(sym_sym_sims):.4f}")
    print(f"  range: [{min(sym_sym_sims):.4f}, {max(sym_sym_sims):.4f}]")
    
    print(f"\nIntra-class similarities (chaos-chaos):")
    print(f"  mean: {np.mean(chaos_chaos_sims):.4f}, std: {np.std(chaos_chaos_sims):.4f}")
    print(f"  range: [{min(chaos_chaos_sims):.4f}, {max(chaos_chaos_sims):.4f}]")
    
    print(f"\nInter-class similarities (symmetry-chaos):")
    print(f"  mean: {np.mean(sym_chaos_sims):.4f}, std: {np.std(sym_chaos_sims):.4f}")
    print(f"  range: [{min(sym_chaos_sims):.4f}, {max(sym_chaos_sims):.4f}]")
    
    print("\n" + "=" * 80)
    print("Analysis complete")
    print("=" * 80)


if __name__ == "__main__":
    analyze_embeddings()

