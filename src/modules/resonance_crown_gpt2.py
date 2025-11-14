"""
Resonance Crown with GPT-2 Models

8 GPT-2 models arranged in a ring around a central crown model.
Each GPT-2 starts with a different color word.
Mexican hat coupling: local excitation, distant inhibition.
Alternating couplings with continuous mixing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple
import sys
from pathlib import Path
import math

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from modules.resonance_gpt2_adapter import patch_gpt2_attention, iter_resonance_heads
except ImportError:
    patch_gpt2_attention = None
    iter_resonance_heads = None


def mexican_hat_coupling(n_models: int, excitation_radius: float = 2.0, inhibition_radius: float = 4.0) -> torch.Tensor:
    """
    Create Mexican hat coupling matrix.
    
    Mexican hat: local excitation, distant inhibition.
    - Nearby models: positive coupling (excitation)
    - Distant models: negative coupling (inhibition)
    
    Args:
        n_models: Number of models in ring
        excitation_radius: Radius for excitation (positive coupling)
        inhibition_radius: Radius for inhibition (negative coupling)
        
    Returns:
        Coupling matrix [n_models, n_models]
    """
    coupling = torch.zeros(n_models, n_models)
    
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                coupling[i, j] = 0.0  # No self-coupling
            else:
                # Circular distance
                dist = min(abs(i - j), n_models - abs(i - j))
                
                # Mexican hat function
                if dist <= excitation_radius:
                    # Excitation (positive)
                    coupling[i, j] = 1.0 / (1.0 + dist**2)
                elif dist <= inhibition_radius:
                    # Inhibition (negative)
                    coupling[i, j] = -0.5 / (1.0 + (dist - excitation_radius)**2)
                else:
                    # Weak coupling
                    coupling[i, j] = 0.1 / (1.0 + dist**2)
    
    return coupling


class ResonanceCrownGPT2(nn.Module):
    """
    Resonance Crown with 8 GPT-2 models around a central crown.
    
    Architecture:
    - 8 GPT-2 models in a ring (each with different color word)
    - 1 central crown model (GPT-2)
    - Mexican hat coupling between ring models
    - Alternating coupling patterns
    - Continuous mixing/generation
    """
    
    COLOR_WORDS = [
        "red", "orange", "yellow", "green",
        "blue", "indigo", "violet", "crimson"
    ]
    
    def __init__(
        self,
        model_name: str = "gpt2",
        n_ring_models: int = 8,
        device: Optional[torch.device] = None,
        use_resonance: bool = True,
        use_full_dragon: bool = True,
        excitation_radius: float = 2.0,
        inhibition_radius: float = 4.0,
    ):
        super().__init__()
        
        self.n_ring_models = n_ring_models
        self.device = device or torch.device("cpu")
        self.use_resonance = use_resonance
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        if use_resonance and patch_gpt2_attention is None:
            raise ImportError("resonance_gpt2_adapter required for resonance mode")
        
        # Load tokenizer
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create ring models (8 GPT-2s with different color words)
        print(f"Creating {n_ring_models} ring models...")
        self.ring_models = nn.ModuleList()
        self.color_words = self.COLOR_WORDS[:n_ring_models]
        
        for i, color_word in enumerate(self.color_words):
            print(f"  Ring model {i+1}/{n_ring_models}: {color_word}")
            model = GPT2LMHeadModel.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            
            if use_resonance:
                # Supercharge with resonance
                try:
                    from scripts.supercharge_model import supercharge_model
                except ImportError:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "supercharge_model",
                        project_root / "scripts" / "supercharge_model.py"
                    )
                    supercharge_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(supercharge_module)
                    supercharge_model = supercharge_module.supercharge_model
                
                model = supercharge_model(
                    model,
                    model_name=model_name,
                    use_full_dragon=use_full_dragon,
                )
            
            self.ring_models.append(model)
        
        # Create central crown model
        print("Creating central crown model...")
        self.crown_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.crown_model.to(self.device)
        self.crown_model.eval()
        
        if use_resonance:
            try:
                from scripts.supercharge_model import supercharge_model
            except ImportError:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "supercharge_model",
                    project_root / "scripts" / "supercharge_model.py"
                )
                supercharge_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(supercharge_module)
                supercharge_model = supercharge_module.supercharge_model
            
            self.crown_model = supercharge_model(
                self.crown_model,
                model_name=model_name,
                use_full_dragon=use_full_dragon,
            )
        
        # Mexican hat coupling matrix
        self.mexican_hat_coupling = mexican_hat_coupling(
            n_ring_models,
            excitation_radius=excitation_radius,
            inhibition_radius=inhibition_radius,
        ).to(self.device)
        
        # Alternating coupling pattern (for time-varying coupling)
        self.coupling_patterns = self._create_alternating_patterns()
        self.current_pattern_idx = 0
    
    def _create_alternating_patterns(self) -> List[torch.Tensor]:
        """Create alternating coupling patterns."""
        patterns = []
        
        # Pattern 1: Standard Mexican hat
        patterns.append(self.mexican_hat_coupling.clone())
        
        # Pattern 2: Rotated (shift excitation/inhibition)
        rotated = torch.roll(self.mexican_hat_coupling, shifts=1, dims=1)
        patterns.append(rotated)
        
        # Pattern 3: Inverted (swap excitation/inhibition)
        inverted = -self.mexican_hat_coupling.clone()
        patterns.append(inverted)
        
        # Pattern 4: Alternating excitation/inhibition
        alternating = torch.zeros_like(self.mexican_hat_coupling)
        for i in range(self.n_ring_models):
            for j in range(self.n_ring_models):
                if i != j:
                    dist = min(abs(i - j), self.n_ring_models - abs(i - j))
                    if dist % 2 == 0:
                        alternating[i, j] = 0.5  # Even distance: excitation
                    else:
                        alternating[i, j] = -0.3  # Odd distance: inhibition
        patterns.append(alternating)
        
        return patterns
    
    def get_current_coupling(self) -> torch.Tensor:
        """Get current coupling pattern (alternating)."""
        pattern = self.coupling_patterns[self.current_pattern_idx]
        self.current_pattern_idx = (self.current_pattern_idx + 1) % len(self.coupling_patterns)
        return pattern
    
    def generate_with_color(
        self,
        model: nn.Module,
        color_word: str,
        max_length: int = 50,
        temperature: float = 0.8,
    ) -> str:
        """
        Generate text starting with a color word.
        
        Args:
            model: GPT-2 model
            color_word: Color word to start with
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        prompt = f"{color_word.capitalize()} "
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    def forward_step(
        self,
        step: int,
        max_length: int = 50,
        temperature: float = 0.8,
    ) -> Dict[str, any]:
        """
        Single forward step: generate from all models and mix.
        
        Args:
            step: Current step number
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Dictionary with outputs and metrics
        """
        # Get current coupling pattern (alternating)
        coupling = self.get_current_coupling()
        
        # Generate from each ring model
        ring_outputs = []
        ring_texts = []
        ring_embeddings = []
        
        for i, (model, color_word) in enumerate(zip(self.ring_models, self.color_words)):
            # Generate text
            text = self.generate_with_color(model, color_word, max_length, temperature)
            ring_texts.append(text)
            
            # Get embeddings instead of token IDs (for mixing)
            tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            
            # Get embeddings from model
            with torch.no_grad():
                if hasattr(model, 'transformer'):
                    embeddings = model.transformer.wte(tokens['input_ids'])
                else:
                    # Fallback: use token IDs as embeddings
                    embeddings = tokens['input_ids'].float()
            
            ring_embeddings.append(embeddings)
            ring_outputs.append(tokens['input_ids'])
        
        # Pad all embeddings to same length
        max_len = max(emb.shape[1] for emb in ring_embeddings)
        padded_embeddings = []
        for emb in ring_embeddings:
            if emb.shape[1] < max_len:
                padding = torch.zeros(emb.shape[0], max_len - emb.shape[1], emb.shape[2], device=emb.device)
                emb = torch.cat([emb, padding], dim=1)
            padded_embeddings.append(emb)
        
        # Apply Mexican hat coupling between ring models
        # Mix embeddings based on coupling strength
        mixed_embeddings = []
        
        for i in range(self.n_ring_models):
            # Weighted sum of embeddings based on coupling
            weighted_sum = torch.zeros_like(padded_embeddings[0])
            
            for j in range(self.n_ring_models):
                weight = coupling[i, j].item()
                if weight != 0:
                    # Mix embeddings (excitation or inhibition)
                    weighted_sum = weighted_sum + weight * padded_embeddings[j]
            
            # Normalize
            mixed_embeddings.append(weighted_sum)
        
        # Generate from crown model (using mixed context)
        # Average of ring embeddings as context
        crown_context_emb = torch.stack(mixed_embeddings).mean(dim=0)
        
        # Convert embeddings back to text for crown input
        # Use the first ring model's text as base, or create a summary
        crown_text = " ".join([text[:50] for text in ring_texts[:3]])  # Use first 3 texts as context
        
        # Generate from crown
        crown_inputs = self.tokenizer(crown_text[:100], return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            crown_outputs = self.crown_model.generate(
                **crown_inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        crown_text_generated = self.tokenizer.decode(crown_outputs[0], skip_special_tokens=True)
        
        # Extract resonance metrics
        metrics = {}
        if self.use_resonance:
            for i, model in enumerate(self.ring_models):
                for head in iter_resonance_heads(model):
                    if hasattr(head, '_last_metrics'):
                        metrics[f'ring_{i}_head'] = head._last_metrics
            
            for head in iter_resonance_heads(self.crown_model):
                if hasattr(head, '_last_metrics'):
                    metrics['crown_head'] = head._last_metrics
        
        return {
            'step': step,
            'ring_texts': ring_texts,
            'crown_text': crown_text_generated,
            'coupling_pattern': self.current_pattern_idx - 1,  # Previous pattern
            'metrics': metrics,
        }
    
    def run_continuous_mixing(
        self,
        n_steps: int = 20,
        max_length: int = 50,
        temperature: float = 0.8,
    ) -> List[Dict]:
        """
        Run continuous mixing for n steps.
        
        Args:
            n_steps: Number of steps
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            List of step results
        """
        results = []
        
        print(f"\nRunning continuous mixing for {n_steps} steps...")
        print("=" * 60)
        
        for step in range(n_steps):
            result = self.forward_step(step, max_length, temperature)
            results.append(result)
            
            if (step + 1) % 5 == 0:
                print(f"\nStep {step + 1}/{n_steps}:")
                print(f"  Coupling pattern: {result['coupling_pattern']}")
                print(f"  Crown: {result['crown_text'][:100]}...")
        
        return results


def create_resonance_crown(
    model_name: str = "gpt2",
    n_ring_models: int = 8,
    device: Optional[torch.device] = None,
    use_resonance: bool = True,
    use_full_dragon: bool = True,
) -> ResonanceCrownGPT2:
    """
    Convenience function to create a Resonance Crown.
    
    Args:
        model_name: GPT-2 model name
        n_ring_models: Number of models in ring (default: 8)
        device: Device to run on
        use_resonance: Enable resonance attention
        use_full_dragon: Enable full dragon mode
        
    Returns:
        ResonanceCrownGPT2 instance
    """
    return ResonanceCrownGPT2(
        model_name=model_name,
        n_ring_models=n_ring_models,
        device=device,
        use_resonance=use_resonance,
        use_full_dragon=use_full_dragon,
    )

