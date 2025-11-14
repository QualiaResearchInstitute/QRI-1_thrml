"""
Resonance Image Editing Network

A complete, isolated image editing network that combines:
1. ResonanceTransformer - for instruction understanding (Kuramoto-based attention)
2. ConsciousnessEditModel - for image editing (consciousness circuit)

This is the "resonance network" we're training for image editing tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from resonance_transformer import ResonanceTransformer
from modules.consciousness_edit_model import ConsciousnessEditModel


class ResonanceImageEditor(nn.Module):
    """
    Complete Resonance Image Editing Network.
    
    This is the isolated, named "resonance network" for image editing.
    
    Architecture:
    1. ResonanceTransformer: Understands instructions using Kuramoto oscillators
    2. ConsciousnessEditModel: Edits images using consciousness circuit
    
    Both components use resonance dynamics (Kuramoto oscillators).
    """
    
    def __init__(
        self,
        # ResonanceTransformer config
        vocab_size: int = 50257,  # GPT-2 vocab size
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        n_sim_steps: int = 15,
        max_seq_len: int = 512,
        
        # ConsciousnessEditModel config
        vision_model_name: str = "google/vit-base-patch16-224",
        n_frequency_bands: int = 4,
        n_circuit_layers: int = 3,
        
        # Tokenizer
        tokenizer_name: str = "gpt2",
        
        device: str = "cuda",
    ):
        """
        Initialize Resonance Image Editor.
        
        Args:
            vocab_size: Vocabulary size for ResonanceTransformer
            d_model: Model dimension (shared between components)
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            n_sim_steps: Kuramoto simulation steps
            max_seq_len: Maximum sequence length
            vision_model_name: Vision model for image encoding
            n_frequency_bands: Number of frequency bands for editing
            n_circuit_layers: Number of consciousness circuit layers
            tokenizer_name: Tokenizer name (for instruction processing)
            device: Device to run on
        """
        super().__init__()
        
        self.device = device
        self.d_model = d_model
        
        # 1. ResonanceTransformer for instruction understanding
        self.resonance_transformer = ResonanceTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_sim_steps=n_sim_steps,
            max_seq_len=max_seq_len,
        )
        
        # 2. ConsciousnessEditModel for image editing
        self.edit_model = ConsciousnessEditModel(
            vision_model_name=vision_model_name,
            n_frequency_bands=n_frequency_bands,
            d_model=d_model,
            n_circuit_layers=n_circuit_layers,
        )
        
        # Tokenizer for instructions
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except ImportError:
            self.tokenizer = None
        
        # Move to device
        self.to(device)
    
    def forward(
        self,
        image: torch.Tensor,
        instruction: Optional[str] = None,
        instruction_ids: Optional[torch.Tensor] = None,
        instruction_embedding: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: Edit image based on instruction.
        
        Args:
            image: [batch, 3, H, W] input image tensor
            instruction: Text instruction (string)
            instruction_ids: Pre-tokenized instruction IDs [batch, seq_len]
            instruction_embedding: Pre-computed instruction embedding [batch, d_model]
            return_metrics: Whether to return detailed metrics
            
        Returns:
            edited_image: [batch, 3, H, W] edited image tensor
            metrics: Dictionary with metrics (if return_metrics=True)
        """
        # Get instruction embedding from ResonanceTransformer
        if instruction_embedding is None:
            if instruction_ids is None:
                if instruction is None:
                    raise ValueError("Must provide instruction, instruction_ids, or instruction_embedding")
                
                # Tokenize instruction
                if self.tokenizer is None:
                    raise RuntimeError("Tokenizer not available. Install transformers or provide instruction_ids.")
                
                instruction_ids = self.tokenizer(
                    instruction if isinstance(instruction, str) else instruction[0],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )['input_ids'].to(self.device)
            
            # Get embedding from ResonanceTransformer
            with torch.no_grad() if not self.training else torch.enable_grad():
                instruction_logits = self.resonance_transformer(instruction_ids)
                instruction_embedding = instruction_logits.mean(dim=1)  # [batch, d_model]
        
        # Edit image using ConsciousnessEditModel
        edited_image, edit_metrics = self.edit_model(
            image,
            instruction_embedding=instruction_embedding,
            return_metrics=return_metrics,
        )
        
        metrics = {}
        if return_metrics:
            metrics = {
                'edit_metrics': edit_metrics,
                'instruction_embedding': instruction_embedding.detach(),
            }
        
        return edited_image, metrics
    
    def encode_instruction(
        self,
        instruction: str,
    ) -> torch.Tensor:
        """
        Encode instruction to embedding using ResonanceTransformer.
        
        Args:
            instruction: Text instruction
            
        Returns:
            [1, d_model] instruction embedding
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available")
        
        instruction_ids = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )['input_ids'].to(self.device)
        
        with torch.no_grad():
            instruction_logits = self.resonance_transformer(instruction_ids)
            instruction_embedding = instruction_logits.mean(dim=1)
        
        return instruction_embedding
    
    def edit_image(
        self,
        image: torch.Tensor,
        instruction: str,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Convenience method: Edit image with text instruction.
        
        Args:
            image: [batch, 3, H, W] input image
            instruction: Text instruction
            return_metrics: Whether to return metrics
            
        Returns:
            edited_image: [batch, 3, H, W] edited image
            metrics: Dictionary with metrics (if return_metrics=True)
        """
        return self.forward(
            image=image,
            instruction=instruction,
            return_metrics=return_metrics,
        )


# Alias for backward compatibility and clarity
ResonanceImageEditingNetwork = ResonanceImageEditor
ResonanceNetwork = ResonanceImageEditor

