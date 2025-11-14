"""
Chain-of-Thought Generator Module

Generates CoT reasoning with diagnosis using Resonance Transformer.
"""

from __future__ import annotations

from typing import Dict, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cot_diagnosis import CoTDiagnostic
from .resonance_gpt2_adapter import set_cot_mode

# CoT special tokens
COT_START_TOKEN = "<think>"
COT_END_TOKEN = "</think>"
ANSWER_START_TOKEN = "<answer>"
ANSWER_END_TOKEN = "</answer>"
PROBLEM_START_TOKEN = "<problem>"
PROBLEM_END_TOKEN = "</problem>"


class CoTGenerator:
    """
    Generates Chain-of-Thought reasoning with diagnosis.
    
    Supports:
    - CoT generation with reasoning steps
    - Direct answer generation (faster)
    - Automatic diagnosis using resonance metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        diagnostic: Optional[CoTDiagnostic] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize CoT generator.
        
        Args:
            model: GPT-2 model with ResonanceGPT2AttentionAdapter
            tokenizer: Tokenizer (must have encode/decode methods)
            diagnostic: Optional CoTDiagnostic instance (creates new if None)
            device: Device to run on (auto-detects if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.diagnostic = diagnostic or CoTDiagnostic(model)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
            
        self.model.to(self.device)
        self.model.eval()
        
        # Get token IDs for special tokens
        self._init_special_tokens()
        
    def _init_special_tokens(self):
        """Initialize special token IDs."""
        # Try to get from tokenizer
        if hasattr(self.tokenizer, "encode"):
            # Try to encode special tokens
            try:
                encoded = self.tokenizer.encode(COT_START_TOKEN, add_special_tokens=False)
                self.cot_start_id = encoded[0] if encoded else None
            except:
                self.cot_start_id = None
            try:
                encoded = self.tokenizer.encode(COT_END_TOKEN, add_special_tokens=False)
                self.cot_end_id = encoded[0] if encoded else None
            except:
                self.cot_end_id = None
            try:
                encoded = self.tokenizer.encode(ANSWER_START_TOKEN, add_special_tokens=False)
                self.answer_start_id = encoded[0] if encoded else None
            except:
                self.answer_start_id = None
        else:
            self.cot_start_id = None
            self.cot_end_id = None
            self.answer_start_id = None
            
        # Fallback: use eos_token_id if available
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            self.eos_token_id = self.tokenizer.eos_token_id
        else:
            self.eos_token_id = None
        
        # Debug: print if special tokens aren't found
        if self.cot_end_id is None:
            print(f"  Warning: {COT_END_TOKEN} not found in vocabulary, using EOS token for stopping")
            
    def generate_with_cot(
        self,
        problem: str,
        max_cot_tokens: int = 200,
        max_answer_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_diagnosis: bool = True,
    ) -> Dict:
        """
        Generate CoT reasoning + answer, optionally with diagnosis.
        
        Args:
            problem: The problem to solve
            max_cot_tokens: Maximum tokens for CoT reasoning
            max_answer_tokens: Maximum tokens for answer
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            return_diagnosis: If True, include resonance-based diagnosis
            
        Returns:
            Dictionary with:
            - problem: Original problem
            - cot_rationale: Generated reasoning steps
            - answer: Final answer
            - diagnosis: Diagnostic scores (if return_diagnosis=True)
            - resonance_metrics: Raw resonance metrics
        """
        # Enable CoT mode
        set_cot_mode(self.model, enabled=True)
        
        # Format prompt with CoT trigger
        prompt = f"{PROBLEM_START_TOKEN}{problem}{PROBLEM_END_TOKEN}\n{COT_START_TOKEN}"
        
        # Encode prompt (already on device from _encode)
        input_ids = self._encode(prompt)
        
        # Generate CoT reasoning
        print("  Generating CoT reasoning...", end="", flush=True)
        cot_ids = self._generate_tokens(
            input_ids,
            max_tokens=max_cot_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token=COT_END_TOKEN,
        )
        print(" Done", flush=True)
        
        # Extract CoT text
        full_cot_text = self._decode(cot_ids[0])
        # Remove prompt from output
        cot_rationale = full_cot_text.split(COT_END_TOKEN)[0].strip()
        if COT_START_TOKEN in cot_rationale:
            cot_rationale = cot_rationale.split(COT_START_TOKEN)[-1].strip()
        
        # Continue with answer generation
        answer_prompt = f"{prompt}{cot_rationale}{COT_END_TOKEN}\n{ANSWER_START_TOKEN}"
        answer_input_ids = self._encode(answer_prompt)
        
        print("  Generating answer...", end="", flush=True)
        answer_ids = self._generate_tokens(
            answer_input_ids,
            max_tokens=max_answer_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token=ANSWER_END_TOKEN,
        )
        print(" Done", flush=True)
        
        # Extract answer text
        full_answer_text = self._decode(answer_ids[0])
        answer = full_answer_text.split(ANSWER_END_TOKEN)[0].strip()
        if ANSWER_START_TOKEN in answer:
            answer = answer.split(ANSWER_START_TOKEN)[-1].strip()
        
        # Collect resonance metrics
        resonance_metrics = self.diagnostic._collect_metrics()
        
        # Diagnose if requested
        diagnosis = None
        if return_diagnosis:
            diagnosis = self.diagnostic.diagnose_cot(
                problem=problem,
                cot_rationale=cot_rationale,
                answer=answer,
                resonance_metrics=resonance_metrics,
            )
        
        return {
            'problem': problem,
            'cot_rationale': cot_rationale,
            'answer': answer,
            'diagnosis': diagnosis,
            'resonance_metrics': resonance_metrics,
            'mode': 'cot',
        }
        
    def generate_direct(
        self,
        problem: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict:
        """
        Generate direct answer without CoT reasoning (faster).
        
        Args:
            problem: The problem to solve
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary with:
            - problem: Original problem
            - answer: Direct answer
            - mode: 'direct'
        """
        # Disable CoT mode for faster inference
        set_cot_mode(self.model, enabled=False)
        
        # Format prompt
        prompt = f"{PROBLEM_START_TOKEN}{problem}{PROBLEM_END_TOKEN}\n{ANSWER_START_TOKEN}"
        input_ids = self._encode(prompt)
        
        # Generate answer
        print("  Generating direct answer...", end="", flush=True)
        answer_ids = self._generate_tokens(
            input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token=ANSWER_END_TOKEN,
        )
        print(" Done", flush=True)
        
        # Extract answer
        full_answer_text = self._decode(answer_ids[0])
        answer = full_answer_text.split(ANSWER_END_TOKEN)[0].strip()
        if ANSWER_START_TOKEN in answer:
            answer = answer.split(ANSWER_START_TOKEN)[-1].strip()
        
        return {
            'problem': problem,
            'answer': answer,
            'mode': 'direct',
        }
        
    def _encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        if hasattr(self.tokenizer, "encode"):
            try:
                # Try modern tokenizer API first
                if hasattr(self.tokenizer, "__call__"):
                    result = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
                    if isinstance(result, dict):
                        input_ids = result.get("input_ids", None)
                        if input_ids is not None:
                            return input_ids.to(self.device)
                    elif isinstance(result, torch.Tensor):
                        return result.to(self.device)
                # Fallback to encode method
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if isinstance(ids, list):
                    return torch.tensor([ids], device=self.device)
                elif isinstance(ids, torch.Tensor):
                    return ids.to(self.device)
            except Exception as e:
                # If encoding fails, try fallback
                pass
        
        # Fallback: character-level
        if hasattr(self.tokenizer, "stoi"):
            ids = [self.tokenizer.stoi.get(ch, 0) for ch in text]
            return torch.tensor([ids], device=self.device)
        
        raise ValueError("Tokenizer must have encode method or stoi mapping")
            
    def _decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
            
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)
        elif hasattr(self.tokenizer, "itos"):
            return "".join([self.tokenizer.itos[i] for i in token_ids if i < len(self.tokenizer.itos)])
        else:
            raise ValueError("Tokenizer must have decode method or itos mapping")
            
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Initial token IDs [batch_size, seq_len]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_token: Optional stop token string
            
        Returns:
            Generated token IDs [batch_size, total_len]
        """
        # Ensure input_ids is a tensor on the correct device
        if not isinstance(input_ids, torch.Tensor):
            if isinstance(input_ids, (list, tuple)):
                input_ids = torch.tensor([input_ids], device=self.device)
            else:
                raise TypeError(f"input_ids must be a tensor, got {type(input_ids)}")
        
        # Ensure tensor is on correct device
        input_ids = input_ids.to(self.device)
        
        # Ensure 2D shape [batch_size, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        generated_ids = input_ids.clone()
        stop_token_id = None
        stop_token_ids = None
        
        if stop_token and hasattr(self.tokenizer, "encode"):
            try:
                if hasattr(self.tokenizer, "__call__"):
                    # Modern tokenizer
                    result = self.tokenizer(stop_token, return_tensors=None, add_special_tokens=False)
                    if isinstance(result, dict):
                        stop_token_ids = result.get("input_ids", [])
                    elif isinstance(result, list):
                        stop_token_ids = result
                    else:
                        stop_token_ids = [result]
                else:
                    # Older tokenizer
                    stop_token_ids = self.tokenizer.encode(stop_token, add_special_tokens=False)
                
                if stop_token_ids:
                    stop_token_id = stop_token_ids[0] if isinstance(stop_token_ids, list) else stop_token_ids
            except Exception as e:
                # If encoding fails, try to find token in vocabulary
                pass
        
        with torch.no_grad():
            for step in range(max_tokens):
                # Progress indicator every 10 steps
                if step > 0 and step % 10 == 0:
                    print(".", end="", flush=True)
                
                # Forward pass
                try:
                    outputs = self.model(generated_ids)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                except Exception as e:
                    print(f"\nWarning: Model forward pass failed: {e}")
                    break
                
                # Get next token logits
                if logits.dim() == 3:
                    next_token_logits = logits[0, -1, :] / temperature
                elif logits.dim() == 2:
                    next_token_logits = logits[-1, :] / temperature
                else:
                    print(f"Warning: Unexpected logits shape: {logits.shape}")
                    break
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1)  # Shape: [1]
                
                # Reshape to [batch_size, 1] for concatenation
                next_token_id = next_token_id.unsqueeze(0)  # Shape: [1, 1]
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                
                # Check for stop token
                next_token_value = next_token_id[0, 0].item()
                
                # Check for stop token
                if stop_token_id is not None:
                    if next_token_value == stop_token_id:
                        break
                    # Check for multi-token stop sequence
                    if stop_token_ids and len(stop_token_ids) > 1:
                        if generated_ids.shape[1] >= len(stop_token_ids):
                            recent_tokens = generated_ids[0, -len(stop_token_ids):].cpu().tolist()
                            if recent_tokens == stop_token_ids:
                                break
                    
                # Check for EOS
                if self.eos_token_id is not None and next_token_value == self.eos_token_id:
                    break
                
                # Safety check: prevent infinite loops
                if step >= max_tokens - 1:
                    break
        
        return generated_ids

