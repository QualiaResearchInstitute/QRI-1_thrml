"""
Resonance Transformer Instruction Refiner

Uses ResonanceTransformer instead of GPT-2 for instruction understanding.
Leverages resonance dynamics for language understanding and refinement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import re

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None

from resonance_transformer import ResonanceTransformer

# Import TextGenerator with fallback
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from scripts.generate_text import TextGenerator
except ImportError:
    # Fallback: create minimal TextGenerator
    class TextGenerator:
        def __init__(self, model, tokenizer, device="auto", **kwargs):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        def generate(self, prompt, max_new_tokens=100, temperature=1.0, **kwargs):
            # Simple generation
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class ResonanceInstructionRefiner:
    """
    Refines instructions using ResonanceTransformer.
    
    What ResonanceTransformer contributes:
    - Resonance dynamics for understanding language patterns
    - CDNS metrics track instruction quality
    - Coherence-based instruction interpretation
    - Learns from successful edit patterns
    """
    
    def __init__(
        self,
        model: Optional[ResonanceTransformer] = None,
        tokenizer=None,
        vocab_size: int = 50257,  # GPT-2 vocab size
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        device: str = "cuda",
    ):
        self.device = device
        
        # Load or create ResonanceTransformer
        if model is None:
            self.model = ResonanceTransformer(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                n_sim_steps=15,
            )
        else:
            self.model = model
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Tokenizer
        if tokenizer is None:
            if TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ImportError("transformers required for tokenizer")
        else:
            self.tokenizer = tokenizer
        
        # Text generator
        self.generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )
        
        # Learned patterns
        self.successful_edits: List[Dict] = []
        self.edit_patterns: List[Dict] = []
        
        # Resonance state tracking
        self.last_resonance_state: Optional[Dict] = None
    
    def refine_instruction(
        self,
        user_instruction: str,
        image_context: Optional[str] = None,
        edit_history: Optional[List[str]] = None,
        return_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Refine instruction using ResonanceTransformer.
        
        ResonanceTransformer provides:
        - Resonance dynamics for understanding language
        - CDNS metrics for instruction quality
        - Coherence-based interpretation
        """
        # Build prompt
        prompt = self._build_refinement_prompt(
            user_instruction,
            image_context,
            edit_history,
        )
        
        # Generate refined instruction with resonance dynamics
        refined_text, resonance_metrics = self._generate_with_resonance(
            prompt,
            max_length=100,
            temperature=0.7,
            return_metrics=return_metrics,
        )
        
        # Parse refined instruction
        refined = self._parse_refined_instruction(
            refined_text,
            user_instruction,
            resonance_metrics,
        )
        
        return refined
    
    def _build_refinement_prompt(
        self,
        instruction: str,
        image_context: Optional[str],
        edit_history: Optional[List[str]],
    ) -> str:
        """Build prompt for instruction refinement."""
        prompt = "Image Editing Instruction Refinement\n\n"
        
        if edit_history:
            prompt += "Previous edits:\n"
            for i, prev_edit in enumerate(edit_history[-3:], 1):
                prompt += f"{i}. {prev_edit}\n"
            prompt += "\n"
        
        if image_context:
            prompt += f"Image context: {image_context}\n\n"
        
        prompt += f"User instruction: {instruction}\n"
        prompt += "Refined instruction (with frequency band details):"
        
        return prompt
    
    def _generate_with_resonance(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        return_metrics: bool = True,
    ) -> Tuple[str, Optional[Dict]]:
        """
        Generate text using ResonanceTransformer with metrics.
        
        Returns:
            generated_text: Generated text
            resonance_metrics: CDNS metrics from generation
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with resonance dynamics
        # We need to capture metrics during generation
        generated_tokens = []
        input_ids = inputs['input_ids']
        
        # Track resonance state
        resonance_metrics_all = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.model(input_ids)
                
                # Try to extract metrics from model if available
                metrics = {}
                if hasattr(self.model, 'last_metrics'):
                    metrics = self.model.last_metrics
                elif hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                    # Try to get metrics from last layer
                    last_layer = self.model.layers[-1]
                    if hasattr(last_layer, 'attention_heads'):
                        # Collect metrics from attention heads
                        head_metrics = []
                        for head in last_layer.attention_heads:
                            if hasattr(head, 'last_metrics'):
                                head_metrics.append(head.last_metrics)
                        if head_metrics:
                            metrics = self._aggregate_head_metrics(head_metrics)
                
                if metrics:
                    resonance_metrics_all.append(metrics)
                
                # Sample next token
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Aggregate metrics
        resonance_metrics = None
        if resonance_metrics_all:
            resonance_metrics = self._aggregate_resonance_metrics(resonance_metrics_all)
        
        return generated_text, resonance_metrics
    
    def _aggregate_resonance_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate resonance metrics across generation steps."""
        aggregated = {}
        
        # Collect all metric keys
        all_keys = set()
        for m in metrics_list:
            if m:
                all_keys.update(m.keys())
        
        for key in all_keys:
            values = []
            for m in metrics_list:
                if m and key in m:
                    val = m[key]
                    if isinstance(val, torch.Tensor):
                        values.append(val.mean().item() if val.numel() > 1 else val.item())
                    else:
                        values.append(val)
            
            if values:
                aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    def _aggregate_head_metrics(self, metric_list: List[Dict]) -> Dict:
        """Aggregate metrics from multiple heads."""
        if not metric_list:
            return {}
        
        aggregated = {}
        for m in metric_list:
            if m:
                for k, v in m.items():
                    if isinstance(v, torch.Tensor):
                        if k not in aggregated:
                            aggregated[k] = []
                        aggregated[k].append(v.mean().item() if v.numel() > 1 else v.item())
        
        # Average
        for k in aggregated:
            aggregated[k] = sum(aggregated[k]) / len(aggregated[k])
        
        return aggregated
    
    def _parse_refined_instruction(
        self,
        generated_text: str,
        original_instruction: str,
        resonance_metrics: Optional[Dict],
    ) -> Dict[str, Any]:
        """Parse refined instruction with resonance metrics."""
        # Extract band weights
        band_weights = {}
        band_matches = re.findall(r'band[_\s]*(\d+)[:\s=]+([\d.]+)', generated_text.lower())
        for band_idx, weight in band_matches:
            band_weights[int(band_idx)] = float(weight)
        
        # Extract operation type
        operation = original_instruction
        if "sharpen" in generated_text.lower():
            operation = "sharpen"
        elif "smooth" in generated_text.lower():
            operation = "smooth"
        elif "enhance" in generated_text.lower():
            operation = "enhance"
        
        # Extract steps
        steps = self._extract_steps(generated_text)
        
        result = {
            'original': original_instruction,
            'refined_text': generated_text,
            'operation': operation,
            'band_weights': band_weights if band_weights else None,
            'decomposed_steps': steps,
        }
        
        # Add resonance metrics
        if resonance_metrics:
            result['resonance_metrics'] = resonance_metrics
            result['coherence'] = resonance_metrics.get('order_parameter', 0.0)
            result['consonance'] = resonance_metrics.get('cdns.C', 0.0)
            result['dissonance'] = resonance_metrics.get('cdns.D', 0.0)
        
        return result
    
    def _extract_steps(self, text: str) -> List[str]:
        """Extract editing steps from generated text."""
        steps = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['step', 'first', 'then', 'next']):
                steps.append(line.strip())
        return steps
    
    def learn_from_edit(
        self,
        instruction: str,
        band_weights: Dict[int, float],
        success: bool,
        edit_metrics: Dict,
        resonance_metrics: Optional[Dict] = None,
    ):
        """
        Learn from edit results.
        
        ResonanceTransformer learns:
        - Which instructions lead to high coherence edits
        - What resonance patterns correlate with success
        - How to predict good edits from resonance state
        """
        pattern = {
            'instruction': instruction,
            'band_weights': band_weights,
            'success': success,
            'edit_throughput': float(edit_metrics.get('throughput', 0.0).mean().item())
            if isinstance(edit_metrics.get('throughput'), torch.Tensor) else 0.0,
        }
        
        # Add resonance metrics if available
        if resonance_metrics:
            pattern['resonance_coherence'] = resonance_metrics.get('order_parameter', 0.0)
            pattern['resonance_consonance'] = resonance_metrics.get('cdns.C', 0.0)
            pattern['resonance_dissonance'] = resonance_metrics.get('cdns.D', 0.0)
        
        if success:
            self.successful_edits.append(pattern)
        else:
            self.edit_patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self.successful_edits) > 1000:
            self.successful_edits.pop(0)
        if len(self.edit_patterns) > 1000:
            self.edit_patterns.pop(0)
    
    def suggest_edit(
        self,
        image_description: str,
        goal: str,
    ) -> Tuple[str, Dict]:
        """
        Suggest edit instruction using ResonanceTransformer.
        
        Uses resonance dynamics to generate suggestions that have
        high coherence and consonance.
        """
        # Find similar successful edits
        similar_edits = self._find_similar_edits(image_description, goal)
        
        # Build suggestion prompt
        prompt = f"Image: {image_description}\nGoal: {goal}\n"
        if similar_edits:
            prompt += "Similar successful edits:\n"
            for edit in similar_edits[:3]:
                prompt += f"- {edit['instruction']}\n"
        prompt += "\nSuggested edit instruction:"
        
        # Generate with resonance
        suggestion, resonance_metrics = self._generate_with_resonance(
            prompt,
            max_length=50,
            temperature=0.7,
        )
        
        suggestion_text = suggestion.split("Suggested edit instruction:")[-1].strip()
        
        return suggestion_text, resonance_metrics or {}
    
    def _find_similar_edits(
        self,
        image_description: str,
        goal: str,
    ) -> List[Dict]:
        """Find similar successful edits."""
        keywords = set(image_description.lower().split() + goal.lower().split())
        
        similar = []
        for edit in self.successful_edits:
            edit_text = edit['instruction'].lower()
            if any(kw in edit_text for kw in keywords):
                similar.append(edit)
        
        # Sort by throughput and coherence
        similar.sort(
            key=lambda x: (
                x.get('edit_throughput', 0) +
                x.get('resonance_coherence', 0) * 0.5
            ),
            reverse=True,
        )
        
        return similar[:5]


class ResonanceEditPlanner:
    """
    Plans complex edits using ResonanceTransformer.
    
    Uses resonance dynamics to plan edit sequences with high coherence.
    """
    
    def __init__(
        self,
        refiner: ResonanceInstructionRefiner,
    ):
        self.refiner = refiner
    
    def plan_edit_sequence(
        self,
        complex_instruction: str,
        image_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Plan edit sequence using ResonanceTransformer.
        
        ResonanceTransformer plans edits that maintain coherence
        across steps.
        """
        prompt = f"Complex Edit Planning\n\n"
        if image_context:
            prompt += f"Image: {image_context}\n"
        prompt += f"Goal: {complex_instruction}\n"
        prompt += "Break down into editing steps:\n"
        
        # Generate plan with resonance
        plan_text, resonance_metrics = self.refiner._generate_with_resonance(
            prompt,
            max_length=200,
            temperature=0.7,
        )
        
        # Parse plan
        steps = self._parse_plan(plan_text, resonance_metrics)
        
        return steps
    
    def _parse_plan(
        self,
        plan_text: str,
        resonance_metrics: Optional[Dict],
    ) -> List[Dict[str, Any]]:
        """Parse edit plan with resonance information."""
        steps = []
        lines = plan_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            step_match = re.match(r'(\d+)\.\s*(.+)', line)
            if step_match:
                step_num = int(step_match.group(1))
                instruction = step_match.group(2)
                
                # Extract band weights
                band_weights = {}
                band_matches = re.findall(r'band[_\s]*(\d+)[:\s=]+([\d.]+)', line.lower())
                for band_idx, weight in band_matches:
                    band_weights[int(band_idx)] = float(weight)
                
                step_dict = {
                    'step': step_num,
                    'instruction': instruction,
                    'band_weights': band_weights if band_weights else None,
                }
                
                # Add resonance info
                if resonance_metrics:
                    step_dict['resonance_coherence'] = resonance_metrics.get('order_parameter', 0.0)
                
                steps.append(step_dict)
        
        return steps


class ResonanceConsciousnessEditor:
    """
    Full integration: ResonanceTransformer + Consciousness Circuit Image Editor.
    
    Uses ResonanceTransformer for instruction understanding instead of GPT-2.
    Benefits:
    - Consistent architecture (resonance throughout)
    - Resonance dynamics for language understanding
    - CDNS metrics track instruction quality
    - Coherence-based instruction interpretation
    """
    
    def __init__(
        self,
        edit_model: Any,  # ConsciousnessEditModel
        resonance_model: Optional[ResonanceTransformer] = None,
        vocab_size: int = 50257,
        d_model: int = 512,
        device: str = "cuda",
    ):
        self.edit_model = edit_model
        self.refiner = ResonanceInstructionRefiner(
            model=resonance_model,
            vocab_size=vocab_size,
            d_model=d_model,
            device=device,
        )
        self.planner = ResonanceEditPlanner(self.refiner)
        
        # Multi-turn context
        self.edit_history: List[Dict] = []
        self.conversation_context: List[str] = []
        
        # Resonance state tracking
        self.resonance_state_history: List[Dict] = []
    
    def edit_with_resonance(
        self,
        image: torch.Tensor,
        user_instruction: str,
        image_description: Optional[str] = None,
        return_metrics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Edit image with ResonanceTransformer assistance.
        
        ResonanceTransformer:
        1. Refines instruction using resonance dynamics
        2. Maintains context through resonance state
        3. Tracks coherence for instruction quality
        4. Learns from successful resonance patterns
        """
        # Refine instruction with ResonanceTransformer
        refined = self.refiner.refine_instruction(
            user_instruction,
            image_context=image_description,
            edit_history=[e['instruction'] for e in self.edit_history[-3:]],
            return_metrics=return_metrics,
        )
        
        # Check if complex edit needs planning
        if self._is_complex_instruction(user_instruction):
            plan = self.planner.plan_edit_sequence(
                user_instruction,
                image_context=image_description,
            )
            
            # Execute plan step by step
            current_image = image
            all_metrics = []
            
            for step in plan:
                step_instruction = step['instruction']
                step_band_weights = step.get('band_weights')
                
                # Get instruction embedding
                instruction_embedding = self._encode_instruction(step_instruction)
                
                # Apply edit
                if step_band_weights:
                    band_weights_tensor = torch.tensor([
                        step_band_weights.get(i, 1.0) for i in range(4)
                    ]).unsqueeze(0).to(image.device)
                else:
                    band_weights_tensor = None
                
                current_image, edit_metrics = self.edit_model(
                    current_image,
                    instruction_embedding=instruction_embedding,
                    band_weights=band_weights_tensor,
                    return_metrics=True,
                )
                
                all_metrics.append({
                    'edit_metrics': edit_metrics,
                    'resonance_coherence': step.get('resonance_coherence', 0.0),
                })
            
            # Store in history
            self.edit_history.append({
                'instruction': user_instruction,
                'refined': refined,
                'plan': plan,
                'metrics': all_metrics,
            })
            
            result_metrics = {
                'refined_instruction': refined,
                'plan': plan,
                'all_metrics': all_metrics,
                'resonance_metrics': refined.get('resonance_metrics'),
            }
            
            return current_image, result_metrics
        
        # Simple edit
        instruction_embedding = self._encode_instruction(refined['refined_text'])
        
        # Get band weights
        band_weights_tensor = None
        if refined.get('band_weights'):
            band_weights_tensor = torch.tensor([
                refined['band_weights'].get(i, 1.0) for i in range(4)
            ]).unsqueeze(0).to(image.device)
        
        # Apply edit
        edited_image, edit_metrics = self.edit_model(
            image,
            instruction_embedding=instruction_embedding,
            band_weights=band_weights_tensor,
            return_metrics=True,
        )
        
        # Learn from edit
        success = float(edit_metrics.get('throughput', torch.tensor(0.0)).mean().item()) > 0.3
        self.refiner.learn_from_edit(
            user_instruction,
            refined.get('band_weights', {}),
            success,
            edit_metrics,
            refined.get('resonance_metrics'),
        )
        
        # Store resonance state
        if refined.get('resonance_metrics'):
            self.resonance_state_history.append(refined['resonance_metrics'])
        
        # Store in history
        self.edit_history.append({
            'instruction': user_instruction,
            'refined': refined,
            'metrics': edit_metrics,
        })
        
        result_metrics = {
            'refined_instruction': refined,
            'edit_metrics': edit_metrics,
            'resonance_metrics': refined.get('resonance_metrics'),
            'coherence': refined.get('coherence', 0.0),
        }
        
        return edited_image, result_metrics
    
    def _is_complex_instruction(self, instruction: str) -> bool:
        """Check if instruction needs decomposition."""
        complex_keywords = [
            'make it', 'transform', 'convert', 'change to', 'turn into',
            'professional', 'artistic', 'dramatic', 'subtle', 'enhance overall',
        ]
        return any(keyword in instruction.lower() for keyword in complex_keywords)
    
    def _encode_instruction(self, instruction: str) -> torch.Tensor:
        """Encode instruction using ResonanceTransformer."""
        inputs = self.refiner.tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.refiner.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get embeddings from ResonanceTransformer
            logits = self.refiner.model(inputs['input_ids'])
            
            # Mean pooling
            embedding = logits.mean(dim=1)  # [batch, d_model]
        
        return embedding
    
    def conversational_edit(
        self,
        image: torch.Tensor,
        user_message: str,
        image_description: Optional[str] = None,
    ) -> Tuple[torch.Tensor, str]:
        """
        Conversational editing with ResonanceTransformer.
        
        ResonanceTransformer maintains context through resonance state,
        not just token sequences.
        """
        # Add to conversation
        self.conversation_context.append(f"User: {user_message}")
        
        # Check if follow-up edit
        if self._is_followup(user_message):
            refined_message = self._interpret_followup(user_message)
        else:
            refined_message = user_message
        
        # Edit image
        edited_image, metrics = self.edit_with_resonance(
            image,
            refined_message,
            image_description,
        )
        
        # Generate response using ResonanceTransformer
        response = self._generate_response(user_message, metrics)
        self.conversation_context.append(f"System: {response}")
        
        return edited_image, response
    
    def _is_followup(self, message: str) -> bool:
        """Check if message is a follow-up edit."""
        followup_keywords = ['more', 'less', 'also', 'and', 'then', 'next', 'further']
        return any(keyword in message.lower() for keyword in followup_keywords)
    
    def _interpret_followup(self, message: str) -> str:
        """Interpret follow-up edit using ResonanceTransformer."""
        context = "\n".join(self.conversation_context[-4:])
        prompt = f"{context}\n\nInterpret this follow-up edit: {message}\nFull instruction:"
        
        interpreted, _ = self.refiner._generate_with_resonance(
            prompt,
            max_length=50,
            temperature=0.7,
        )
        
        return interpreted.split("Full instruction:")[-1].strip()
    
    def _generate_response(
        self,
        user_message: str,
        metrics: Dict,
    ) -> str:
        """Generate conversational response using ResonanceTransformer."""
        coherence = metrics.get('coherence', 0.0)
        throughput = float(metrics.get('edit_metrics', {}).get('throughput', torch.tensor(0.0)).mean().item())
        
        prompt = f"User: {user_message}\n"
        prompt += f"Edit completed. Coherence: {coherence:.2f}, Throughput: {throughput:.2f}\n"
        prompt += "System response:"
        
        response, _ = self.refiner._generate_with_resonance(
            prompt,
            max_length=30,
            temperature=0.7,
        )
        
        return response.split("System response:")[-1].strip()
    
    def get_resonance_knowledge(self) -> Dict[str, Any]:
        """Get what ResonanceTransformer has learned."""
        return {
            'successful_patterns': len(self.refiner.successful_edits),
            'total_edits': len(self.edit_history),
            'resonance_states': len(self.resonance_state_history),
            'avg_coherence': sum(
                s.get('order_parameter', 0.0) for s in self.resonance_state_history
            ) / len(self.resonance_state_history) if self.resonance_state_history else 0.0,
            'top_patterns': self.refiner.successful_edits[-10:],
        }

