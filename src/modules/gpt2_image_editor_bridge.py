"""
GPT-2 + Consciousness Circuit Image Editor Bridge

Integrates GPT-2 language model with consciousness-circuit image editing:
- Instruction understanding and refinement
- Multi-turn editing with context
- Learning from edit patterns
- Cross-modal reasoning
- Edit planning and decomposition
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import json

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    GPT2LMHeadModel = None
    GPT2Tokenizer = None

from modules.consciousness_edit_model import ConsciousnessEditModel
from modules.multi_model_orchestrator import MultiModelResonanceOrchestrator


class GPT2InstructionRefiner:
    """
    GPT-2 refines and expands user instructions for better editing.
    
    What GPT-2 learns:
    - How to interpret vague instructions ("make it better" → specific edits)
    - How to decompose complex edits into steps
    - How to maintain context across multiple edits
    - Patterns from successful edits
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        fine_tuned_path: Optional[str] = None,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for GPT-2 integration")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if fine_tuned_path:
            self.model = GPT2LMHeadModel.from_pretrained(fine_tuned_path)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        self.model.eval()
        
        # Instruction patterns learned from edits
        self.edit_patterns: List[Dict] = []
        self.successful_edits: List[Dict] = []
    
    def refine_instruction(
        self,
        user_instruction: str,
        image_context: Optional[str] = None,
        edit_history: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Refine user instruction into detailed edit specification.
        
        GPT-2 learns to:
        - Expand vague instructions ("sharpen" → "enhance high frequencies, band 3: 1.5")
        - Decompose complex edits ("make it artistic" → multiple steps)
        - Maintain context from previous edits
        """
        # Build prompt
        prompt = self._build_refinement_prompt(
            user_instruction,
            image_context,
            edit_history,
        )
        
        # Generate refined instruction
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse refined instruction
        refined = self._parse_refined_instruction(generated_text, user_instruction)
        
        return refined
    
    def _build_refinement_prompt(
        self,
        instruction: str,
        image_context: Optional[str],
        edit_history: Optional[List[str]],
    ) -> str:
        """Build prompt for GPT-2 to refine instruction."""
        prompt = "Image Editing Instruction Refinement\n\n"
        
        if edit_history:
            prompt += "Previous edits:\n"
            for i, prev_edit in enumerate(edit_history[-3:], 1):  # Last 3 edits
                prompt += f"{i}. {prev_edit}\n"
            prompt += "\n"
        
        if image_context:
            prompt += f"Image context: {image_context}\n\n"
        
        prompt += f"User instruction: {instruction}\n"
        prompt += "Refined instruction (with frequency band details):"
        
        return prompt
    
    def _parse_refined_instruction(
        self,
        generated_text: str,
        original_instruction: str,
    ) -> Dict[str, Any]:
        """Parse GPT-2's refined instruction."""
        # Extract band weights if mentioned
        import re
        
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
        
        return {
            'original': original_instruction,
            'refined_text': generated_text,
            'operation': operation,
            'band_weights': band_weights if band_weights else None,
            'decomposed_steps': self._extract_steps(generated_text),
        }
    
    def _extract_steps(self, text: str) -> List[str]:
        """Extract editing steps from generated text."""
        # Simple extraction - in practice, use more sophisticated parsing
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
        metrics: Dict,
    ):
        """
        Learn from edit results to improve future refinements.
        
        GPT-2 learns:
        - Which instructions lead to successful edits
        - What band weight patterns work well
        - How to predict good edits from instructions
        """
        pattern = {
            'instruction': instruction,
            'band_weights': band_weights,
            'success': success,
            'throughput': float(metrics.get('throughput', 0.0).mean().item())
            if isinstance(metrics.get('throughput'), torch.Tensor) else 0.0,
        }
        
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
    ) -> str:
        """
        Suggest edit instruction based on image and goal.
        
        GPT-2 learns to suggest edits that work well.
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
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
            )
        
        suggestion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return suggestion.split("Suggested edit instruction:")[-1].strip()
    
    def _find_similar_edits(
        self,
        image_description: str,
        goal: str,
    ) -> List[Dict]:
        """Find similar successful edits (simplified - use embeddings in practice)."""
        # Simple keyword matching
        keywords = set(image_description.lower().split() + goal.lower().split())
        
        similar = []
        for edit in self.successful_edits:
            edit_text = edit['instruction'].lower()
            if any(kw in edit_text for kw in keywords):
                similar.append(edit)
        
        # Sort by throughput (success metric)
        similar.sort(key=lambda x: x['throughput'], reverse=True)
        return similar[:5]


class GPT2EditPlanner:
    """
    GPT-2 plans complex edits by decomposing them into steps.
    
    What GPT-2 learns:
    - How to break down complex edits ("make it look professional" → multiple steps)
    - Optimal edit sequences
    - Dependencies between edits
    """
    
    def __init__(
        self,
        gpt2_model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
    ):
        self.model = gpt2_model
        self.tokenizer = tokenizer
    
    def plan_edit_sequence(
        self,
        complex_instruction: str,
        image_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Plan a sequence of edits for complex instruction.
        
        Example:
        "make it look professional" →
        1. enhance structure (band 0: 1.2)
        2. slightly sharpen (band 3: 1.3)
        3. enhance textures (band 1: 1.4)
        """
        prompt = f"Complex Edit Planning\n\n"
        if image_context:
            prompt += f"Image: {image_context}\n"
        prompt += f"Goal: {complex_instruction}\n"
        prompt += "Break down into editing steps:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
            )
        
        plan_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse plan into steps
        steps = self._parse_plan(plan_text)
        
        return steps
    
    def _parse_plan(self, plan_text: str) -> List[Dict[str, Any]]:
        """Parse edit plan into steps."""
        steps = []
        lines = plan_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Extract step number and instruction
            import re
            step_match = re.match(r'(\d+)\.\s*(.+)', line)
            if step_match:
                step_num = int(step_match.group(1))
                instruction = step_match.group(2)
                
                # Extract band weights if mentioned
                band_weights = {}
                band_matches = re.findall(r'band[_\s]*(\d+)[:\s=]+([\d.]+)', line.lower())
                for band_idx, weight in band_matches:
                    band_weights[int(band_idx)] = float(weight)
                
                steps.append({
                    'step': step_num,
                    'instruction': instruction,
                    'band_weights': band_weights if band_weights else None,
                })
        
        return steps


class GPT2ConsciousnessEditor:
    """
    Full integration: GPT-2 + Consciousness Circuit Image Editor.
    
    GPT-2 contributes:
    1. Instruction understanding and refinement
    2. Multi-turn context maintenance
    3. Edit planning and decomposition
    4. Learning from successful edits
    5. Cross-modal reasoning (text → visual concepts)
    """
    
    def __init__(
        self,
        edit_model: ConsciousnessEditModel,
        gpt2_model_name: str = "gpt2",
        fine_tune_gpt2: bool = False,
    ):
        self.edit_model = edit_model
        self.refiner = GPT2InstructionRefiner(gpt2_model_name)
        self.planner = GPT2EditPlanner(
            self.refiner.model,
            self.refiner.tokenizer,
        )
        
        # Multi-turn context
        self.edit_history: List[Dict] = []
        self.conversation_context: List[str] = []
        
        # Learning from edits
        self.learned_patterns: Dict[str, Dict] = {}
    
    def edit_with_gpt2(
        self,
        image: torch.Tensor,
        user_instruction: str,
        image_description: Optional[str] = None,
        return_refined: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Edit image with GPT-2 assistance.
        
        GPT-2:
        1. Refines user instruction
        2. Maintains context from previous edits
        3. Suggests optimal band weights
        4. Learns from results
        """
        # Refine instruction with GPT-2
        refined = self.refiner.refine_instruction(
            user_instruction,
            image_context=image_description,
            edit_history=[e['instruction'] for e in self.edit_history[-3:]],
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
                
                # Get instruction embedding (use refined instruction)
                instruction_embedding = self._encode_instruction(step_instruction)
                
                # Apply edit
                if step_band_weights:
                    band_weights_tensor = torch.tensor([
                        step_band_weights.get(i, 1.0) for i in range(4)
                    ]).unsqueeze(0).to(image.device)
                else:
                    band_weights_tensor = None
                
                current_image, metrics = self.edit_model(
                    current_image,
                    instruction_embedding=instruction_embedding,
                    band_weights=band_weights_tensor,
                    return_metrics=True,
                )
                
                all_metrics.append(metrics)
            
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
                'final_throughput': all_metrics[-1].get('throughput', torch.tensor(0.0)),
            }
            
            return current_image, result_metrics
        
        # Simple edit - use refined instruction
        instruction_embedding = self._encode_instruction(refined['refined_text'])
        
        # Get band weights from refined instruction
        band_weights_tensor = None
        if refined.get('band_weights'):
            band_weights_tensor = torch.tensor([
                refined['band_weights'].get(i, 1.0) for i in range(4)
            ]).unsqueeze(0).to(image.device)
        
        # Apply edit
        edited_image, metrics = self.edit_model(
            image,
            instruction_embedding=instruction_embedding,
            band_weights=band_weights_tensor,
            return_metrics=True,
        )
        
        # Learn from edit
        success = float(metrics.get('throughput', torch.tensor(0.0)).mean().item()) > 0.3
        self.refiner.learn_from_edit(
            user_instruction,
            refined.get('band_weights', {}),
            success,
            metrics,
        )
        
        # Store in history
        self.edit_history.append({
            'instruction': user_instruction,
            'refined': refined,
            'metrics': metrics,
        })
        
        result_metrics = {
            'refined_instruction': refined,
            'metrics': metrics,
            'throughput': metrics.get('throughput', torch.tensor(0.0)),
        }
        
        if return_refined:
            result_metrics['refined_text'] = refined['refined_text']
        
        return edited_image, result_metrics
    
    def _is_complex_instruction(self, instruction: str) -> bool:
        """Check if instruction needs decomposition."""
        complex_keywords = [
            'make it', 'transform', 'convert', 'change to', 'turn into',
            'professional', 'artistic', 'dramatic', 'subtle', 'enhance overall',
        ]
        return any(keyword in instruction.lower() for keyword in complex_keywords)
    
    def _encode_instruction(self, instruction: str) -> torch.Tensor:
        """Encode instruction to embedding (simplified - use CLIP in practice)."""
        # In practice, use CLIP or similar for text encoding
        # This is a placeholder
        inputs = self.refiner.tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.refiner.model.transformer(**inputs)
            # Use mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1)  # [batch, d_model]
        
        return embedding
    
    def conversational_edit(
        self,
        image: torch.Tensor,
        user_message: str,
        image_description: Optional[str] = None,
    ) -> Tuple[torch.Tensor, str]:
        """
        Conversational editing with GPT-2 maintaining context.
        
        GPT-2 learns to:
        - Understand follow-up edits ("make it sharper")
        - Maintain context across turns
        - Suggest next edits
        """
        # Add to conversation
        self.conversation_context.append(f"User: {user_message}")
        
        # Check if this is a follow-up edit
        if self._is_followup(user_message):
            # GPT-2 interprets relative to previous edits
            refined_message = self._interpret_followup(user_message)
        else:
            refined_message = user_message
        
        # Edit image
        edited_image, metrics = self.edit_with_gpt2(
            image,
            refined_message,
            image_description,
        )
        
        # GPT-2 generates response
        response = self._generate_response(user_message, metrics)
        self.conversation_context.append(f"System: {response}")
        
        return edited_image, response
    
    def _is_followup(self, message: str) -> bool:
        """Check if message is a follow-up edit."""
        followup_keywords = ['more', 'less', 'also', 'and', 'then', 'next', 'further']
        return any(keyword in message.lower() for keyword in followup_keywords)
    
    def _interpret_followup(self, message: str) -> str:
        """Interpret follow-up edit using GPT-2."""
        context = "\n".join(self.conversation_context[-4:])  # Last 4 turns
        prompt = f"{context}\n\nInterpret this follow-up edit: {message}\nFull instruction:"
        
        inputs = self.refiner.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.refiner.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
            )
        
        interpreted = self.refiner.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return interpreted.split("Full instruction:")[-1].strip()
    
    def _generate_response(
        self,
        user_message: str,
        metrics: Dict,
    ) -> str:
        """Generate conversational response using GPT-2."""
        throughput = float(metrics.get('throughput', torch.tensor(0.0)).mean().item())
        
        prompt = f"User: {user_message}\n"
        prompt += f"Edit completed. Throughput: {throughput:.2f}\n"
        prompt += "System response:"
        
        inputs = self.refiner.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        
        with torch.no_grad():
            outputs = self.refiner.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
            )
        
        response = self.refiner.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("System response:")[-1].strip()
    
    def get_learned_knowledge(self) -> Dict[str, Any]:
        """Get what GPT-2 has learned."""
        return {
            'successful_patterns': len(self.refiner.successful_edits),
            'total_edits': len(self.edit_history),
            'conversation_turns': len(self.conversation_context),
            'top_patterns': self.refiner.successful_edits[-10:],
        }

