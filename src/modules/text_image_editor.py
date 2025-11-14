"""
Text-Programmable Image Editor

Allows editing images through natural language descriptions and text commands.
Maps text descriptions to frequency band manipulations in the consciousness circuit.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image

from modules.image_circuit_editor import (
    ImageCircuitMapper,
    FrequencyBandEditor,
    create_band_presets,
)


class TextImageEditor:
    """
    Text-programmable image editor that interprets natural language edits.
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        n_frequency_bands: int = 4,
    ):
        self.mapper = ImageCircuitMapper(
            model_name=model_name,
            patch_size=16,
            n_frequency_bands=n_frequency_bands,
        )
        self.editor = FrequencyBandEditor(
            mapper=self.mapper,
            n_frequency_bands=n_frequency_bands,
        )
        self.presets = create_band_presets()
        
        # Text-to-edit mapping patterns
        self.edit_patterns = self._build_edit_patterns()
    
    def _build_edit_patterns(self) -> Dict[str, Dict]:
        """Build pattern matching rules for text edits."""
        return {
            # Sharpening
            r'sharpen|sharp|enhance.*detail|make.*crisp|increase.*clarity': {
                'preset': 'sharpen',
                'band_weights': {0: 0.8, 1: 1.0, 2: 1.3, 3: 1.5},
            },
            # Smoothing
            r'smooth|soften|blur|reduce.*noise|make.*smooth': {
                'preset': 'smooth',
                'band_weights': {0: 1.0, 1: 0.7, 2: 0.3, 3: 0.1},
            },
            # Enhance low frequencies (structure)
            r'enhance.*structure|improve.*composition|strengthen.*colors|boost.*global': {
                'preset': 'enhance_low',
                'band_weights': {0: 1.5, 1: 1.0, 2: 0.8, 3: 0.5},
            },
            # Enhance mid frequencies (textures)
            r'enhance.*texture|improve.*pattern|strengthen.*texture|boost.*texture': {
                'preset': 'enhance_mid',
                'band_weights': {0: 0.8, 1: 1.5, 2: 1.5, 3: 0.8},
            },
            # Enhance high frequencies (details)
            r'enhance.*detail|improve.*edge|strengthen.*detail|boost.*detail': {
                'preset': 'enhance_high',
                'band_weights': {0: 0.5, 1: 0.8, 2: 1.0, 3: 1.5},
            },
            # Low-pass filter
            r'low.*pass|remove.*detail|keep.*structure|only.*low': {
                'preset': 'low_pass',
                'band_weights': {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0},
            },
            # High-pass filter
            r'high.*pass|remove.*structure|keep.*detail|only.*high': {
                'preset': 'high_pass',
                'band_weights': {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0},
            },
            # Intensity modifiers
            r'slightly|a bit|a little': {'intensity': 0.5},
            r'moderately|somewhat': {'intensity': 0.7},
            r'strongly|heavily|very|extremely': {'intensity': 1.5},
            r'very.*slightly|just.*a.*bit': {'intensity': 0.3},
        }
    
    def parse_edit_command(self, text: str) -> Dict[str, Any]:
        """
        Parse a text edit command into frequency band configuration.
        
        Args:
            text: Natural language edit description
        
        Returns:
            Dictionary with band_weights and other parameters
        """
        text_lower = text.lower().strip()
        
        # Initialize result
        result = {
            'band_weights': None,
            'intensity': 1.0,
            'n_iterations': 5,
            'preset': None,
        }
        
        # Extract intensity modifier
        intensity = 1.0
        for pattern, config in self.edit_patterns.items():
            if re.search(pattern, text_lower):
                if 'intensity' in config:
                    intensity = config['intensity']
                elif 'band_weights' in config:
                    result['band_weights'] = config['band_weights'].copy()
                    result['preset'] = config.get('preset')
        
        # Extract numeric values for bands
        band_matches = re.findall(r'band[_\s]*(\d+)[:\s=]+([\d.]+)', text_lower)
        if band_matches:
            if result['band_weights'] is None:
                result['band_weights'] = {}
            for band_idx, weight in band_matches:
                result['band_weights'][int(band_idx)] = float(weight) * intensity
        
        # Extract iteration count
        iter_match = re.search(r'(\d+)\s*iterations?', text_lower)
        if iter_match:
            result['n_iterations'] = int(iter_match.group(1))
        
        # Apply intensity to band weights
        if result['band_weights']:
            for band_idx in result['band_weights']:
                # Scale around 1.0 (neutral)
                original = result['band_weights'][band_idx]
                if original > 1.0:
                    # Amplify: scale up
                    result['band_weights'][band_idx] = 1.0 + (original - 1.0) * intensity
                elif original < 1.0:
                    # Attenuate: scale down
                    result['band_weights'][band_idx] = 1.0 - (1.0 - original) * intensity
        
        # If no band weights found, try preset matching
        if result['band_weights'] is None:
            for preset_name, preset_weights in self.presets.items():
                if preset_name.replace('_', ' ') in text_lower or preset_name in text_lower:
                    result['band_weights'] = preset_weights.copy()
                    result['preset'] = preset_name
                    break
        
        return result
    
    def edit_from_text(
        self,
        image: Image.Image,
        edit_text: str,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Edit image based on text description.
        
        Args:
            image: Input PIL Image
            edit_text: Natural language edit description
        
        Returns:
            edited_image: Edited PIL Image
            metrics: Dictionary with editing metrics and parsed command
        """
        # Parse edit command
        edit_config = self.parse_edit_command(edit_text)
        
        # Apply edit
        edited_image, metrics = self.editor.edit_image(
            image,
            band_weights=edit_config['band_weights'],
            n_circuit_iterations=edit_config['n_iterations'],
        )
        
        # Add parsed command to metrics
        metrics['parsed_command'] = edit_config
        metrics['edit_text'] = edit_text
        
        return edited_image, metrics
    
    def edit_from_script(
        self,
        image: Image.Image,
        script: Union[str, List[str]],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Apply multiple edits from a script (list of commands).
        
        Args:
            image: Input PIL Image
            script: String (newline-separated) or list of edit commands
        
        Returns:
            edited_image: Final edited Image
            metrics: Dictionary with all edit metrics
        """
        if isinstance(script, str):
            commands = [line.strip() for line in script.split('\n') if line.strip()]
        else:
            commands = script
        
        current_image = image
        all_metrics = []
        
        for i, command in enumerate(commands):
            if command.startswith('#'):  # Comment
                continue
            
            print(f"  [{i+1}/{len(commands)}] {command}")
            current_image, metrics = self.edit_from_text(current_image, command)
            all_metrics.append({
                'command': command,
                'metrics': metrics,
            })
        
        return current_image, {'all_edits': all_metrics}


class ImageEditScript:
    """
    Script-based image editing with chaining and variables.
    """
    
    def __init__(self, editor: TextImageEditor):
        self.editor = editor
        self.variables: Dict[str, Image.Image] = {}
    
    def execute(
        self,
        script: str,
        input_image: Optional[Image.Image] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Execute an image editing script.
        
        Script syntax:
        - `load <name> <path>`: Load image into variable
        - `edit <name> <command>`: Edit variable
        - `save <name> <path>`: Save variable to file
        - `# comment`: Comments
        - `set <name> = <command>`: Create/edit variable
        
        Example:
            load img1 photo.jpg
            edit img1 sharpen
            edit img1 slightly smooth
            save img1 output.jpg
        """
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        
        result_image = input_image
        results = {}
        
        for line in lines:
            if line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            command = parts[0].lower()
            
            if command == 'load':
                if len(parts) < 3:
                    continue
                var_name = parts[1]
                path = ' '.join(parts[2:])
                self.variables[var_name] = Image.open(path)
                print(f"Loaded {var_name} from {path}")
            
            elif command == 'edit':
                if len(parts) < 3:
                    continue
                var_name = parts[1]
                edit_command = ' '.join(parts[2:])
                
                if var_name not in self.variables:
                    print(f"Warning: Variable {var_name} not found")
                    continue
                
                edited, metrics = self.editor.edit_from_text(
                    self.variables[var_name],
                    edit_command,
                )
                self.variables[var_name] = edited
                results[var_name] = metrics
                print(f"Edited {var_name}: {edit_command}")
            
            elif command == 'save':
                if len(parts) < 3:
                    continue
                var_name = parts[1]
                path = ' '.join(parts[2:])
                
                if var_name not in self.variables:
                    print(f"Warning: Variable {var_name} not found")
                    continue
                
                self.variables[var_name].save(path)
                print(f"Saved {var_name} to {path}")
                result_image = self.variables[var_name]
            
            elif command == 'set':
                # set var_name = edit_command
                if '=' not in line:
                    continue
                var_part, edit_part = line.split('=', 1)
                var_name = var_part.replace('set', '').strip()
                edit_command = edit_part.strip()
                
                if var_name not in self.variables:
                    print(f"Warning: Variable {var_name} not found")
                    continue
                
                edited, metrics = self.editor.edit_from_text(
                    self.variables[var_name],
                    edit_command,
                )
                self.variables[var_name] = edited
        
        return result_image, results


def create_edit_prompt_template() -> str:
    """Create a template for LLM-based edit generation."""
    return """
You are an image editing assistant. Given an image and an edit request, generate a frequency band configuration.

Available frequency bands:
- Band 0 (Low): Global structure, colors, composition
- Band 1 (Mid-low): Large-scale textures, patterns
- Band 2 (Mid-high): Fine textures, details
- Band 3 (High): Edges, sharp details, noise

Edit request: {edit_request}

Generate a JSON configuration:
{{
    "band_weights": {{
        "0": <weight for band 0 (0.0-2.0)>,
        "1": <weight for band 1 (0.0-2.0)>,
        "2": <weight for band 2 (0.0-2.0)>,
        "3": <weight for band 3 (0.0-2.0)>
    }},
    "n_iterations": <number of circuit iterations (1-10)>
}}

Example for "sharpen the image":
{{
    "band_weights": {{"0": 0.8, "1": 1.0, "2": 1.3, "3": 1.5}},
    "n_iterations": 5
}}
"""

