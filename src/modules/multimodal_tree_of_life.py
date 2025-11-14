"""
Multi-Modal Tree of Life Resonance Network

Combines text and image models in the Tree of Life:
- Some Sefirot are image models (Vision Transformers)
- Some Sefirot are text models (Language Models)
- Generates images throughout recursion
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Dict, Tuple, List, Union
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from resonance_transformer import ResonanceAttentionHead
except ImportError:
    try:
        from resonance_transformer.resonance_transformer import ResonanceAttentionHead
    except ImportError:
        ResonanceAttentionHead = None

try:
    from modules.pid_autopilot import PIDConfig
except ImportError:
    PIDConfig = None


class ImageDecoder(nn.Module):
    """
    Simple decoder to convert latent representations to images.
    
    Uses a simple MLP to decode features to RGB images.
    """
    
    def __init__(self, d_model: int, image_size: int = 64, channels: int = 3):
        super().__init__()
        self.d_model = d_model
        self.image_size = image_size
        self.channels = channels
        
        # Decoder: d_model -> image_size^2 * channels
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, image_size * image_size * channels),
            nn.Tanh(),  # Output in [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.
        
        Args:
            x: [batch, seq_len, d_model] or [batch, d_model]
            
        Returns:
            image: [batch, channels, image_size, image_size]
        """
        if x.dim() == 3:
            # Average over sequence length
            x = x.mean(dim=1)  # [batch, d_model]
        
        # Decode
        x_flat = self.decoder(x)  # [batch, image_size^2 * channels]
        
        # Reshape to image
        batch_size = x_flat.shape[0]
        image = x_flat.view(batch_size, self.channels, self.image_size, self.image_size)
        
        # Normalize to [0, 1]
        image = (image + 1.0) / 2.0
        image = torch.clamp(image, 0.0, 1.0)
        
        return image


class MultiModalTreeOfLifeNetwork(nn.Module):
    """
    Multi-modal Tree of Life network with image and text models.
    
    Sefirot assignments:
    - Image models: Chokhmah (Wisdom), Binah (Understanding), Tiferet (Beauty), Malkhut (Kingdom)
    - Text models: Keter (Crown), Chesed (Kindness), Gevurah (Severity), Netzach (Victory), Hod (Glory), Yesod (Foundation)
    """
    
    SEFIROT_NAMES = [
        "Keter",      # 0 - Crown (text)
        "Chokhmah",   # 1 - Wisdom (image)
        "Binah",      # 2 - Understanding (image)
        "Chesed",     # 3 - Kindness (text)
        "Gevurah",    # 4 - Severity (text)
        "Tiferet",    # 5 - Beauty (image)
        "Netzach",    # 6 - Victory (text)
        "Hod",        # 7 - Glory (text)
        "Yesod",      # 8 - Foundation (text)
        "Malkhut",    # 9 - Kingdom (image - final output)
    ]
    
    # Which Sefirot are image models
    IMAGE_SEFIROT = [1, 2, 5, 9]  # Chokhmah, Binah, Tiferet, Malkhut
    TEXT_SEFIROT = [0, 3, 4, 6, 7, 8]  # Others
    
    def __init__(
        self,
        d_model: int = 64,
        image_size: int = 64,
        use_gematria_frequencies: bool = True,
        use_four_worlds: bool = True,
        enable_tikkun: bool = True,
        n_sim_steps: int = 15,
        target_R: float = 0.6,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.n_sefirot = 10
        self.d_model = d_model
        self.image_size = image_size
        self.use_four_worlds = use_four_worlds
        self.device = device or torch.device("cpu")
        
        # Each Sefirah is a ResonanceAttentionHead
        pid_config = None
        if enable_tikkun and PIDConfig is not None:
            pid_config = PIDConfig(target_R=target_R)
        
        self.sefirot = nn.ModuleList([
            ResonanceAttentionHead(
                d_model=d_model,
                n_sim_steps=n_sim_steps,
                use_pid_autopilot=enable_tikkun and pid_config is not None,
                pid_config=pid_config,
                use_critical_tuning=True,
                use_adaptive_coupling=True,
                track_metrics=True,
                track_cdns=True,
            )
            for _ in range(self.n_sefirot)
        ])
        
        # Image decoders for image Sefirot
        self.image_decoders = nn.ModuleDict({
            str(i): ImageDecoder(d_model, image_size, channels=3)
            for i in self.IMAGE_SEFIROT
        })
        
        # Coupling matrix for paths
        self.path_coupling = nn.Parameter(
            torch.randn(self.n_sefirot, self.n_sefirot, device=self.device) * 0.1
        )
        
        # Path mask (22 paths)
        self.path_mask = self._build_path_mask()
        
        # Gematria-based natural frequencies
        if use_gematria_frequencies:
            self.natural_frequencies = self._compute_gematria_frequencies()
        else:
            self.natural_frequencies = nn.Parameter(
                torch.randn(self.n_sefirot, d_model, device=self.device) * 0.1
            )
        
        # Four Worlds layers
        if use_four_worlds:
            self.atzilut_layer = self._build_world_layer("atzilut")
            self.beriah_layer = self._build_world_layer("beriah")
            self.yetzirah_layer = self._build_world_layer("yetzirah")
            self.assiyah_layer = self._build_world_layer("assiyah")
    
    def _build_path_mask(self) -> torch.Tensor:
        """Build mask for the 22 paths."""
        from modules.tree_of_life_network import TreeOfLifeResonanceNetwork
        PATHS = TreeOfLifeResonanceNetwork.PATHS
        
        mask = torch.zeros(self.n_sefirot, self.n_sefirot, device=self.device)
        for i, j in PATHS:
            mask[i, j] = 1.0
            mask[j, i] = 1.0
        return mask
    
    def _compute_gematria_frequencies(self) -> nn.Parameter:
        """Compute natural frequencies based on Gematria values."""
        from modules.tree_of_life_network import TreeOfLifeResonanceNetwork
        GEMATRIA_VALUES = TreeOfLifeResonanceNetwork.GEMATRIA_VALUES
        
        gematria_values = torch.tensor([
            GEMATRIA_VALUES[name] for name in self.SEFIROT_NAMES
        ], dtype=torch.float32, device=self.device)
        
        max_val = gematria_values.max()
        frequencies = (gematria_values / max_val) * 2 * math.pi
        frequencies = frequencies.unsqueeze(-1).expand(-1, self.d_model)
        
        return nn.Parameter(frequencies)
    
    def _build_world_layer(self, world_name: str) -> nn.Module:
        """Build a layer for one of the Four Worlds."""
        if world_name == "atzilut":
            return nn.Identity()
        elif world_name == "beriah":
            return nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.Tanh(),
            )
        elif world_name == "yetzirah":
            return nn.Identity()
        elif world_name == "assiyah":
            return nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.d_model, self.d_model),
            )
        else:
            return nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        return_images: bool = True,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Process input through Tree of Life.
        
        Args:
            x: Input [batch, seq_len, d_model]
            return_images: If True, generate images from image Sefirot
            return_metrics: If True, return metrics dictionary
            
        Returns:
            output: Processed output [batch, seq_len, d_model]
            images: Dictionary of images from image Sefirot (if return_images)
            metrics: Optional metrics dictionary
        """
        batch_size, seq_len, _ = x.shape
        
        # Process through each Sefirah
        sefirot_outputs = []
        sefirot_metrics = []
        
        for i, sefirah in enumerate(self.sefirot):
            # Add natural frequency bias
            x_freq = x + self.natural_frequencies[i].unsqueeze(0).unsqueeze(0)
            
            # Process through Sefirah
            output = sefirah(x_freq, return_metrics=False)
            sefirot_outputs.append(output)
            
            # Collect metrics
            if hasattr(sefirah, '_last_metrics'):
                sefirot_metrics.append(sefirah._last_metrics)
        
        # Stack Sefirot outputs
        sefirot_tensor = torch.stack(sefirot_outputs, dim=1)  # [batch, n_sefirot, seq_len, d_model]
        
        # Apply path coupling
        coupling = self.path_coupling * self.path_mask
        coupling = torch.softmax(coupling, dim=-1)
        coupling_expanded = coupling.unsqueeze(0)
        
        # Route information through paths
        routed = torch.einsum('bij,bjsd->bisd', coupling_expanded, sefirot_tensor)
        
        # Process through Four Worlds
        if self.use_four_worlds:
            routed_atzilut = self.atzilut_layer(routed)
            routed_beriah = self.beriah_layer(routed_atzilut)
            routed_yetzirah = self.yetzirah_layer(routed_beriah)
            routed_assiyah = self.assiyah_layer(routed_yetzirah)
            routed = routed_assiyah
        
        # Final output: Malkhut
        final_output = routed[:, -1, :, :]  # [batch, seq_len, d_model]
        
        # Generate images from image Sefirot
        images = {}
        if return_images:
            for i in self.IMAGE_SEFIROT:
                sefirah_name = self.SEFIROT_NAMES[i]
                sefirah_output = routed[:, i, :, :]  # [batch, seq_len, d_model]
                
                # Decode to image
                decoder = self.image_decoders[str(i)]
                image = decoder(sefirah_output)  # [batch, channels, image_size, image_size]
                
                images[sefirah_name] = image
        
        # Build metrics
        metrics = None
        if return_metrics:
            metrics = {
                'sefirot_outputs': [out.detach().cpu() for out in sefirot_outputs],
                'path_coupling': coupling.detach().cpu(),
                'routed_output': routed.detach().cpu(),
                'sefirot_metrics': sefirot_metrics,
            }
        
        return final_output, images, metrics
    
    def get_sefirah_names(self) -> List[str]:
        """Get list of Sefirah names."""
        return self.SEFIROT_NAMES
    
    def get_image_sefirot(self) -> List[int]:
        """Get list of image Sefirah indices."""
        return self.IMAGE_SEFIROT


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: [channels, height, width] or [batch, channels, height, width]
        
    Returns:
        PIL Image or list of PIL Images
    """
    if tensor.dim() == 4:
        # Batch
        images = []
        for i in range(tensor.shape[0]):
            img_tensor = tensor[i]
            img_array = (img_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
            img_array = np.transpose(img_array, (1, 2, 0))  # CHW -> HWC
            images.append(Image.fromarray(img_array))
        return images
    else:
        # Single image
        img_array = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        img_array = np.transpose(img_array, (1, 2, 0))
        return Image.fromarray(img_array)

