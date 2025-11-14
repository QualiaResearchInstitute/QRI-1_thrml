"""
Tree of Life Resonance Network

A resonance network based on the Kabbalistic Tree of Life:
- 10 Sefirot as oscillators
- 22 paths (Netivot) as coupling matrices
- Four Worlds as network layers
- Tikkun (repair) as criticality maintenance
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from resonance_transformer import ResonanceAttentionHead
except ImportError:
    # Fallback import
    try:
        from resonance_transformer.resonance_transformer import ResonanceAttentionHead
    except ImportError:
        raise ImportError("Could not import ResonanceAttentionHead")

try:
    from modules.pid_autopilot import PIDConfig
except ImportError:
    PIDConfig = None


class TreeOfLifeResonanceNetwork(nn.Module):
    """
    Resonance network based on the Kabbalistic Tree of Life.
    
    The Tree of Life consists of:
    - 10 Sefirot (divine emanations/attributes)
    - 22 paths (Netivot) connecting the Sefirot
    - Four Worlds (Atzilut, Beriah, Yetzirah, Assiyah)
    
    Each Sefirah is implemented as a ResonanceAttentionHead (oscillator),
    and the paths are coupling matrices that route information between Sefirot.
    """
    
    SEFIROT_NAMES = [
        "Keter",      # 0 - Crown (the source)
        "Chokhmah",   # 1 - Wisdom (masculine, active)
        "Binah",      # 2 - Understanding (feminine, receptive)
        "Chesed",     # 3 - Kindness (mercy, expansion)
        "Gevurah",    # 4 - Severity (judgment, contraction)
        "Tiferet",    # 5 - Beauty (harmony, balance)
        "Netzach",    # 6 - Victory (endurance)
        "Hod",        # 7 - Glory (splendor)
        "Yesod",      # 8 - Foundation (connection)
        "Malkhut",    # 9 - Kingdom (manifestation)
    ]
    
    # Gematria values for each Sefirah (Hebrew letter values)
    GEMATRIA_VALUES = {
        "Keter": 620,      # כתר
        "Chokhmah": 73,    # חכמה
        "Binah": 67,       # בינה
        "Chesed": 72,      # חסד
        "Gevurah": 216,    # גבורה
        "Tiferet": 1081,   # תפארת
        "Netzach": 148,    # נצח
        "Hod": 15,         # הוד
        "Yesod": 80,       # יסוד
        "Malkhut": 496,    # מלכות
    }
    
    # 22 paths of the Tree of Life (simplified structure)
    # Full Tree has specific connections following the traditional diagram
    PATHS = [
        # Upper triangle (Keter, Chokhmah, Binah)
        (0, 1),  # Keter → Chokhmah
        (0, 2),  # Keter → Binah
        (1, 2),  # Chokhmah ↔ Binah (Da'at - knowledge, hidden Sefirah)
        
        # Middle triangle (Chesed, Gevurah, Tiferet)
        (1, 3),  # Chokhmah → Chesed
        (2, 4),  # Binah → Gevurah
        (3, 5),  # Chesed → Tiferet
        (4, 5),  # Gevurah → Tiferet
        (3, 4),  # Chesed ↔ Gevurah (balance)
        
        # Lower triangle (Netzach, Hod, Yesod)
        (5, 6),  # Tiferet → Netzach
        (5, 7),  # Tiferet → Hod
        (6, 8),  # Netzach → Yesod
        (7, 8),  # Hod → Yesod
        (6, 7),  # Netzach ↔ Hod (balance)
        
        # Foundation to Kingdom
        (8, 9),  # Yesod → Malkhut
        
        # Additional paths for completeness (22 total)
        (0, 5),  # Keter → Tiferet (direct path)
        (1, 5),  # Chokhmah → Tiferet
        (2, 5),  # Binah → Tiferet
        (3, 6),  # Chesed → Netzach
        (4, 7),  # Gevurah → Hod
        (5, 8),  # Tiferet → Yesod
        (5, 9),  # Tiferet → Malkhut
        (6, 9),  # Netzach → Malkhut
        (7, 9),  # Hod → Malkhut
    ]
    
    def __init__(
        self,
        d_model: int = 64,
        n_sefirot: int = 10,
        use_gematria_frequencies: bool = True,
        use_four_worlds: bool = True,
        enable_tikkun: bool = True,
        n_sim_steps: int = 15,
        target_R: float = 0.6,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Tree of Life Resonance Network.
        
        Args:
            d_model: Model dimension
            n_sefirot: Number of Sefirot (default: 10)
            use_gematria_frequencies: Use Gematria values for natural frequencies
            use_four_worlds: Process through Four Worlds layers
            enable_tikkun: Enable Tikkun (criticality maintenance via PID)
            n_sim_steps: Number of Kuramoto simulation steps
            target_R: Target order parameter for criticality (default: 0.6)
            device: Device to run on
        """
        super().__init__()
        
        self.n_sefirot = n_sefirot
        self.d_model = d_model
        self.use_four_worlds = use_four_worlds
        self.device = device or torch.device("cpu")
        
        # Each Sefirah is a ResonanceAttentionHead (oscillator)
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
            for _ in range(n_sefirot)
        ])
        
        # Coupling matrix for paths (22 paths)
        self.path_coupling = nn.Parameter(
            torch.randn(n_sefirot, n_sefirot, device=self.device) * 0.1
        )
        
        # Mask paths (only 22 paths are active)
        self.path_mask = self._build_path_mask()
        
        # Gematria-based natural frequencies
        if use_gematria_frequencies:
            self.natural_frequencies = self._compute_gematria_frequencies()
        else:
            self.natural_frequencies = nn.Parameter(
                torch.randn(n_sefirot, d_model, device=self.device) * 0.1
            )
        
        # Four Worlds layers
        if use_four_worlds:
            self.atzilut_layer = self._build_world_layer("atzilut")  # R ≈ 1.0
            self.beriah_layer = self._build_world_layer("beriah")    # R ≈ 0.8-0.9
            self.yetzirah_layer = self._build_world_layer("yetzirah") # R ≈ 0.6
            self.assiyah_layer = self._build_world_layer("assiyah")  # R ≈ 0.3-0.5
    
    def _build_path_mask(self) -> torch.Tensor:
        """Build mask for the 22 paths."""
        mask = torch.zeros(self.n_sefirot, self.n_sefirot, device=self.device)
        for i, j in self.PATHS:
            mask[i, j] = 1.0
            mask[j, i] = 1.0  # Bidirectional paths
        return mask
    
    def _compute_gematria_frequencies(self) -> nn.Parameter:
        """
        Compute natural frequencies based on Gematria values.
        
        Gematria assigns numerical values to Hebrew letters.
        These values determine the natural frequencies of each Sefirah.
        """
        gematria_values = torch.tensor([
            self.GEMATRIA_VALUES[name] for name in self.SEFIROT_NAMES
        ], dtype=torch.float32, device=self.device)
        
        # Normalize to [0, 2π] range
        max_val = gematria_values.max()
        frequencies = (gematria_values / max_val) * 2 * math.pi
        
        # Expand to d_model dimensions
        frequencies = frequencies.unsqueeze(-1).expand(-1, self.d_model)
        
        return nn.Parameter(frequencies)
    
    def _build_world_layer(self, world_name: str) -> nn.Module:
        """
        Build a layer for one of the Four Worlds.
        
        Each world operates at a different synchronization level:
        - Atzilut: Pure phase dynamics, R ≈ 1.0 (full sync)
        - Beriah: Slight amplitude modulation, R ≈ 0.8-0.9
        - Yetzirah: Full phase-amplitude dynamics, R ≈ 0.6 (critical)
        - Assiyah: Noise and dissipation, R ≈ 0.3-0.5
        """
        if world_name == "atzilut":
            # Pure phase dynamics, no amplitude decay
            return nn.Identity()
        elif world_name == "beriah":
            # Slight amplitude modulation
            return nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.Tanh(),
            )
        elif world_name == "yetzirah":
            # Full phase-amplitude dynamics (default)
            return nn.Identity()
        elif world_name == "assiyah":
            # Noise and dissipation
            return nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.d_model, self.d_model),
            )
        else:
            return nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Process input through Tree of Life.
        
        Information flows:
        1. Through each Sefirah (oscillator)
        2. Through paths (coupling matrices)
        3. Through Four Worlds (layers)
        4. To Malkhut (Kingdom - manifestation)
        
        Args:
            x: Input [batch, seq_len, d_model]
            return_metrics: If True, return metrics dictionary
            
        Returns:
            output: Processed output [batch, seq_len, d_model]
            metrics: Optional metrics dictionary
        """
        batch_size, seq_len, _ = x.shape
        
        # Process through each Sefirah
        sefirot_outputs = []
        sefirot_metrics = []
        
        for i, sefirah in enumerate(self.sefirot):
            # Add natural frequency bias (Gematria-based)
            x_freq = x + self.natural_frequencies[i].unsqueeze(0).unsqueeze(0)
            
            # Process through Sefirah
            output = sefirah(x_freq, return_metrics=False)
            sefirot_outputs.append(output)
            
            # Collect metrics if available
            if hasattr(sefirah, '_last_metrics'):
                sefirot_metrics.append(sefirah._last_metrics)
        
        # Stack Sefirot outputs [batch, n_sefirot, seq_len, d_model]
        sefirot_tensor = torch.stack(sefirot_outputs, dim=1)
        
        # Apply path coupling (22 paths)
        coupling = self.path_coupling * self.path_mask
        # Normalize coupling matrix
        coupling = torch.softmax(coupling, dim=-1)
        
        # Route information through paths
        # coupling: [n_sefirot, n_sefirot]
        # sefirot_tensor: [batch, n_sefirot, seq_len, d_model]
        # Expand coupling for batch dimension
        coupling_expanded = coupling.unsqueeze(0)  # [1, n_sefirot, n_sefirot]
        
        # Route: [batch, n_sefirot, seq_len, d_model]
        routed = torch.einsum('bij,bjsd->bisd', coupling_expanded, sefirot_tensor)
        
        # Process through Four Worlds (if enabled)
        if self.use_four_worlds:
            routed_atzilut = self.atzilut_layer(routed)
            routed_beriah = self.beriah_layer(routed_atzilut)
            routed_yetzirah = self.yetzirah_layer(routed_beriah)
            routed_assiyah = self.assiyah_layer(routed_yetzirah)
            routed = routed_assiyah
        else:
            routed = routed
        
        # Final output: Malkhut (Kingdom - manifestation)
        # Take the last Sefirah (Malkhut) as the final output
        final_output = routed[:, -1, :, :]  # [batch, seq_len, d_model]
        
        # Build metrics dictionary
        metrics = None
        if return_metrics:
            metrics = {
                'sefirot_outputs': [out.detach().cpu() for out in sefirot_outputs],
                'path_coupling': coupling.detach().cpu(),
                'routed_output': routed.detach().cpu(),
                'sefirot_metrics': sefirot_metrics,
            }
        
        return final_output, metrics
    
    def get_sefirah_names(self) -> List[str]:
        """Get list of Sefirah names."""
        return self.SEFIROT_NAMES
    
    def get_path_structure(self) -> List[Tuple[int, int]]:
        """Get list of paths (connections between Sefirot)."""
        return self.PATHS


def create_tree_of_life_network(
    d_model: int = 64,
    use_gematria: bool = True,
    enable_tikkun: bool = True,
    device: Optional[torch.device] = None,
) -> TreeOfLifeResonanceNetwork:
    """
    Convenience function to create a Tree of Life network.
    
    Args:
        d_model: Model dimension
        use_gematria: Use Gematria-based frequencies
        enable_tikkun: Enable Tikkun (criticality maintenance)
        device: Device to run on
        
    Returns:
        TreeOfLifeResonanceNetwork instance
    """
    return TreeOfLifeResonanceNetwork(
        d_model=d_model,
        use_gematria_frequencies=use_gematria,
        use_four_worlds=True,
        enable_tikkun=enable_tikkun,
        device=device,
    )

