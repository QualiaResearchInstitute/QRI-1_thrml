"""
CRV Oscillator Feature Extractor: Maps oscillator dynamics to interpretable features.

Extracts discrete, interpretable features from continuous oscillator dynamics
(phases, amplitudes, coupling) for use in attribution graphs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from modules.msf_analysis import MSFAnalyzer


class OscillatorFeatureExtractor:
    """
    Extracts interpretable features from oscillator dynamics.
    
    Maps continuous phases/amplitudes to discrete feature nodes:
    - Phase coherence clusters (synchronized groups)
    - Amplitude peaks (highly active oscillators)
    - Coupling hubs (highly connected oscillators)
    - CDNS-derived features (consonance/dissonance patterns)
    """
    
    def __init__(
        self,
        n_phase_clusters: int = 8,
        amplitude_threshold_percentile: float = 75.0,
        coupling_hub_percentile: float = 90.0,
    ):
        """
        Initialize feature extractor.
        
        Args:
            n_phase_clusters: Number of phase coherence clusters to identify
            amplitude_threshold_percentile: Percentile for amplitude peak threshold
            coupling_hub_percentile: Percentile for coupling hub threshold
        """
        self.n_phase_clusters = n_phase_clusters
        self.amplitude_threshold_percentile = amplitude_threshold_percentile
        self.coupling_hub_percentile = coupling_hub_percentile
    
    def extract_features(
        self,
        phases: torch.Tensor,
        amplitudes: Optional[torch.Tensor] = None,
        coupling_matrix: Optional[torch.Tensor] = None,
        cdns: Optional[Dict[str, torch.Tensor]] = None,
        msf_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract interpretable oscillator features.
        
        Args:
            phases: Phase values [batch, heads, seq_len, n_sim_steps] or [batch, seq_len, n_sim_steps]
            amplitudes: Amplitude values (same shape as phases)
            coupling_matrix: Coupling matrix [batch, heads, seq_len, seq_len] or [batch, seq_len, seq_len]
            cdns: CDNS metrics dictionary
        
        Returns:
            Dictionary with extracted features:
            - phase_clusters: Cluster assignments [batch, seq_len]
            - phase_coherence: Coherence within clusters [batch, n_clusters]
            - amplitude_peaks: Binary mask for amplitude peaks [batch, seq_len]
            - coupling_hubs: Binary mask for coupling hubs [batch, seq_len]
            - cdns_features: CDNS-derived features
        """
        features = {}
        
        # Handle different input shapes
        if phases.dim() == 4:  # [batch, heads, seq_len, n_sim_steps]
            # Average over heads for now (can be enhanced to per-head features)
            phases = phases.mean(dim=1)  # [batch, seq_len, n_sim_steps]
            if amplitudes is not None:
                amplitudes = amplitudes.mean(dim=1)
            if coupling_matrix is not None and coupling_matrix.dim() == 4:
                coupling_matrix = coupling_matrix.mean(dim=1)  # [batch, seq_len, seq_len]
        
        # Use final timestep for clustering (or average over time)
        if phases.dim() == 3:  # [batch, seq_len, n_sim_steps]
            phases_final = phases[:, :, -1]  # [batch, seq_len]
        else:
            phases_final = phases  # [batch, seq_len]
        
        # Extract phase coherence clusters
        phase_clusters, phase_coherence = self._extract_phase_clusters(phases_final)
        features['phase_clusters'] = phase_clusters
        features['phase_coherence'] = phase_coherence
        
        # Extract amplitude peaks
        if amplitudes is not None:
            if amplitudes.dim() == 3:
                amplitudes_final = amplitudes[:, :, -1]  # [batch, seq_len]
            else:
                amplitudes_final = amplitudes
            
            amplitude_peaks = self._extract_amplitude_peaks(amplitudes_final)
            features['amplitude_peaks'] = amplitude_peaks
        
        # Extract coupling hubs
        if coupling_matrix is not None:
            coupling_hubs = self._extract_coupling_hubs(coupling_matrix)
            features['coupling_hubs'] = coupling_hubs
        
        # Extract CDNS features
        if cdns is not None:
            cdns_features = self._extract_cdns_features(cdns, phases_final)
            features['cdns_features'] = cdns_features

        # MSF stability analysis (optional, requires coupling_matrix)
        if coupling_matrix is not None:
            try:
                # Determine shapes: if coupling_matrix dim==4 (heads), already averaged above
                # Ensure tensor on CPU-safe for any .item() calls later
                msf_cfg = msf_params or {}
                # Default dt consistent with ResonanceAttentionHead default
                dt = float(msf_cfg.get('dt', 0.01))
                alpha = float(msf_cfg.get('alpha', 0.0))
                # Effective sigma: user-specified or mean |K| as a proxy
                if 'sigma' in msf_cfg and msf_cfg['sigma'] is not None:
                    sigma_eff = float(msf_cfg['sigma'])
                else:
                    K_safe = torch.nan_to_num(coupling_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                    try:
                        sigma_eff = float(K_safe.abs().mean().detach().item())
                    except Exception:
                        sigma_eff = float(0.0)
                analyzer = MSFAnalyzer()
                msf_out = analyzer.compute_transverse_lyapunov_exponents(
                    coupling_matrix=coupling_matrix,
                    cluster_assignments=phase_clusters,
                    dt=dt,
                    sigma=sigma_eff,
                    alpha=alpha,
                )
                regimes = analyzer.predict_dynamical_regime(msf_out['Lambda'])
                sigma_crit = analyzer.predict_critical_coupling(
                    coupling_matrix=coupling_matrix,
                    cluster_assignments=phase_clusters,
                    dt=dt,
                    alpha=alpha,
                )
                features['msf'] = {
                    'Lambda': msf_out['Lambda'],
                    'lambda_max': msf_out['lambda_max'],
                    'stable_flags': msf_out['stable_flags'],
                    'regimes': regimes,
                    'sigma_crit': sigma_crit,
                    'gamma': msf_out.get('gamma', None),
                    'cos_alpha': msf_out.get('cos_alpha', None),
                    'sigma_eff': sigma_eff,
                    'alpha_used': alpha,
                    'dt_used': dt,
                }
            except Exception:
                # Fail gracefully without breaking feature extraction
                pass
        
        return features
    
    def _extract_phase_clusters(
        self,
        phases: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cluster phases by synchronization (coherence).
        
        Args:
            phases: Phase values [batch, seq_len]
        
        Returns:
            cluster_assignments: [batch, seq_len] cluster IDs
            coherence: [batch, n_clusters] coherence within each cluster
        """
        batch_size, seq_len = phases.shape
        cluster_assignments = torch.zeros(batch_size, seq_len, dtype=torch.long)
        coherence = torch.zeros(batch_size, self.n_phase_clusters)
        
        for b in range(batch_size):
            phases_b = phases[b].detach().cpu().numpy()
            
            # Convert phases to 2D coordinates [cos(θ), sin(θ)]
            coords = np.stack([np.cos(phases_b), np.sin(phases_b)], axis=-1)
            
            # Cluster using KMeans
            try:
                kmeans = KMeans(n_clusters=self.n_phase_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(coords)
                cluster_assignments[b] = torch.from_numpy(clusters)
                
                # Compute coherence within each cluster
                for c in range(self.n_phase_clusters):
                    mask = clusters == c
                    if mask.sum() > 0:
                        cluster_phases = phases_b[mask]
                        # Coherence = |mean(e^{iθ})|
                        coherence_val = np.abs(np.mean(np.exp(1j * cluster_phases)))
                        coherence[b, c] = coherence_val
            except Exception:
                # Fallback: assign all to cluster 0
                cluster_assignments[b] = 0
                coherence[b, 0] = 1.0
        
        return cluster_assignments, coherence
    
    def _extract_amplitude_peaks(
        self,
        amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Identify amplitude peaks (highly active oscillators).
        
        Args:
            amplitudes: Amplitude values [batch, seq_len]
        
        Returns:
            Binary mask [batch, seq_len] where True indicates amplitude peak
        """
        batch_size, seq_len = amplitudes.shape
        peaks = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        for b in range(batch_size):
            amps_b = amplitudes[b].detach().cpu().numpy()
            threshold = np.percentile(amps_b, self.amplitude_threshold_percentile)
            peaks[b] = torch.from_numpy(amps_b >= threshold)
        
        return peaks
    
    def _extract_coupling_hubs(
        self,
        coupling_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Identify coupling hubs (highly connected oscillators).
        
        Args:
            coupling_matrix: Coupling matrix [batch, seq_len, seq_len]
        
        Returns:
            Binary mask [batch, seq_len] where True indicates coupling hub
        """
        batch_size, seq_len, _ = coupling_matrix.shape
        hubs = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        for b in range(batch_size):
            coupling_b = coupling_matrix[b].detach().cpu().numpy()
            # Compute node strength (sum of incoming + outgoing connections)
            node_strength = coupling_b.sum(axis=0) + coupling_b.sum(axis=1)
            threshold = np.percentile(node_strength, self.coupling_hub_percentile)
            hubs[b] = torch.from_numpy(node_strength >= threshold)
        
        return hubs
    
    def _extract_cdns_features(
        self,
        cdns: Dict[str, torch.Tensor],
        phases: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract CDNS-derived features.
        
        Args:
            cdns: CDNS metrics dictionary
            phases: Phase values [batch, seq_len]
        
        Returns:
            Dictionary with CDNS features
        """
        features = {}
        
        # Extract individual CDNS metrics
        for key in ['consonance', 'dissonance', 'noise', 'signal']:
            if key in cdns:
                val = cdns[key]
                if isinstance(val, torch.Tensor):
                    # Average over spatial dimensions if needed
                    while val.dim() > 1:
                        val = val.mean(dim=-1)
                    features[f'cdns_{key}'] = val
        
        # Compute phase spread (variance of phases)
        if phases.dim() == 2:  # [batch, seq_len]
            # Convert to complex representation
            complex_phases = torch.exp(1j * phases)
            phase_spread = torch.var(torch.angle(complex_phases), dim=-1)
            features['phase_spread'] = phase_spread
        
        return features
    
    def get_feature_nodes(
        self,
        features: Dict[str, torch.Tensor],
        batch_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Convert extracted features to discrete feature nodes for attribution graph.
        
        Args:
            features: Extracted features dictionary
            batch_idx: Batch index to extract nodes for
        
        Returns:
            List of feature node dictionaries with:
            - node_id: Unique identifier
            - node_type: Type of feature (phase_cluster, amplitude_peak, coupling_hub, etc.)
            - position: Position in sequence (if applicable)
            - value: Feature value
        """
        nodes = []
        node_id = 0
        
        # Phase cluster nodes
        if 'phase_clusters' in features:
            clusters = features['phase_clusters'][batch_idx]
            coherence = features['phase_coherence'][batch_idx]
            
            for cluster_id in range(self.n_phase_clusters):
                mask = clusters == cluster_id
                positions = torch.where(mask)[0].tolist()
                if len(positions) > 0:
                    nodes.append({
                        'node_id': node_id,
                        'node_type': 'phase_cluster',
                        'cluster_id': int(cluster_id),
                        'positions': positions,
                        'coherence': float(coherence[cluster_id].item()),
                        'size': len(positions),
                    })
                    node_id += 1
        
        # Amplitude peak nodes
        if 'amplitude_peaks' in features:
            peaks = features['amplitude_peaks'][batch_idx]
            peak_positions = torch.where(peaks)[0].tolist()
            for pos in peak_positions:
                nodes.append({
                    'node_id': node_id,
                    'node_type': 'amplitude_peak',
                    'position': int(pos),
                })
                node_id += 1
        
        # Coupling hub nodes
        if 'coupling_hubs' in features:
            hubs = features['coupling_hubs'][batch_idx]
            hub_positions = torch.where(hubs)[0].tolist()
            for pos in hub_positions:
                nodes.append({
                    'node_id': node_id,
                    'node_type': 'coupling_hub',
                    'position': int(pos),
                })
                node_id += 1
        
        # CDNS feature nodes (global features, not position-specific)
        if 'cdns_features' in features:
            cdns_feat = features['cdns_features']
            for key, value in cdns_feat.items():
                if isinstance(value, torch.Tensor):
                    val = value[batch_idx].item() if value.dim() > 0 else value.item()
                    nodes.append({
                        'node_id': node_id,
                        'node_type': 'cdns_feature',
                        'feature_name': key,
                        'value': float(val),
                    })
                    node_id += 1
        
        return nodes
