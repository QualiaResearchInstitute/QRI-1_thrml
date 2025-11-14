"""
Telemetry Logger for Resonance Transformer

Captures CDNS metrics, order parameter variance, spectral bands, and other
resonance dynamics metrics per batch for analysis and auxiliary loss computation.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Any
import torch
import numpy as np


class TelemetryLogger:
    """
    Logs resonance metrics per batch for analysis and training feedback.
    
    Tracks:
    - CDNS metrics (consonance, dissonance, noise, signal)
    - Order parameter variance
    - Spectral band energies
    - Criticality indices
    - Harmonic components (if Hodge decomposition enabled)
    """
    
    def __init__(self, log_dir: Optional[str] = None, log_every_n_batches: int = 1):
        """
        Initialize telemetry logger.
        
        Args:
            log_dir: Directory to save telemetry logs (None = no file logging)
            log_every_n_batches: Log every N batches (1 = every batch)
        """
        self.log_dir = log_dir
        self.log_every_n_batches = log_every_n_batches
        self.batch_count = 0
        
        # In-memory storage
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        
        # Create log directory if needed
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
    
    def log_batch(
        self,
        metrics: Dict[str, Any],
        batch_idx: int,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """
        Log metrics from a single batch.
        
        Args:
            metrics: Dictionary of metrics from resonance heads
            batch_idx: Current batch index
            epoch: Current epoch
        
        Returns:
            Aggregated scalar metrics dictionary
        """
        self.batch_count += 1
        
        # Only log every N batches
        if self.batch_count % self.log_every_n_batches != 0:
            return {}
        
        aggregated = self._aggregate_metrics(metrics)
        
        # Store in history
        for key, value in aggregated.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(float(value))
        
        # Write to file if enabled
        if self.log_dir:
            self._write_batch_log(aggregated, batch_idx, epoch)
        
        return aggregated
    
    def _aggregate_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Aggregate metrics across layers and heads.
        
        Handles different metric formats:
        - List of layer metrics (one per layer)
        - List of head metrics (one per head)
        - Nested dictionaries
        """
        aggregated: Dict[str, float] = {}
        
        # Handle empty metrics
        if not metrics:
            return aggregated
        
        # Extract CDNS metrics
        cdns_metrics = self._extract_cdns(metrics)
        aggregated.update(cdns_metrics)
        
        # Extract order parameter metrics
        order_metrics = self._extract_order_parameter(metrics)
        aggregated.update(order_metrics)
        
        # Extract spectral metrics
        spectral_metrics = self._extract_spectral(metrics)
        aggregated.update(spectral_metrics)
        
        # Extract criticality metrics
        criticality_metrics = self._extract_criticality(metrics)
        aggregated.update(criticality_metrics)
        
        # Extract harmonic metrics (if Hodge decomposition enabled)
        harmonic_metrics = self._extract_harmonic(metrics)
        aggregated.update(harmonic_metrics)
        
        return aggregated
    
    def _extract_cdns(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract CDNS metrics (consonance, dissonance, noise, signal)."""
        cdns = {}
        
        # Try to find CDNS in various formats
        cdns_data = None
        
        # Format 1: Direct 'cdns' key
        if 'cdns' in metrics:
            cdns_data = metrics['cdns']
        
        # Format 2: Nested in layer/head structure
        elif isinstance(metrics, list):
            # Aggregate across layers/heads
            all_cdns = []
            for item in metrics:
                if isinstance(item, list):
                    # List of head metrics
                    for head_metric in item:
                        if isinstance(head_metric, dict) and 'cdns' in head_metric:
                            all_cdns.append(head_metric['cdns'])
                elif isinstance(item, dict) and 'cdns' in item:
                    all_cdns.append(item['cdns'])
            
            if all_cdns:
                # Average across heads/layers
                cdns_data = self._average_cdns_dicts(all_cdns)
        
        if cdns_data:
            if isinstance(cdns_data, dict):
                # Extract scalar values
                if 'consonance' in cdns_data:
                    val = cdns_data['consonance']
                    cdns['cdns_consonance'] = self._to_float(val)
                
                if 'dissonance' in cdns_data:
                    val = cdns_data['dissonance']
                    cdns['cdns_dissonance'] = self._to_float(val)
                
                # Full CDNS (noise/signal)
                if 'noise' in cdns_data:
                    val = cdns_data['noise']
                    cdns['cdns_noise'] = self._to_float(val)
                
                if 'signal' in cdns_data:
                    val = cdns_data['signal']
                    cdns['cdns_signal'] = self._to_float(val)
        
        # Also check for cdns_full
        if 'cdns_full' in metrics:
            cdns_full = metrics['cdns_full']
            if isinstance(cdns_full, dict):
                if 'noise' in cdns_full:
                    cdns['cdns_noise'] = self._to_float(cdns_full['noise'])
                if 'signal' in cdns_full:
                    cdns['cdns_signal'] = self._to_float(cdns_full['signal'])
        
        return cdns
    
    def _extract_order_parameter(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract order parameter and variance metrics."""
        order = {}
        
        # Final order parameter
        if 'final_order_parameter' in metrics:
            val = metrics['final_order_parameter']
            order['order_parameter'] = self._to_float(val)
        
        # Order parameter variance
        if 'order_param_variance' in metrics:
            val = metrics['order_param_variance']
            order['order_param_variance'] = self._to_float(val)
        
        # Also check nested structures
        if isinstance(metrics, list):
            r_vals = []
            var_vals = []
            for item in metrics:
                if isinstance(item, list):
                    for head_metric in item:
                        if isinstance(head_metric, dict):
                            if 'final_order_parameter' in head_metric:
                                r_vals.append(self._to_float(head_metric['final_order_parameter']))
                            if 'order_param_variance' in head_metric:
                                var_vals.append(self._to_float(head_metric['order_param_variance']))
                elif isinstance(item, dict):
                    if 'final_order_parameter' in item:
                        r_vals.append(self._to_float(item['final_order_parameter']))
                    if 'order_param_variance' in item:
                        var_vals.append(self._to_float(item['order_param_variance']))
            
            if r_vals:
                order['order_parameter'] = float(np.mean(r_vals))
            if var_vals:
                order['order_param_variance'] = float(np.mean(var_vals))
        
        return order
    
    def _extract_spectral(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract spectral band energies."""
        spectral = {}
        
        # Check for spectral band metrics
        if 'spectral_bands' in metrics:
            bands = metrics['spectral_bands']
            if isinstance(bands, (list, torch.Tensor)):
                bands_array = self._to_numpy(bands)
                if bands_array is not None:
                    for i, energy in enumerate(bands_array):
                        spectral[f'spectral_band_{i}'] = float(energy)
        
        return spectral
    
    def _extract_criticality(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract criticality index."""
        criticality = {}
        
        if 'criticality_index' in metrics:
            val = metrics['criticality_index']
            criticality['criticality_index'] = self._to_float(val)
        
        return criticality
    
    def _extract_harmonic(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract harmonic component metrics (Hodge decomposition)."""
        harmonic = {}
        
        if 'harmonic_dim' in metrics:
            harmonic['harmonic_dim'] = self._to_float(metrics['harmonic_dim'])
        
        if 'harmonic_strength' in metrics:
            harmonic['harmonic_strength'] = self._to_float(metrics['harmonic_strength'])
        
        return harmonic
    
    def _average_cdns_dicts(self, cdns_list: List[Dict]) -> Dict:
        """Average multiple CDNS dictionaries."""
        if not cdns_list:
            return {}
        
        result = {}
        keys = set()
        for cdns in cdns_list:
            if isinstance(cdns, dict):
                keys.update(cdns.keys())
        
        for key in keys:
            values = []
            for cdns in cdns_list:
                if isinstance(cdns, dict) and key in cdns:
                    val = self._to_float(cdns[key])
                    if val is not None:
                        values.append(val)
            
            if values:
                result[key] = float(np.mean(values))
        
        return result
    
    def _to_float(self, value: Any) -> Optional[float]:
        """Convert value to float, handling tensors and arrays."""
        if value is None:
            return None
        
        if isinstance(value, torch.Tensor):
            # Take mean if multi-dimensional
            if value.numel() > 1:
                value = value.mean()
            return float(value.detach().cpu().item())
        
        if isinstance(value, (np.ndarray, np.generic)):
            if value.size > 1:
                value = np.mean(value)
            return float(value)
        
        if isinstance(value, (int, float)):
            return float(value)
        
        return None
    
    def _to_numpy(self, value: Any) -> Optional[np.ndarray]:
        """Convert value to numpy array."""
        if value is None:
            return None
        
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        
        if isinstance(value, np.ndarray):
            return value
        
        if isinstance(value, (list, tuple)):
            return np.array(value)
        
        return None
    
    def _write_batch_log(self, aggregated: Dict[str, float], batch_idx: int, epoch: int):
        """Write batch log to file."""
        if not self.log_dir:
            return
        
        log_entry = {
            'epoch': epoch,
            'batch': batch_idx,
            'metrics': aggregated,
        }
        
        log_file = os.path.join(self.log_dir, 'telemetry.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of logged metrics."""
        summary = {}
        
        for key, values in self.metrics_history.items():
            if values:
                summary[f'{key}_mean'] = float(np.mean(values))
                summary[f'{key}_std'] = float(np.std(values))
                summary[f'{key}_min'] = float(np.min(values))
                summary[f'{key}_max'] = float(np.max(values))
        
        return summary
    
    def save_summary(self, filepath: str):
        """Save summary statistics to JSON file."""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)


