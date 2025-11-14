"""
Frequency Domain Analysis for Oscillator Networks

Provides explicit spectral analysis of oscillator dynamics:
- FFT analysis: Frequency content of phase dynamics
- Spectral coherence: Frequency-specific synchronization
- Frequency-dependent coupling: Coupling that depends on frequency
- Multi-frequency synchronization: Different frequencies sync separately
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


def compute_fft_spectrum(
    phases_history: List[torch.Tensor],  # List of [batch, seq_len] snapshots
    dt: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    Compute FFT spectrum of phase dynamics.
    
    Args:
        phases_history: Phase snapshots over time
        dt: Time step
    
    Returns:
        {
            'frequencies': frequency values [n_frequencies],
            'spectrum': FFT spectrum [batch, seq_len, n_frequencies],
            'power': power spectrum [batch, seq_len, n_frequencies],
            'dominant_frequencies': dominant frequencies per oscillator [batch, seq_len],
        }
    """
    if len(phases_history) < 2:
        return {}
    
    # Stack into time series
    phases_series = torch.stack(phases_history, dim=2)  # [batch, seq_len, time]
    batch_size, seq_len, time_steps = phases_series.shape
    
    # Compute FFT
    # Use real FFT for efficiency (phases are real)
    fft_result = torch.fft.rfft(phases_series, dim=2)  # [batch, seq_len, n_freq]
    
    # Compute frequencies
    frequencies = torch.fft.rfftfreq(time_steps, d=dt)  # [n_freq]
    
    # Compute power spectrum
    power_spectrum = torch.abs(fft_result) ** 2  # [batch, seq_len, n_freq]
    
    # Find dominant frequencies (peaks in power spectrum)
    dominant_freq_indices = torch.argmax(power_spectrum, dim=2)  # [batch, seq_len]
    dominant_frequencies = frequencies[dominant_freq_indices]  # [batch, seq_len]
    
    return {
        'frequencies': frequencies,
        'spectrum': fft_result,
        'power': power_spectrum,
        'dominant_frequencies': dominant_frequencies,
    }


def compute_spectral_coherence(
    phases_history: List[torch.Tensor],
    dt: float = 0.01,
    frequency_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute spectral coherence: synchronization at each frequency.
    
    Coherence measures how well phases are synchronized at each frequency.
    
    Args:
        phases_history: Phase snapshots over time
        dt: Time step
        frequency_range: (min_freq, max_freq) to analyze
    
    Returns:
        {
            'frequencies': frequency values [n_frequencies],
            'coherence': coherence at each frequency [batch, n_frequencies],
            'mean_coherence': average coherence [batch],
        }
    """
    if len(phases_history) < 2:
        return {}
    
    # Stack into time series
    phases_series = torch.stack(phases_history, dim=2)  # [batch, seq_len, time]
    batch_size, seq_len, time_steps = phases_series.shape
    
    # Convert to complex representation: exp(i*phase)
    complex_phases = torch.exp(1j * phases_series)  # [batch, seq_len, time]
    # Complex-valued FFT along time dimension
    fft_complex = torch.fft.fft(complex_phases, dim=2)  # [batch, seq_len, n_freq]
    
    # Compute coherence: |mean(FFT)| / mean(|FFT|)
    # High coherence = phases synchronized at this frequency
    mean_fft = fft_complex.mean(dim=1)  # [batch, n_freq]
    mean_abs_fft = torch.abs(fft_complex).mean(dim=1)  # [batch, n_freq]
    
    coherence = torch.abs(mean_fft) / (mean_abs_fft + 1e-10)  # [batch, n_freq]
    
    # Compute frequencies
    frequencies = torch.fft.fftfreq(time_steps, d=dt)  # [n_freq]
    
    # Filter frequency range if specified
    if frequency_range is not None:
        min_freq, max_freq = frequency_range
        freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        frequencies = frequencies[freq_mask]
        coherence = coherence[:, freq_mask]
    
    return {
        'frequencies': frequencies,
        'coherence': coherence,
        'mean_coherence': coherence.mean(dim=1),  # [batch]
    }


def compute_frequency_dependent_coupling(
    coupling_matrix: torch.Tensor,
    phases_history: List[torch.Tensor],
    frequency_bands: List[float],
    coupling_per_band: List[float],
    dt: float = 0.01,
) -> torch.Tensor:
    """
    Compute frequency-dependent coupling.
    
    Different frequency bands get different coupling strengths.
    
    Args:
        coupling_matrix: Base coupling [batch, seq_len, seq_len]
        phases_history: Phase snapshots over time
        frequency_bands: Frequency band boundaries [f0, f1, f2, ...]
        coupling_per_band: Coupling strength per band [K0, K1, K2, ...]
        dt: Time step
    
    Returns:
        Frequency-weighted coupling [batch, seq_len, seq_len]
    """
    # Compute frequency content
    fft_analysis = compute_fft_spectrum(phases_history, dt)
    if not fft_analysis:
        return coupling_matrix
    
    dominant_freqs = fft_analysis['dominant_frequencies']  # [batch, seq_len]
    
    batch_size, seq_len = dominant_freqs.shape
    
    # Create frequency-dependent coupling
    freq_coupling = coupling_matrix.clone()
    
    for b in range(batch_size):
        for i in range(seq_len):
            freq_i = dominant_freqs[b, i].item()
            
            # Find which band this frequency belongs to
            band_idx = 0
            for band_boundary in frequency_bands[1:]:
                if freq_i < band_boundary:
                    break
                band_idx += 1
            
            # Apply coupling strength for this band
            if band_idx < len(coupling_per_band):
                coupling_strength = coupling_per_band[band_idx]
                freq_coupling[b, i, :] *= coupling_strength
                freq_coupling[b, :, i] *= coupling_strength
    
    return freq_coupling


def detect_multi_frequency_synchronization(
    phases_history: List[torch.Tensor],
    dominant_frequencies: List[float],
    dt: float = 0.01,
    tolerance: float = 0.1,
) -> Dict[str, Any]:
    """
    Detect synchronization at multiple frequencies.
    
    Groups oscillators by dominant frequency and checks synchronization
    within each frequency group.
    
    Args:
        phases_history: Phase snapshots over time
        dominant_frequencies: List of frequencies to check [f1, f2, ...]
        dt: Time step
        tolerance: Frequency matching tolerance
    
    Returns:
        {
            'frequency_groups': groups of oscillators per frequency,
            'sync_per_frequency': synchronization per frequency [batch, n_frequencies],
            'dominant_frequencies': dominant frequencies,
        }
    """
    if len(phases_history) < 1:
        return {}
    
    # Compute frequency content
    fft_analysis = compute_fft_spectrum(phases_history, dt)
    if not fft_analysis:
        return {}
    
    dominant_freqs = fft_analysis['dominant_frequencies']  # [batch, seq_len]
    
    batch_size, seq_len = dominant_freqs.shape
    n_frequencies = len(dominant_frequencies)
    
    # Group oscillators by frequency
    frequency_groups = []
    sync_per_freq = []
    
    for b in range(batch_size):
        groups_b = []
        sync_b = []
        
        for freq_idx, target_freq in enumerate(dominant_frequencies):
            # Find oscillators with this frequency
            freq_mask = torch.abs(dominant_freqs[b] - target_freq) < tolerance
            oscillator_indices = torch.where(freq_mask)[0].tolist()
            
            groups_b.append(oscillator_indices)
            
            if len(oscillator_indices) > 1:
                # Compute order parameter for this frequency group
                group_phases = phases_history[-1][b, oscillator_indices]  # Latest phases
                complex_phases = torch.exp(1j * group_phases)
                order_param = torch.abs(complex_phases.mean())
                sync_b.append(order_param.item())
            else:
                sync_b.append(0.0)
        
        frequency_groups.append(groups_b)
        sync_per_freq.append(sync_b)
    
    return {
        'frequency_groups': frequency_groups,
        'sync_per_frequency': torch.tensor(sync_per_freq, device=dominant_freqs.device),  # [batch, n_frequencies]
        'dominant_frequencies': dominant_frequencies,
    }


class FrequencyDomainAnalyzer:
    """
    Frequency domain analyzer for oscillator networks.
    
    Provides FFT analysis, spectral coherence, and frequency-dependent coupling.
    """
    
    def __init__(
        self,
        history_length: int = 50,
        dt: float = 0.01,
        frequency_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize frequency domain analyzer.
        
        Args:
            history_length: Number of phase snapshots to keep for FFT
            dt: Time step
            frequency_range: (min_freq, max_freq) to analyze
        """
        self.history_length = history_length
        self.dt = dt
        self.frequency_range = frequency_range
        self.phases_history = []
    
    def update_history(self, phases: torch.Tensor):
        """
        Update phase history.
        
        Args:
            phases: Current phases [batch, seq_len]
        """
        self.phases_history.append(phases.clone())
        if len(self.phases_history) > self.history_length:
            self.phases_history.pop(0)
    
    def analyze(self) -> Dict[str, torch.Tensor]:
        """
        Perform frequency domain analysis.
        
        Returns:
            Dictionary with FFT analysis and spectral coherence results
        """
        if len(self.phases_history) < 10:
            return {}
        
        # FFT analysis
        fft_analysis = compute_fft_spectrum(self.phases_history, self.dt)
        
        # Spectral coherence
        spectral_coherence = compute_spectral_coherence(
            self.phases_history,
            self.dt,
            self.frequency_range,
        )
        
        return {
            'fft_analysis': fft_analysis,
            'spectral_coherence': spectral_coherence,
        }
    
    def reset(self):
        """Reset phase history."""
        self.phases_history = []
