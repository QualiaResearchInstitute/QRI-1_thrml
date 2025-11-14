"""
Tests for stochastic dynamics implementation.
"""

import torch
import pytest
import numpy as np
from modules.stochastic_dynamics import (
    StochasticDynamics,
    generate_gaussian_noise,
    generate_levy_noise,
    generate_colored_noise,
)


def test_gaussian_noise():
    """Test Gaussian noise generation."""
    device = torch.device("cpu")
    shape = (2, 10)
    strength = 0.01
    
    noise = generate_gaussian_noise(shape, strength, device)
    
    assert noise.shape == shape
    assert noise.device == device
    # Check that noise is approximately zero-mean
    assert torch.abs(noise.mean()) < 0.1
    # Check that standard deviation is approximately correct
    assert 0.005 < noise.std() < 0.02


def test_colored_noise():
    """Test colored noise generation."""
    device = torch.device("cpu")
    shape = (2, 10)
    strength = 0.01
    correlation = 0.5
    
    noise = generate_colored_noise(shape, strength, correlation, device)
    
    assert noise.shape == shape
    assert noise.device == device
    # Check temporal correlation
    # For AR(1) process: x[t] = ρ*x[t-1] + ε[t]
    # The difference x[t] - ρ*x[t-1] should be approximately ε[t] (white noise)
    # which has std ≈ strength, so allow up to ~5 std deviations
    for b in range(shape[0]):
        for t in range(1, shape[1]):
            # Check that consecutive values follow AR(1) structure
            diff = torch.abs(noise[b, t] - correlation * noise[b, t-1])
            # Should be approximately the white noise component (allow up to 5 std)
            assert diff < strength * 5


def test_stochastic_dynamics_init():
    """Test StochasticDynamics initialization."""
    sd = StochasticDynamics(
        noise_type="gaussian",
        noise_strength=0.01,
        stochastic_resonance=True,
    )
    
    assert sd.noise_type == "gaussian"
    assert sd.noise_strength == 0.01
    assert sd.stochastic_resonance == True
    assert sd.optimal_noise_level is None


def test_stochastic_dynamics_apply_noise():
    """Test noise application to phases."""
    device = torch.device("cpu")
    sd = StochasticDynamics(
        noise_type="gaussian",
        noise_strength=0.01,
    )
    
    batch_size, seq_len = 2, 10
    phases = torch.randn(batch_size, seq_len, device=device) * np.pi
    amplitudes = torch.ones(batch_size, seq_len, device=device) * 0.5
    dt = 0.01
    
    phases_noisy, amplitudes_noisy = sd.apply_noise(phases, amplitudes, dt)
    
    assert phases_noisy.shape == phases.shape
    assert amplitudes_noisy.shape == amplitudes.shape
    # Phases should be wrapped to [0, 2π)
    assert torch.all(phases_noisy >= 0)
    assert torch.all(phases_noisy < 2 * np.pi)
    # Amplitudes should be non-negative
    assert torch.all(amplitudes_noisy >= 0)


def test_stochastic_resonance_detection():
    """Test stochastic resonance detection."""
    sd = StochasticDynamics(
        noise_type="gaussian",
        noise_strength=0.01,
        stochastic_resonance=True,
    )
    
    # Create a synthetic resonance curve (peak at noise=0.01)
    order_history = [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3]
    noise_history = [0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05]
    
    resonance = sd.detect_stochastic_resonance(order_history, noise_history)
    
    assert 'optimal_noise' in resonance
    assert 'resonance_strength' in resonance
    assert 'resonance_detected' in resonance
    # Peak should be detected at noise=0.01
    assert resonance['optimal_noise'] == 0.01
    assert resonance['resonance_strength'] == 0.7


def test_stochastic_dynamics_integration():
    """Test integration with ResonanceAttentionHead."""
    try:
        from resonance_transformer import ResonanceAttentionHead
        
        # Create head with stochastic dynamics
        head = ResonanceAttentionHead(
            d_model=64,
            use_stochastic_dynamics=True,
            noise_type="gaussian",
            noise_strength=0.01,
            stochastic_resonance=False,
        )
        
        assert head.use_stochastic_dynamics == True
        assert head.stochastic_dynamics is not None
        assert head.stochastic_dynamics.noise_type == "gaussian"
        assert head.stochastic_dynamics.noise_strength == 0.01
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 64)
        
        output = head(x, return_metrics=True)
        
        # Should return output and metrics
        assert isinstance(output, tuple)
        assert len(output) == 2
        output_tensor, metrics = output
        assert output_tensor.shape == (batch_size, seq_len, 64)
        assert 'noise' in metrics
        
    except ImportError:
        pytest.skip("ResonanceAttentionHead not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

