"""
Tests for Phase 6 features: Frequency Domain Analysis and Multi-Scale Structures
"""

import torch
import pytest
from resonance_transformer import ResonanceAttentionHead


def test_frequency_domain_analysis_import():
    """Test that frequency domain analysis module can be imported."""
    try:
        from modules.frequency_domain_analysis import (
            FrequencyDomainAnalyzer,
            compute_fft_spectrum,
            compute_spectral_coherence,
        )
        assert FrequencyDomainAnalyzer is not None
        assert compute_fft_spectrum is not None
        assert compute_spectral_coherence is not None
    except ImportError:
        pytest.skip("Frequency domain analysis module not available")


def test_multiscale_structures_import():
    """Test that multi-scale structures module can be imported."""
    try:
        from modules.multiscale_structures import (
            MultiScaleStructures,
            construct_hierarchical_coupling,
            generate_scale_free_topology,
        )
        assert MultiScaleStructures is not None
        assert construct_hierarchical_coupling is not None
        assert generate_scale_free_topology is not None
    except ImportError:
        pytest.skip("Multi-scale structures module not available")


def test_frequency_domain_analyzer():
    """Test FrequencyDomainAnalyzer basic functionality."""
    try:
        from modules.frequency_domain_analysis import FrequencyDomainAnalyzer
        
        analyzer = FrequencyDomainAnalyzer(history_length=20, dt=0.01)
        
        # Add some phase history
        phases = torch.randn(2, 10)  # [batch, seq_len]
        for _ in range(15):
            analyzer.update_history(phases)
        
        # Analyze
        results = analyzer.analyze()
        assert isinstance(results, dict)
        
        # Reset
        analyzer.reset()
        assert len(analyzer.phases_history) == 0
    except ImportError:
        pytest.skip("Frequency domain analysis module not available")


def test_hierarchical_coupling():
    """Test hierarchical coupling construction."""
    try:
        from modules.multiscale_structures import construct_hierarchical_coupling
        
        seq_len = 20
        local_coupling, global_coupling = construct_hierarchical_coupling(
            seq_len=seq_len,
            local_scale=5,
            global_scale=15,
            local_weight=0.7,
            global_weight=0.3,
        )
        
        assert local_coupling.shape == (seq_len, seq_len)
        assert global_coupling.shape == (seq_len, seq_len)
        
        # Local coupling should be stronger for nearby nodes
        assert local_coupling[0, 1] > local_coupling[0, 10]
        
        # Global coupling should be weaker
        assert global_coupling[0, 10] < local_coupling[0, 1]
    except ImportError:
        pytest.skip("Multi-scale structures module not available")


def test_scale_free_topology():
    """Test scale-free topology generation."""
    try:
        from modules.multiscale_structures import generate_scale_free_topology
        
        seq_len = 20
        adjacency = generate_scale_free_topology(
            seq_len=seq_len,
            hub_fraction=0.2,
            power_law_exponent=2.5,
        )
        
        assert adjacency.shape == (seq_len, seq_len)
        # Should be symmetric (undirected)
        assert torch.allclose(adjacency, adjacency.T)
        
        # Hubs should have more connections
        hub_degrees = adjacency[:4].sum(dim=1)  # First 4 nodes are hubs
        spoke_degrees = adjacency[4:].sum(dim=1)  # Rest are spokes
        assert hub_degrees.mean() > spoke_degrees.mean()
    except ImportError:
        pytest.skip("Multi-scale structures module not available")


def test_resonance_head_with_frequency_domain():
    """Test ResonanceAttentionHead with frequency domain analysis enabled."""
    d_model = 64
    seq_len = 32
    batch_size = 2
    
    head = ResonanceAttentionHead(
        d_model=d_model,
        analyze_frequency_domain=True,
        frequency_analysis_history_length=20,
        use_frequency_dependent_coupling=False,  # Don't use this in test to avoid needing history
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # First forward pass (no history yet)
    output, metrics = head(x, return_metrics=True)
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Multiple forward passes to build history
    for _ in range(15):
        output, metrics = head(x, return_metrics=True)
    
    # Check if frequency domain metrics are present
    if 'frequency_domain' in metrics:
        freq_metrics = metrics['frequency_domain']
        assert isinstance(freq_metrics, dict)


def test_resonance_head_with_multiscale():
    """Test ResonanceAttentionHead with multi-scale structures enabled."""
    d_model = 64
    seq_len = 32
    batch_size = 2
    
    head = ResonanceAttentionHead(
        d_model=d_model,
        use_hierarchical_coupling=True,
        local_scale=5,
        global_scale=15,
        local_weight=0.7,
        global_weight=0.3,
        use_scale_free_topology=False,  # Test hierarchical only first
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, metrics = head(x, return_metrics=True)
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Check if nested synchronization metrics are present
    if 'nested_synchronization' in metrics:
        nested_metrics = metrics['nested_synchronization']
        assert isinstance(nested_metrics, dict)
        assert 'local_order_param' in nested_metrics
        assert 'global_order_param' in nested_metrics
        assert 'nested_order_param' in nested_metrics


def test_resonance_head_with_scale_free_topology():
    """Test ResonanceAttentionHead with scale-free topology enabled."""
    d_model = 64
    seq_len = 32
    batch_size = 2
    
    head = ResonanceAttentionHead(
        d_model=d_model,
        use_scale_free_topology=True,
        hub_fraction=0.2,
        power_law_exponent=2.5,
        use_hierarchical_coupling=False,  # Test scale-free only
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, metrics = head(x, return_metrics=True)
    assert output.shape == (batch_size, seq_len, d_model)


def test_resonance_head_phase6_combined():
    """Test ResonanceAttentionHead with both Phase 6 features enabled."""
    d_model = 64
    seq_len = 32
    batch_size = 2
    
    head = ResonanceAttentionHead(
        d_model=d_model,
        # Frequency domain
        analyze_frequency_domain=True,
        frequency_analysis_history_length=20,
        # Multi-scale
        use_hierarchical_coupling=True,
        local_scale=5,
        global_scale=15,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Multiple forward passes to build history
    for _ in range(15):
        output, metrics = head(x, return_metrics=True)
        assert output.shape == (batch_size, seq_len, d_model)
    
    # Check metrics
    if 'frequency_domain' in metrics:
        assert isinstance(metrics['frequency_domain'], dict)
    
    if 'nested_synchronization' in metrics:
        assert isinstance(metrics['nested_synchronization'], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

