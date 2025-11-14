"""
Test Hodge decomposition implementation.
"""

import torch
import sys
import pathlib

# Add project root to path
_project_root = pathlib.Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from resonance_transformer import ResonanceAttentionHead, hodge_decompose_coupling


def test_hodge_decomposition_function():
    """Test the hodge_decompose_coupling function."""
    print("Testing hodge_decompose_coupling function...")
    
    batch_size = 2
    seq_len = 10
    
    # Create a simple coupling matrix
    coupling_matrix = torch.randn(batch_size, seq_len, seq_len)
    # Make symmetric
    coupling_matrix = 0.5 * (coupling_matrix + coupling_matrix.transpose(-2, -1))
    # Make positive (for realistic coupling)
    coupling_matrix = torch.abs(coupling_matrix)
    
    # Decompose
    hodge = hodge_decompose_coupling(coupling_matrix, return_harmonic_dim=True)
    
    # Check outputs
    assert 'exact' in hodge
    assert 'coexact' in hodge
    assert 'harmonic' in hodge
    assert 'harmonic_dim' in hodge
    
    # Check shapes
    assert hodge['exact'].shape == (batch_size, seq_len, seq_len)
    assert hodge['coexact'].shape == (batch_size, seq_len, seq_len)
    assert hodge['harmonic'].shape == (batch_size, seq_len, seq_len)
    assert hodge['harmonic_dim'].shape == (batch_size,)
    
    # Check reconstruction: exact + harmonic should approximately equal original
    reconstructed = hodge['exact'] + hodge['harmonic'] + hodge['coexact']
    diff = torch.abs(reconstructed - coupling_matrix).max()
    print(f"  Reconstruction error: {diff.item():.6f}")
    assert diff < 1e-5, f"Reconstruction error too large: {diff.item()}"
    
    # Check harmonic dimension is reasonable (should be >= 0, <= seq_len)
    harmonic_dim = hodge['harmonic_dim']
    assert torch.all(harmonic_dim >= 0), "Harmonic dimension should be >= 0"
    assert torch.all(harmonic_dim <= seq_len), "Harmonic dimension should be <= seq_len"
    
    print("  ✓ hodge_decompose_coupling function works correctly")
    return True


def test_resonance_attention_head_with_hodge():
    """Test ResonanceAttentionHead with Hodge decomposition enabled."""
    print("Testing ResonanceAttentionHead with Hodge decomposition...")
    
    batch_size = 2
    seq_len = 8
    d_model = 16
    
    # Create attention head with Hodge decomposition
    head = ResonanceAttentionHead(
        d_model=d_model,
        use_hodge_decomposition=True,
        harmonic_leak_rate=0.001,
        standard_leak_rate=0.1,
        n_sim_steps=5,  # Fewer steps for faster test
        track_metrics=True
    )
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, metrics = head(x, return_metrics=True)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), f"Output shape mismatch: {output.shape}"
    
    # Check metrics include Hodge metrics
    assert 'hodge' in metrics, "Metrics should include Hodge decomposition metrics"
    assert 'harmonic_dim' in metrics['hodge'], "Hodge metrics should include harmonic_dim"
    assert 'harmonic_strength' in metrics['hodge'], "Hodge metrics should include harmonic_strength"
    assert 'standard_strength' in metrics['hodge'], "Hodge metrics should include standard_strength"
    
    print(f"  Harmonic dimension: {metrics['hodge']['harmonic_dim']:.2f}")
    print(f"  Harmonic strength: {metrics['hodge']['harmonic_strength']:.6f}")
    print(f"  Standard strength: {metrics['hodge']['standard_strength']:.6f}")
    
    print("  ✓ ResonanceAttentionHead with Hodge decomposition works correctly")
    return True


def test_hodge_without_enabled():
    """Test that Hodge decomposition doesn't affect behavior when disabled."""
    print("Testing ResonanceAttentionHead without Hodge decomposition...")
    
    batch_size = 2
    seq_len = 8
    d_model = 16
    
    # Create attention head without Hodge decomposition
    head = ResonanceAttentionHead(
        d_model=d_model,
        use_hodge_decomposition=False,
        n_sim_steps=5,
        track_metrics=True
    )
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, metrics = head(x, return_metrics=True)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model)
    
    # Hodge metrics should not be present when disabled
    assert 'hodge' not in metrics or metrics.get('hodge', {}).get('harmonic_dim', 0.0) == 0.0
    
    print("  ✓ ResonanceAttentionHead without Hodge decomposition works correctly")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hodge Decomposition Implementation")
    print("=" * 60)
    
    try:
        test_hodge_decomposition_function()
        print()
        
        test_resonance_attention_head_with_hodge()
        print()
        
        test_hodge_without_enabled()
        print()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

