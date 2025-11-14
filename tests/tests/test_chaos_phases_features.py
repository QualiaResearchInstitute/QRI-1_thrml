import math
import torch

from resonance_transformer import ResonanceAttentionHead, ResonanceTransformerBlock

def _rand_input(batch=2, seq_len=8, d_model=32, device="cpu"):
    torch.manual_seed(0)
    return torch.randn(batch, seq_len, d_model, device=device)


def test_dual_timescale_metrics_present():
    x = _rand_input()
    head = ResonanceAttentionHead(
        d_model=x.shape[-1],
        n_sim_steps=5,
        use_stuart_landau=True,
        use_heun=True,
        track_metrics=True,
        use_dual_timescale=True,
        fast_tau=0.005,
        slow_tau=0.1,
        fast_coupling_strength=1.0,
        slow_coupling_strength=0.5,
        use_voltage_gating=False,
    )
    _ = head(x)  # forward
    assert isinstance(getattr(head, "metrics", {}), dict)
    assert "dual_timescale" in head.metrics
    dtm = head.metrics["dual_timescale"]
    assert "mean_phase_difference" in dtm
    assert "phase_locking_value" in dtm
    assert "timescale_separation_index" in dtm
    assert dtm["timescale_separation_index"] > 0.0


def test_time_reversible_metrics_present():
    x = _rand_input()
    head = ResonanceAttentionHead(
        d_model=x.shape[-1],
        n_sim_steps=5,
        use_stuart_landau=True,
        use_heun=True,
        track_metrics=True,
        use_time_reversible=True,
        tr_adaptive_steps=True,
        tr_convergence_threshold=1e-4,
    )
    _ = head(x)
    metr = getattr(head, "metrics", {})
    assert "time_reversible" in metr
    assert "reversibility_error" in metr["time_reversible"]


def test_relu_and_hybrid_metrics_present():
    x = _rand_input()
    # ReLU
    head_relu = ResonanceAttentionHead(
        d_model=x.shape[-1],
        n_sim_steps=3,
        use_stuart_landau=True,
        use_heun=True,
        track_metrics=True,
        coupling_type="relu",
    )
    _ = head_relu(x)
    metr_r = getattr(head_relu, "metrics", {})
    assert "relu_coupling" in metr_r
    assert metr_r["relu_coupling"]["coupling_type"] == "relu"

    # Hybrid
    head_hybrid = ResonanceAttentionHead(
        d_model=x.shape[-1],
        n_sim_steps=3,
        use_stuart_landau=True,
        use_heun=True,
        track_metrics=True,
        coupling_type="hybrid",
        relu_weight=0.7,
    )
    _ = head_hybrid(x)
    metr_h = getattr(head_hybrid, "metrics", {})
    assert "relu_coupling" in metr_h
    assert metr_h["relu_coupling"]["coupling_type"] == "hybrid"
    assert 0.0 <= float(metr_h["relu_coupling"]["relu_weight"]) <= 1.0


def test_chimera_death_smoke():
    x = _rand_input()
    head = ResonanceAttentionHead(
        d_model=x.shape[-1],
        n_sim_steps=5,
        use_stuart_landau=True,
        use_heun=True,
        track_metrics=True,
        detect_chimera_death=True,
        amplitude_death_threshold=1e-3,
    )
    _ = head(x)
    metr = getattr(head, "metrics", {})
    # Either we detect metrics or we recorded a non-fatal error
    assert ("chimera_death" in metr) or ("chimera_death_errors" in metr)


def test_lag_synchronization_smoke():
    x = _rand_input()
    head = ResonanceAttentionHead(
        d_model=x.shape[-1],
        n_sim_steps=5,
        use_stuart_landau=True,
        use_heun=True,
        track_metrics=True,
        detect_lag_synchronization=True,
        lag_similarity_threshold=0.5,
        max_lag=5,
    )
    _ = head(x)
    metr = getattr(head, "metrics", {})
    assert ("lag_synchronization" in metr) or ("lag_sync_errors" in metr)


def test_remote_synchronization_block_metrics_present():
    batch, seq_len, d_model = 2, 8, 32
    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, d_model)
    block = ResonanceTransformerBlock(
        d_model=d_model,
        n_heads=2,
        n_sim_steps=3,
        use_remote_synchronization=True,
    )
    _ = block(x)
    metr = getattr(block, "metrics", {})
    assert "remote_synchronization" in metr
    rs = metr["remote_synchronization"]
    assert "rs_pairs" in rs
    assert "rs_ratio" in rs


def test_bifurcation_analyzer_module():
    # Import module and exercise detection with synthetic history
    from modules.bifurcation_analyzer import BifurcationAnalyzer
    analyzer = BifurcationAnalyzer()
    phases = torch.zeros(1, 10)  # phases not central for these heuristics
    K = 1.0
    # History with a jump to trigger pitchfork-like detection
    R_hist = [0.1, 0.12, 0.15, 0.16, 0.5, 0.52, 0.53, 0.55, 0.56, 0.57]
    current_R = R_hist[-1]
    out = analyzer.detect_bifurcations(phases, K, current_R, R_hist)
    assert isinstance(out, dict)
    assert "any_detected" in out
    assert "hopf" in out and "pitchfork" in out and "saddle_node" in out


def test_information_flow_module():
    from modules.information_flow import InformationFlowAnalyzer
    analyzer = InformationFlowAnalyzer(
        transfer_entropy_k=1, transfer_entropy_l=1, transfer_entropy_bins=5, granger_max_lag=2
    )
    # Build simple directed influence phases_history
    torch.manual_seed(0)
    batch, seq_len, T = 1, 5, 6
    phases_history = []
    base = torch.linspace(0, math.pi, T)
    for t in range(T):
        # oscillator 0 drives 1 weakly
        ph = torch.zeros(batch, seq_len)
        ph[:, 0] = base[t]
        ph[:, 1] = base[max(0, t - 1)]  # delayed copy
        phases_history.append(ph)
    res = analyzer.compute_information_flow(phases_history, compute_transfer_entropy=True, compute_granger=True)
    assert isinstance(res, dict)
    assert ("transfer_entropy" in res) or ("granger_causality" in res)


def test_adaptive_coupling_smoke():
    x = _rand_input()
    head = ResonanceAttentionHead(
        d_model=x.shape[-1],
        n_sim_steps=4,
        use_stuart_landau=True,
        use_heun=True,
        track_metrics=True,
        use_adaptive_coupling=True,
        adaptation_rate=0.01,
        adaptation_signal="order_parameter",
        adaptation_target=0.6,
    )
    _ = head(x)  # should not crash
    assert getattr(head, "use_adaptive_coupling", False) is True
