import torch

from modules.resonance_executive import ExecutiveConfig, ResonanceExecutive
from modules.recursive_dynamics_probes import RecursiveDynamicsMetrics


class DummyHead(torch.nn.Module):
    def __init__(self, order_value: float):
        super().__init__()
        self.metrics = {
            "final_order_parameter": torch.tensor([order_value]),
            "order_param_variance": torch.tensor([0.02]),
            "cdns": {"dissonance": torch.tensor([0.1])},
            "hodge": {"harmonic_dim": 0.5},
        }
        self.target_R = 0.5
        self.harmonic_leak_rate = 0.001
        self.lambda_dissonance = 0.01
        self.use_temporal_multiplex = True
        self.dt = 0.01


class DummyModel(torch.nn.Module):
    def __init__(self, order_value: float):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        head = DummyHead(order_value)
        self.layers[0].attention_heads = torch.nn.ModuleList([head])


def _build_probe_metrics(**overrides):
    base = dict(
        mi_with_input=0.1,
        mi_with_priors=0.2,
        latent_dimensionality=8,
        num_distinct_states=8,
        has_limit_cycle=False,
        cycle_length=None,
        attractor_stability=0.0,
        spectral_gap=0.0,
        dominant_eigenvalue=0.0,
        critical_depth=None,
        is_chaotic=False,
        lyapunov_exponent=None,
        symmetry_score=0.1,
        coherence_score=0.5,
        history=[],
        safety_events=[],
        abort_reason=None,
        triggered_thresholds=[],
    )
    base.update(overrides)
    return RecursiveDynamicsMetrics(**base)


def test_executive_adjusts_target_r_upwards_when_order_low():
    model = DummyModel(order_value=0.4)
    exec_cfg = ExecutiveConfig(
        target_order_parameter=0.6,
        order_param_band=0.05,
        pid_adjust_rate=0.02,
        decision_log_path=None,
    )
    executive = ResonanceExecutive(model, exec_cfg)
    head = model.layers[0].attention_heads[0]
    original_target = head.target_R

    decision = executive.step(step=1, epoch=1, probe_metrics=None)

    assert head.target_R > original_target
    assert decision.adjustments, "Expected adjustments when order is below target"


def test_executive_reacts_to_probe_coherence_and_chaos():
    model = DummyModel(order_value=0.65)
    exec_cfg = ExecutiveConfig(
        target_order_parameter=0.6,
        order_param_band=0.05,
        pid_adjust_rate=0.02,
        max_probe_coherence=0.4,
        decision_log_path=None,
    )
    executive = ResonanceExecutive(model, exec_cfg)
    head = model.layers[0].attention_heads[0]

    probe_metrics = _build_probe_metrics(coherence_score=0.6, is_chaotic=True, lyapunov_exponent=0.2)
    executive.step(step=1, epoch=1, probe_metrics=probe_metrics)

    assert head.lambda_dissonance > 0.01, "Coherence clamp should scale lambda_dissonance upward"
    assert not head.use_temporal_multiplex, "Chaos should disable temporal multiplexing"
    assert head.dt < 0.01, "Chaos should reduce dt to stabilize dynamics"

