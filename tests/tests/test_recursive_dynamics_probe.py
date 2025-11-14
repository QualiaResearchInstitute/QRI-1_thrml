import torch
import torch.nn as nn

from modules.recursive_dynamics_probes import (
    ProbeSafetyConfig,
    RecursiveDynamicsProbe,
)


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 32, embed_dim: int = 16):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        logits = self.head(x)
        return logits


def test_probe_records_history_and_triggers_safety_event():
    device = torch.device("cpu")
    model = DummyModel().to(device)
    probe = RecursiveDynamicsProbe()
    cfg = ProbeSafetyConfig(min_mi_with_input=5.0)
    input_ids = torch.randint(0, 32, (1, 8), device=device)

    snapshots = []

    metrics = probe.probe_recursive_dynamics(
        model=model,
        input_ids=input_ids,
        num_iterations=5,
        safety_config=cfg,
        callback=lambda snapshot: snapshots.append(snapshot),
    )

    assert metrics.history, "Expected per-iteration history to be recorded"
    assert snapshots == metrics.history, "Callback snapshots should match metrics history"
    assert metrics.abort_reason is not None, "Safety event should trigger an abort reason"
    trigger_metrics = {event.metric for event in metrics.safety_events}
    assert "mi_with_input" in trigger_metrics
    assert "mi_with_input" in metrics.triggered_thresholds

