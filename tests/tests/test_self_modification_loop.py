import torch
import torch.nn as nn

from modules.self_modification_loop import SelfModificationLoop
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


def test_self_modification_loop_halts_on_safety_trigger():
    device = torch.device("cpu")
    baseline_model = DummyModel().to(device)
    safety_cfg = ProbeSafetyConfig(min_mi_with_input=5.0)
    probe = RecursiveDynamicsProbe(safety_config=safety_cfg)
    loop = SelfModificationLoop(probe=probe)

    input_ids = torch.randint(0, 32, (1, 8), device=device)

    def mutate_fn(model, step_idx):
        # Return a fresh copy of the model so parameters don't alias
        new_model = DummyModel().to(device)
        return new_model, {"step": step_idx}

    def accept_fn(_, __):
        return True

    results = loop.run(
        initial_model=baseline_model,
        input_ids=input_ids,
        steps=3,
        mutate_fn=mutate_fn,
        accept_fn=accept_fn,
        probe_iterations=4,
        safety_config=safety_cfg,
    )

    assert results, "Expected at least one loop result"
    first = results[0]
    assert first.safety_triggered, "Safety should trigger on first step"
    assert first.abort_reason is not None
    assert first.candidate_metrics.safety_events, "Safety events should be recorded"
    assert len(results) == 1, "Loop should halt after safety trigger"

