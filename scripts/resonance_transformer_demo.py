#!/usr/bin/env python3
"""
Minimal, skeptic-friendly demo for the Resonance Transformer stack.

Usage (from collab_package root):

    pip install -r env/requirements.txt
    export PYTHONPATH=src
    python scripts/resonance_transformer_demo.py

This will:
1. Run a ResonanceAttentionHead on "coherent" vs "dissonant" hidden states and
   compare Kuramoto-style metrics (order parameter, metastability, criticality).
2. Run a small ResonanceTransformer forward pass to show end-to-end behavior.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from resonance_transformer import (
    ResonanceAttentionHead,
    ResonanceTransformer,
)


@dataclass
class DemoConfig:
    batch_size: int = 2
    seq_len: int = 16
    d_model: int = 64
    n_sim_steps: int = 15
    vocab_size: int = 1000
    n_layers: int = 2
    n_heads: int = 4


def _summarize_metrics(label: str, metrics: dict) -> None:
    """Print a concise summary of key Kuramoto metrics for a given run."""
    def _safe_mean(name: str) -> float | None:
        value = metrics.get(name)
        if value is None:
            return None
        try:
            # Tensor or array-like
            if hasattr(value, "mean"):
                return float(value.mean().item())
            return float(value)
        except Exception:
            return None

    R = _safe_mean("final_order_parameter")
    metastability = _safe_mean("order_param_variance")
    criticality = _safe_mean("criticality_index")

    parts = [f"[{label}]"]
    if R is not None:
        parts.append(f"R={R:.3f}")
    if metastability is not None:
        parts.append(f"metastability={metastability:.3f}")
    if criticality is not None:
        parts.append(f"criticality={criticality:.3f}")

    print("  " + " | ".join(parts))


def run_head_demo(cfg: DemoConfig) -> None:
    """
    Demonstrate that coherent vs dissonant hidden states yield different Kuramoto
    order parameters and related metrics.
    """
    print("=== ResonanceAttentionHead demo (Kuramoto metrics) ===")

    head = ResonanceAttentionHead(
        d_model=cfg.d_model,
        n_sim_steps=cfg.n_sim_steps,
        track_metrics=True,
    )

    # "Coherent" batch: tokens in each sequence share a strong common component.
    base = torch.randn(cfg.batch_size, 1, cfg.d_model)
    coherent = base.repeat(1, cfg.seq_len, 1) + 0.01 * torch.randn(
        cfg.batch_size, cfg.seq_len, cfg.d_model
    )

    # "Dissonant" batch: tokens are independently sampled.
    dissonant = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model)

    out_coh, metrics_coh = head(coherent, return_metrics=True)
    out_dis, metrics_dis = head(dissonant, return_metrics=True)

    print(f"Input shape:  batch={cfg.batch_size}, seq_len={cfg.seq_len}, d_model={cfg.d_model}")
    print(f"Output shape (coherent):  {tuple(out_coh.shape)}")
    print(f"Output shape (dissonant): {tuple(out_dis.shape)}")

    _summarize_metrics("coherent", metrics_coh)
    _summarize_metrics("dissonant", metrics_dis)

    print("Heuristic expectation: coherent inputs should exhibit higher R (order parameter)")
    print()


def run_transformer_demo(cfg: DemoConfig) -> None:
    """Run a small end-to-end ResonanceTransformer forward pass."""
    print("=== ResonanceTransformer demo (end-to-end forward) ===")

    model = ResonanceTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_sim_steps=cfg.n_sim_steps,
    )

    input_ids = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(cfg.batch_size, cfg.seq_len),
    )

    logits = model(input_ids)

    print(f"Input ids shape:    {tuple(input_ids.shape)}")
    print(f"Logits shape:       {tuple(logits.shape)}")
    print(f"Expected logits:    (batch, seq_len, vocab_size) = "
          f"({cfg.batch_size}, {cfg.seq_len}, {cfg.vocab_size})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal demo for the Resonance Transformer (attention head + full model)."
    )
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Only run the attention head demo.",
    )
    parser.add_argument(
        "--skip-head",
        action="store_true",
        help="Only run the full ResonanceTransformer demo.",
    )
    args = parser.parse_args()

    cfg = DemoConfig()

    if not args.skip_head:
        run_head_demo(cfg)

    if not args.skip_transformer:
        run_transformer_demo(cfg)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()


