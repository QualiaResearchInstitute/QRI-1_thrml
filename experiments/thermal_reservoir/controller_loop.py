"""
Simple hackathon-style controller loop around `ThrmlReservoir`.

This is intentionally minimal: it treats the reservoir as a black-box
dynamical system, sweeps over a few temperature schedules, and logs
summary stats (magnetization, energy) you can later plug into a more
complex controller or visualization.
"""

from __future__ import annotations

import logging
from typing import Sequence

import jax.numpy as jnp

from .reservoir_thrml import ReservoirConfig, ThrmlReservoir


log = logging.getLogger(__name__)


def run_temperature_scan(
    betas: Sequence[float] = (0.3, 0.5, 0.8, 1.0, 1.2, 1.5),
) -> None:
    """
    Run a simple temperature scan over the reservoir and log summaries.

    This is roughly analogous to "annealing as reservoir probing":
    we explore different dynamical regimes by changing inverse temperature.
    """
    cfg = ReservoirConfig()
    reservoir = ThrmlReservoir(cfg)
    reservoir.reset()

    magnetizations, mean_energies = reservoir.sweep_beta(
        betas=betas,
        n_samples=100,
        n_warmup=100,
        steps_per_sample=2,
    )

    for beta, m, e in zip(betas, magnetizations, mean_energies):
        log.info("beta=%.3f | magnetization=%.4f | mean_energy=%.4f", beta, float(m), float(e))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    betas = jnp.linspace(0.3, 1.5, 8)
    run_temperature_scan(betas)


if __name__ == "__main__":
    main()


