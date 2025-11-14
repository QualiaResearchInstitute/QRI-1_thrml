"""
High-level wrapper around Extropic's `thrml` library that exposes a
thermal "reservoir" interface for hackathon-style experiments.

Design goals
------------
- Treat a thrml Ising EBM as a high-dimensional thermal field.
- Provide simple methods to:
  - reset the reservoir state
  - run annealing/sampling schedules
  - compute basic summary statistics for downstream controllers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


@dataclass
class ReservoirConfig:
    """Configuration for a simple 1D Ising chain reservoir."""

    n_spins: int = 64
    beta: float = 1.0
    coupling: float = 0.5
    seed: int = 0
    # Sampling defaults
    n_warmup: int = 100
    n_samples: int = 100
    steps_per_sample: int = 2


class ThrmlReservoir:
    """
    Thin wrapper around a `thrml` IsingEBM + sampling program.

    This intentionally keeps the interface very small and hackathon-friendly:

    - `reset`   : reinitialize the model state
    - `run`     : run a sampling schedule and return states + summaries
    - `sweep_beta`: convenience helper to scan over temperature regimes
    """

    def __init__(self, config: Optional[ReservoirConfig] = None):
        self.config = config or ReservoirConfig()

        # Build nodes and edges for a simple 1D chain.
        self.nodes = [SpinNode() for _ in range(self.config.n_spins)]
        self.edges = [(self.nodes[i], self.nodes[i + 1]) for i in range(self.config.n_spins - 1)]

        # Biases and weights (homogeneous for now).
        self.biases = jnp.zeros((self.config.n_spins,))
        self.weights = jnp.ones((self.config.n_spins - 1,)) * self.config.coupling
        self.beta = jnp.array(self.config.beta)

        # Core EBM.
        self.model = IsingEBM(
            self.nodes,
            self.edges,
            self.biases,
            self.weights,
            self.beta,
        )

        # Two-color block Gibbs sampling: even/odd spins.
        free_blocks = [Block(self.nodes[::2]), Block(self.nodes[1::2])]
        self.program = IsingSamplingProgram(self.model, free_blocks, clamped_blocks=[])

        # Random key + initial state.
        self._main_key = jax.random.key(self.config.seed)
        self.state = self._init_state()

    def _split_key(self) -> jax.Array:
        self._main_key, k = jax.random.split(self._main_key, 2)
        return k

    def _init_state(self):
        key = self._split_key()
        return hinton_init(key, self.model, self.program.free_blocks, ())

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset random key (optional) and reinitialize the reservoir state."""
        if seed is not None:
            self._main_key = jax.random.key(seed)
        self.state = self._init_state()

    def _build_schedule(
        self,
        n_warmup: Optional[int] = None,
        n_samples: Optional[int] = None,
        steps_per_sample: Optional[int] = None,
    ) -> SamplingSchedule:
        return SamplingSchedule(
            n_warmup=n_warmup or self.config.n_warmup,
            n_samples=n_samples or self.config.n_samples,
            steps_per_sample=steps_per_sample or self.config.steps_per_sample,
        )

    def run(
        self,
        n_warmup: Optional[int] = None,
        n_samples: Optional[int] = None,
        steps_per_sample: Optional[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run a sampling schedule and return (states, energies).

        - states: array with shape (n_samples, n_spins)
        - energies: array with shape (n_samples,)
        """
        schedule = self._build_schedule(n_warmup, n_samples, steps_per_sample)
        key = self._split_key()

        # We treat all nodes as a single "block" to read out samples.
        readout_blocks = [Block(self.nodes)]
        samples = sample_states(key, self.program, schedule, self.state, [], readout_blocks)

        # Update internal state to the last sample for continuity.
        self.state = samples[-1]

        # Compute simple energy trace for convenience.
        energies = jax.vmap(self.model.energy)(samples)
        return samples, energies

    def sweep_beta(
        self,
        betas: Sequence[float],
        n_samples: int = 50,
        n_warmup: int = 50,
        steps_per_sample: int = 2,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sweep over a list of beta (inverse temperature) values and collect summaries.

        Returns:
            magnetizations: shape (len(betas),)
            mean_energies: shape (len(betas),)
        """
        magnetizations = []
        mean_energies = []

        for beta in betas:
            # Update model temperature.
            self.beta = jnp.array(beta)
            self.model = IsingEBM(
                self.nodes,
                self.edges,
                self.biases,
                self.weights,
                self.beta,
            )
            # Rebuild program with the new model.
            free_blocks = [Block(self.nodes[::2]), Block(self.nodes[1::2])]
            self.program = IsingSamplingProgram(self.model, free_blocks, clamped_blocks=[])

            samples, energies = self.run(
                n_warmup=n_warmup,
                n_samples=n_samples,
                steps_per_sample=steps_per_sample,
            )

            # Global magnetization per sample, then mean across samples.
            mags = jnp.mean(samples, axis=1)
            magnetizations.append(jnp.mean(mags))
            mean_energies.append(jnp.mean(energies))

        return jnp.stack(magnetizations), jnp.stack(mean_energies)


