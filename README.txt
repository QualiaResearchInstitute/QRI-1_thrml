Tactile Visualizer + Thermal Reservoir + Resonance Transformer
==============================================================

This folder is a self-contained package intended to become the root of a
shared GitHub repo for collaborators. It contains:

- core resonance-transformer code (`src/`),
- STV/EBM + thermal reservoir experiments (`experiments/`),
- oscilleditor + WebGPU reference visual frontends (`viz/`),
- theory + engineering specs (`docs/`),
- and basic tests (`tests/`).

Quick Start
-----------
1. Create a virtualenv and install dependencies:

   ```bash
   pip install -r env/requirements.txt
   ```

2. Run tests (optional but recommended):

   ```bash
   pytest
   ```

3. Resonance Transformer demo (attention head + full model):

   ```bash
   export PYTHONPATH=src
   python scripts/resonance_transformer_demo.py
   ```

4. Thermal reservoir demo:

   ```bash
   python -m experiments.thermal_reservoir.controller_loop
   ```

5. STV / EBM experiments:
   See `experiments/ebm_llm_kuramoto/README.md` and `RESEARCH_ROADMAP.md`.

6. Visual frontends (oscilleditor / WebGPU reference):
   - `viz/oscilleditor/index.html` can be opened in a browser.
   - `viz/reference_webgpu/README.txt` contains integration notes for WebGPU.

Directory Layout
----------------
- `env/requirements.txt`:
  Python dependencies for this package (copied from the main project).

- `src/resonance_transformer/`:
  Core resonance transformer library code.

- `src/modules/`:
  Supporting modules used by resonance transformer and experiments.

- `experiments/ebm_llm_kuramoto/`:
  Symmetry Theory of Valence (STV) and EBM-driven experiments.

- `experiments/thermal_reservoir/`:
  Thermal reservoir experiments built on Extropic's `thrml` library.

- `viz/oscilleditor/`:
  Oscilleditor prototype for image-based oscillation/kernel editing.

- `viz/reference_webgpu/`:
  Replication Realtime WebGPU skeleton and notes for GPU visual integration.

- `docs/`:
  - `tactile_visualizer_implementation_plan.txt`: engineering spec for the tactile visualizer stack.
  - `tactile_visualizer_theory_alignment.txt`: mapping between QRI theory and module structure.
  - `CONSCIOUSNESS_PHYSICS.md`, `EXTREME_RECURSION_RESONANCE.md`: broader theoretical context.

- `tests/`:
  Test suite copied from the main project; use as a starting point for validation.


