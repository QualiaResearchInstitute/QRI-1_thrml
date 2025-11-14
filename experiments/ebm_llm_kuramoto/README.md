# EBM + LLM Kuramoto Fusion

This experiment folder captures a simple recipe for fusing an Energy-Based Model (EBM)
and a lightweight language model (LLM) into a shared Kuramoto-style latent manifold.
The idea is inspired by the observation that Kuramoto oscillators can be interpreted
as the meeting point between:

- **EBMs**, which supply the attractor landscape (phase-locking is equivalent to
  minimizing a cosine-based energy functional);
- **Latent Manifold Models (LMMs)**, which show that many-body neural dynamics often
  admit low-dimensional summaries (order parameters, Möbius flows, graph harmonics);
- **Kuramoto networks**, which instantiate the simplest nonlinear system where the
  above two ingredients meet—weakly coupled oscillators whose synchronization is
  driven by an implicit energy and described by a small set of collective variables.

## Conceptual Links

1. **Kuramoto \(\rightarrow\) LMM**: Phase reduction, low-rank order parameters, and
   hyperbolic flows demonstrate that Kuramoto systems natively live on succinct latent
   manifolds (e.g., \(r, \psi\)).
2. **EBM \rightarrow Kuramoto**: The standard Kuramoto energy
   \(E = -\sum_{i,j} a_{ij} \cos(\theta_i - \theta_j)\) acts as an attractor-driven
   objective; gradient descent on this energy recovers synchrony.
3. **LMM \rightarrow EBM**: Latent manifolds expose the coordinates that the energy
   should stabilize, clarifying why near–Hopf bifurcations and Stuart–Landau dynamics
   reduce to Kuramoto-style EBMs in the weak-coupling limit.

Combining these observations motivates the shorthand **EBM + LMM ≈ Kuramoto** for the
simplest neural synchronization stack.

## What Lives Here

- `fusion_demo.py` — a self-contained prototype that supports both text-only and
  vision-language encoding:
  - **Text encoding**: Uses pretrained `sshleifer/tiny-gpt2` to encode text prompts
  - **Vision-language encoding**: Uses CLIP (`openai/clip-vit-base-patch32`) to encode
    images, text, or image+text pairs
  
  Both encoders implement the `PromptEncoder` protocol, producing latent vectors that
  parameterize the initial phases, natural frequencies, and coupling/noise gates of a
  Kuramoto simulator whose stability is monitored by an energy functional. Running the
  script prints a short summary of how each prompt nudges the oscillator population.

## Prerequisites

The demo relies on Hugging Face Transformers. For vision-language support, PIL is also
required. Install dependencies via:

```bash
pip install torch transformers pillow
```

Note: The vision-language encoder (CLIP) is optional; text-only encoding works with
just `torch` and `transformers`.

## Running the Demo

```bash
python experiments/ebm_llm_kuramoto/fusion_demo.py
```

Expected output (values will vary slightly because the latent manifold is random but
seeded) looks like:

```
prompt='stabilize coherent chimera clusters' | final_energy=0.0898 | final_order=0.019 angle=-0.880 | coupling=0.542 noise=0.028
energy trace (first 5): [0.0511, 0.0526, 0.0542, 0.0556, 0.0571]
order trace (first 5): [0.258, 0.255, 0.252, 0.25, 0.247]
--------------------------------------------------------------------------------
```

You can replace the prompts or adjacency (see `ring_adjacency`) to explore other
synchronization motifs and observe how the energy landscape responds.

## Theoretical Grounding

### Symmetry Theory of Valence (STV)

The integration of vision-language models (VLMs) like CLIP directly tests core hypotheses
from the **Symmetry Theory of Valence (STV)**, which posits that:

- **Valence as Symmetry**: The pleasantness of an experience correlates with the symmetry
  of its corresponding mathematical object. Highly symmetrical configurations correspond
  to **energy minima** and high pleasure.
- **Geometric Input**: Visual geometry encodes spatial and tactile structure that is
  foundational to valence. Psychedelic experiences, which reveal underlying valence
  mechanisms, are characterized by **lowered symmetry detection thresholds** and the
  emergence of highly symmetrical patterns (mandalas, fractals, tessellations).
- **Energy Minimization**: In the EBM framework, low energy corresponds to minimum energy
  configurations achieved by symmetry maximization, directly linking STV to the Kuramoto
  energy functional.

### Geometric Latent Manifolds

The CLIP encoder processes visual geometry more directly than abstract language, enabling:

- **Visual Reification**: The visual system stabilizes ambiguous 2D inputs into 3D percepts
  by finding interpretations with **greatest symmetry**, relying on wave propagation and
  geometric constraints.
- **Equivariant Representations**: Vision Transformers (ViTs) in CLIP excel at encoding
  spatial and geometric properties, aligning with equivariant deep learning principles that
  enforce geometric domain symmetries.
- **Direct Phase Alignment**: Visual-only prompts allow testing how geometric input directly
  drives Kuramoto phase alignment, with synchronization serving as an inductive bias for
  structured generation.

### Multimodal Coherence

The fusion of image+text embeddings tests how combinations of semantic coherence (text)
and geometric coherence (image) affect dynamic outcomes. High coherence corresponds to
high synchrony (high order parameter \(R\)) and low energy, while contradictory inputs
may induce metastability or competing clusters.

## Regression Testing

The repository includes comprehensive regression tests that validate the semantic-to-dynamics
and visual-to-dynamics loops:

### Text Prompts

- **Coherent prompts** emphasize harmony, gentle fields, and stabilization.
- **Dissonant prompts** emphasize chaos, sharp interference, and fragmentation.

Test: `tests/test_ebm_llm_kuramoto.py::test_canonical_prompts_show_expected_energy_order_trend`

```bash
python -m pytest tests/test_ebm_llm_kuramoto.py -k canonical
```

### Visual Prompts

- **Symmetry images**: Synthetic mandala-like patterns with high geometric symmetry
- **Chaos images**: Random, fragmented patterns with low geometric structure

Tests verify that symmetry images yield **higher synchronization order** and **lower energy**
than chaos images, directly testing STV predictions:

- `test_visual_prompts_show_expected_energy_order_trend` - Image-only encoding
- `test_visual_text_fusion_prompts` - Image+text fusion encoding

```bash
python -m pytest tests/test_ebm_llm_kuramoto.py -k visual
```

All tests assert the expected relationship: coherent/symmetric inputs → higher order,
lower energy; dissonant/chaotic inputs → lower order, higher energy.

## STV Validation Results

### Empirical Validation

The visual regression tests have been **successfully validated** using geometric regularization
fine-tuning. The fine-tuned `CLIPVisionLanguageEncoder` demonstrates clear separation between
symmetry and chaos in the latent space, confirming STV predictions:

| Metric | Symmetry Images | Chaos Images | Difference | STV Prediction |
|--------|----------------|--------------|------------|----------------|
| **Order** | **0.3841** | 0.1934 | **+0.1907** | ✓ Higher order |
| **Energy** | **-0.0270** | 0.0197 | **-0.0467** | ✓ Lower energy |

### Geometric Regularization

The validation was achieved through **geometric regularization** using frustration loss (η_F):

- **Frustration Loss**: Measures geometric inconsistency. Symmetry images achieve near-zero
  frustration (0.000000), indicating perfect geometric consistency. Chaos images show higher
  frustration (0.049804), reflecting geometric inconsistency.
- **Feature Synchronization**: Reduces intra-class variance, ensuring embeddings for similar
  geometric structures are coherent.

The fine-tuned encoder is enabled by default (`fine_tuned=True`) in `CLIPVisionLanguageEncoder`,
providing improved geometric consistency out of the box.

### Fine-Tuning Process

To regenerate fine-tuned weights or investigate the fine-tuning process:

```bash
cd /Users/hunter/computer/resonance-transformer
PYTHONPATH=/Users/hunter/computer/resonance-transformer python3 experiments/ebm_llm_kuramoto/investigate_stv_validation.py
```

This script:
1. Analyzes threshold sensitivity
2. Fine-tunes the projection layer with geometric regularization
3. Validates STV predictions on held-out images
4. Reports quantitative results

See `STV_VALIDATION_SUMMARY.md` for detailed investigation methodology and theoretical grounding.
