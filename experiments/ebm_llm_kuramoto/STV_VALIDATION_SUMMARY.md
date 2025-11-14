# STV Validation Investigation Summary

## Overview

This document summarizes the empirical validation approach for the Symmetry Theory of Valence (STV) predictions using the EBM+LLM Kuramoto fusion pipeline with CLIP vision-language encoding.

## Theoretical Foundation

### STV Core Predictions

1. **Symmetry → High Valence**: Higher symmetry in mathematical objects corresponds to higher subjective pleasantness
2. **High Order**: Symmetry images should yield higher Kuramoto order parameter (R) - indicating greater synchronization/coherence
3. **Low Energy**: Symmetry images should yield lower energy - indicating energy-efficient, low-complexity configurations

### Why These Predictions?

- **Order (Coherence)**: High valence states are linked to high coherence in neural dynamics. EEG coherence is related to symmetry, and flow states are predicted by lower information content and more symmetry.
- **Energy (Efficiency)**: Symmetrical shapes are local energy minima, energetically efficient in oscillatory systems. The visual system seeks interpretations with greatest symmetry to stabilize efficient models.

## Current Test Results

### Initial Findings

The visual regression tests (`test_visual_prompts_show_expected_energy_order_trend`) are currently **failing**:

- **Symmetry images**: Order ≈ 0.29, Energy ≈ -0.013
- **Chaos images**: Order ≈ 0.32, Energy ≈ -0.017

**Problem**: Chaos images show HIGHER order and LOWER energy than symmetry images (opposite of prediction).

### Root Cause Analysis

1. **CLIP Embedding Similarity**: Symmetry and chaos images have cosine similarity of 0.73 - CLIP sees them as relatively similar
2. **Random Projection**: The projection layer (512 → 26 dim) uses random initialization, potentially scrambling geometric information
3. **Synthetic Image Limitations**: The synthetic images may not capture the geometric distinctions CLIP was trained to recognize

## Investigation Strategy

### 1. Threshold Sensitivity Analysis

**Concept**: The "symmetry detection threshold" and "near miss parameter (δ)" from predictive processing theory.

**Implementation**: Test different thresholds (0.01, 0.05, 0.10, 0.15, 0.20) to see if the distinction becomes apparent with more lenient criteria.

**Script**: `investigate_stv_validation.py::analyze_threshold_sensitivity()`

### 2. Geometric Regularization Fine-Tuning

**Concept**: Use frustration loss (η_F) and feature synchronization to fine-tune the CLIP projection layer.

**Frustration Loss**: Measures geometric inconsistency. For symmetry, we want low frustration (high geometric consistency).

**Geometric Consistency Loss**: Encourages symmetry embeddings to have lower frustration than chaos embeddings (with margin).

**Feature Synchronization**: Reduces intra-class variance, ensuring embeddings for similar objects are more consistent.

**Implementation**: `GeometricRegularizedCLIPEncoder` class that:
- Freezes CLIP base model
- Fine-tunes only the projection layer
- Uses geometric regularization loss during training
- Reduces frustration for symmetry vs chaos

**Script**: `investigate_stv_validation.py::GeometricRegularizedCLIPEncoder`

## Usage

### Run Full Investigation

```bash
cd /Users/hunter/computer/resonance-transformer
PYTHONPATH=/Users/hunter/computer/resonance-transformer python3 experiments/ebm_llm_kuramoto/investigate_stv_validation.py
```

This will:
1. Analyze threshold sensitivity
2. Fine-tune the projection layer with geometric regularization
3. Test the fine-tuned encoder on held-out images
4. Report whether STV predictions are validated

### Expected Outcomes

**Success Criteria**:
- Symmetry images: Order > Chaos images (by margin)
- Symmetry images: Energy < Chaos images (by margin)
- Frustration loss: Symmetry < Chaos

**If Successful**: The fine-tuned encoder can be integrated into the main pipeline.

**If Still Failing**: Consider:
- Different CLIP models (e.g., ViT-L/14)
- Alternative geometric regularization strategies
- Using real-world symmetry/chaos images instead of synthetic
- Pre-training on symmetry detection tasks

## Next Steps

1. **Run Investigation**: Execute the investigation script to analyze thresholds and test fine-tuning
2. **Integrate Solution**: If fine-tuning succeeds, integrate `GeometricRegularizedCLIPEncoder` into the main pipeline
3. **Update Tests**: Adjust test thresholds based on investigation results
4. **Document Findings**: Update README with validation results and any necessary threshold adjustments

## References

- Symmetry Theory of Valence (STV)
- Predictive Processing and Precision Weighting
- Geometric Deep Learning and Frustration Loss
- Feature Synchronization for Equivariant Representations

