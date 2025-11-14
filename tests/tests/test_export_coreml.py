#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
import shutil
import pytest

try:
    import coremltools as ct  # type: ignore
    COREML_AVAILABLE = True
except Exception:
    COREML_AVAILABLE = False

try:
    import torch  # noqa: F401
except Exception:
    pytest.skip("PyTorch not available; skipping Core ML export test", allow_module_level=True)

try:
    # Ensure shim import path works
    from resonance_transformer import ResonanceTransformer  # noqa: F401
except Exception:
    pytest.skip("resonance_transformer import failed; skipping Core ML export test", allow_module_level=True)


@pytest.mark.skipif(not COREML_AVAILABLE, reason="coremltools not installed")
def test_coreml_single_bucket_export():
    """
    Smoke test: run export for a tiny fixed seq_len to ensure ct.convert completes.
    Uses seq_len=8 to keep conversion fast.
    """
    from resonance_transformer.export_coreml import export_coreml_resonance  # type: ignore

    tmpdir = tempfile.mkdtemp(prefix="rt_coreml_")
    try:
        outfile = os.path.join(tmpdir, "T8.mlpackage")
        export_coreml_resonance(
            seq_len=8,
            vocab_size=256,
            d_model=64,
            n_layers=1,
            n_heads=4,
            n_sim_steps=5,
            outfile=outfile,
        )
        assert os.path.exists(outfile), "Expected .mlpackage not found after export"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not COREML_AVAILABLE, reason="coremltools not installed")
def test_coreml_multi_bucket_export():
    """
    Smoke test: export two tiny buckets to ensure loop works.
    """
    from resonance_transformer.export_coreml import export_coreml_buckets  # type: ignore

    tmpdir = tempfile.mkdtemp(prefix="rt_coreml_buckets_")
    try:
        export_coreml_buckets(
            buckets=[8, 12],
            vocab_size=128,
            d_model=64,
            n_layers=1,
            n_heads=4,
            n_sim_steps=4,
            outdir=tmpdir,
        )
        for T in (8, 12):
            path = os.path.join(tmpdir, f"T{T}.mlpackage")
            assert os.path.exists(path), f"Missing bucket export: {path}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
