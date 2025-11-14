"""
Package shim for resonance_transformer

- Re-exports core classes from the monolithic resonance_transformer.py module
- Provides a package namespace so resonance_transformer.export_coreml imports work
"""

from __future__ import annotations

import importlib.util
import importlib.machinery
from pathlib import Path
from types import ModuleType
from typing import Any

# Load the monolithic implementation file (resonance_transformer.py) alongside this package
# In the standalone collab_package repo, the implementation lives next to this __init__.py.
_impl_path = Path(__file__).resolve().parent / "resonance_transformer.py"

def _load_impl_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_resonance_transformer_impl", str(_impl_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec to load implementation from {_impl_path}")
    mod = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert isinstance(loader, importlib.machinery.SourceFileLoader)
    loader.exec_module(mod)
    return mod

_impl = _load_impl_module()

# Re-export commonly used classes/symbols from the implementation
# Add more symbols here if tests or downstream code require them.
ResonanceAttentionHead = getattr(_impl, "ResonanceAttentionHead", None)  # type: ignore
ResonanceTransformerBlock = getattr(_impl, "ResonanceTransformerBlock", None)  # type: ignore
ResonanceTransformer = getattr(_impl, "ResonanceTransformer", None)  # type: ignore
CriticalCouplingTuner = getattr(_impl, "CriticalCouplingTuner", None)  # type: ignore
CouplingKernel = getattr(_impl, "CouplingKernel", None)  # type: ignore
hodge_decompose_coupling = getattr(_impl, "hodge_decompose_coupling", None)  # type: ignore

# Expose anything else explicitly if needed
__all__ = [
    "ResonanceAttentionHead",
    "ResonanceTransformerBlock",
    "ResonanceTransformer",
    "CriticalCouplingTuner",
    "CouplingKernel",
    "hodge_decompose_coupling",
]
