"""
Package shim for Core ML export utilities.

Re-exports the CLI-friendly functions from scripts/export_coreml.py so callers
can import:
  from resonance_transformer.export_coreml import export_coreml_resonance, export_coreml_buckets, ...

This keeps tests and downstream code working when treating `resonance_transformer` as a package.
"""

from __future__ import annotations

# Prefer importing from top-level 'scripts' package if available; otherwise
# fall back to loading the file by path, so this works even if 'scripts' isn't a package.
try:
    from scripts.export_coreml import (  # type: ignore
        export_coreml_resonance,
        export_coreml_buckets,
        export_coreml_constraints_only,
        export_coreml_constraints_buckets,
    )
except Exception:
    import importlib.util
    import importlib.machinery
    from pathlib import Path
    _export_path = Path(__file__).resolve().parent.parent / "scripts" / "export_coreml.py"
    spec = importlib.util.spec_from_file_location("_rt_export_coreml_impl", str(_export_path))
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Unable to locate Core ML export script at {_export_path}")
    _mod = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert isinstance(loader, importlib.machinery.SourceFileLoader)
    loader.exec_module(_mod)

    export_coreml_resonance = getattr(_mod, "export_coreml_resonance")
    export_coreml_buckets = getattr(_mod, "export_coreml_buckets")
    export_coreml_constraints_only = getattr(_mod, "export_coreml_constraints_only")
    export_coreml_constraints_buckets = getattr(_mod, "export_coreml_constraints_buckets")

__all__ = [
    "export_coreml_resonance",
    "export_coreml_buckets",
    "export_coreml_constraints_only",
    "export_coreml_constraints_buckets",
]
