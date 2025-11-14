"""
Genome Registry: Versioned, append-only storage for GPT-2 repair deltas (DNA).

Provides:
- Address schema: layer/head/band/target/param
- Delta types: lora, alpha, kernel_weight
- Content-addressed, versioned JSON storage
- Atomic writes with cross-platform file locking
- Schema validation (stdlib + optional jsonschema)
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


# Address schema: layer/head/band/target/param
@dataclass(frozen=True)
class Address:
    """Canonical address for a delta target in GPT-2."""
    layer: int
    param: Literal['lora', 'alpha', 'kernel_weight']
    head: Optional[int] = None  # None means '-' (all heads or matrix-level)
    band: Optional[int] = None  # None means '-' (not band-specific)
    target: Optional[Literal['q', 'k', 'v', 'o', 'mlp_in', 'mlp_out']] = None  # None means '-'
    
    def __post_init__(self):
        """Validate address constraints."""
        if self.param == 'alpha':
            if self.head is None:
                raise ValueError("alpha requires head != None")
            if self.band is not None or self.target is not None:
                raise ValueError("alpha requires band=None and target=None")
        elif self.param == 'lora':
            if self.target is None:
                raise ValueError("lora requires target in {'q','k','v','o','mlp_in','mlp_out'}")
            if self.band is not None:
                raise ValueError("lora requires band=None")
        elif self.param == 'kernel_weight':
            if self.band is None:
                raise ValueError("kernel_weight requires band != None")
            if self.target is not None:
                raise ValueError("kernel_weight requires target=None")
            if self.head is not None:
                raise ValueError("kernel_weight requires head=None (band-level)")
    
    def to_canonical(self) -> str:
        """Return canonical string: L{layer}/H{head or '-'}/B{band or '-'}/T{target or '-'}/{param}"""
        h = str(self.head) if self.head is not None else '-'
        b = str(self.band) if self.band is not None else '-'
        t = str(self.target) if self.target is not None else '-'
        return f"L{self.layer}/H{h}/B{b}/T{t}/{self.param}"
    
    @classmethod
    def from_canonical(cls, s: str) -> Address:
        """Parse canonical string back to Address."""
        # L{layer}/H{head}/B{band}/T{target}/{param}
        parts = s.split('/')
        if len(parts) != 5:
            raise ValueError(f"Invalid canonical address: {s}")
        layer = int(parts[0][1:])
        head_str = parts[1][1:]
        head = int(head_str) if head_str != '-' else None
        band_str = parts[2][1:]
        band = int(band_str) if band_str != '-' else None
        target_str = parts[3][1:]
        target = target_str if target_str != '-' else None
        param = parts[4]
        return cls(layer=layer, head=head, band=band, target=target, param=param)


@dataclass
class DeltaSpec:
    """Specification for a single delta."""
    address: Address
    type: Literal['lora', 'alpha', 'kernel_weight']
    payload: Dict[str, Any]  # Type-specific payload
    provenance: Dict[str, Any] = field(default_factory=dict)
    metrics_delta: Dict[str, float] = field(default_factory=dict)
    delta_id: Optional[str] = None  # Content-addressed hash
    
    def __post_init__(self):
        """Compute delta_id if not provided."""
        if self.delta_id is None:
            self.delta_id = self._compute_id()
    
    def _compute_id(self) -> str:
        """Compute content-addressed hash with deterministic serialization."""
        # Create canonical dict with sorted keys and quantized floats
        canonical = {
            'address': self.address.to_canonical(),
            'type': self.type,
            'payload': self._quantize_floats(self.payload),
        }
        json_str = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    @staticmethod
    def _quantize_floats(obj: Any, precision: int = 7) -> Any:
        """Recursively quantize floats to fixed precision for deterministic hashing."""
        if isinstance(obj, float):
            return round(obj, precision)
        elif isinstance(obj, dict):
            return {k: DeltaSpec._quantize_floats(v, precision) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DeltaSpec._quantize_floats(item, precision) for item in obj]
        else:
            return obj
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            'address': {
                'layer': self.address.layer,
                'param': self.address.param,
                'head': self.address.head,
                'band': self.address.band,
                'target': self.address.target,
            },
            'type': self.type,
            'payload': self.payload,
            'provenance': self.provenance,
            'metrics_delta': self.metrics_delta,
            'delta_id': self.delta_id,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DeltaSpec:
        """Deserialize from dict."""
        addr_dict = d['address']
        address = Address(
            layer=addr_dict['layer'],
            param=addr_dict['param'],
            head=addr_dict.get('head'),
            band=addr_dict.get('band'),
            target=addr_dict.get('target'),
        )
        return cls(
            address=address,
            type=d['type'],
            payload=d['payload'],
            provenance=d.get('provenance', {}),
            metrics_delta=d.get('metrics_delta', {}),
            delta_id=d.get('delta_id'),
        )


@dataclass
class Guardrails:
    """Guardrails for delta proposals."""
    max_norms: Dict[str, float] = field(default_factory=lambda: {
        'lora_fro_rel': 0.05,  # ||U@V||_F / ||W||_F
        'alpha_abs': 0.1,
        'kernel_weight_l2_rel': 0.05,
    })
    hysteresis: Dict[str, int] = field(default_factory=lambda: {
        'min_wins': 3,
        'window': 10,
        'cooldown_steps': 100,
    })
    collapse: Dict[str, Any] = field(default_factory=lambda: {
        'R_min': 0.2,
        'cdns_max': 0.25,
        'abort_on_nan': True,
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Guardrails:
        return cls(**d)


@dataclass
class GenomeVersionMeta:
    """Metadata for a genome version."""
    version: int
    created_at: str
    n_deltas: int
    summary_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class GenomeVersion:
    """A single versioned genome."""
    model_hash: str
    env: str
    version: int
    created_at: str
    guardrails: Guardrails
    deltas: List[DeltaSpec]
    summary_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            'model_hash': self.model_hash,
            'env': self.env,
            'version': self.version,
            'created_at': self.created_at,
            'guardrails': self.guardrails.to_dict(),
            'deltas': [d.to_dict() for d in self.deltas],
            'summary_metrics': self.summary_metrics,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GenomeVersion:
        """Deserialize from dict."""
        return cls(
            model_hash=d['model_hash'],
            env=d['env'],
            version=d['version'],
            created_at=d['created_at'],
            guardrails=Guardrails.from_dict(d['guardrails']),
            deltas=[DeltaSpec.from_dict(delta) for delta in d['deltas']],
            summary_metrics=d.get('summary_metrics', {}),
        )


class GenomeRegistry:
    """Versioned registry for GPT-2 repair deltas."""
    
    def __init__(self, model_hash: str, env: str, root_dir: Union[str, Path] = "artifacts/genome"):
        """
        Initialize registry.
        
        Args:
            model_hash: Stable hash of model config + shapes
            env: Environment identifier (e.g., 'dev', 'prod')
            root_dir: Root directory for genome storage
        """
        self.model_hash = model_hash
        self.env = env
        self.root_dir = Path(root_dir)
        self.genome_dir = self.root_dir / model_hash / env
        self.genome_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.genome_dir / "index.json"
        self._lock_path = self.genome_dir / ".lock"
    
    def _acquire_lock(self):
        """Acquire file lock (cross-platform)."""
        if platform.system() == 'Windows':
            # Windows: use msvcrt
            try:
                import msvcrt
                lock_file = open(self._lock_path, 'w')
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                return lock_file
            except ImportError:
                # Fallback: best-effort advisory lock file
                self._lock_path.touch()
                return open(self._lock_path, 'w')
        else:
            # POSIX: use fcntl
            lock_file = open(self._lock_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            return lock_file
    
    def _release_lock(self, lock_file):
        """Release file lock."""
        try:
            if platform.system() == 'Windows':
                try:
                    import msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except ImportError:
                    pass
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        except Exception:
            pass
    
    def _load_index(self) -> Dict[str, Any]:
        """Load index.json, creating if missing."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {
            'model_hash': self.model_hash,
            'env': self.env,
            'latest_version': 0,
            'versions': [],
        }
    
    def _save_index(self, index: Dict[str, Any]):
        """Save index.json atomically."""
        tmp_path = self.index_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(index, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.index_path)
        # Try to fsync directory (best-effort)
        try:
            dir_fd = os.open(str(self.genome_dir), os.O_RDONLY)
            os.fsync(dir_fd)
            os.close(dir_fd)
        except Exception:
            pass
    
    def get_latest(self) -> Optional[GenomeVersion]:
        """Get latest genome version."""
        index = self._load_index()
        latest_version = index.get('latest_version', 0)
        if latest_version == 0:
            return None
        return self._load_version(latest_version)
    
    def _load_version(self, version: int) -> GenomeVersion:
        """Load a specific version."""
        version_path = self.genome_dir / f"dna_v{version}.json"
        if not version_path.exists():
            raise ValueError(f"Version {version} not found")
        with open(version_path, 'r') as f:
            data = json.load(f)
        return GenomeVersion.from_dict(data)
    
    def add_version(
        self,
        deltas: List[DeltaSpec],
        guardrails: Guardrails,
        metrics: Optional[Dict[str, float]] = None,
    ) -> GenomeVersion:
        """
        Add a new version atomically.
        
        Args:
            deltas: List of delta specs to add
            guardrails: Guardrails used for this version
            metrics: Summary metrics (optional)
        
        Returns:
            The new GenomeVersion
        """
        lock_file = self._acquire_lock()
        try:
            index = self._load_index()
            latest_version = index.get('latest_version', 0)
            new_version = latest_version + 1
            
            genome = GenomeVersion(
                model_hash=self.model_hash,
                env=self.env,
                version=new_version,
                created_at=datetime.utcnow().isoformat() + 'Z',
                guardrails=guardrails,
                deltas=deltas,
                summary_metrics=metrics or {},
            )
            
            # Validate schema
            self.validate_schema(genome.to_dict())
            
            # Write version file atomically
            version_path = self.genome_dir / f"dna_v{new_version}.json"
            tmp_path = version_path.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(genome.to_dict(), f, indent=2, sort_keys=True)
            os.replace(tmp_path, version_path)
            
            # Update index
            index['latest_version'] = new_version
            meta = GenomeVersionMeta(
                version=new_version,
                created_at=genome.created_at,
                n_deltas=len(deltas),
                summary_metrics=genome.summary_metrics,
            )
            index['versions'] = index.get('versions', [])
            index['versions'].append({
                'version': meta.version,
                'created_at': meta.created_at,
                'n_deltas': meta.n_deltas,
                'summary_metrics': meta.summary_metrics,
            })
            self._save_index(index)
            
            return genome
        finally:
            self._release_lock(lock_file)
    
    def resolve(self, address: Address) -> Optional[DeltaSpec]:
        """
        Resolve the latest effective delta for an address.
        
        Returns the most recent delta matching the address, or None.
        """
        latest = self.get_latest()
        if latest is None:
            return None
        
        # Find matching delta (most recent wins)
        matching = None
        for delta in reversed(latest.deltas):
            if (delta.address.layer == address.layer and
                delta.address.head == address.head and
                delta.address.band == address.band and
                delta.address.target == address.target and
                delta.address.param == address.param):
                matching = delta
                break
        
        return matching
    
    def list_versions(self) -> List[GenomeVersionMeta]:
        """List all version metadata."""
        index = self._load_index()
        versions = []
        for vmeta in index.get('versions', []):
            versions.append(GenomeVersionMeta(
                version=vmeta['version'],
                created_at=vmeta['created_at'],
                n_deltas=vmeta['n_deltas'],
                summary_metrics=vmeta.get('summary_metrics', {}),
            ))
        return versions
    
    def validate_schema(self, obj: Dict[str, Any]) -> None:
        """
        Validate schema (stdlib checks + optional jsonschema).
        
        Raises ValueError on invalid schema.
        """
        # Basic stdlib checks
        required_fields = ['model_hash', 'env', 'version', 'created_at', 'guardrails', 'deltas']
        for field in required_fields:
            if field not in obj:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(obj['deltas'], list):
            raise ValueError("deltas must be a list")
        
        for delta in obj['deltas']:
            if 'address' not in delta or 'type' not in delta:
                raise ValueError("Delta missing address or type")
            if delta['type'] not in ['lora', 'alpha', 'kernel_weight']:
                raise ValueError(f"Invalid delta type: {delta['type']}")
        
        # Optional jsonschema validation
        if JSONSCHEMA_AVAILABLE:
            schema = self._get_json_schema()
            try:
                jsonschema.validate(obj, schema)
            except jsonschema.ValidationError as e:
                raise ValueError(f"Schema validation failed: {e.message}") from e
    
    @staticmethod
    def _get_json_schema() -> Dict[str, Any]:
        """Return JSON schema for genome version."""
        return {
            "type": "object",
            "required": ["model_hash", "env", "version", "created_at", "guardrails", "deltas"],
            "properties": {
                "model_hash": {"type": "string"},
                "env": {"type": "string"},
                "version": {"type": "integer", "minimum": 1},
                "created_at": {"type": "string"},
                "guardrails": {"type": "object"},
                "deltas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["address", "type", "payload"],
                        "properties": {
                            "address": {"type": "object"},
                            "type": {"type": "string", "enum": ["lora", "alpha", "kernel_weight"]},
                            "payload": {"type": "object"},
                        },
                    },
                },
            },
        }


def compute_model_hash(model) -> str:
    """
    Compute stable hash of model config + shapes (excludes weights).
    
    Args:
        model: HuggingFace GPT-2 model
    
    Returns:
        Hex digest of model hash
    """
    import torch
    
    # Collect config + shape info
    config = getattr(model, 'config', None)
    if config is None:
        raise ValueError("Model missing config")
    
    # Hash config dict (sorted keys)
    config_dict = {k: v for k, v in config.to_dict().items() if k != 'torch_dtype'}
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    
    # Hash parameter shapes and dtypes
    shapes_info = []
    for name, param in model.named_parameters():
        shapes_info.append(f"{name}:{tuple(param.shape)}:{str(param.dtype)}")
    shapes_str = '\n'.join(sorted(shapes_info))
    
    # Combine and hash
    combined = f"{config_str}\n{shapes_str}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]  # 16 chars for brevity


__all__ = [
    'Address',
    'DeltaSpec',
    'Guardrails',
    'GenomeVersion',
    'GenomeVersionMeta',
    'GenomeRegistry',
    'compute_model_hash',
]


