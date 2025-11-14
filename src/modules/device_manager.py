"""
Device Manager for optimal ANE/MPS/CPU splitting on Apple Silicon.

This module provides intelligent device routing for different operations:
- ANE (via CoreML): Neural network operations (linear layers, embeddings)
- MPS: Large tensor operations that fit in memory
- CPU: Operations that exceed MPS memory limits or require control flow

ANE Support:
- Detects CoreML availability and ANE-capable hardware
- Provides utilities for hybrid PyTorch/CoreML workflows
- Recommends when to use CoreML export for ANE acceleration
"""

from __future__ import annotations

import os
import time
import platform
import contextlib
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:  # pragma: no cover
    from modules.telemetry import TelemetryWriter


class DeviceManager:
    """
    Manages optimal device placement for Apple Silicon devices.
    
    Strategy:
    - MPS: Default for most operations, but monitor memory usage
    - CPU: Fallback when MPS memory is exhausted or for specific operations
    - ANE: Used via CoreML export (not directly accessible in PyTorch)
    """
    
    def __init__(
        self,
        primary_device: Optional[torch.device] = None,
        mps_memory_limit_gb: float = 25.0,  # Conservative limit for MPS
        enable_auto_fallback: bool = True,
        enable_ane_detection: bool = True,
        telemetry: Optional["TelemetryWriter"] = None,
    ):
        """
        Initialize device manager.
        
        Args:
            primary_device: Primary device to use (auto-detects if None)
            mps_memory_limit_gb: Memory limit in GB for MPS before falling back to CPU
            enable_auto_fallback: Automatically fall back to CPU when MPS OOM
            enable_ane_detection: Detect ANE availability via CoreML
        """
        self.enable_auto_fallback = enable_auto_fallback
        self.mps_memory_limit_bytes = mps_memory_limit_gb * 1024 * 1024 * 1024
        
        # Detect available devices
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.has_cuda = torch.cuda.is_available()
        
        # Detect ANE availability (via CoreML)
        self.has_coreml = False
        self.has_ane = False
        self.is_apple_silicon = False
        
        if enable_ane_detection:
            self._detect_ane_availability()
        
        # Set primary device
        if primary_device is not None:
            self.primary_device = primary_device
        elif self.has_cuda:
            self.primary_device = torch.device("cuda")
        elif self.has_mps:
            self.primary_device = torch.device("mps")
        else:
            self.primary_device = torch.device("cpu")
        
        # CPU device (always available)
        self.cpu_device = torch.device("cpu")
        
        # Track MPS memory usage
        self._mps_allocated = 0.0
        
        # CoreML model cache (for hybrid workflows)
        self._coreml_models: Dict[str, Any] = {}

        # Telemetry hook (optional)
        self.telemetry = telemetry
    
    def _detect_ane_availability(self) -> None:
        """Detect if ANE is available via CoreML."""
        # Check if we're on Apple Silicon
        self.is_apple_silicon = platform.machine() == "arm64" and platform.system() == "Darwin"
        
        # Try to import CoreML
        try:
            import coremltools as ct  # type: ignore
            self.has_coreml = True
            
            # ANE is available on Apple Silicon devices (M1, M2, M3, etc.)
            # CoreML will automatically use ANE when available
            if self.is_apple_silicon:
                self.has_ane = True
        except ImportError:
            self.has_coreml = False
            self.has_ane = False
        
    def get_device_for_operation(
        self,
        operation_type: str,
        tensor_size_bytes: Optional[int] = None,
        requires_grad: bool = False,
    ) -> torch.device:
        """
        Get the best device for a given operation.
        
        Args:
            operation_type: Type of operation ('linear', 'embedding', 'matmul', 
                          'elementwise', 'iterative', 'large_tensor')
            tensor_size_bytes: Estimated size of tensors in bytes
            requires_grad: Whether gradients are needed
        
        Returns:
            Best device for the operation
        """
        trace_meta = {
            "operation_type": operation_type,
            "tensor_size_bytes": tensor_size_bytes,
            "requires_grad": bool(requires_grad),
        }
        with self._trace("device.select", device=str(self.primary_device), metadata=trace_meta):
            # For operations that require control flow or are iterative, prefer CPU
            if operation_type in ('iterative', 'control_flow'):
                selected = self.cpu_device
            # For very large tensors that might exceed MPS memory, use CPU
            elif tensor_size_bytes is not None and tensor_size_bytes > self.mps_memory_limit_bytes:
                selected = self.cpu_device
            # Check if MPS is available and not exhausted
            elif self.has_mps and self.primary_device.type == "mps":
                if tensor_size_bytes is not None:
                    estimated_total = self._mps_allocated + tensor_size_bytes
                    if estimated_total > self.mps_memory_limit_bytes:
                        selected = self.cpu_device
                    else:
                        selected = self.primary_device
                else:
                    selected = self.primary_device
            else:
                # Fallback to CPU
                selected = self.cpu_device

            trace_meta["selected_device"] = str(selected)
            return selected
    
    def get_device_for_tensor(self, tensor: torch.Tensor) -> torch.device:
        """Get the device of a tensor."""
        return tensor.device
    
    def should_use_cpu_for_large_operation(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        num_tensors: int = 1,
    ) -> bool:
        """
        Check if a large operation should use CPU instead of MPS.
        
        Args:
            shape: Shape of the tensor(s)
            dtype: Data type
            num_tensors: Number of tensors involved
        
        Returns:
            True if CPU should be used
        """
        if not self.has_mps or self.primary_device.type != "mps":
            return True
        
        # Calculate total memory needed
        element_size = torch.tensor(0, dtype=dtype).element_size()
        total_elements = num_tensors * torch.tensor(shape).prod().item()
        total_bytes = total_elements * element_size
        
        # Use CPU if exceeds limit
        return total_bytes > self.mps_memory_limit_bytes
    
    @contextmanager
    def auto_device_context(self, operation_type: str = "general"):
        """
        Context manager that automatically handles device placement and fallback.
        
        Usage:
            with device_manager.auto_device_context("large_tensor"):
                # Operations that might OOM on MPS
                result = large_operation()
        """
        try:
            yield self
        except RuntimeError as e:
            if self.enable_auto_fallback and "out of memory" in str(e).lower():
                # Clear MPS cache and retry on CPU
                if self.has_mps:
                    torch.mps.empty_cache()
                # The caller should retry on CPU
                raise RuntimeError(
                    f"MPS OOM detected. Consider using CPU device for this operation. "
                    f"Original error: {e}"
                ) from e
            raise
    
    def move_to_optimal_device(
        self,
        tensor: torch.Tensor,
        operation_type: str = "general",
    ) -> torch.Tensor:
        """
        Move tensor to optimal device based on operation type.
        
        Args:
            tensor: Input tensor
            operation_type: Type of operation
        
        Returns:
            Tensor on optimal device
        """
        optimal_device = self.get_device_for_operation(
            operation_type,
            tensor_size_bytes=tensor.numel() * tensor.element_size(),
        )
        
        if tensor.device != optimal_device:
            return tensor.to(optimal_device)
        return tensor
    
    def clear_mps_cache(self):
        """Clear MPS cache to free memory."""
        if self.has_mps:
            torch.mps.empty_cache()
            self._mps_allocated = 0.0
    
    def get_memory_info(self) -> dict:
        """Get memory information for available devices."""
        info = {
            "primary_device": str(self.primary_device),
            "has_mps": self.has_mps,
            "has_cuda": self.has_cuda,
            "has_coreml": self.has_coreml,
            "has_ane": self.has_ane,
            "is_apple_silicon": self.is_apple_silicon,
        }
        
        if self.has_mps:
            try:
                # MPS doesn't have direct memory query, but we can track allocations
                info["mps_estimated_allocated_gb"] = self._mps_allocated / (1024**3)
                info["mps_memory_limit_gb"] = self.mps_memory_limit_bytes / (1024**3)
            except Exception:
                pass
        
        if self.has_cuda:
            try:
                info["cuda_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                info["cuda_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            except Exception:
                pass
        
        return info
    
    def get_device_state_snapshot(self) -> Dict[str, Dict[str, float]]:
        """
        Capture lightweight device telemetry for contextual routing.
        
        Returns:
            Dict mapping device type -> telemetry (memory headroom, load, temperature estimate)
        """
        info = self.get_memory_info()
        snapshot: Dict[str, Dict[str, float]] = {}
        timestamp = time.time()
        
        def _headroom(allocated: float, limit: float) -> float:
            if limit <= 0:
                return 1.0
            return max(0.0, min(1.0, 1.0 - (allocated / limit)))
        
        # CPU telemetry (approximate using system load)
        cpu_entry: Dict[str, float] = {"memory_headroom": 1.0, "timestamp": timestamp}
        try:
            load1, _, _ = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            cpu_entry["load"] = min(1.0, load1 / cpu_count)
        except OSError:
            cpu_entry["load"] = 0.0
        cpu_entry["temperature_c"] = 0.0  # Placeholder (no direct sensor access)
        snapshot["cpu"] = cpu_entry
        
        if info.get("has_mps"):
            allocated = info.get("mps_estimated_allocated_gb", 0.0)
            limit = info.get("mps_memory_limit_gb", max(allocated, 1.0))
            mps_entry = {
                "memory_headroom": _headroom(allocated, limit),
                "load": 1.0 - _headroom(allocated, limit),
                "temperature_c": 0.0,
                "timestamp": timestamp,
            }
            snapshot["mps"] = mps_entry
            self._report_headroom("mps", mps_entry["memory_headroom"])
        
        if info.get("has_cuda"):
            try:
                device_index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_index)
                total_gb = props.total_memory / (1024 ** 3)
            except Exception:
                total_gb = info.get("cuda_reserved_gb", 0.0)
            allocated = info.get("cuda_allocated_gb", 0.0)
            headroom = _headroom(allocated, total_gb if total_gb else max(allocated, 1.0))
            cuda_entry = {
                "memory_headroom": headroom,
                "load": 1.0 - headroom,
                "temperature_c": 0.0,
                "timestamp": timestamp,
            }
            snapshot["cuda"] = cuda_entry
            self._report_headroom("cuda", headroom)
        
        if info.get("has_ane"):
            ane_entry = {
                "memory_headroom": 1.0,
                "load": 0.0,
                "temperature_c": 0.0,
                "timestamp": timestamp,
            }
            snapshot["ane"] = ane_entry
            self._report_headroom("ane", 1.0)
        
        return snapshot

    def _report_headroom(self, device: str, headroom: float) -> None:
        if self.telemetry is None:
            return
        try:
            self.telemetry.observe_device_headroom(device, headroom)
        except Exception:
            pass

    def _trace(self, op_name: str, *, device: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        if self.telemetry is None:
            return contextlib.nullcontext()
        return self.telemetry.trace_operation(op_name, device=device, metadata=metadata or {})
    
    def should_use_coreml_for_operation(
        self,
        operation_type: str,
        seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> bool:
        """
        Determine if an operation should use CoreML/ANE instead of PyTorch.
        
        Args:
            operation_type: Type of operation ('linear', 'embedding', 'attention', etc.)
            seq_len: Sequence length (for fixed-shape models)
            batch_size: Batch size
        
        Returns:
            True if CoreML/ANE should be used
        """
        if not self.has_coreml or not self.has_ane:
            return False
        
        # ANE is best for:
        # - Linear/fully-connected layers
        # - Embeddings
        # - Fixed-shape operations (CoreML requires fixed shapes)
        # - Inference workloads (not training)
        
        ane_friendly_ops = {'linear', 'embedding', 'matmul', 'conv', 'attention'}
        
        if operation_type.lower() not in ane_friendly_ops:
            return False
        
        # CoreML works best with fixed shapes
        # If we have fixed seq_len and batch_size, CoreML is a good option
        if seq_len is not None and batch_size is not None:
            # Prefer CoreML for common sequence lengths (buckets)
            common_buckets = {128, 256, 512, 1024, 2048}
            if seq_len in common_buckets:
                return True
        
        return False
    
    def recommend_coreml_export(
        self,
        operation_type: str,
        seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Provide recommendations for CoreML export.
        
        Returns:
            Dictionary with recommendations and export instructions
        """
        recommendation = {
            "should_export": False,
            "reason": "",
            "export_command": "",
            "benefits": [],
        }
        
        if not self.has_coreml or not self.has_ane:
            recommendation["reason"] = "CoreML/ANE not available"
            return recommendation
        
        should_use = self.should_use_coreml_for_operation(operation_type, seq_len, batch_size)
        
        if should_use:
            recommendation["should_export"] = True
            recommendation["reason"] = f"Operation '{operation_type}' would benefit from ANE acceleration"
            recommendation["benefits"] = [
                "Lower power consumption",
                "Faster inference for neural network operations",
                "Better performance for fixed-shape operations",
            ]
            
            if seq_len is not None:
                recommendation["export_command"] = (
                    f"python scripts/export_coreml.py "
                    f"--seq_len {seq_len} "
                    f"--outfile artifacts/coreml/T{seq_len}.mlpackage"
                )
        else:
            recommendation["reason"] = (
                f"Operation '{operation_type}' may not benefit from ANE, "
                f"or requires dynamic shapes (CoreML prefers fixed shapes)"
            )
        
        return recommendation
    
    def register_coreml_model(self, key: str, model_path: str) -> None:
        """
        Register a CoreML model for hybrid PyTorch/CoreML workflows.
        
        Args:
            key: Unique key to identify the model
            model_path: Path to the .mlpackage file
        """
        if not self.has_coreml:
            raise RuntimeError("CoreML not available. Cannot register CoreML model.")
        
        # Lazy load - just store the path for now
        # Actual loading would happen when needed (requires CoreML runtime)
        self._coreml_models[key] = {
            "path": model_path,
            "loaded": False,
            "model": None,
        }
    
    def get_coreml_model_info(self) -> Dict[str, Any]:
        """Get information about registered CoreML models."""
        return {
            "has_coreml": self.has_coreml,
            "has_ane": self.has_ane,
            "registered_models": list(self._coreml_models.keys()),
            "model_count": len(self._coreml_models),
        }


# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Get or create the global device manager."""
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager


def set_device_manager(manager: DeviceManager):
    """Set the global device manager."""
    global _global_device_manager
    _global_device_manager = manager


def get_optimal_device(
    operation_type: str = "general",
    tensor_size_bytes: Optional[int] = None,
) -> torch.device:
    """
    Convenience function to get optimal device for an operation.
    
    Args:
        operation_type: Type of operation
        tensor_size_bytes: Estimated tensor size in bytes
    
    Returns:
        Optimal device
    """
    return get_device_manager().get_device_for_operation(
        operation_type,
        tensor_size_bytes,
    )
